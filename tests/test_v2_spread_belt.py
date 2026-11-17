from types import SimpleNamespace

import math
import numpy as np
import pytest

from orogen.v2.tectonics.crust import BeltSide, CrustFlag, CrustRaster, CrustType, OrogenyType
from orogen.v2.tectonics.spread_belt import spread_belt


def fixture(types=None):
    n = 5
    crust = CrustRaster(n)
    types = types or [CrustType["CONTINENT"]] * n
    for i, kind in enumerate(types):
        crust.create(i, kind, 0.0)
    theta = np.arange(n, dtype=float) * 0.01
    xyz = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(n))).astype(np.float32).ravel()
    offsets = np.array([0, 1, 3, 5, 7, 8], dtype=np.int32)
    adjacency = np.array([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int32)
    mesh = SimpleNamespace(xyz=xyz, adj_offset=offsets, adj_list=adjacency, edge_km=100.0)
    plate = SimpleNamespace(crust=crust)
    config = {
        "apronKm": 100.0, "cratonRimKm": 200.0, "cratonRimCost": 2.0,
        "cratonRimShare": 0.25, "weakCellCost": 0.5, "orogenyQuietMyr": 10.0,
        "beltWorkShare": 0.5, "thickeningPerWorkKm": 0.1,
    }
    belt = {
        "type": OrogenyType["ANDEAN"], "widthKm": 1000.0, "workKm": 20.0,
        "time": 5.0, "side": BeltSide["UPPER"], "shear": 15,
    }
    return plate, mesh, config, belt


def test_spread_belt_records_thickens_and_builds_relief():
    plate, mesh, config, belt = fixture()
    reached = spread_belt(plate, mesh, [0], belt, config)
    assert reached == [0, 1, 2, 3, 4]
    assert np.all(plate.crust.oroType == OrogenyType["ANDEAN"])
    assert plate.crust.beltSide[0] == BeltSide["UPPER"]
    assert plate.crust.beltShear[0] == 15
    assert plate.crust.thickKm[0] > plate.crust.thickKm[-1] >= 0
    assert plate.crust.rootKm[0] == pytest.approx(plate.crust.thickKm[0] * 1.25)
    assert plate.crust.upliftKm[0] > 0
    assert plate.crust.rugged[0] > 0


def test_spread_belt_restricts_cells_and_supports_no_root_keep_shear():
    plate, mesh, config, belt = fixture()
    plate.crust.beltShear[0] = 42
    belt.update({"noRoot": True, "keepShear": True})
    reached = spread_belt(plate, mesh, [0], belt, config, allow=lambda cell: cell != 2)
    assert reached == [0, 1]
    assert plate.crust.thickKm.sum() == 0
    assert plate.crust.rootKm.sum() == 0
    assert plate.crust.beltShear[0] == 42
    assert plate.crust.oroType[3] == OrogenyType["NONE"]


def test_spread_belt_handles_weak_cells_and_craton_rim():
    plate, mesh, config, belt = fixture(
        [CrustType["CONTINENT"], CrustType["CRATON"], CrustType["CONTINENT"],
         CrustType["CONTINENT"], CrustType["CONTINENT"]]
    )
    plate.crust.flags[2] |= CrustFlag["SUTURE"]
    reached = spread_belt(plate, mesh, [0], belt, config)
    assert reached[0] == 0
    assert 1 in reached
    assert plate.crust.thickKm[1] == 0
    assert plate.crust.oroWork[1] < plate.crust.oroWork[0]


def test_spread_belt_preserves_previous_belt_on_escalation():
    plate, mesh, config, belt = fixture()
    c = plate.crust
    c.record_orogeny(0, OrogenyType["ANDEAN"], 3, 1, config["orogenyQuietMyr"])
    c.beltReach[0] = 100
    c.beltU[0] = 0.2
    c.beltSide[0] = BeltSide["LOWER"]
    c.beltShear[0] = -20
    c.rugged[0] = 2
    belt["type"] = OrogenyType["HIMALAYAN"]
    spread_belt(plate, mesh, [0], belt, config)
    assert c.prevType[0] == OrogenyType["ANDEAN"]
    assert c.prevSide[0] == BeltSide["LOWER"]
    assert c.prevShear[0] == -20
    assert c.prevRugged[0] == 2
    assert 0 < c.rugged[0] < c.prevRugged[0] + 1


def test_spread_belt_traverses_apron_from_ocean_to_land_only():
    plate, mesh, config, belt = fixture(
        [CrustType["OCEAN"], CrustType["OCEAN"], CrustType["CONTINENT"],
         CrustType["OCEAN"], CrustType["CONTINENT"]]
    )
    plate.crust.mask[4] = 0
    belt["widthKm"] = 500.0
    reached = spread_belt(plate, mesh, [0, 1], belt, config)
    assert reached == [2]
    assert plate.crust.oroType[2] == OrogenyType["ANDEAN"]
    assert plate.crust.oroType[3] == OrogenyType["NONE"]


def test_spread_belt_rejects_coastal_ocean_when_front_is_on_land():
    plate, mesh, config, belt = fixture(
        [CrustType["CONTINENT"], CrustType["OCEAN"], CrustType["CONTINENT"],
         CrustType["CONTINENT"], CrustType["CONTINENT"]]
    )
    assert spread_belt(plate, mesh, [0], belt, config) == [0]


def test_spread_belt_stale_heap_entry_and_craton_limit():
    plate, mesh, config, belt = fixture()
    angles = np.deg2rad([0.0, -1.0, 8.0, 10.0, 11.0])
    xyz = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(5))).astype(np.float32).ravel()
    mesh = SimpleNamespace(
        xyz=xyz,
        adj_offset=np.array([0, 2, 4, 6, 8, 8], dtype=np.int32),
        adj_list=np.array([1, 2, 0, 3, 0, 3, 1, 2], dtype=np.int32),
        edge_km=100.0,
    )
    # There are two routes to cell 3: the first enqueues a longer tentative path,
    # and the second improves it, leaving a stale heap record to be discarded.
    belt["widthKm"] = 2000.0
    assert spread_belt(plate, mesh, [0], belt, config) == [0, 1, 2, 3]
    plate, mesh, config, belt = fixture(
        [CrustType["CONTINENT"], CrustType["CRATON"], CrustType["CONTINENT"],
         CrustType["CONTINENT"], CrustType["CONTINENT"]]
    )
    config["cratonRimKm"] = 1.0
    assert spread_belt(plate, mesh, [0], belt, config) == [0]


def test_spread_belt_excludes_craton_source_without_rim():
    plate, mesh, config, belt = fixture(
        [CrustType["CRATON"], CrustType["OCEAN"], CrustType["OCEAN"],
         CrustType["OCEAN"], CrustType["OCEAN"]]
    )
    config["cratonRimKm"] = 0.0
    plate.crust.mask[1:] = 0
    assert spread_belt(plate, mesh, [0], belt, config) == []


def test_spread_belt_thickens_existing_sedimentary_cover():
    plate, mesh, config, belt = fixture()
    plate.crust.coverKm[0] = 2.0
    config["geology"] = {"records": True, "crustKm": 35.0}
    spread_belt(plate, mesh, [0], belt, config)
    assert plate.crust.coverKm[0] > 2.0
    assert plate.crust.coverMaxAt[0] == pytest.approx(5.0)
    assert plate.crust.unroofKm[0] < 0


def test_spread_belt_validates_mesh_and_source_and_ignores_empty_work():
    plate, mesh, config, belt = fixture()
    assert spread_belt(plate, mesh, [], belt, config) == []
    wrong = SimpleNamespace(xyz=np.zeros(3), adj_offset=mesh.adj_offset,
                            adj_list=mesh.adj_list, edge_km=mesh.edge_km)
    with pytest.raises(ValueError):
        spread_belt(plate, wrong, [0], belt, config)
    with pytest.raises(IndexError):
        spread_belt(plate, mesh, [99], belt, config)
    belt["works"] = [0]
    assert spread_belt(plate, mesh, [0], belt, config) == []


def test_spread_belt_uses_source_specific_width_work_shear_and_amplitude():
    plate, mesh, config, belt = fixture()
    belt.update({
        "widths": [130.0, 500.0],
        "works": [3.0, 40.0],
        "shears": [10.0, -20.0],
        "amps": [0.5, 1.2],
    })
    reached = spread_belt(plate, mesh, [0, 4], belt, config)
    assert reached == [0, 1, 2, 3, 4]
    assert plate.crust.beltReach[0] == pytest.approx(130.0)
    assert plate.crust.beltReach[1] == pytest.approx(500.0)
    assert plate.crust.beltShear[1] == -20
    assert plate.crust.oroWork[4] == pytest.approx(40.0)


def test_spread_belt_stops_when_ocean_apron_exceeds_limit():
    plate, mesh, config, belt = fixture(
        [CrustType["OCEAN"], CrustType["OCEAN"], CrustType["OCEAN"],
         CrustType["CONTINENT"], CrustType["CONTINENT"]]
    )
    config["apronKm"] = 10.0
    mesh.edge_km = 0.0
    assert spread_belt(plate, mesh, [0], belt, config) == []
