import math
from types import SimpleNamespace

import numpy as np

from orogen.v2.tectonics.crust import BeltSide, CrustFlag, CrustRaster, CrustType
from orogen.v2.tectonics.ops import plate_accretion
from orogen.v2.tectonics.ops.plate_accretion import (
    accrete_terrane,
    close_seas_between,
    landmass_of,
)
from orogen.v2.tectonics.plate import Plate


class Index:
    def __init__(self, points, far_index=0):
        self.points = np.asarray(points, dtype=float)
        self.far_index = int(far_index)

    def nearest(self, x, y, z):
        point = np.asarray([x, y, z], dtype=float)
        distance = np.sum((self.points - point) ** 2, axis=1)
        nearest = int(np.argmin(distance))
        return nearest if distance[nearest] < 1e-6 else self.far_index


class Mesh:
    def __init__(self, points, neighbours, edge_km=100.0, cell_area=100.0):
        self.num_cells = len(points)
        self.xyz = np.asarray(points, dtype=np.float32).reshape(-1)
        self.edge_km = float(edge_km)
        self.cell_area_km2 = float(cell_area)
        self.edge_rad = self.edge_km / 6371.0
        self.index = Index(points)
        offsets = [0]
        adjacency = []
        for row in neighbours:
            adjacency.extend(row)
            offsets.append(len(adjacency))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)


def close_fixture(points, neighbours):
    mesh = Mesh(points, neighbours)
    crust = __import__(
        "orogen.v2.tectonics.crust", fromlist=["CrustRaster"]
    ).CrustRaster(mesh.num_cells)
    crust.create(0, CrustType["CONTINENT"], -100.0, 10)
    crust.create(mesh.num_cells - 1, CrustType["CONTINENT"], -200.0, 20)
    origin = np.zeros(mesh.num_cells, dtype=np.int8)
    origin[0] = 1
    origin[-1] = 2
    upper = Plate(0, mesh.num_cells)
    upper.capCenter = [1.0, 0.0, 0.0]
    cfg = {
        "sealStraitKm": 100.0,
        "inlandSeaMaxKm2": 0.0,
    }
    return mesh, crust, origin, upper, cfg


def accretion_fixture():
    points = [
        [1.0, 0.0, 0.0],
        [0.5, math.sqrt(0.75), 0.0],
        [-0.5, math.sqrt(0.75), 0.0],
    ]
    mesh = Mesh(points, [[1], [0, 2], [1]])
    host = Plate(0, 3)
    donor = Plate(1, 3)
    host.crust.create(0, CrustType["CONTINENT"], -100.0, 10)
    donor.crust.create(1, CrustType["OCEAN"], -50.0, 21)
    donor.crust.create(2, CrustType["CONTINENT"], -80.0, 30)
    donor.crust.flags[1] = CrustFlag["TRENCH"]
    world = SimpleNamespace(
        mesh=mesh,
        config={
            "terraneForeArcKm": 100.0,
            "terraneMinAreaKm2": 100.0,
            "sealStraitKm": 100.0,
            "inlandSeaMaxKm2": 0.0,
            "belts": {"ural": {"baseKm": 120.0, "perKm": 1.0, "maxKm": 400.0}},
            "beltSpeedFactorMax": 1.6,
            "beltSpeedFactorMin": 0.6,
            "beltSpeedRefCmYr": 5.0,
            "beltSpeedExponent": 0.5,
        },
        time=12.0,
    )
    return world, host, donor


def test_close_seas_between_fills_a_narrow_strait_and_marks_suture():
    mesh, crust, origin, upper, cfg = close_fixture(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0]],
        [[1], [0, 2], [1]],
    )
    crust.create(1, CrustType["OCEAN"], -20.0, 99)

    sealed = close_seas_between(crust, origin, mesh, upper, cfg, 5.0)

    assert sealed == 1
    assert crust.type[1] == CrustType["CONTINENT"]
    assert crust.formedAt[1] == 5.0
    assert crust.flags[1] == CrustFlag["SUTURE"]
    assert crust.slabKm[1] == 0.0
    assert origin[1] == 3


def test_close_seas_between_keeps_wide_open_water_and_large_inland_seas():
    mesh, crust, origin, upper, cfg = close_fixture(
        [
            [1, 0, 0],
            [0.5, 0.866, 0],
            [-0.5, 0.866, 0],
            [-1, 0, 0],
        ],
        [[1], [0, 2], [1, 3], [2]],
    )
    crust.create(1, CrustType["OCEAN"], -20.0, 99)
    crust.create(2, CrustType["OCEAN"], -20.0, 98)

    assert close_seas_between(crust, origin, mesh, upper, cfg, 5.0) == 0
    assert crust.type[1] == CrustType["OCEAN"]
    assert crust.type[2] == CrustType["OCEAN"]

    cfg["inlandSeaMaxKm2"] = 1000.0
    # Make the far-side seed an ocean cell; it remains open even when small.
    upper.capCenter = [-0.5, -0.866, 0.0]
    assert close_seas_between(crust, origin, mesh, upper, cfg, 5.0) == 0


def test_close_seas_between_fills_a_small_enclosed_sea():
    mesh, crust, origin, upper, cfg = close_fixture(
        [
            [1, 0, 0],
            [0.5, 0.866, 0],
            [0, 1, 0],
            [-0.5, 0.866, 0],
            [-1, 0, 0],
        ],
        [[1], [0, 2], [1, 3], [2, 4], [3]],
    )
    crust.create(1, CrustType["OCEAN"], -20.0, 99)
    crust.create(2, CrustType["OCEAN"], -20.0, 98)
    crust.create(3, CrustType["OCEAN"], -20.0, 97)
    # No outside ocean is reachable from the far-side land seed.
    cfg["inlandSeaMaxKm2"] = 500.0

    sealed = close_seas_between(crust, origin, mesh, upper, cfg, 7.0)

    assert sealed == 3
    assert all(crust.type[cell] == CrustType["CONTINENT"] for cell in (1, 2, 3))
    assert all(origin[cell] == 3 for cell in (1, 2, 3))


def test_accrete_terrane_moves_land_and_forearc_closes_the_sea(monkeypatch):
    world, host, donor = accretion_fixture()
    belt_calls = []
    monkeypatch.setattr(
        plate_accretion,
        "spread_belt",
        lambda plate, mesh, sources, belt, cfg, allow: belt_calls.append(
            (plate.id, list(sources), belt["side"])
        ),
    )

    result = accrete_terrane(world, host, donor, [2], rate_cm_yr=5.0)

    assert result == {"cells": 1, "sealed": 1, "absorbed": 0}
    assert donor.crust.count == 0
    assert donor.alive is False
    assert host.crust.type.tolist() == [
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
    ]
    assert host.crust.flags[1] & CrustFlag["SUTURE"]
    assert host.crust.flags[2] & CrustFlag["SUTURE"]
    assert host.crust.terrane[2] == 30
    assert belt_calls == [
        (host.id, [0], BeltSide["UPPER"]),
        (host.id, [1, 2], BeltSide["LOWER"]),
    ]
    assert host.stats["cells"] == 3
    assert donor.stats["cells"] == 0


def test_accrete_terrane_absorbs_land_when_no_neighbour_is_free(monkeypatch):
    mesh = Mesh(
        [[1, 0, 0], [-1, 0, 0]],
        [[1], [0]],
    )
    host = Plate(0, 2)
    donor = Plate(1, 2)
    host.crust.create(0, CrustType["CONTINENT"], -100.0, 10)
    host.crust.create(1, CrustType["CRATON"], -200.0, 20)
    donor.crust.create(0, CrustType["CONTINENT"], -50.0, 30)
    world = SimpleNamespace(
        mesh=mesh,
        config={
            "terraneForeArcKm": 100.0,
            "terraneMinAreaKm2": 100.0,
            "sealStraitKm": 100.0,
            "inlandSeaMaxKm2": 0.0,
            "belts": {"ural": {"baseKm": 100, "perKm": 1, "maxKm": 300}},
            "beltSpeedFactorMax": 1.6,
            "beltSpeedFactorMin": 0.6,
            "beltSpeedRefCmYr": 5.0,
            "beltSpeedExponent": 0.5,
        },
        time=10.0,
    )
    monkeypatch.setattr(plate_accretion, "spread_belt", lambda *_args: None)

    result = accrete_terrane(world, host, donor, [0], 5.0)

    assert result == {"cells": 0, "sealed": 0, "absorbed": 1}
    assert donor.alive is False
    assert host.crust.terrane[0] == 10


def test_accrete_terrane_marks_small_fragment_as_scrap(monkeypatch):
    world, host, donor = accretion_fixture()
    world.config["terraneMinAreaKm2"] = 101.0
    monkeypatch.setattr(
        plate_accretion,
        "spread_belt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no belt for scrap")
        ),
    )

    result = accrete_terrane(world, host, donor, [2], 5.0)

    assert result["scrap"] is True
    assert result["sealed"] == 0
    assert host.crust.mask[2] == 1


def test_landmass_of_returns_connected_land_containing_all_land_seeds():
    world, host, _donor = accretion_fixture()
    host.crust.create(1, CrustType["CONTINENT"], -50.0, 11)
    host.crust.create(2, CrustType["CRATON"], -25.0, 12)
    result = landmass_of(host, world.mesh, [0])

    assert result == [0, 1, 2]


def test_close_seas_between_converts_craton_source_when_sealing_two_cell_strait():
    points = [
        [1, 0, 0],
        [0.5, 0.866, 0],
        [-0.5, 0.866, 0],
        [-1, 0, 0],
    ]
    mesh = Mesh(points, [[1], [0, 2], [1, 3], [2]])
    crust = CrustRaster(4)
    crust.create(0, CrustType["CONTINENT"], -100.0, 10)
    crust.create(1, CrustType["OCEAN"], -20.0, 11)
    crust.create(2, CrustType["OCEAN"], -30.0, 12)
    crust.create(3, CrustType["CRATON"], -200.0, 30)
    origin = np.asarray([1, 0, 0, 2], dtype=np.int8)
    upper = Plate(0, 4)
    upper.capCenter = [1.0, 0.0, 0.0]
    cfg = {"sealStraitKm": 200.0, "inlandSeaMaxKm2": 0.0}

    assert close_seas_between(crust, origin, mesh, upper, cfg, 8.0) == 2
    assert crust.type[2] == CrustType["CONTINENT"]
    assert crust.terrane[2] == 30
    assert crust.flags[2] == CrustFlag["SUTURE"]
    assert origin.tolist() == [1, 3, 3, 2]


def test_accrete_terrane_skips_an_isolated_ocean_cell_when_building_seam(monkeypatch):
    points = [
        [1.0, 0.0, 0.0],
        [0.5, math.sqrt(0.75), 0.0],
        [-0.5, math.sqrt(0.75), 0.0],
        [0.0, -1.0, 0.0],
    ]
    mesh = Mesh(points, [[1], [0, 2], [1], []])
    host = Plate(0, 4)
    donor = Plate(1, 4)
    host.crust.create(0, CrustType["CONTINENT"], -100.0, 10)
    host.crust.create(3, CrustType["OCEAN"], -10.0, 40)
    donor.crust.create(1, CrustType["OCEAN"], -50.0, 21)
    donor.crust.create(2, CrustType["CONTINENT"], -80.0, 30)
    donor.crust.flags[1] = CrustFlag["TRENCH"]
    world = SimpleNamespace(
        mesh=mesh,
        config={
            "terraneForeArcKm": 100.0,
            "terraneMinAreaKm2": 100.0,
            "sealStraitKm": 100.0,
            "inlandSeaMaxKm2": 0.0,
            "belts": {"ural": {"baseKm": 100.0, "perKm": 1.0, "maxKm": 300.0}},
            "beltSpeedFactorMax": 1.6,
            "beltSpeedFactorMin": 0.6,
            "beltSpeedRefCmYr": 5.0,
            "beltSpeedExponent": 0.5,
        },
        time=10.0,
    )
    monkeypatch.setattr(plate_accretion, "spread_belt", lambda *_args, **_kwargs: None)

    result = accrete_terrane(world, host, donor, [2], 5.0)

    assert result["cells"] == 1
    assert host.crust.type[3] == CrustType["OCEAN"]


def test_accretion_leaves_unclaimed_donor_crust_and_skips_occupied_forearc(monkeypatch):
    world, host, donor = accretion_fixture()
    donor.crust.create(0, CrustType["OCEAN"], -60.0, 20)
    host.cell_at = lambda *_args: 0
    monkeypatch.setattr(
        plate_accretion,
        "free_cell_beside",
        lambda *_args: 1,
    )
    monkeypatch.setattr(plate_accretion, "spread_belt", lambda *_args, **_kwargs: None)

    result = accrete_terrane(world, host, donor, [2], rate_cm_yr=5.0)

    assert result["cells"] == 1
    assert donor.crust.mask[0]
    assert donor.alive is True
    assert host.crust.type[1] == CrustType["CONTINENT"]
