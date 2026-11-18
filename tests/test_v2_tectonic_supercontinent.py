from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.supercontinent import is_continental, landmasses


class TestMesh:
    pass


def make_land_world(types, cell_area=300000.0, neck_km=80.0):
    n = len(types)
    plates = [Plate(0, n)]
    for i, kind in enumerate(types):
        if kind is not None:
            plates[0].crust.create(i, kind, 0.0)
    angles = np.arange(n, dtype=float) * 0.02
    xyz = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(n)))
    adjacency = []
    offsets = [0]
    for cell in range(n):
        if cell > 0:
            adjacency.append(cell - 1)
        if cell + 1 < n:
            adjacency.append(cell + 1)
        offsets.append(len(adjacency))
    mesh = TestMesh()
    mesh.num_cells = n
    mesh.xyz = xyz.astype(np.float32).ravel()
    mesh.adj_offset = np.asarray(offsets, dtype=np.int32)
    mesh.adj_list = np.asarray(adjacency, dtype=np.int32)
    mesh.cell_area_km2 = cell_area
    world = SimpleNamespace(
        mesh=mesh,
        config={"cycleNeckKm": neck_km},
        owner=np.array(
            [0 if kind is not None else -1 for kind in types], dtype=np.int16
        ),
        ownerCell=np.arange(n, dtype=np.int32),
        plates=plates,
    )
    return world


def test_landmasses_finds_largest_component_and_share():
    world = make_land_world(
        [
            CrustType["OCEAN"],
            CrustType["CONTINENT"],
            CrustType["CONTINENT"],
            CrustType["CRATON"],
            CrustType["CONTINENT"],
            CrustType["OCEAN"],
        ]
    )
    result = landmasses(world)
    assert result["largest"] == [1, 2, 3, 4]
    assert result["share"] == pytest.approx(1.0)
    assert result["count"] == 1


def test_landmasses_counts_disconnected_large_pieces_and_selects_the_largest():
    world = make_land_world(
        [
            CrustType["CONTINENT"],
            CrustType["CRATON"],
            CrustType["CONTINENT"],
            CrustType["OCEAN"],
            CrustType["CONTINENT"],
            CrustType["CONTINENT"],
            CrustType["CRATON"],
            CrustType["CONTINENT"],
        ],
        cell_area=300000.0,
    )
    result = landmasses(world)
    assert result["count"] == 1
    assert result["largest"] == [4, 5, 6, 7]
    assert result["share"] == pytest.approx(4 / 7)


def test_landmasses_handles_no_land_and_sub_threshold_pieces():
    world = make_land_world([CrustType["OCEAN"]] * 5)
    assert landmasses(world) == {"largest": [], "share": 0.0, "count": 0}
    world = make_land_world(
        [
            CrustType["CONTINENT"],
            CrustType["CONTINENT"],
            CrustType["OCEAN"],
            CrustType["CRATON"],
            CrustType["OCEAN"],
        ],
        cell_area=200000.0,
    )
    result = landmasses(world)
    assert result["count"] == 0
    assert result["largest"]
    assert result["share"] == pytest.approx(len(result["largest"]) / 3)


def test_is_continental_rejects_unowned_ocean_and_arc_cells():
    world = make_land_world(
        [
            CrustType["CONTINENT"],
            CrustType["CRATON"],
            CrustType["OCEAN"],
            CrustType["ARC"],
        ]
    )
    assert is_continental(world, 0)
    assert is_continental(world, 1)
    assert not is_continental(world, 2)
    assert not is_continental(world, 3)
    world.owner[0] = -1
    assert not is_continental(world, 0)
