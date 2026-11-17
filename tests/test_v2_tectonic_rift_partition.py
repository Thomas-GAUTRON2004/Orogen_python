import math

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustRaster, CrustType
from orogen.v2.tectonics.ops.rift_partition import (
    _through_ocean,
    elongation_of,
    exterior_sea,
    masked_cells,
    partition,
    sides_of,
)


class Mesh:
    def __init__(self, points, neighbours, edge_km=1000.0):
        self.num_cells = len(points)
        self.xyz = np.asarray(points, dtype=np.float32).reshape(-1)
        self.edge_km = float(edge_km)
        offsets = [0]
        adjacency = []
        for row in neighbours:
            adjacency.extend(row)
            offsets.append(len(adjacency))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)


def make_rift_fixture():
    c, s = math.cos, math.sin
    points = [
        [c(0.05), s(0.05), 0.0],       # left land component
        [1.0, 0.0, 0.0],               # first rift cell
        [c(0.1), 0.0, s(0.1)],         # second rift cell
        [c(0.05), -s(0.05), 0.0],      # right land component
        [c(0.1), s(0.05), s(0.1)],     # left land component
        [c(0.1), -s(0.05), s(0.1)],    # right land component
        [c(0.08), s(0.08), 0.0],       # left-side ocean
        [c(0.08), -s(0.08), 0.0],      # right-side ocean
    ]
    neighbours = [
        [1, 4, 6],
        [0, 2, 3],
        [1, 4, 5],
        [1, 5, 7],
        [0, 2],
        [2, 3],
        [0],
        [3],
    ]
    mesh = Mesh(points, neighbours)
    crust = CrustRaster(8)
    crust.create(0, CrustType["CRATON"], 0.0, 10)
    crust.create(1, CrustType["OCEAN"], 0.0)
    crust.create(2, CrustType["OCEAN"], 0.0)
    crust.create(3, CrustType["CONTINENT"], 0.0, 20)
    crust.create(4, CrustType["CONTINENT"], 0.0)
    crust.create(5, CrustType["CRATON"], 0.0, 30)
    crust.create(6, CrustType["OCEAN"], 0.0)
    crust.create(7, CrustType["OCEAN"], 0.0)
    return mesh, crust


def test_masked_cells_and_rift_partition_assign_both_land_sides_and_ocean():
    mesh, crust = make_rift_fixture()
    assert masked_cells(crust).tolist() == list(range(8))

    side, land, cratons = partition(mesh, crust, [1, 2])

    assert side.dtype == np.int8
    assert land == [0, 3, 2]
    assert cratons == [0, 1, 1]
    assert side[0] == side[4] == side[6] == 1
    assert side[3] == side[5] == side[7] == 2


def test_sides_of_reports_share_elongation_and_cratons_on_both_sides():
    mesh, crust = make_rift_fixture()
    result = sides_of(mesh, crust, [1, 2])

    assert result["share"] == pytest.approx(0.4)
    assert result["elongation"] == 1.0
    assert result["cratonsBothSides"] is True


def test_empty_land_and_single_cell_path_are_handled():
    mesh, _ = make_rift_fixture()
    crust = CrustRaster(8)
    for cell in range(8):
        crust.create(cell, CrustType["OCEAN"], 0.0)

    side, land, cratons = partition(mesh, crust, [1])
    assert land == [0, 0, 0]
    assert cratons == [0, 0, 0]
    assert np.all(side == 1)
    assert sides_of(mesh, crust, [1]) == {
        "share": 0.0, "elongation": 1.0, "cratonsBothSides": False
    }


def test_through_ocean_extends_both_ends_and_short_paths_are_unchanged():
    c, s = math.cos, math.sin
    points = [
        [c(0.05), 0.0, -s(0.05)],
        [1.0, 0.0, 0.0],
        [c(0.05), 0.0, s(0.05)],
        [c(0.1), 0.0, s(0.1)],
    ]
    mesh = Mesh(points, [[1], [0, 2], [1, 3], [2]], edge_km=1000.0)
    crust = CrustRaster(4)
    crust.create(0, CrustType["OCEAN"], 0.0)
    crust.create(1, CrustType["CONTINENT"], 0.0)
    crust.create(2, CrustType["CONTINENT"], 0.0)
    crust.create(3, CrustType["OCEAN"], 0.0)

    assert _through_ocean(mesh, crust, [1, 2]) == [0, 1, 2, 3]
    assert _through_ocean(mesh, crust, [1]) == [1]


def test_exterior_sea_floods_from_unmasked_cells():
    mesh = Mesh(
        [[1, 0, 0], [0.9, 0.1, 0], [0.8, 0.2, 0], [0.7, 0.3, 0], [0.6, 0.4, 0]],
        [[1], [0, 2], [1, 3], [2, 4], [3]],
    )
    crust = CrustRaster(5)
    crust.create(1, CrustType["OCEAN"], 0.0)
    crust.create(2, CrustType["OCEAN"], 0.0)
    crust.create(3, CrustType["CONTINENT"], 0.0)
    crust.create(4, CrustType["OCEAN"], 0.0)

    assert exterior_sea(mesh, crust).tolist() == [1, 1, 1, 0, 0]


def test_exterior_sea_uses_largest_ocean_when_plate_covers_mesh():
    mesh = Mesh(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]],
    )
    crust = CrustRaster(6)
    for cell in range(6):
        kind = CrustType["CONTINENT"] if cell == 3 else CrustType["OCEAN"]
        crust.create(cell, kind, 0.0)

    assert exterior_sea(mesh, crust).tolist() == [1, 1, 1, 0, 0, 0]


def test_elongation_handles_small_degenerate_and_rank_one_blocks():
    points = [
        [1, 0, 0], [-1, 0, 0], [0, 1, 0],
        [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ]
    mesh = Mesh(points, [[] for _ in points])
    side = np.ones(6, dtype=np.int8)
    assert elongation_of(mesh, side, 1) == pytest.approx(1.0)

    duplicate_mesh = Mesh([[1, 0, 0]] * 6, [[] for _ in range(6)])
    assert elongation_of(duplicate_mesh, side, 1) == 1.0

    line_mesh = Mesh(
        [[1, 0, 0], [-1, 0, 0]] * 3,
        [[] for _ in range(6)],
    )
    assert elongation_of(line_mesh, side, 1) == 10.0
    assert elongation_of(mesh, side, 2) == 1.0
