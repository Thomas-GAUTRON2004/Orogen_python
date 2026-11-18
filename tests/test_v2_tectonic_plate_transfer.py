import math
from types import SimpleNamespace

import numpy as np

from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.ops.plate_transfer import (
    free_cell_beside,
    transfer_cells,
)


class Index:
    def __init__(self, mapping):
        self.mapping = mapping

    def nearest(self, x, y, z):
        key = (round(float(x), 5), round(float(y), 5), round(float(z), 5))
        return self.mapping[key]


class Mesh:
    def __init__(self, points, neighbours):
        self.num_cells = len(points)
        self.xyz = np.asarray(points, dtype=np.float32).reshape(-1)
        offsets = [0]
        adjacency = []
        for row in neighbours:
            adjacency.extend(row)
            offsets.append(len(adjacency))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)
        self.index = Index(
            {tuple(round(float(v), 5) for v in p): i for i, p in enumerate(points)}
        )


def test_transfer_cells_resamples_to_target_frame_and_rotates_far_field():
    points = [[1, 0, 0], [0, 1, 0], [-1, 0, 0]]
    mesh = Mesh(points, [[1], [0, 2], [1]])
    world = SimpleNamespace(mesh=mesh)
    target = Plate(0, 3)
    source = Plate(1, 3)
    angle = math.pi / 2.0
    source.set_quat([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
    source.crust.create(0, CrustType["CONTINENT"], -20.0, 51)
    source.crust.flags[0] = CrustFlag["TRENCH"] | CrustFlag["LIP"]
    source.crust.farDx[0] = 127
    source.crust.volcKm[0] = 12.0

    assert transfer_cells(world, source, target, [0]) == 1

    assert source.crust.count == 0
    assert target.crust.mask.tolist() == [0, 1, 0]
    assert target.crust.type[1] == CrustType["CONTINENT"]
    assert target.crust.terrane[1] == 51
    assert target.crust.flags[1] & CrustFlag["LIP"]
    assert not (target.crust.flags[1] & CrustFlag["TRENCH"])
    assert target.crust.farDx[1] == 0
    assert target.crust.farDy[1] == 127
    assert target.crust.volcKm[1] == 12.0
    assert source.reshaped and target.reshaped
    assert source.stats["cells"] == 0
    assert target.stats["cells"] == 1


def test_transfer_cells_places_overlapping_land_in_nearest_free_neighbour():
    points = [
        [1.0, 0.0, 0.0],
        [0.9, 0.435, 0.0],
        [0.0, 1.0, 0.0],
    ]
    mesh = Mesh(points, [[1, 2], [0], [0]])
    world = SimpleNamespace(mesh=mesh)
    target = Plate(0, 3)
    source = Plate(1, 3)
    target.crust.create(0, CrustType["CONTINENT"], -100.0, 1)
    source.crust.create(0, CrustType["CRATON"], -200.0, 2)
    source.crust.flags[0] = CrustFlag["TRENCH"]

    assert transfer_cells(world, source, target, [0]) == 1
    assert target.crust.type[0] == CrustType["CONTINENT"]
    assert target.crust.type[1] == CrustType["CRATON"]
    assert target.crust.terrane[1] == 2
    assert not (target.crust.flags[1] & CrustFlag["TRENCH"])


def test_transfer_cells_does_not_overwrite_existing_host_ocean():
    points = [[1, 0, 0], [0, 1, 0]]
    mesh = Mesh(points, [[1], [0]])
    world = SimpleNamespace(mesh=mesh)
    target = Plate(0, 2)
    source = Plate(1, 2)
    target.crust.create(1, CrustType["OCEAN"], -10.0, 20)
    source.crust.create(1, CrustType["OCEAN"], -100.0, 21)

    assert transfer_cells(world, source, target, [1]) == 0
    assert source.crust.count == 0
    assert target.crust.count == 1
    assert target.crust.terrane[1] == 20


def test_transfer_cells_absorbs_land_when_no_free_neighbour_exists():
    points = [[1, 0, 0], [0, 1, 0]]
    mesh = Mesh(points, [[1], [0]])
    world = SimpleNamespace(mesh=mesh)
    target = Plate(0, 2)
    source = Plate(1, 2)
    target.crust.create(0, CrustType["CONTINENT"], -10.0, 20)
    target.crust.create(1, CrustType["CRATON"], -20.0, 21)
    source.crust.create(0, CrustType["CONTINENT"], -100.0, 30)

    assert transfer_cells(world, source, target, [0]) == 0
    assert source.crust.count == 0
    assert target.crust.count == 2
    assert target.crust.terrane[0] == 20


def test_ocean_is_copied_into_an_empty_target_cell():
    points = [[1, 0, 0], [0, 1, 0]]
    mesh = Mesh(points, [[1], [0]])
    world = SimpleNamespace(mesh=mesh)
    target = Plate(0, 2)
    source = Plate(1, 2)
    source.crust.create(0, CrustType["OCEAN"], -100.0, 31)

    assert transfer_cells(world, source, target, [0]) == 1
    assert target.crust.type[0] == CrustType["OCEAN"]
    assert target.crust.terrane[0] == 31


def test_free_cell_beside_returns_nearest_free_cell_or_minus_one():
    points = [
        [1.0, 0.0, 0.0],
        [0.9, 0.435, 0.0],
        [0.0, 1.0, 0.0],
    ]
    mesh = Mesh(points, [[1, 2], [0], [0]])
    plate = Plate(0, 3)
    assert free_cell_beside(plate, mesh, 0, [1.0, 0.0, 0.0], lambda _cell: False) == 1
    assert free_cell_beside(plate, mesh, 0, [1.0, 0.0, 0.0], lambda _cell: True) == -1
