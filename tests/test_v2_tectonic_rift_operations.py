import math
from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType
from orogen.v2.tectonics.ops import choose_junction, split_plate
from orogen.v2.tectonics import ops
from orogen.v2.tectonics.world import TectonicWorld


class Mesh:
    def __init__(self, points, neighbours, edge_km=1000.0, edge_rad=0.01):
        self.num_cells = len(points)
        self.xyz = np.asarray(points, dtype=np.float32).reshape(-1)
        self.edge_km = float(edge_km)
        self.edge_rad = float(edge_rad)
        offsets = [0]
        adjacency = []
        for row in neighbours:
            adjacency.extend(row)
            offsets.append(len(adjacency))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)


def partition_mesh(extra_right=0):
    c, s = math.cos, math.sin
    points = [
        [c(0.05), s(0.05), 0.0],
        [1.0, 0.0, 0.0],
        [c(0.1), 0.0, s(0.1)],
        [c(0.05), -s(0.05), 0.0],
        [c(0.1), s(0.05), s(0.1)],
        [c(0.1), -s(0.05), s(0.1)],
        [c(0.08), s(0.08), 0.0],
        [c(0.08), -s(0.08), 0.0],
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
    for index in range(extra_right):
        angle = -0.08 - 0.03 * (index + 1)
        point = [c(0.15) * c(angle), c(0.15) * s(angle), s(0.15)]
        length = math.sqrt(sum(value * value for value in point))
        points.append([value / length for value in point])
        neighbours.append([])
    if extra_right:
        start = 8
        neighbours[5].append(start)
        neighbours[start].append(5)
        for cell in range(start, start + extra_right - 1):
            neighbours[cell].append(cell + 1)
            neighbours[cell + 1].append(cell)
    return Mesh(points, neighbours)


def make_world(extra_right=0):
    mesh = partition_mesh(extra_right)
    world = TectonicWorld(
        mesh,
        73,
        {"riftShoulderKm": 1000.0, "riftShoulderWorkKm": 20.0},
    )
    world.time = 12.5
    plate = world.add_plate("parent")
    crust = plate.crust
    crust.create(0, CrustType["CRATON"], 0.0, 10)
    crust.create(1, CrustType["OCEAN"], 0.0)
    crust.create(2, CrustType["OCEAN"], 0.0)
    crust.create(3, CrustType["CONTINENT"], 0.0, 20)
    crust.create(4, CrustType["CONTINENT"], 0.0, 21)
    crust.create(5, CrustType["CRATON"], 0.0, 30)
    crust.create(6, CrustType["OCEAN"], 0.0)
    crust.create(7, CrustType["OCEAN"], 0.0)
    for cell in range(8, 8 + extra_right):
        crust.create(cell, CrustType["CONTINENT"], 0.0, 40 + cell)
    return world, plate


def square_junction_fixture():
    points = []
    neighbours = []
    for row in range(7):
        latitude = (row - 3) * 0.01
        for col in range(7):
            longitude = (col - 3) * 0.01
            points.append(
                [
                    math.cos(latitude) * math.cos(longitude),
                    math.cos(latitude) * math.sin(longitude),
                    math.sin(latitude),
                ]
            )
            cell = row * 7 + col
            adjacent = []
            if row:
                adjacent.append(cell - 7)
            if col:
                adjacent.append(cell - 1)
            if col < 6:
                adjacent.append(cell + 1)
            if row < 6:
                adjacent.append(cell + 7)
            neighbours.append(adjacent)
    mesh = Mesh(points, neighbours, edge_km=63.71)
    world = TectonicWorld(
        mesh,
        17,
        {"riftMinArmKm": 50.0, "riftWeaknessCost": 0.2},
    )
    plate = world.add_plate("landmass")
    for row in range(1, 6):
        for col in range(1, 6):
            cell = row * 7 + col
            plate.crust.create(cell, CrustType["CONTINENT"], 0.0, cell)
    return world, plate


def test_split_plate_moves_crust_wings_sets_rift_records_and_spreads_shoulders(
    monkeypatch,
):
    world, plate = make_world()
    plate.set_quat([math.cos(0.1), 0.0, 0.0, math.sin(0.1)])
    plate.omega = [0.01, 0.02, 0.03]
    plate.hue = 42
    plate.wings = {0: [1, 0, 0], 3: [0, 1, 0], 5: [0, 0, 1]}
    calls = []

    def fake_spread(target, mesh, sources, belt, config, allow):
        calls.append(
            {
                "plate": target.id,
                "sources": list(sources),
                "belt": dict(belt),
                "failed0": allow(0),
                "failed3": allow(3),
                "failed5": allow(5),
            }
        )

    monkeypatch.setattr("orogen.v2.tectonics.ops.rift.spread_belt", fake_spread)
    fresh = split_plate(
        world,
        plate,
        [1, 2],
        [0, 3],
        lambda left, right: ("west", "east"),
        across=[0.0, 0.0, 1.0],
    )

    assert len(world.plates) == 2
    assert fresh.id == 1
    assert plate.name == "west" and fresh.name == "east"
    assert plate.born == fresh.born == world.time
    assert plate.bornOfRift and fresh.bornOfRift
    assert plate.riftPartner == fresh.id
    assert fresh.riftPartner == plate.id
    assert plate.hue == 42 and fresh.hue == 1
    assert np.allclose(fresh.quat, plate.quat)
    assert fresh.omega == plate.omega == [0.01, 0.02, 0.03]

    assert plate.crust.count == 5
    assert fresh.crust.count == 3
    assert plate.crust.mask.tolist() == [1, 1, 1, 0, 1, 0, 1, 0]
    assert fresh.crust.mask.tolist() == [0, 0, 0, 1, 0, 1, 0, 1]
    assert plate.wings == {0: [1, 0, 0]}
    assert fresh.wings == {3: [0, 1, 0], 5: [0, 0, 1]}

    assert plate.crust.flags[1] & CrustFlag["RIFT_MARGIN"]
    assert plate.crust.flags[2] & CrustFlag["RIFT_MARGIN"]
    assert fresh.crust.flags[3] & CrustFlag["RIFT_MARGIN"]
    assert fresh.crust.flags[5] & CrustFlag["RIFT_MARGIN"]
    assert plate.crust.flags[0] & CrustFlag["FAILED_RIFT"]
    assert fresh.crust.flags[3] & CrustFlag["FAILED_RIFT"]
    assert np.all(plate.crust.riftAt[[0, 1, 2]] == pytest.approx(world.time))
    assert np.all(fresh.crust.riftAt[[3, 5]] == pytest.approx(world.time))

    assert len(calls) == 2
    assert calls[0]["plate"] == plate.id
    assert calls[0]["sources"] == [1, 2]
    assert calls[0]["failed0"] is False
    assert calls[1]["plate"] == fresh.id
    assert calls[1]["sources"] == [5]
    assert calls[1]["failed3"] is False
    assert calls[1]["failed5"] is True

    assert np.allclose(plate.riftPushDir, [0, 0, 1])
    assert np.allclose(fresh.riftPushDir, [0, 0, -1])
    assert plate.reshaped and fresh.reshaped
    assert not plate.dirty and not fresh.dirty
    assert plate.stats["cells"] == 5
    assert fresh.stats["cells"] == 3


def test_split_plate_larger_new_half_keeps_parent_hue_and_default_push(monkeypatch):
    world, plate = make_world(extra_right=3)
    plate.hue = 55
    monkeypatch.setattr(
        "orogen.v2.tectonics.ops.rift.spread_belt",
        lambda *_args, **_kwargs: None,
    )
    fresh = split_plate(
        world,
        plate,
        [1, 2],
        [],
        lambda _left, _right: ("west", "east"),
    )

    assert fresh.crust.count > plate.crust.count
    assert fresh.hue == 55
    assert plate.hue == fresh.id
    assert np.isfinite(plate.riftPushDir).all()
    assert np.isfinite(fresh.riftPushDir).all()


def test_choose_junction_weights_weak_interior_and_skips_cratons():
    world, plate = square_junction_fixture()
    plate.crust.type[2 * 7 + 2] = CrustType["CRATON"]
    selected = 3 * 7 + 3
    plate.crust.flags[selected] |= CrustFlag["SUTURE"]

    assert choose_junction(world, plate, lambda: 0.45) == selected


def test_choose_junction_returns_minus_one_without_interior():
    world, plate = square_junction_fixture()
    for cell in range(len(plate.crust.mask)):
        if plate.crust.mask[cell]:
            row, col = divmod(cell, 7)
            if row not in (1, 5) and col not in (1, 5):
                plate.crust.remove(cell)
    assert choose_junction(world, plate, lambda: 0.0) == -1


def test_choose_junction_failed_rift_is_also_a_weak_seed():
    world, plate = square_junction_fixture()
    selected = 3 * 7 + 3
    plate.crust.flags[selected] |= CrustFlag["FAILED_RIFT"]
    assert choose_junction(world, plate, lambda: 0.45) == selected


def test_choose_junction_falls_back_to_last_candidate_if_rng_overshoots():
    world, plate = square_junction_fixture()
    candidates = [row * 7 + col for row in range(2, 5) for col in range(2, 5)]
    assert choose_junction(world, plate, lambda: 1.1) == candidates[-1]
