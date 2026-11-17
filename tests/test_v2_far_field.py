"""Tests for far-field push accumulation and inland diffusion."""

from types import SimpleNamespace
import math

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.far_field import (
    evolve_far_field,
    far_take,
    feed_far_field,
    recent_push,
    record_push,
    spread_push,
)
from orogen.v2.tectonics.plate import Plate


class LineMesh:
    def __init__(self, count=5):
        angle = np.arange(count, dtype=np.float64) * 0.01
        points = np.column_stack((
            np.cos(angle), np.sin(angle), np.zeros(count)
        ))
        self.xyz = points.astype(np.float64).ravel()
        self.num_cells = count
        self.edge_km = 63.71
        self.adj_offset = np.arange(0, 2 * count, 2, dtype=np.int32)
        edges = []
        offsets = [0]
        for cell in range(count):
            neighbours = []
            if cell > 0:
                neighbours.append(cell - 1)
            if cell + 1 < count:
                neighbours.append(cell + 1)
            edges.extend(neighbours)
            offsets.append(len(edges))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(edges, dtype=np.int32)


def far_config():
    return {
        "farFieldEveryMyr": 2.0,
        "farReachKm": 1000.0,
        "farFrontRefKm": 1000.0,
        "farMaxKm": 1000.0,
        "farCratonCost": 0.3,
        "farWeakCost": 1.6,
        "farCratonTake": 0.15,
        "farWeakTake": 1.6,
        "farFoldMyr": 100.0,
    }


def make_world(types=None):
    mesh = LineMesh(5)
    plate = Plate(0, 5)
    values = types or [
        CrustType["CONTINENT"],
        CrustType["CRATON"],
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
        CrustType["OCEAN"],
    ]
    for cell, crust_type in enumerate(values):
        plate.crust.create(cell, crust_type, -20.0, cell + 1)
    world = SimpleNamespace(
        mesh=mesh,
        plates=[plate],
        config=far_config(),
        time=10.0,
    )
    return world, plate


def test_feed_far_field_accumulates_scalar_and_per_cell_work():
    world, plate = make_world()
    feed_far_field(world, plate, "collision:upper", [0, 1], 10.0, 500.0, 0.5)
    feed_far_field(world, plate, "collision:upper", [1, 2], [2.0, 4.0], 700.0, 0.25)
    feed_far_field(world, plate, "collision:upper", [0], 100.0, 900.0, 0.0)

    group = world.farPending["collision:upper"]
    assert group["plate"] is plate
    assert group["frontKm"] == 700.0
    assert group["cells"] == {0: 5.0, 1: 5.5, 2: 1.0}


def test_feed_far_field_skips_empty_and_non_positive_share():
    world, plate = make_world()
    feed_far_field(world, plate, "zero", [0], 10.0, 100.0, 0.0)
    feed_far_field(world, plate, "empty", [], 10.0, 100.0, 1.0)
    assert not hasattr(world, "farPending")


def test_feed_far_field_replaces_key_when_plate_owner_changes():
    world, plate = make_world()
    other = Plate(1, 5)
    feed_far_field(world, plate, "front", [0], 5.0, 100.0, 1.0)
    feed_far_field(world, other, "front", [2], 7.0, 200.0, 1.0)
    assert world.farPending["front"]["plate"] is other
    assert world.farPending["front"]["cells"] == {2: 7.0}


def test_spread_push_uses_craton_and_weak_path_costs_and_cell_take_shares():
    world, plate = make_world()
    plate.crust.flags[2] |= CrustFlag["SUTURE"]

    result = spread_push(world, plate, {0: 100.0}, 1000.0, world.time)

    assert result["cells"] == [0, 1, 2, 3]
    assert result["from"].tolist() == [0, 0, 0, 0]
    assert result["dist"][0] == 0.0
    edge = 63.71
    assert result["dist"][1] == pytest.approx(edge * 0.3)
    assert result["dist"][2] == pytest.approx(edge * (0.3 + 1.6))
    assert result["dist"][3] == pytest.approx(edge * (0.3 + 1.6 + 1.0))
    assert result["inc"][1] == pytest.approx(plate.crust.farWork[1] * 0.15)
    assert result["inc"][2] == pytest.approx(plate.crust.farWork[2] * 1.6)
    assert plate.crust.farAt[3] == world.time
    assert plate.crust.farWork[4] == 0.0  # ocean receives no record


def test_spread_push_selects_strongest_arrival_from_multiple_sources():
    world, plate = make_world([
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
    ])
    result = spread_push(
        world, plate, {0: 10.0, 4: 100.0}, 1000.0, world.time
    )
    assert 2 in result["cells"]
    index = result["cells"].index(2)
    assert result["from"][index] == 4
    assert plate.crust.farWork[2] > 0.0


def test_spread_push_respects_max_path_cost_and_empty_sources():
    world, plate = make_world()
    world.config["farMaxKm"] = 10.0
    result = spread_push(world, plate, {0: 100.0}, 1000.0, world.time)
    assert result["cells"] == [0]
    empty = spread_push(world, plate, {1: -1.0, 9: 3.0}, 1000.0, world.time)
    assert empty["cells"] == []
    assert empty["inc"].size == 0


def test_spread_push_rejects_bad_mesh_size_and_non_positive_reach():
    world, plate = make_world()
    world.mesh.xyz = np.zeros(3)
    with pytest.raises(ValueError, match="same cell count"):
        spread_push(world, plate, {0: 1.0}, 1000.0, 0.0)

    world, plate = make_world()
    world.config["farReachKm"] = 0.0
    with pytest.raises(ValueError, match="must be positive"):
        spread_push(world, plate, {0: 1.0}, 1000.0, 0.0)


def test_record_push_accumulates_weighted_distance_and_recent_component():
    world, plate = make_world()
    crust = plate.crust
    crust.farWork[1] = 10.0
    crust.farDist[1] = 100.0
    crust.farFold[1] = 4.0
    crust.farAt[1] = 5.0

    record_push(
        crust, 1, 5.0, 200.0, world.mesh.xyz, 0, 15.0, 100.0
    )

    assert crust.farWork[1] == pytest.approx(15.0)
    assert crust.farDist[1] == pytest.approx((10 * 100 + 5 * 200) / 15)
    assert crust.farFold[1] == pytest.approx(4 * math.exp(-0.1) + 5)
    assert crust.farAt[1] == 15.0
    assert crust.farDy[1] > 0


def test_record_push_keeps_axis_sign_blind_and_handles_source_or_antipode():
    world, plate = make_world()
    crust = plate.crust
    record_push(crust, 2, 5.0, 100.0, world.mesh.xyz, 1, 10.0, 100.0)
    first_axis = np.asarray(
        [crust.farDx[2], crust.farDy[2], crust.farDz[2]], dtype=float
    )
    record_push(crust, 2, 5.0, 200.0, world.mesh.xyz, 3, 11.0, 100.0)
    second_axis = np.asarray(
        [crust.farDx[2], crust.farDy[2], crust.farDz[2]], dtype=float
    )
    assert np.dot(first_axis, second_axis) > 0.0

    record_push(crust, 1, 2.0, 50.0, world.mesh.xyz, 1, 12.0, 100.0)
    assert crust.farWork[1] == pytest.approx(2.0)

    xyz = world.mesh.xyz.copy()
    xyz[3:6] = -xyz[0:3]
    record_push(crust, 3, 1.0, 5.0, xyz, 0, 12.0, 100.0)
    assert crust.farWork[3] == pytest.approx(1.0)


def test_far_take_prefers_cratons_then_weak_flags():
    config = far_config()
    assert far_take(config, CrustType["CRATON"], CrustFlag["SUTURE"]) == 0.15
    assert far_take(config, CrustType["CONTINENT"], CrustFlag["FAILED_RIFT"]) == 1.6
    assert far_take(config, CrustType["CONTINENT"], 0) == 1.0


def test_recent_push_decays_and_never_grows_backwards_in_time():
    assert recent_push(10.0, 5.0, 15.0, 100.0) == pytest.approx(10.0 * math.exp(-0.1))
    assert recent_push(10.0, 15.0, 5.0, 100.0) == 10.0
    assert recent_push(0.0, 0.0, 100.0, 100.0) == 0.0


def test_evolve_far_field_spreads_when_clock_is_due_and_skips_dead_plates(monkeypatch):
    world, plate = make_world()
    dead = Plate(1, 5)
    dead.alive = False
    calls = []

    from orogen.v2.tectonics import far_field
    monkeypatch.setattr(
        far_field, "spread_push",
        lambda w, p, cells, front, time: calls.append((p.id, dict(cells), front, time)) or {},
    )
    feed_far_field(world, plate, "alive", [0], 10.0, 200.0, 1.0)
    feed_far_field(world, dead, "dead", [1], 20.0, 300.0, 1.0)

    assert evolve_far_field(world, 1.0) == 0
    assert world.farClock == 1.0
    assert calls == []
    assert evolve_far_field(world, 1.0) == 0
    assert world.farClock == 0.0
    assert calls == [(0, {0: 10.0}, 200.0, world.time)]
    assert world.farPending == {}
    assert world.farLastWindow == 2.0


def test_evolve_far_field_resets_clock_when_nothing_is_pending():
    world, _plate = make_world()
    assert evolve_far_field(world, 2.0) == 0
    assert world.farClock == 0.0
    assert not hasattr(world, "farLastWindow")


@pytest.mark.parametrize("dt", [-1.0, float("inf"), float("nan")])
def test_evolve_far_field_rejects_invalid_dt(dt):
    world, _plate = make_world()
    with pytest.raises(ValueError, match="finite and non-negative"):
        evolve_far_field(world, dt)
