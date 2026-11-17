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
    assert abs(float(np.dot(first_axis, second_axis))) > 0.99 * float(np.dot(first_axis, first_axis))

    record_push(crust, 1, 2.0, 50.0, world.mesh.xyz, 1, 12.0, 100.0)
    assert crust.farWork[1] == pytest.approx(2.0)

    xyz = world.mesh.xyz.copy()
    xyz[9:12] = -xyz[0:3]
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



def test_spread_push_does_not_cross_missing_crust_cells():
    world, plate = make_world()
    plate.crust.remove(2)
    result = spread_push(world, plate, {0: 10.0}, 1000.0, world.time)
    assert 1 in result["cells"]
    assert 2 not in result["cells"]
    assert 3 not in result["cells"]


def test_spread_push_drops_arrivals_below_recording_threshold():
    world, plate = make_world([
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
    ])
    world.config["farReachKm"] = 1.0
    world.config["farMaxKm"] = 1000.0
    result = spread_push(world, plate, {0: 1.0}, 1000.0, world.time)
    assert result["cells"] == [0]
    assert plate.crust.farWork[1] == 0.0

def _reactivation_config():
    config = far_config()
    config.update({
        "farReactivate": 1.0,
        "farReactMinKmPerMyr": 10.0,
        "farNearKm": 100.0,
        "farBeltKm": [110.0, 320.0],
        "farBeltWorkKm": 60.0,
        "farRejuvenateCrests": 0.75,
        "farRejuvenateUplift": 0.35,
        "orogenyQuietMyr": 5.0,
    })
    return config


def _push(cells, inc, sources, distances):
    return {
        "cells": list(cells),
        "inc": np.asarray(inc, dtype=np.float32),
        "from": np.asarray(sources, dtype=np.int32),
        "dist": np.asarray(distances, dtype=np.float64),
    }


def test_line_strike_finds_the_long_axis_of_a_weak_line():
    from orogen.v2.tectonics.far_field import line_strike

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    for cell in range(5):
        plate.crust.flags[cell] |= CrustFlag["SUTURE"]

    strike = line_strike(plate.crust, world.mesh, 2)
    assert strike is not None
    assert abs(strike[1]) > 0.99
    assert abs(strike[0]) < 0.02


def test_line_strike_filters_missing_cells_and_non_weak_neighbours():
    from orogen.v2.tectonics.far_field import line_strike

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    for cell in range(5):
        plate.crust.flags[cell] |= CrustFlag["SUTURE"]
    plate.crust.remove(0)
    plate.crust.flags[4] = 0
    assert line_strike(plate.crust, world.mesh, 2) is None


def test_line_strike_returns_none_for_too_few_weak_cells():
    from orogen.v2.tectonics.far_field import line_strike

    world, plate = make_world()
    plate.crust.flags[2] |= CrustFlag["SUTURE"]
    plate.crust.flags[1] |= CrustFlag["FAILED_RIFT"]
    assert line_strike(plate.crust, world.mesh, 2) is None


def test_line_strike_uses_stable_tangent_basis_near_pole():
    from orogen.v2.tectonics.far_field import line_strike

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    for cell in range(5):
        plate.crust.flags[cell] |= CrustFlag["SUTURE"]
    points = []
    for offset in (-0.02, -0.01, 0.0, 0.01, 0.02):
        points.extend([math.sin(offset), 0.0, math.cos(offset)])
    world.mesh.xyz = np.asarray(points, dtype=np.float64)
    strike = line_strike(plate.crust, world.mesh, 2)
    assert strike is not None
    assert abs(strike[0]) > 0.99


def test_line_strike_rejects_degenerate_position():
    from orogen.v2.tectonics.far_field import line_strike

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    for cell in range(5):
        plate.crust.flags[cell] |= CrustFlag["SUTURE"]
    world.mesh.xyz[6:9] = 0.0
    assert line_strike(plate.crust, world.mesh, 2) is None


def test_smoothstep_has_flat_ends_and_smooth_middle():
    from orogen.v2.tectonics.far_field import _smoothstep

    assert _smoothstep(-1.0) == 0.0
    assert _smoothstep(2.0) == 1.0
    assert _smoothstep(0.5) == pytest.approx(0.5)


def test_reactivate_spreads_a_rootless_belt_on_a_distant_weak_line(monkeypatch):
    from orogen.v2.tectonics import far_field

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    world.config.update(_reactivation_config())
    plate.crust.flags[2] |= CrustFlag["SUTURE"]
    monkeypatch.setattr(far_field, "line_strike", lambda *_args: [0.0, 0.0, 1.0])
    calls = []
    monkeypatch.setattr(
        far_field, "spread_belt",
        lambda p, m, sources, belt, cfg: calls.append((p, m, list(sources), dict(belt), cfg)),
    )
    far_field.reactivate(
        world, plate, _push([2], [100.0], [0], [1000.0]), 4.0
    )

    assert len(calls) == 1
    assert calls[0][2] == [2]
    belt = calls[0][3]
    assert belt["type"] == 8
    assert belt["noRoot"] is True
    assert belt["works"][0] == pytest.approx(60.0)
    assert belt["widths"][0] == pytest.approx(110.0 + 210.0 * math.tanh(1.0))


def test_reactivate_uses_no_strike_when_a_weak_line_has_insufficient_geometry(monkeypatch):
    from orogen.v2.tectonics import far_field

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    world.config.update(_reactivation_config())
    plate.crust.flags[2] |= CrustFlag["FAILED_RIFT"]
    monkeypatch.setattr(far_field, "line_strike", lambda *_args: None)
    calls = []
    monkeypatch.setattr(far_field, "spread_belt", lambda *args: calls.append(args))
    far_field.reactivate(world, plate, _push([2], [100.0], [0], [150.0]), 0.0)
    assert len(calls) == 1


def test_reactivate_skips_cratons_near_fronts_and_active_ranges(monkeypatch):
    from orogen.v2.tectonics import far_field

    world, plate = make_world([
        CrustType["CRATON"], CrustType["CONTINENT"],
        CrustType["CONTINENT"], CrustType["CONTINENT"],
        CrustType["CONTINENT"],
    ])
    world.config.update(_reactivation_config())
    plate.crust.flags[0] |= CrustFlag["SUTURE"]
    plate.crust.flags[1] |= CrustFlag["SUTURE"]
    plate.crust.flags[2] |= CrustFlag["SUTURE"]
    plate.crust.oroType[2] = 1
    plate.crust.oroLast[2] = world.time
    monkeypatch.setattr(far_field, "line_strike", lambda *_args: None)
    calls = []
    monkeypatch.setattr(far_field, "spread_belt", lambda *args: calls.append(args))
    far_field.reactivate(
        world,
        plate,
        _push([0, 1, 2], [100.0] * 3, [4] * 3, [1000.0, 50.0, 1000.0]),
        4.0,
    )
    assert calls == []


def test_reactivate_skips_invalid_or_coincident_sources_and_subthreshold_work(monkeypatch):
    from orogen.v2.tectonics import far_field

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    world.config.update(_reactivation_config())
    plate.crust.flags[2] |= CrustFlag["SUTURE"]
    world.mesh.xyz[0:3] = -world.mesh.xyz[6:9]
    monkeypatch.setattr(far_field, "line_strike", lambda *_args: None)
    calls = []
    monkeypatch.setattr(far_field, "spread_belt", lambda *args: calls.append(args))
    far_field.reactivate(
        world, plate, _push([2], [100.0], [0], [1000.0]), 4.0
    )  # An antipodal source has no well-defined tangent direction.
    far_field.reactivate(
        world,
        plate,
        _push([2, 2], [100.0] * 2, [-1, 9], [1000.0] * 2),
        4.0,
    )
    assert calls == []

    world.mesh.xyz[0:3] = np.asarray([1.0, 0.0, 0.0])
    world.config["farReactMinKmPerMyr"] = 1000.0
    far_field.reactivate(
        world, plate, _push([2], [100.0], [0], [1000.0]), 4.0
    )
    assert calls == []


def test_reactivate_rejects_invalid_belt_widths_and_malformed_push(monkeypatch):
    from orogen.v2.tectonics import far_field

    world, plate = make_world()
    world.config.update(_reactivation_config())
    world.config["farBeltKm"] = [320.0, 110.0]
    with pytest.raises(ValueError, match="ordered non-negative"):
        far_field.reactivate(world, plate, _push([], [], [], []), 1.0)
    world.config["farBeltKm"] = [-1.0, 110.0]
    with pytest.raises(ValueError, match="ordered non-negative"):
        far_field.reactivate(world, plate, _push([], [], [], []), 1.0)
    world.config.update(_reactivation_config())
    with pytest.raises(ValueError, match="equal lengths"):
        far_field.reactivate(
            world, plate, {"cells": [0], "inc": [], "from": [], "dist": []}, 1.0
        )


def test_reactivate_rejuvenates_an_old_range_without_seeding_a_new_belt(monkeypatch):
    from orogen.v2.tectonics import far_field

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    world.config.update(_reactivation_config())
    for cell in range(5):
        plate.crust.flags[cell] |= CrustFlag["SUTURE"]
        plate.crust.oroType[cell] = 3
        plate.crust.oroLast[cell] = 0.0
        plate.crust.beltReach[cell] = 200.0
        plate.crust.beltU[cell] = 0.2
        plate.crust.beltSide[cell] = 1
    monkeypatch.setattr(far_field, "line_strike", lambda *_args: [0.0, 0.0, 1.0])
    monkeypatch.setattr(
        far_field, "spread_belt",
        lambda *_args: pytest.fail("old ranges must be rejuvenated in place"),
    )
    before_rugged = plate.crust.rugged.copy()
    before_uplift = plate.crust.upliftKm.copy()
    far_field.reactivate(
        world, plate, _push([2], [100.0], [0], [1000.0]), 0.0
    )
    assert np.all(plate.crust.rugged >= before_rugged)
    assert np.all(plate.crust.upliftKm >= before_uplift)
    assert plate.crust.rugged[2] > 0.0
    assert plate.crust.upliftKm[2] > 0.0


def test_old_range_predicate_requires_a_quiet_non_cratonic_belt():
    from orogen.v2.tectonics.far_field import _old_range

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    world.config.update(_reactivation_config())
    crust = plate.crust
    assert not _old_range(world, crust, 2)
    crust.oroType[2] = 6
    crust.beltReach[2] = 100.0
    crust.oroLast[2] = 0.0
    assert not _old_range(world, crust, 2)
    crust.oroType[2] = 3
    assert _old_range(world, crust, 2)
    crust.oroLast[2] = world.time
    assert not _old_range(world, crust, 2)
    crust.oroLast[2] = 0.0
    crust.type[2] = CrustType["CRATON"]
    assert not _old_range(world, crust, 2)


def test_evolve_far_field_runs_reactivation_with_the_spread_window(monkeypatch):
    from orogen.v2.tectonics import far_field

    world, plate = make_world()
    world.config.update(_reactivation_config())
    calls = []
    monkeypatch.setattr(
        far_field, "spread_push",
        lambda *_args: {"cells": [1], "inc": [1.0], "from": [0], "dist": [500.0]},
    )
    monkeypatch.setattr(
        far_field, "reactivate",
        lambda w, p, push, window: calls.append((w, p, push, window)),
    )
    feed_far_field(world, plate, "front", [0], 10.0, 200.0, 1.0)
    evolve_far_field(world, 2.0)
    assert len(calls) == 1
    assert calls[0][0] is world
    assert calls[0][1] is plate
    assert calls[0][3] == pytest.approx(2.0)


def test_rejuvenation_handles_multiple_sources_and_rejects_zero_uplift_scale():
    from orogen.v2.tectonics import far_field

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    world.config.update(_reactivation_config())
    crust = plate.crust
    for cell in range(5):
        crust.oroType[cell] = 3
        crust.oroLast[cell] = 0.0
        crust.beltReach[cell] = 200.0
        crust.beltU[cell] = 0.2
        crust.beltSide[cell] = 1
    far_field._rejuvenate(
        world, plate, [2, 2], [150.0, 100.0], [50.0, 40.0]
    )
    assert np.any(crust.rugged > 0.0)
    world.config["heights"] = {
        "upliftBuildKm": {8: 0.0},
    }
    with pytest.raises(ValueError, match="must be positive"):
        far_field._rejuvenate(world, plate, [2], [150.0], [50.0])



def test_reactivate_rejects_non_positive_near_field_scale():
    from orogen.v2.tectonics import far_field

    world, plate = make_world()
    world.config.update(_reactivation_config())
    world.config["farNearKm"] = 0.0
    with pytest.raises(ValueError, match="farNearKm and farBeltWorkKm"):
        far_field.reactivate(world, plate, _push([], [], [], []), 1.0)


def test_rejuvenation_does_not_cross_cells_outside_the_old_belt():
    from orogen.v2.tectonics import far_field

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    world.config.update(_reactivation_config())
    crust = plate.crust
    crust.oroType[2] = 3
    crust.oroLast[2] = 0.0
    crust.beltReach[2] = 200.0
    crust.beltU[2] = 0.1
    far_field._rejuvenate(world, plate, [2], [120.0], [50.0])
    assert crust.rugged[2] > 0.0
    assert crust.rugged[1] == 0.0
    assert crust.rugged[3] == 0.0


def test_rejuvenation_skips_ocean_cells():
    from orogen.v2.tectonics import far_field

    world, plate = make_world([CrustType["CONTINENT"]] * 5)
    world.config.update(_reactivation_config())
    crust = plate.crust
    crust.type[2] = CrustType["OCEAN"]
    crust.oroType[2] = 3
    crust.oroLast[2] = 0.0
    crust.beltReach[2] = 200.0
    before = crust.rugged.copy()
    far_field._rejuvenate(world, plate, [2], [120.0], [50.0])
    assert np.array_equal(crust.rugged, before)
    assert np.array_equal(crust.upliftKm, np.zeros(5, dtype=np.float32))
