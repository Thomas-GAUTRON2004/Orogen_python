from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics import rifting
from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.rifting import (
    _force_rift,
    _heat_under_continents,
    _with_loose_shape,
    pace_of,
    rift_continents,
    search_junctions,
)


def base_config(**overrides):
    config = {
        "riftRatePerMyr": 1.0,
        "riftMinGapMyr": 30.0,
        "riftGlobalGapMyr": 20.0,
        "riftMinContinentKm2": 100.0,
        "riftMaxGapMyr": 250.0,
        "plumeRiftWaitMyr": 30.0,
        "superTenureMyr": 40.0,
        "riftTortuosityMax": 5.0,
        "riftMinChordKm": 500.0,
        "riftMinPieceShare": 0.1,
        "riftCoastHugKm": 100.0,
        "riftSpacingMyr": 60.0,
        "riftDueMyr": 100.0,
        "riftDueRampMyr": 50.0,
        "riftSizeCap": 3.0,
        "riftTenureMyr": 60.0,
        "riftTenureHotMyr": 20.0,
        "riftHeatGain": 0.0,
        "riftHeatFloor": 0.2,
        "riftHeatPower": 1.0,
        "riftHeatTension": 0.5,
        "dispersalRiftBoost": 2.0,
        "gatherRiftFactor": 0.3,
    }
    config.update(overrides)
    return config


def make_world(*, time=100.0, config=None, dynamics=None, cells=4):
    plate = Plate(0, cells, born=0.0)
    plate.name = "P"
    for cell in range(cells):
        plate.crust.create(cell, CrustType["CONTINENT"], 0.0, cell + 1)
    plate.stats["continentalCells"] = cells
    plate.dynamics = dynamics
    mesh = SimpleNamespace(
        cell_area_km2=100.0,
        edge_km=100.0,
        num_cells=cells,
        adj_offset=np.zeros(cells + 1, dtype=np.int32),
        adj_list=np.zeros(0, dtype=np.int32),
    )
    return SimpleNamespace(
        mesh=mesh,
        plates=[plate],
        time=float(time),
        config=base_config() if config is None else config,
        cycle=None,
        pendingPlumeRifts=[],
        lastRiftAt=None,
        rng=lambda: 0.0,
        events=[],
    )


def test_pace_of_clamps_before_min_gap_and_smoothly_rises():
    config = base_config()
    assert pace_of(10.0, config) == 0.0
    assert 0.0 < pace_of(45.0, config) < 1.0
    assert pace_of(100.0, config) == 1.0
    assert pace_of(150.0, config) > 1.0


def test_search_junctions_preserves_order_and_skips_negative_cells(monkeypatch):
    world = SimpleNamespace(config={})
    plate = object()
    request = {
        "plate": plate,
        "junctions": [-1, 4, 9],
        "pull": [1, 0, 0],
        "needCratonsBothSides": False,
        "ctx": {"cached": True},
    }
    calls = []
    monkeypatch.setattr(
        "orogen.v2.tectonics.ops.rift.find_rift",
        lambda _w, _p, junction, pull, need, ctx: (
            calls.append((junction, pull, need, ctx)) or {"value": junction}
        ),
    )
    result = search_junctions(world, request)
    assert result == [None, {"value": 4}, {"value": 9}]
    assert [call[0] for call in calls] == [4, 9]
    assert all(call[2] is False and call[3] is request["ctx"] for call in calls)


def test_heat_under_continents_averages_only_owned_continental_cells():
    world = make_world(cells=3)
    world.owner = np.asarray([0, 0, -1], dtype=np.int16)
    world.ownerCell = np.asarray([0, 1, 2], dtype=np.int32)
    world.plates[0].crust.type[1] = CrustType["OCEAN"]
    world.mantleHeat = np.asarray([0.2, 0.9, 0.7], dtype=np.float32)
    assert _heat_under_continents(world) == {0: pytest.approx(0.2)}


def test_pending_plume_rift_executes_and_clears_request(monkeypatch):
    world = make_world(config=base_config(riftRatePerMyr=0.0), dynamics=None)
    world.pendingPlumeRifts = [
        {
            "plate": 0,
            "cell": 2,
            "xyz": [1.0, 0.0, 0.0],
            "cause": {"text": "plume", "kind": "plume", "hotspot": 3},
        }
    ]
    calls = []
    monkeypatch.setattr(
        rifting,
        "try_rift",
        lambda _w, _p, junctions, cause, need: (
            calls.append((list(junctions), cause, need)) or True
        ),
    )
    assert rift_continents(world, 1.0) == 1
    assert world.pendingPlumeRifts == []
    assert calls[0][0] == [2]
    assert calls[0][2] is False


def test_pending_plume_rift_waits_while_piece_is_too_young():
    config = base_config(
        riftMinGapMyr=30.0,
        riftMinContinentKm2=100.0,
        plumeRiftWaitMyr=30.0,
    )
    world = make_world(time=5.0, config=config)
    world.pendingPlumeRifts = [
        {
            "plate": 0,
            "cell": 2,
            "since": 0.0,
            "junctions": [2],
            "cause": {"text": "plume", "kind": "plume", "hotspot": 1},
        }
    ]
    assert rift_continents(world, 1.0) == 0
    assert len(world.pendingPlumeRifts) == 1


def test_pending_plume_rift_expires_after_wait_window():
    config = base_config(
        riftMinGapMyr=150.0,
        riftMaxGapMyr=500.0,
        riftMinContinentKm2=1000.0,
        plumeRiftWaitMyr=20.0,
    )
    world = make_world(time=50.0, config=config)
    world.pendingPlumeRifts = [
        {
            "plate": 0,
            "cell": 2,
            "since": 0.0,
            "junctions": [2],
            "cause": {"text": "plume", "kind": "plume", "hotspot": 1},
        }
    ]
    assert rift_continents(world, 1.0) == 0
    assert world.pendingPlumeRifts == []


def test_pending_plume_rift_retries_with_loose_shape_and_restores_config(monkeypatch):
    config = base_config(
        riftRatePerMyr=0.0,
        riftTortuosityMax=8.0,
        riftMinChordKm=700.0,
        riftMinPieceShare=0.2,
        riftCoastHugKm=300.0,
    )
    world = make_world(time=10.0, config=config)
    world.pendingPlumeRifts = [
        {
            "plate": 0,
            "cell": 2,
            "since": 0.0,
            "junctions": [2],
            "cause": {"text": "plume", "kind": "plume", "hotspot": 1},
        }
    ]
    outcomes = iter([False, True])
    calls = []

    def fake_try(_world, _plate, _junctions, _cause, _need):
        calls.append(dict(world.config))
        return next(outcomes)

    monkeypatch.setattr(rifting, "try_rift", fake_try)
    assert rift_continents(world, 1.0) == 1
    assert len(calls) == 2
    assert calls[0]["riftTortuosityMax"] == 8.0
    assert calls[1]["riftTortuosityMax"] == 2.0
    assert world.config["riftTortuosityMax"] == 8.0
    assert world.config["riftMinChordKm"] == 700.0


def test_world_global_gap_keeps_plume_request_waiting():
    config = base_config(riftGlobalGapMyr=30.0, plumeRiftWaitMyr=20.0)
    world = make_world(time=5.0, config=config)
    world.lastRiftAt = 0.0
    world.pendingPlumeRifts = [
        {
            "plate": 0,
            "cell": 2,
            "since": 0.0,
            "junctions": [2],
            "cause": {"text": "plume", "kind": "plume", "hotspot": 1},
        }
    ]
    assert rift_continents(world, 1.0) == 0
    assert len(world.pendingPlumeRifts) == 1


def test_due_plate_gets_forced_rift_before_spontaneous_pass(monkeypatch):
    world = make_world(
        time=200.0, config=base_config(riftMaxGapMyr=150.0), dynamics={"tension": 0.4}
    )
    world.plates[0].bigSince = 0.0
    calls = []
    monkeypatch.setattr(
        rifting,
        "_force_rift",
        lambda _world, plate, age: calls.append((plate.id, age)) or True,
    )
    assert rift_continents(world, 1.0) == 1
    assert calls == [(0, 200.0)]


def test_spontaneous_rift_uses_tension_size_and_seeded_junctions(monkeypatch):
    world = make_world(
        time=100.0,
        config=base_config(
            riftRatePerMyr=2.0,
            riftMinGapMyr=30.0,
            riftMaxGapMyr=1000.0,
            riftMinContinentKm2=100.0,
            riftSpacingMyr=60.0,
            riftDueMyr=100.0,
        ),
        dynamics={"tension": 0.6},
    )
    selected = []
    monkeypatch.setattr(rifting, "choose_junction", lambda _w, _p, _rng: 7)
    monkeypatch.setattr(
        rifting,
        "try_rift",
        lambda _w, _p, junctions, cause: selected.append((junctions, cause)) or True,
    )
    assert rift_continents(world, 1.0) == 1
    assert selected[0][0] == [7] * 12
    assert selected[0][1]["kind"] == "tension"


def test_spontaneous_rift_uses_mantle_heat_gather_and_dispersal(monkeypatch):
    cycle = {
        "phase": "dispersing",
        "gather": 0.5,
        "largestPlates": [0],
        "clump": [0],
    }
    config = base_config(
        riftRatePerMyr=2.0,
        riftMinGapMyr=30.0,
        riftMaxGapMyr=1000.0,
        riftMinContinentKm2=100.0,
        riftHeatGain=0.8,
        riftHeatFloor=0.3,
        riftHeatPower=1.0,
        riftHeatTension=0.7,
        dispersalRiftBoost=2.0,
        gatherRiftFactor=0.25,
        riftTenureMyr=60.0,
        riftTenureHotMyr=20.0,
    )
    world = make_world(
        time=100.0,
        config=config,
        dynamics={"tension": 0.2},
    )
    world.cycle = cycle
    monkeypatch.setattr(rifting, "_heat_under_continents", lambda _w: {0: 0.9})
    monkeypatch.setattr(rifting, "choose_junction", lambda *_args: 3)
    seen = []
    monkeypatch.setattr(
        rifting,
        "try_rift",
        lambda _w, _p, junctions, cause: seen.append(cause) or True,
    )
    assert rift_continents(world, 1.0) == 1
    assert "mantle heat 0.90" in seen[0]["text"]


def test_force_rift_tries_craton_free_and_loose_shapes_then_restores(monkeypatch):
    world = make_world(config=base_config())
    monkeypatch.setattr(rifting, "choose_junction", lambda *_args: 1)
    outcomes = iter([False, False, False, True])
    calls = []

    def fake_try(_world, _plate, junctions, _cause, need=True):
        calls.append((len(junctions), need, world.config["riftTortuosityMax"]))
        return next(outcomes)

    monkeypatch.setattr(rifting, "try_rift", fake_try)
    assert _force_rift(world, world.plates[0], 150.0) is True
    assert calls == [
        (48, True, 5.0),
        (48, False, 5.0),
        (48, True, 2.0),
        (48, False, 2.0),
    ]
    assert world.config["riftTortuosityMax"] == 5.0


def test_loose_shape_restores_configuration_even_when_action_raises():
    config = base_config()
    with pytest.raises(RuntimeError):
        _with_loose_shape(
            config,
            lambda: (_ for _ in ()).throw(RuntimeError("failure")),
        )
    assert config["riftTortuosityMax"] == 5.0
    assert config["riftMinChordKm"] == 500.0


def test_rift_continents_short_circuits_when_disabled_and_no_pending():
    world = make_world(config=base_config(riftRatePerMyr=0.0))
    assert rift_continents(world, 1.0) == 0


def test_dead_plate_is_skipped_for_lifecycle_and_pending_requests():
    world = make_world(config=base_config())
    world.plates[0].alive = False
    world.pendingPlumeRifts = [
        {
            "plate": 0,
            "cell": 1,
            "since": 0.0,
            "junctions": [1],
            "cause": {"text": "plume", "kind": "plume", "hotspot": 1},
        }
    ]
    assert rift_continents(world, 1.0) == 0
    assert world.pendingPlumeRifts == []


def test_assembled_supercontinent_holds_plume_rift_until_tenure_expires():
    config = base_config(
        riftMinGapMyr=5.0,
        superTenureMyr=40.0,
        plumeRiftWaitMyr=30.0,
    )
    world = make_world(time=10.0, config=config)
    world.cycle = {
        "phase": "assembled",
        "formedAt": 1.0,
        "largestPlates": [0],
        "gather": 0.0,
    }
    world.pendingPlumeRifts = [
        {
            "plate": 0,
            "cell": 1,
            "since": 0.0,
            "junctions": [1],
            "cause": {"text": "plume", "kind": "plume", "hotspot": 1},
        }
    ]
    assert rift_continents(world, 1.0) == 0
    assert len(world.pendingPlumeRifts) == 1


def test_second_paced_plume_request_waits_after_first_split(monkeypatch):
    config = base_config(riftMinGapMyr=1.0, plumeRiftWaitMyr=30.0)
    world = make_world(time=50.0, config=config)
    request = {
        "plate": 0,
        "cell": 1,
        "since": 49.0,
        "junctions": [1],
        "cause": {"text": "plume", "kind": "plume", "hotspot": 1},
    }
    world.pendingPlumeRifts = [dict(request), dict(request)]
    calls = []
    monkeypatch.setattr(
        rifting,
        "try_rift",
        lambda *_args: calls.append(True) or True,
    )
    assert rift_continents(world, 1.0) == 1
    assert len(calls) == 1
    assert len(world.pendingPlumeRifts) == 1


def test_unpaced_recently_clumped_piece_uses_hot_tenure():
    config = base_config(
        riftRatePerMyr=1.0,
        riftMinGapMyr=0.0,
        riftTenureHotMyr=20.0,
    )
    world = make_world(time=10.0, config=config, dynamics={"tension": 1.0})
    world.cycle = {"clump": [0]}
    assert rift_continents(world, 1.0) == 0


def test_unpaced_hot_rift_born_piece_uses_interpolated_tenure(monkeypatch):
    config = base_config(
        riftRatePerMyr=1.0,
        riftMinGapMyr=0.0,
        riftHeatGain=1.0,
        riftTenureMyr=60.0,
        riftTenureHotMyr=20.0,
    )
    world = make_world(time=10.0, config=config, dynamics={"tension": 1.0})
    world.plates[0].bornOfRift = True
    monkeypatch.setattr(rifting, "_heat_under_continents", lambda _w: {0: 1.0})
    assert rift_continents(world, 1.0) == 0


def test_unpaced_cool_piece_uses_base_tenure(monkeypatch):
    config = base_config(
        riftRatePerMyr=1.0,
        riftMinGapMyr=0.0,
        riftTenureMyr=30.0,
        riftTenureHotMyr=20.0,
    )
    world = make_world(time=10.0, config=config, dynamics={"tension": 1.0})
    monkeypatch.setattr(rifting, "_heat_under_continents", lambda _w: None)
    assert rift_continents(world, 1.0) == 0


def test_spontaneous_rift_is_skipped_when_random_draw_exceeds_chance(monkeypatch):
    config = base_config(
        riftRatePerMyr=0.01,
        riftMinGapMyr=0.0,
        riftTenureMyr=0.0,
        riftTenureHotMyr=0.0,
        riftMaxGapMyr=1000.0,
    )
    world = make_world(
        time=100.0,
        config=config,
        dynamics={"tension": 0.1},
    )
    world.rng = lambda: 0.99
    monkeypatch.setattr(rifting, "try_rift", lambda *_args: pytest.fail("must not try"))
    assert rift_continents(world, 1.0) == 0


def test_disabled_rifting_keeps_updating_big_continent_clock():
    world = make_world(
        time=40.0,
        config=base_config(riftRatePerMyr=0.0),
    )
    assert rift_continents(world, 1.0) == 0
    assert world.plates[0].bigSince == 40.0


def test_rifting_stage_short_circuits_only_without_rate_configuration():
    world = make_world(config={})
    assert rift_continents(world, 1.0) == 0

def test_pending_plume_request_is_discarded_when_both_rift_attempts_fail(monkeypatch):
    world = make_world(time=100.0, config=base_config())
    world.pendingPlumeRifts = [
        {
            "plate": 0,
            "cell": 1,
            "since": 0.0,
            "junctions": [1],
            "cause": {"text": "plume", "kind": "plume", "hotspot": 2},
        }
    ]
    monkeypatch.setattr(rifting, "try_rift", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(rifting, "_retry_plume_rift", lambda *_args: False)

    assert rift_continents(world, 1.0) == 0
    assert world.pendingPlumeRifts == []


def test_unpaced_spontaneous_rift_continues_after_a_failed_attempt(monkeypatch):
    world = make_world(
        time=100.0,
        config=base_config(
            riftRatePerMyr=100.0,
            riftMinGapMyr=0.0,
            riftMaxGapMyr=1000.0,
        ),
        dynamics={"tension": 1.0},
    )
    monkeypatch.setattr(rifting, "choose_junction", lambda *_args: 3)
    monkeypatch.setattr(rifting, "try_rift", lambda *_args, **_kwargs: False)

    assert rift_continents(world, 1.0) == 0
