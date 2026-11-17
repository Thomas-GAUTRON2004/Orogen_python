from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.supercontinent import track_cycle


class Mesh:
    pass


def make_world():
    mesh = Mesh()
    mesh.num_cells = 6
    angles = np.arange(6, dtype=float) * 0.02
    xyz = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6)))
    mesh.xyz = xyz.astype(np.float32).ravel()
    mesh.adj_offset = np.array([0, 1, 3, 5, 7, 9, 10], dtype=np.int32)
    mesh.adj_list = np.array([1, 0, 2, 1, 3, 2, 4, 3, 5, 4], dtype=np.int32)
    mesh.cell_area_km2 = 300000.0
    mesh.edge_km = 100.0

    from orogen.v2.tectonics.plate import Plate
    plate = Plate(0, 6)
    plate.crust.create(1, CrustType["CONTINENT"], 0)
    plate.crust.create(2, CrustType["CONTINENT"], 0)
    plate.crust.create(3, CrustType["CRATON"], 0)
    plate.crust.create(4, CrustType["CONTINENT"], 0)
    cycle_random = iter([0.25, 0.5])
    events = []

    world = SimpleNamespace(
        mesh=mesh, plates=[plate],
        owner=np.array([-1, 0, 0, 0, 0, -1], dtype=np.int16),
        ownerCell=np.arange(6, dtype=np.int32),
        mantleHeat=np.arange(6, dtype=np.float32),
        time=0.0, events=events,
        rngCycle=lambda: next(cycle_random),
        config={
            "mantleOutflow": 0.025, "mantleInflow": 0.015, "riftHeatGain": 5.0,
            "cycleCheckMyr": 10.0, "cycleUpwellingKm": 20000.0,
            "cycleNeckKm": 400.0, "superShare": 0.7, "dispersedShare": 0.6,
            "cycleHoldMyr": 10.0, "cyclePeriodMinMyr": 600.0,
            "cyclePeriodMaxMyr": 720.0, "cycleGatherMyr": 150.0,
            "cycleDisperseMinMyr": 150.0, "cycleRampMyr": 30.0,
            "dispersalMaxShare": 0.45, "superHoldMarginKm": 500.0,
        },
    )

    def log(kind, message, data=None):
        world.events.append({
            "kind": kind, "text": message, "data": {} if data is None else data,
            "time": world.time,
        })

    world.log = log
    return world


def test_track_cycle_waits_for_sustained_supercontinent_then_assembles():
    world = make_world()
    cycle = track_cycle(world)
    assert cycle["up"] is None
    assert cycle["phase"] == "none"
    world.time = 10.0
    cycle = track_cycle(world)
    assert cycle["state"] == "assembled"
    assert cycle["phase"] == "assembled"
    assert cycle["period"] == pytest.approx(660.0)
    assert np.linalg.norm(cycle["up"]) == pytest.approx(1.0)
    assert np.linalg.norm(cycle["down"]) == pytest.approx(1.0)
    assert np.dot(cycle["up"], cycle["down"]) == pytest.approx(0.0, abs=1e-12)
    assert cycle["hold"] == 1.0
    assert cycle["upHeat"] == pytest.approx(2.5)
    assert world.events[-1]["kind"] == "supercontinent"
    world.config["cycleUpwellingKm"] = 1.0
    cycle["near"] = None
    world.time = 11.0
    cycle = track_cycle(world)
    assert len(cycle["near"]) == 0
    assert cycle["upHeat"] == 0.0


def test_track_cycle_schedules_rift_breakup_and_gathering_phase():
    world = make_world()
    track_cycle(world)
    world.time = 10.0
    cycle = track_cycle(world)
    world.events.append({"kind": "rift", "time": 20.0, "text": "rift", "data": {}})
    world.time = 20.0
    cycle = track_cycle(world)
    assert cycle["breakupAt"] == 20.0
    assert cycle["gatherAt"] == pytest.approx(520.0)
    assert cycle["phase"] == "dispersing"
    assert cycle["clump"] == [0]
    assert cycle["hold"] == 0.0

    world.time = 535.0
    cycle = track_cycle(world)
    assert cycle["phase"] == "gathering"
    assert cycle["gather"] == pytest.approx(0.5)
    assert cycle["clump"] is None


def test_track_cycle_logs_breakup_after_low_share_is_held():
    world = make_world()
    track_cycle(world)
    world.time = 10.0
    track_cycle(world)
    world.events.append({"kind": "rift", "time": 20.0, "text": "rift", "data": {}})
    world.time = 20.0
    track_cycle(world)
    world.owner[:] = -1
    world.time = 30.0
    cycle = track_cycle(world)
    assert cycle["state"] == "assembled"
    world.time = 40.0
    cycle = track_cycle(world)
    assert cycle["state"] == "dispersed"
    assert world.events[-1]["kind"] == "dispersed"
    assert cycle["share"] == 0.0


def test_track_cycle_ignores_old_rifts_and_disabled_cycle():
    world = make_world()
    track_cycle(world)
    world.time = 10.0
    track_cycle(world)
    world.events.append({"kind": "rift", "time": -1.0, "text": "old", "data": {}})
    world.time = 20.0
    cycle = track_cycle(world)
    assert cycle["breakupAt"] is None
    assert cycle["phase"] == "assembled"

    disabled = make_world()
    disabled.config.update({"mantleOutflow": 0.0, "mantleInflow": 0.0, "riftHeatGain": 0.0})
    assert track_cycle(disabled) is None
    assert not hasattr(disabled, "cycle")
