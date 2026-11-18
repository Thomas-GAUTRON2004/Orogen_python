from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.stepper import step_world


class Mesh:
    pass


def make_world():
    mesh = Mesh()
    mesh.num_cells = 4
    mesh.xyz = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
        ],
        dtype=np.float32,
    )
    mesh.edge_rad = 0.5
    mesh.edge_km = 100.0
    mesh.cell_area_km2 = 10000.0
    mesh.adj_offset = np.array([0, 1, 3, 5, 6], dtype=np.int32)
    mesh.adj_list = np.array([1, 0, 2, 1, 3, 2], dtype=np.int32)
    mesh.index = SimpleNamespace(nearest=lambda x, y, z: 0)

    plate = Plate(0, 4)
    for cell in range(4):
        plate.crust.create(cell, CrustType["CONTINENT"], 0.0)
    plate.omega = [0.0, 0.0, 0.01]
    plate.refresh_extent(mesh.xyz)

    world = SimpleNamespace(
        mesh=mesh,
        plates=[plate],
        time=0.0,
        step=0,
        lastDt=0.0,
        occupancyStamp=0,
        owner=np.full(4, -1, dtype=np.int16),
        ownerCell=np.zeros(4, dtype=np.int32),
        claimCount=np.zeros(4, dtype=np.uint8),
        landClaims=np.zeros(4, dtype=np.uint8),
        collider=np.full(4, -1, dtype=np.int16),
        oceanOverrider=np.full(4, -1, dtype=np.int16),
        consumedAt=np.zeros(4, dtype=np.uint8),
        provisional=np.zeros(4, dtype=np.uint8),
        boundaryKind=np.zeros(4, dtype=np.uint8),
        convergence=np.zeros(4, dtype=np.float32),
        budget={
            "created": 0,
            "consumed": 0,
            "erodedCells": 0,
            "underthrustCells": 0,
            "recycledCells": 0,
        },
        config={
            "boundaryActiveCmYr": 0.3,
            "obliqueLimit": 0.35,
            "segmentPolarity": False,
            "maxStepMyr": 4.0,
            "maxCellsPerStep": 1.5,
            "maxSpeedOceanic": 25.0,
            "relaxMyr": 6.0,
            "mantleInsulateMyr": 100.0,
            "mantleCoolMyr": 200.0,
            "mobility": 1.0,
            "dragOcean": 1.0,
            "dragContinent": 5.0,
            "dragCraton": 8.0,
            "continentalPlateShare": 0.15,
            "continentalSoftCmYr": 7.0,
            "oceanicSoftCmYr": 12.0,
            "maxSpeedContinental": 10.0,
        },
        events=[],
        collisions={},
        arcScrapes={},
        mantleHeat=np.zeros(4, dtype=np.float32),
        hotspotsReady=False,
        rngCycle=lambda: 0.25,
        log=lambda *args, **kwargs: None,
    )
    return world


def test_step_world_advances_resolves_and_updates_motion(monkeypatch):
    world = make_world()
    called = []
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.resolve", lambda w: {"created": 2, "consumed": 1}
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.classify_contact_boundaries", lambda w, dt: 3
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.track_cycle", lambda w: {"phase": "dispersing"}
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plate_motion", lambda w: called.append(w)
    )
    result = step_world(world, 2.0)
    assert world.time == pytest.approx(2.0)
    assert world.step == 1
    assert world.lastDt == 2.0
    assert world.plates[0].quat != [1.0, 0.0, 0.0, 0.0]
    assert result["resolve"] == {"created": 2, "consumed": 1}
    assert result["classifiedBoundaries"] == 3
    assert result["cyclePhase"] == "dispersing"
    assert result["motionUpdated"] is True
    assert called == [world]


def test_step_world_adaptive_dt_and_external_motion(monkeypatch):
    world = make_world()
    selected = []
    monkeypatch.setattr("orogen.v2.tectonics.stepper.choose_step", lambda w: 1.25)
    monkeypatch.setattr("orogen.v2.tectonics.stepper.resolve", lambda w: {})
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.classify_contact_boundaries", lambda w, dt: 0
    )
    monkeypatch.setattr("orogen.v2.tectonics.stepper.track_cycle", lambda w: None)
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plate_motion",
        lambda w: selected.append(True),
    )
    result = step_world(world, update_motion=False, update_cycle=False)
    assert result["dt"] == pytest.approx(1.25)
    assert result["cyclePhase"] is None
    assert result["motionUpdated"] is False
    assert selected == []


@pytest.mark.parametrize("dt", [0.0, -1.0, float("inf"), float("nan")])
def test_step_world_rejects_invalid_dt(dt):
    with pytest.raises(ValueError):
        step_world(make_world(), dt)


def test_step_world_runs_rifting_when_enabled(monkeypatch):
    world = make_world()
    world.config["riftRatePerMyr"] = 0.2
    monkeypatch.setattr("orogen.v2.tectonics.stepper.resolve", lambda _w: {})
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.classify_contact_boundaries", lambda _w, _dt: 0
    )
    monkeypatch.setattr("orogen.v2.tectonics.stepper.track_cycle", lambda _w: None)
    monkeypatch.setattr("orogen.v2.tectonics.stepper.rift_continents", lambda _w, dt: 2)
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plate_motion", lambda _w: None
    )
    result = step_world(world, 1.0)
    assert result["rifts"] == 2


def test_step_world_runs_subduction_stage_and_rebuilds_after_topology_changes(
    monkeypatch,
):
    world = make_world()
    world.config["subductionInitRatePerMyr"] = 0.5
    world.config["riftRatePerMyr"] = 0.0
    calls = []

    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.resolve",
        lambda _world: calls.append("resolve") or {"created": 0},
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.classify_contact_boundaries",
        lambda _world, dt: calls.append(("classify", dt)) or 0,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plumes",
        lambda _world, _dt: None,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.track_cycle",
        lambda _world: None,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.rift_continents",
        lambda _world, _dt: 1,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.initiate_subduction",
        lambda _world, _dt: 2,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plate_motion",
        lambda _world: calls.append("motion"),
    )

    result = step_world(world, 1.0)

    assert result["rifts"] == 1
    assert result["subductions"] == 2
    assert result["resolve"] == {"created": 0}
    assert result["subductions"] == 2
    assert calls.count("resolve") == 2
    assert calls.count(("classify", 0.0)) == 1
    assert calls[-1] == "motion"


def test_step_world_runs_subduction_without_rifting_config(monkeypatch):
    world = make_world()
    world.config["subductionInitRatePerMyr"] = 0.5
    subductions = []
    monkeypatch.setattr("orogen.v2.tectonics.stepper.resolve", lambda _world: {})
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.classify_contact_boundaries", lambda _world, _dt: 0
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plumes", lambda _world, _dt: None
    )
    monkeypatch.setattr("orogen.v2.tectonics.stepper.track_cycle", lambda _world: None)
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.initiate_subduction",
        lambda _world, dt: subductions.append(dt) or 1,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plate_motion", lambda _world: None
    )

    result = step_world(world, 1.0)

    assert result["rifts"] == 0
    assert result["subductions"] == 1
    assert subductions == [1.0]


@pytest.mark.parametrize("changed", [False, True])
def test_step_world_schedules_collision_tracking_and_rebuilds_after_crust_changes(
    monkeypatch, changed
):
    world = make_world()
    world.config["collisionForgetMyr"] = 8.0
    calls = []

    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.resolve",
        lambda _world: calls.append("resolve") or {"created": 0},
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.classify_contact_boundaries",
        lambda _world, dt: calls.append(("classify", dt)) or 0,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plumes",
        lambda _world, _dt: None,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.track_cycle",
        lambda _world: None,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.track_collisions",
        lambda _world, dt: (
            calls.append(("collisions", dt))
            or {
                "tracked": 1,
                "started": 1,
                "underthrustCells": 2 if changed else 0,
                "changed": changed,
            }
        ),
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plate_motion",
        lambda _world: calls.append("motion"),
    )

    result = step_world(world, 2.0)

    assert result["collisions"]["tracked"] == 1
    assert ("collisions", 2.0) in calls
    assert calls.count("resolve") == (2 if changed else 1)
    assert calls.count(("classify", 0.0)) == (1 if changed else 0)
    assert calls[-1] == "motion"


def test_step_world_schedules_far_field_clock(monkeypatch):
    world = make_world()
    world.config["farFieldEveryMyr"] = 4.0
    calls = []
    monkeypatch.setattr("orogen.v2.tectonics.stepper.resolve", lambda _world: {})
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.classify_contact_boundaries",
        lambda _world, _dt: 0,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plumes", lambda _world, _dt: None
    )
    monkeypatch.setattr("orogen.v2.tectonics.stepper.track_cycle", lambda _world: None)
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.evolve_far_field",
        lambda _world, dt: calls.append(dt) or 0,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plate_motion", lambda _world: None
    )

    result = step_world(world, 1.5, update_motion=False)

    assert result["farField"] == 0
    assert calls == [1.5]



def test_step_world_runs_back_arc_stage_and_steers_after_motion(monkeypatch):
    world = make_world()
    calls = []
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.resolve",
        lambda _world: calls.append("resolve") or {},
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.classify_contact_boundaries",
        lambda _world, _dt: calls.append("classify") or 0,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plumes",
        lambda _world, _dt: None,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.track_cycle",
        lambda _world: None,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.open_back_arcs",
        lambda _world, dt: calls.append(("backarcs", dt)) or 1,
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.update_plate_motion",
        lambda _world: calls.append("motion"),
    )
    monkeypatch.setattr(
        "orogen.v2.tectonics.stepper.steer_back_arc_strips",
        lambda _world: calls.append("steer") or 1,
    )

    result = step_world(world, 1.0)

    assert result["backArcs"] == 1
    assert calls.count("resolve") == 2
    assert calls.index(("backarcs", 1.0)) < calls.index("motion") < calls.index("steer")
