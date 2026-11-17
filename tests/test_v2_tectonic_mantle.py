import math
from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.mantle import (
    dispersal_field, flow_of_cycle, mantle_flow_at, repel_at,
)
from orogen.v2.tectonics.forces import FORCE_KINDS, gather_boundary_loads
from orogen.v2.tectonics.plate import Plate


def test_flow_of_cycle_outflow_and_inflow_are_tangent_and_directional():
    up = [1.0, 0.0, 0.0]
    down = [0.0, 1.0, 0.0]
    config = {"mantleOutflow": 0.1, "mantleInflow": 0.2, "mantleInflowPackKm": 1500}
    cycle = {"up": up, "down": down, "upHeat": 1.0, "gather": 0.0, "hold": 0.0}
    outflow = flow_of_cycle(cycle, config, math.sqrt(0.5), math.sqrt(0.5), 0.0)
    assert outflow[0] < 0
    assert outflow[1] > 0
    assert abs(sum(outflow[i] * [math.sqrt(0.5), math.sqrt(0.5), 0][i] for i in range(3))) < 1e-12

    cycle["upHeat"] = 0.0
    cycle["gather"] = 1.0
    gathered = flow_of_cycle(cycle, config, 1.0, 0.0, 0.0)
    assert gathered[1] > 0
    assert gathered[0] == pytest.approx(0.0)

    cycle["gather"] = 0.0
    cycle["hold"] = 1.0
    cycle["holdKm"] = 1000.0
    held = flow_of_cycle(cycle, config, 0.99, 0.1, 0.0)
    assert held[1] < 0


def test_flow_of_cycle_no_upwelling_and_world_wrapper_zero_output():
    assert flow_of_cycle(None, {}, 1, 0, 0) == [0.0, 0.0, 0.0]
    out = [5.0, 5.0, 5.0]
    world = SimpleNamespace(config={}, cycle=None)
    assert mantle_flow_at(world, 1, 0, 0, out) == [0.0, 0.0, 0.0]
    cycle = {"up": [1.0, 0.0, 0.0], "upHeat": 1.0, "gather": 0.0, "hold": 0.0}
    assert flow_of_cycle(cycle, {}, 1.0, 0.0, 0.0) == [0.0, 0.0, 0.0]


def make_dispersal_world():
    plates = [Plate(i, 2) for i in range(2)]
    for plate in plates:
        plate.crust.create(0, CrustType["CONTINENT"], 0)
        plate.crust.create(1, CrustType["CRATON"], 0)
    xyz = np.array([
        1.0, 0.0, 0.0, 0.99, 0.1, 0.0,
        0.0, 1.0, 0.0, 0.1, 0.99, 0.0,
    ])
    xyz = xyz.reshape(-1, 3)
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    mesh = SimpleNamespace(num_cells=4, xyz=xyz.ravel(), cell_area_km2=600000.0, edge_km=173.20508075688772)
    world = SimpleNamespace(
        mesh=mesh, plates=plates,
        owner=np.array([0, 0, 1, 1]),
        ownerCell=np.array([0, 1, 0, 1]),
        landClaims=np.zeros(4, dtype=np.uint8),
        collider=np.full(4, -1, dtype=np.int16),
        config={"dispersalRepulsion": 0.3, "dispersalRepulsionKm": 2000.0},
        cycle={"phase": "dispersing"},
    )
    return world


def test_dispersal_field_builds_large_plate_centres_and_repels_away():
    world = make_dispersal_world()
    field = dispersal_field(world)
    assert field is not None
    assert {item["id"] for item in field["centres"]} == {0, 1}
    vector = repel_at(field, 0, 1.0, 0.0, 0.0)
    assert vector[0] > 0
    assert np.linalg.norm(vector) > 0
    assert np.dot(vector, [1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_dispersal_field_honours_threshold_touching_and_disabled_phase():
    world = make_dispersal_world()
    world.config["dispersalRepulsion"] = 0.0
    assert dispersal_field(world) is None
    world.config["dispersalRepulsion"] = 0.3
    world.cycle["phase"] = "gathering"
    assert dispersal_field(world) is None
    world.cycle["phase"] = "dispersing"
    world.plates[1].isBackground = True
    field = dispersal_field(world)
    assert [item["id"] for item in field["centres"]] == [0]
    assert repel_at(field, 0, 1.0, 0.0, 0.0) == [0.0, 0.0, 0.0]
    world.plates[1].isBackground = False
    world.landClaims[0] = 2
    world.collider[0] = 1
    field = dispersal_field(world)
    assert repel_at(field, 0, 1.0, 0.0, 0.0) == [0.0, 0.0, 0.0]
    assert repel_at(None, 0, 1, 0, 0) == [0.0, 0.0, 0.0]
    assert repel_at({"centres": [], "touching": set(), "k": 1, "reach": 0}, 0, 1, 0, 0) == [0.0, 0.0, 0.0]


def test_gather_boundary_loads_adds_mantle_force_only_to_continent_craton(monkeypatch):
    world = make_dispersal_world()
    world.owner = np.array([0, 0, 1, 1])
    world.ownerCell = np.array([0, 1, 0, 1])
    world.cycle = {
        "up": [1.0, 0.0, 0.0], "down": [0.0, 1.0, 0.0],
        "upHeat": 1.0, "gather": 0.0, "hold": 0.0,
        "phase": "assembled",
    }
    world.config.update({
        "mantleOutflow": 0.1, "mantleInflow": 0.0,
        "slabPull": 1.0, "slabSuction": 0.25, "slabPullAgeRefMyr": 80.0,
        "slabMatureKm": 600.0, "ridgeMaxAgeMyr": 40.0,
        "riftPushMyr": 20.0, "riftPush": 0.5, "ridgePush": 0.05,
    })
    world.time = 100.0
    world.plates[0].crust.type[1] = CrustType["OCEAN"]
    monkeypatch.setattr("orogen.v2.tectonics.forces.contact_at", lambda *args: -1)
    loads = gather_boundary_loads(world)
    assert loads[0]["cells"][4] == 1
    assert loads[1]["cells"][4] == 2
    assert len(loads[0]["byKind"]) == len(FORCE_KINDS) == 5
    assert len(loads[1]["forceByKind"]) == 5
    assert np.linalg.norm(loads[1]["flowTorque"]) > 0
