from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.forces import FORCE_KINDS, gather_boundary_loads
from orogen.v2.tectonics.plate import Plate


def make_world(monkeypatch, kind):
    born = 100.0 if kind == "rift" else 0.0
    a, b = Plate(0, 1, born=born), Plate(1, 1, born=born)
    type_a = CrustType["OCEAN"] if kind in ("pull", "ridge") else CrustType["CONTINENT"]
    type_b = CrustType["CONTINENT"] if kind == "pull" else CrustType["OCEAN"]
    a.crust.create(0, type_a, 0.0)
    b.crust.create(0, type_b, 0.0)
    if kind == "pull":
        a.crust.formedAt[0] = 0
        b.crust.arm_trench(0, 600)
    elif kind == "suction":
        a.crust.arm_trench(0, 600)
        b.crust.type[0] = CrustType["OCEAN"]
        b.crust.formedAt[0] = 0
    elif kind == "ridge":
        a.crust.formedAt[0] = 95
        b.crust.formedAt[0] = 95
    elif kind == "rift":
        a.crust.flags[0] |= CrustFlag["RIFT_MARGIN"]
        b.crust.flags[0] |= CrustFlag["RIFT_MARGIN"]
        a.riftPartner, b.riftPartner = 1, 0
    xyz = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
    mesh = SimpleNamespace(
        num_cells=2, xyz=xyz, edge_km=100.0 * np.sqrt(3.0), cell_area_km2=10000.0
    )
    world = SimpleNamespace(
        mesh=mesh, owner=np.array([0, 1]), ownerCell=np.array([0, 0]),
        landClaims=np.zeros(2, dtype=np.uint8), collider=np.full(2, -1),
        plates=[a, b], config={
            "slabPull": 1.0, "slabSuction": 0.25, "slabPullAgeRefMyr": 80.0,
            "slabMatureKm": 600.0, "ridgeMaxAgeMyr": 40.0,
            "riftPushMyr": 20.0, "riftPush": 0.5, "ridgePush": 0.05,
        }, time=100.0, occupancyStamp=1,
    )

    def contact(_world, cell, out):
        if cell == 0:
            out[:] = [0.0, 1.0, 0.0, 1.0]
            return 1
        return -1

    monkeypatch.setattr("orogen.v2.tectonics.forces.contact_at", contact)
    return world


def test_gather_boundary_loads_builds_slab_pull_and_slab_record(monkeypatch):
    world = make_world(monkeypatch, "pull")
    loads = gather_boundary_loads(world)
    load = loads[0]
    assert load["cells"][0] == 1
    assert load["torque"][2] == pytest.approx(1.0)
    assert load["slabs"][0]["other"] == 1
    assert load["slabs"][0]["length"] == pytest.approx(1.0)
    assert load["contacts"] == []
    assert len(FORCE_KINDS) == 4


def test_gather_boundary_loads_builds_suction_and_young_ridge(monkeypatch):
    world = make_world(monkeypatch, "suction")
    loads = gather_boundary_loads(world)
    assert loads[0]["cells"][1] == 1
    assert loads[0]["torque"][2] == pytest.approx(0.25)

    world = make_world(monkeypatch, "ridge")
    loads = gather_boundary_loads(world)
    assert loads[0]["cells"][2] == 1
    assert loads[0]["torque"][2] == pytest.approx(-0.05)


def test_gather_boundary_loads_detects_partner_rift_push_and_direction(monkeypatch):
    world = make_world(monkeypatch, "rift")
    loads = gather_boundary_loads(world)
    assert loads[0]["cells"][3] == 1
    assert loads[0]["torque"][2] == pytest.approx(-0.5)
    world.plates[0].riftPushDir = [0, 1, 0]
    loads = gather_boundary_loads(world)
    assert loads[0]["cells"][3] == 1
    assert loads[0]["torque"][2] == pytest.approx(0.5)
    world.plates[0].riftPartner = -1
    assert gather_boundary_loads(world)[0]["cells"][3] == 0


def test_gather_boundary_loads_skips_invalid_collider_and_background(monkeypatch):
    world = make_world(monkeypatch, "pull")
    world.landClaims[0] = 2
    world.collider[0] = 99
    assert gather_boundary_loads(world)[0]["cells"] == [0, 0, 0, 0]
    world.landClaims[0] = 0
    world.plates[0].isBackground = True
    assert 0 not in gather_boundary_loads(world)


def test_gather_boundary_loads_validates_arrays_and_zero_edge_length(monkeypatch):
    world = make_world(monkeypatch, "pull")
    world.ownerCell = np.array([0])
    with pytest.raises(ValueError):
        gather_boundary_loads(world)
    world = make_world(monkeypatch, "pull")

    def no_length(_world, cell, out):
        out[:] = [0.0, 1.0, 0.0, 0.0]
        return 1

    monkeypatch.setattr("orogen.v2.tectonics.forces.contact_at", no_length)
    assert gather_boundary_loads(world)[0]["cells"] == [0, 0, 0, 0]


def test_gather_boundary_loads_ignores_missing_owner_contact_and_unknown_plate(monkeypatch):
    world = make_world(monkeypatch, "pull")
    world.owner[:] = [-1, -1]
    assert gather_boundary_loads(world) == {0: {
        "torque": [0.0, 0.0, 0.0], "flowTorque": [0.0, 0.0, 0.0],
        "byKind": [[0.0, 0.0, 0.0] for _ in FORCE_KINDS],
        "forceByKind": [[0.0, 0.0, 0.0] for _ in FORCE_KINDS],
        "torqueMagnitudeSum": 0.0, "cells": [0, 0, 0, 0], "slabs": [], "contacts": [],
    }, 1: {
        "torque": [0.0, 0.0, 0.0], "flowTorque": [0.0, 0.0, 0.0],
        "byKind": [[0.0, 0.0, 0.0] for _ in FORCE_KINDS],
        "forceByKind": [[0.0, 0.0, 0.0] for _ in FORCE_KINDS],
        "torqueMagnitudeSum": 0.0, "cells": [0, 0, 0, 0], "slabs": [], "contacts": [],
    }}
    world = make_world(monkeypatch, "pull")

    def no_contact(_world, cell, out):
        return -1

    monkeypatch.setattr("orogen.v2.tectonics.forces.contact_at", no_contact)
    assert sum(gather_boundary_loads(world)[0]["cells"]) == 0

    def unknown_contact(_world, cell, out):
        out[:] = [0.0, 1.0, 0.0, 1.0]
        return 1

    monkeypatch.setattr("orogen.v2.tectonics.forces.contact_at", unknown_contact)
    world.owner[1] = 99
    assert sum(gather_boundary_loads(world)[0]["cells"]) == 0


def test_gather_boundary_loads_handles_zero_force_and_clips_maturity(monkeypatch):
    world = make_world(monkeypatch, "pull")
    world.plates[0].crust.formedAt[0] = 101.0
    assert sum(gather_boundary_loads(world)[0]["cells"]) == 0
    world.plates[0].crust.formedAt[0] = 0
    world.plates[1].crust.slabKm[0] = 0
    assert sum(gather_boundary_loads(world)[0]["cells"]) == 0


def test_gather_boundary_loads_rejects_bad_xyz_layout(monkeypatch):
    world = make_world(monkeypatch, "pull")
    world.mesh.xyz = np.zeros(3)
    with pytest.raises(ValueError):
        gather_boundary_loads(world)


def configure_collision_world(monkeypatch, owner_neighbour=True):
    world = make_world(monkeypatch, "pull")
    a, b = world.plates
    a.crust.type[0] = CrustType["CONTINENT"]
    b.crust.type[0] = CrustType["CONTINENT"]
    a.capCenter = [0.0, 0.0, 1.0]
    b.capCenter = [0.0, 1.0, 0.0]
    b.cell_at = lambda index, x, y, z: 0
    world.mesh.index = object()
    world.mesh.adj_offset = np.array([0, 1, 2], dtype=np.int32)
    world.mesh.adj_list = np.array([1, 0], dtype=np.int32)
    world.landClaims[0] = 2
    world.collider[0] = 1
    if not owner_neighbour:
        world.owner[1] = -1
    return world


def test_gather_boundary_loads_records_collision_contacts_with_opposite_normals(monkeypatch):
    world = configure_collision_world(monkeypatch)
    loads = gather_boundary_loads(world)
    a_contact = loads[0]["contacts"][0]
    b_contact = loads[1]["contacts"][0]
    assert a_contact["other"] == 1
    assert b_contact["other"] == 0
    assert a_contact["weight"] == 1.0
    assert a_contact["n"] == pytest.approx([0.0, 1.0, 0.0])
    assert b_contact["n"] == pytest.approx([0.0, -1.0, 0.0])
    assert a_contact["r"] == [1.0, 0.0, 0.0]


def test_gather_boundary_loads_uses_craton_lock_weight_and_fallback_normal(monkeypatch):
    world = configure_collision_world(monkeypatch, owner_neighbour=False)
    world.config["cratonLockFactor"] = 3.5
    world.plates[0].crust.type[0] = CrustType["CRATON"]
    loads = gather_boundary_loads(world)
    contact = loads[0]["contacts"][0]
    assert contact["weight"] == pytest.approx(3.5)
    assert np.linalg.norm(contact["n"]) == pytest.approx(1.0)
    assert contact["n"] == pytest.approx([0.0, 1 / np.sqrt(2), -1 / np.sqrt(2)])
    world.plates[1].capCenter = world.plates[0].capCenter.copy()
    zero_normal = gather_boundary_loads(world)[0]["contacts"][0]["n"]
    assert zero_normal == [0.0, 0.0, 0.0]


def test_gather_boundary_loads_skips_arc_and_young_rift_margin_jitter(monkeypatch):
    world = configure_collision_world(monkeypatch)
    world.plates[1].crust.type[0] = CrustType["ARC"]
    assert gather_boundary_loads(world)[0]["contacts"] == []

    world = configure_collision_world(monkeypatch)
    a, b = world.plates
    a.born = b.born = 95.0
    a.riftPartner, b.riftPartner = 1, 0
    a.crust.flags[0] |= CrustFlag["RIFT_MARGIN"]
    b.crust.flags[0] |= CrustFlag["RIFT_MARGIN"]
    assert gather_boundary_loads(world)[0]["contacts"] == []


def test_gather_boundary_loads_keeps_collision_load_on_non_background_side(monkeypatch):
    world = configure_collision_world(monkeypatch)
    world.plates[1].isBackground = True
    loads = gather_boundary_loads(world)
    assert list(loads) == [0]
    assert len(loads[0]["contacts"]) == 1


def test_gather_boundary_loads_records_collision_on_only_loaded_side(monkeypatch):
    world = configure_collision_world(monkeypatch)
    world.plates[0].isBackground = True
    loads = gather_boundary_loads(world)
    assert list(loads) == [1]
    assert len(loads[1]["contacts"]) == 1
    assert loads[1]["contacts"][0]["other"] == 0
    assert loads[1]["contacts"][0]["n"] == pytest.approx([0.0, -1.0, 0.0])


def test_front_normal_ignores_disputed_and_oceanic_neighbours(monkeypatch):
    world = configure_collision_world(monkeypatch)
    world.landClaims[1] = 2
    loads = gather_boundary_loads(world)
    assert np.linalg.norm(loads[0]["contacts"][0]["n"]) == pytest.approx(1.0)

    world = configure_collision_world(monkeypatch)
    world.plates[1].crust.type[0] = CrustType["OCEAN"]
    loads = gather_boundary_loads(world)
    assert np.linalg.norm(loads[0]["contacts"][0]["n"]) == pytest.approx(1.0)
