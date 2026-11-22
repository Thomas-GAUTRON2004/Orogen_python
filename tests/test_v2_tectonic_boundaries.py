""""Tests for the deterministic boundary classification primitives."""

import numpy as np

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.boundaries import (
    Motion,
    _feed_subduction_far_field,
    classify_contact_boundaries,
    classify_motion,
    feed_trench,
    near_land,
    vote_segments,
)
from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.world import BoundaryKind, TectonicWorld


def _world():
    mesh = build_sim_mesh(48, 31)
    return mesh, TectonicWorld(mesh, 31, overrides={"apronKm": 100.0})


def test_classify_motion_all_four_states_and_obliquity():
    assert classify_motion(-2, 0.1, 0.5, 1.0) == Motion.OPENING
    assert classify_motion(2, 0.1, 0.5, 1.0) == Motion.CLOSING
    assert classify_motion(0.1, 2, 0.5, 1.0) == Motion.SLIDING
    assert classify_motion(0.1, 0.1, 0.5, 1.0) == Motion.LOCKED
    assert classify_motion(2, 5, 0.5, 0.5) == Motion.SLIDING


def test_vote_segments_empty_clears_previous_votes():
    mesh, world = _world()
    world.oceanOverrider.fill(7)
    vote_segments(world, [])
    assert np.all(world.oceanOverrider == -1)


def test_vote_segments_connected_weighted_and_tied_groups():
    mesh, world = _world()
    a, b, c = (
        0,
        int(mesh.adj_list[mesh.adj_offset[0]]),
        int(mesh.adj_list[mesh.adj_offset[0] + 1]),
    )
    contacts = [
        {"w": a, "key": "0:1", "lowerId": 0, "upperId": 1, "prefersLower": True, "weight": 2},
        {"w": b, "key": "0:1", "lowerId": 0, "upperId": 1, "prefersLower": False, "weight": 1},
        {"w": c, "key": "2:3", "lowerId": 2, "upperId": 3, "prefersLower": False, "weight": 1},
    ]
    vote_segments(world, contacts)
    assert world.oceanOverrider[a] == 0
    assert world.oceanOverrider[b] == 0
    assert world.oceanOverrider[c] == 3


def test_vote_segments_tie_uses_first_contact_preference():
    mesh, world = _world()
    a = 0
    world.oceanOverrider.fill(9)
    vote_segments(
        world,
        [
            {"w": a, "key": "0:1", "lowerId": 0, "upperId": 1, "prefersLower": True, "weight": 1},
            {"w": int(mesh.adj_list[mesh.adj_offset[a]]), "key": "0:1", "lowerId": 0, "upperId": 1, "prefersLower": False, "weight": 1},
        ],
    )
    assert world.oceanOverrider[a] == 0


def test_near_land_requires_non_ocean_crust_within_apron():
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["OCEAN"], 0, 1)
    assert near_land(plate, mesh, 0, {"apronKm": 0.0}) is False
    neighbour = int(mesh.adj_list[mesh.adj_offset[0]])
    plate.crust.create(neighbour, CrustType["CONTINENT"], 0, 2)
    assert near_land(plate, mesh, 0, {"apronKm": mesh.edge_km}) is True


def test_feed_trench_requires_consumption_and_records_existing_trench():
    mesh, world = _world()
    plate = world.add_plate("P")
    crust = plate.crust
    crust.create(0, CrustType["OCEAN"], 0, 1)
    assert feed_trench(world, 0, crust, 0, 10.0) is False
    world.consumedAt[0] = 1
    assert feed_trench(world, 0, crust, 0, 10.0) is True
    assert crust.flags[0] & CrustFlag["TRENCH"]
    assert crust.slabKm[0] == 10.0
    assert feed_trench(world, 0, crust, 0, 0.0) is False
    assert crust.slabKm[0] == 10.0


def test_feed_trench_can_start_from_adjacent_consumption():
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["OCEAN"], 0, 1)
    neighbour = int(mesh.adj_list[mesh.adj_offset[0]])
    world.consumedAt[neighbour] = 1
    assert feed_trench(world, 0, plate.crust, 0, 4.0) is True
    assert plate.crust.slabKm[0] == 4.0


def test_feed_subduction_far_field_clears_flat_slab_without_land_sources():
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["OCEAN"], 0, 1)
    plate.flatSlabKm = 12.0
    world.config.update({"farAndeanRateCmYr": 1.0, "orogenyQuietMyr": 100.0, "apronKm": 0.0})

    _feed_subduction_far_field(world, {plate: {"cells": [0], "flat": 1}}, 1.0)

    assert plate.flatSlabKm == 0.0


def test_feed_subduction_far_field_skips_nonpositive_dt():
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["CONTINENT"], 0, 1)
    plate.flatSlabKm = 12.0
    world.config.update({"farAndeanRateCmYr": 1.0, "orogenyQuietMyr": 100.0})

    _feed_subduction_far_field(world, {plate: {"cells": [0], "flat": 1}}, 0.0)

    assert plate.flatSlabKm == 12.0


def test_classify_contact_boundaries_covers_non_closing_and_provisional_closing(monkeypatch):
    import orogen.v2.tectonics.boundaries as boundaries

    mesh, world = _world()
    world.config.update({"boundaryActiveCmYr": 0.5, "obliqueLimit": 1.0, "segmentPolarity": False})
    a, b = world.add_plate("A"), world.add_plate("B")
    a.crust.create(0, CrustType["CONTINENT"], 0, 1)
    b.crust.create(0, CrustType["CONTINENT"], 0, 2)
    world.owner[:] = -1
    world.owner[:2] = [a.id, b.id]
    world.ownerCell[:2] = [0, 0]
    world.provisional[0] = True

    def contact(_world, cell, normal):
        if cell == 0:
            normal[:] = [1.0, 0.0, 0.0, 1.0]
            return 1
        return -1

    monkeypatch.setattr(boundaries, "contact_at", contact)
    monkeypatch.setattr(boundaries, "overrides", lambda *_args: True)

    monkeypatch.setattr(boundaries, "classify_motion", lambda *_args: Motion.OPENING)
    assert classify_contact_boundaries(world, 1.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["RIDGE"]

    monkeypatch.setattr(boundaries, "classify_motion", lambda *_args: Motion.LOCKED)
    assert classify_contact_boundaries(world, 1.0) == 0
    assert world.boundaryKind[0] == BoundaryKind["NONE"]

    # Exercise a real overriding contact whose trench is not fed by consumed crust.
    world.provisional[0] = False
    monkeypatch.setattr(boundaries, "classify_motion", lambda *_args: Motion.CLOSING)
    assert classify_contact_boundaries(world, 1.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["OVERRIDING"]


def test_feed_subduction_far_field_rejects_non_positive_reference_rate():
    import pytest

    from orogen.v2.tectonics.boundaries import _feed_subduction_far_field

    _, world = _world()
    world.config["farAndeanRateCmYr"] = 0.0

    with pytest.raises(ValueError, match="farAndeanRateCmYr must be positive"):
        _feed_subduction_far_field(world, {object(): {}}, 1.0)


def test_feed_subduction_far_field_processes_land_front(monkeypatch):
    import orogen.v2.tectonics.boundaries as boundaries

    _, world = _world()
    plate = world.add_plate("A")
    plate.crust.create(0, CrustType["CONTINENT"], 0, 1)
    front = {
        "cells": [0],
        "kms": [10.0],
        "headOn": [1.0],
        "ages": [20.0],
        "shears": [0.0],
        "flat": 0,
        "km": 10.0,
    }
    captured = {}
    monkeypatch.setattr(boundaries, "belt_shape", lambda *_args: 1.0)
    monkeypatch.setattr(
        boundaries,
        "feed_far_field",
        lambda *args: captured.update(
            {"cells": args[3], "works": args[4], "front_km": args[5]}
        ),
    )

    boundaries._feed_subduction_far_field(world, {plate: front}, 1.0)

    assert plate.lastFedAt == world.time
    assert captured["cells"] == [0]
    assert captured["works"] == [10.0 * world.config["andeanWorkShare"] * world.config["farAndeanShare"]]
    assert captured["front_km"] > 0.0

    monkeypatch.setattr(boundaries, "overrides", lambda *_args: None)
    assert classify_contact_boundaries(world, 1.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["COLLISION"]
