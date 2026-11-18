"""Tests for the partial world contact-classification pass."""

import importlib

import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.boundaries import classify_contact_boundaries
from orogen.v2.tectonics.crust import CrustFlag, CrustType, OrogenyType
from orogen.v2.tectonics.world import BoundaryKind, TectonicWorld


def _setup():
    mesh = build_sim_mesh(48, 19)
    world = TectonicWorld(
        mesh,
        19,
        {
            "boundaryActiveCmYr": 0.3,
            "obliqueLimit": 0.35,
            "closingCmYr": 0.2,
            "segmentPolarity": False,
        },
    )
    a, b = world.add_plate("A"), world.add_plate("B")
    a.crust.create(0, CrustType["OCEAN"], -100, 1)
    b.crust.create(1, CrustType["OCEAN"], -80, 2)
    a.refresh_extent(mesh.xyz)
    b.refresh_extent(mesh.xyz)
    world.owner[0], world.owner[1] = a.id, b.id
    world.ownerCell[0], world.ownerCell[1] = 0, 1
    world.landClaims[0] = 1
    p = np.asarray(mesh.xyz).reshape(-1, 3)[0].astype(float)
    n = np.asarray([0.0, 0.0, 1.0])
    n -= np.dot(n, p) * p
    n /= np.linalg.norm(n)
    return mesh, world, a, b, p, n


def test_contact_classifier_marks_ridge_and_resets_outputs(monkeypatch):
    module = importlib.import_module("orogen.v2.tectonics.boundaries")
    mesh, world, a, b, p, n = _setup()
    a.omega = list(np.cross(p, n) * -0.01)
    monkeypatch.setattr(
        module,
        "contact_at",
        lambda w, cell, out: (
            (out.__setitem__(slice(0, 4), [*n, 1]) or 1) if cell == 0 else -1
        ),
    )
    world.boundaryKind.fill(9)
    world.convergence.fill(9)
    count = classify_contact_boundaries(world, 1.0)
    assert count == 1
    assert world.boundaryKind[0] == BoundaryKind["RIDGE"]
    assert world.convergence[0] < 0
    assert world.boundaryKind[1] == BoundaryKind["NONE"]


def test_contact_classifier_closing_sets_override_and_trench(monkeypatch):
    module = importlib.import_module("orogen.v2.tectonics.boundaries")
    mesh, world, a, b, p, n = _setup()
    a.omega = list(np.cross(p, n) * 0.01)
    world.consumedAt[0] = 1
    monkeypatch.setattr(
        module,
        "contact_at",
        lambda w, cell, out: (
            (out.__setitem__(slice(0, 4), [*n, 1]) or 1) if cell == 0 else -1
        ),
    )
    count = classify_contact_boundaries(world, 2.0)
    assert count == 1
    assert world.boundaryKind[0] in (
        BoundaryKind["OVERRIDING"],
        BoundaryKind["SUBDUCTING"],
    )
    assert world.convergence[0] > 0
    if world.boundaryKind[0] == BoundaryKind["OVERRIDING"]:
        assert a.crust.flags[0] & CrustFlag["TRENCH"]


def test_contact_classifier_marks_land_overlap_and_clears_interior_ocean_trench(
    monkeypatch,
):
    module = importlib.import_module("orogen.v2.tectonics.boundaries")
    mesh, world, a, b, p, n = _setup()
    world.landClaims[0] = 2
    a.crust.flags[0] |= CrustFlag["TRENCH"]
    monkeypatch.setattr(module, "contact_at", lambda w, cell, out: -1)
    assert classify_contact_boundaries(world, 0.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["COLLISION"]
    world.landClaims[0] = 1
    world.owner[1] = -1
    assert classify_contact_boundaries(world, 0.0) == 0
    assert not (a.crust.flags[0] & CrustFlag["TRENCH"])


def test_contact_classifier_handles_foreign_missing_and_transform(monkeypatch):
    module = importlib.import_module("orogen.v2.tectonics.boundaries")
    mesh, world, a, b, p, n = _setup()
    # The opposite owner can disappear between contact discovery and classification.
    monkeypatch.setattr(
        module, "contact_at", lambda w, cell, out: 1 if cell == 0 else -1
    )
    world.owner[1] = -1
    assert classify_contact_boundaries(world, 1.0) == 0
    # Restore a foreign plate and make relative motion purely tangential to the contact.
    world.owner[1] = b.id
    tangent = np.cross(p, n)
    tangent /= np.linalg.norm(tangent)
    a.omega = list(np.cross(p, tangent) * 0.01)
    monkeypatch.setattr(
        module,
        "contact_at",
        lambda w, cell, out: (
            (out.__setitem__(slice(0, 4), [*n, 1]) or 1) if cell == 0 else -1
        ),
    )
    assert classify_contact_boundaries(world, 1.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["TRANSFORM"]


def test_contact_classifier_continental_collision_and_subducting_trench(monkeypatch):
    module = importlib.import_module("orogen.v2.tectonics.boundaries")
    mesh, world, a, b, p, n = _setup()
    a.crust.type[0] = CrustType["CONTINENT"]
    b.crust.type[1] = CrustType["CRATON"]
    a.omega = list(np.cross(p, n) * 0.01)
    monkeypatch.setattr(
        module,
        "contact_at",
        lambda w, cell, out: (
            (out.__setitem__(slice(0, 4), [*n, 1]) or 1) if cell == 0 else -1
        ),
    )
    assert classify_contact_boundaries(world, 1.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["COLLISION"]

    # Oceanic A goes down beneath continental B; a stale trench is cleared.
    a.crust.type[0] = CrustType["OCEAN"]
    b.crust.type[1] = CrustType["CONTINENT"]
    a.crust.flags[0] |= CrustFlag["TRENCH"]
    assert classify_contact_boundaries(world, 1.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["SUBDUCTING"]
    assert not (a.crust.flags[0] & CrustFlag["TRENCH"])


def test_contact_classifier_votes_ocean_segments_and_feeds_overriding_trench(
    monkeypatch,
):
    module = importlib.import_module("orogen.v2.tectonics.boundaries")
    mesh, world, a, b, p, n = _setup()
    # A is younger, so it overrides B under the physical age rule.
    a.crust.formedAt[0] = -20
    b.crust.formedAt[1] = -100
    a.omega = list(np.cross(p, n) * 0.01)
    world.consumedAt[0] = 1
    world.config["segmentPolarity"] = True
    monkeypatch.setattr(
        module,
        "contact_at",
        lambda w, cell, out: (
            (out.__setitem__(slice(0, 4), [*n, 1]) or 1) if cell == 0 else -1
        ),
    )
    assert classify_contact_boundaries(world, 1.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["OVERRIDING"]
    assert a.crust.flags[0] & CrustFlag["TRENCH"]
    assert world.oceanOverrider[0] == a.id


def _subduction_far_field_config():
    return {
        "apronKm": 100.0,
        "orogenyQuietMyr": 10.0,
        "flatSlabMaxAgeMyr": 35.0,
        "flatSlabAdvanceCmYr": 2.0,
        "flatSlabOnShare": 0.5,
        "flatSlabOffShare": 0.3,
        "andeanWorkShare": 0.4,
        "beltObliqueMin": 0.2,
        "farFieldEveryMyr": 4.0,
        "farReachKm": 1100.0,
        "farFrontRefKm": 2000.0,
        "farMaxKm": 3200.0,
        "farCratonCost": 0.3,
        "farWeakCost": 1.6,
        "farCratonTake": 0.15,
        "farWeakTake": 1.6,
        "farFoldMyr": 120.0,
        "farAndeanShare": 0.35,
        "farAndeanRateCmYr": 6.0,
        "farFlatShare": 1.0,
    }


def _subduction_contact(monkeypatch, slab_age_myr=10.0, omega_scale=0.01):
    import orogen.v2.tectonics.boundaries as module

    mesh, world, upper, lower, position, normal = _setup()
    world.config.update(_subduction_far_field_config())
    world.time = 10.0
    upper.crust.type[0] = CrustType["CONTINENT"]
    lower.crust.formedAt[1] = world.time - slab_age_myr
    upper.omega = list(np.cross(position, normal) * omega_scale)
    world.consumedAt[0] = 1

    def contact(_world, cell, out):
        if cell == 0:
            out[:] = [*normal, 1.0]
            return 1
        return -1

    monkeypatch.setattr(module, "contact_at", contact)
    monkeypatch.setattr(module, "overrides", lambda *_args: True)
    return module, mesh, world, upper, lower


@pytest.mark.parametrize(
    ("slab_age_myr", "omega_scale", "flat", "far_share"),
    [
        (10.0, 0.01, True, 1.0),
        (60.0, 0.01, False, 0.35),
        (10.0, 0.002, False, 0.35 * (1.2742 / 6.0) ** 2),
    ],
)
def test_contact_classifier_feeds_flat_and_steep_subduction_far_field(
    monkeypatch, slab_age_myr, omega_scale, flat, far_share
):
    module, mesh, world, upper, _lower = _subduction_contact(
        monkeypatch, slab_age_myr, omega_scale
    )
    calls = []
    monkeypatch.setattr(
        module,
        "feed_far_field",
        lambda *args: calls.append(args),
    )

    assert classify_contact_boundaries(world, 1.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["OVERRIDING"]
    assert bool(upper.flatSlab) is flat
    assert upper.lastFedAt == pytest.approx(world.time)
    assert len(calls) == 1

    _, fed_plate, key, cells, pushes, front_km, share = calls[0]
    assert fed_plate is upper
    assert key == f"s{upper.id}"
    assert cells == [0]
    expected_km = omega_scale * 6371.0
    expected_work = expected_km * 0.4
    assert pushes == pytest.approx([expected_work * far_share])
    assert front_km == pytest.approx(mesh.edge_km * 0.87)
    assert share == 1.0
    if flat:
        assert upper.flatSlabKm == pytest.approx(expected_km)
        assert world.events[-1]["kind"] == "slab-flat"
    else:
        assert upper.flatSlabKm == 0.0
        assert world.events == []


def test_subduction_flat_slab_hysteresis_and_quiet_gap_reset(monkeypatch):
    import orogen.v2.tectonics.boundaries as module

    mesh, world, upper, _lower, _position, _normal = _setup()
    world.config.update(_subduction_far_field_config())
    world.time = 10.0
    upper.crust.type[0] = CrustType["CONTINENT"]
    upper.crust.create(2, CrustType["CONTINENT"], -20.0, 3)
    upper.flatSlab = True
    upper.flatSlabKm = 100.0
    upper.lastFedAt = world.time
    feed_calls = []
    monkeypatch.setattr(module, "feed_far_field", lambda *args: feed_calls.append(args))
    monkeypatch.setattr(module, "near_land", lambda *_args: True)

    # Half of the cells qualify: a previously flat slab stays flat above the
    # lower off-threshold, even though it does not exceed the on-threshold.
    front = {
        "cells": [0, 2],
        "kms": [10.0, 10.0],
        "headOn": [1.0, 1.0],
        "flat": 1,
        "km": 20.0,
    }
    module._feed_subduction_far_field(world, {upper: front}, 1.0)
    assert upper.flatSlab is True
    assert upper.flatSlabKm == pytest.approx(110.0)
    assert world.events == []
    assert len(feed_calls) == 1

    # A long quiet interval resets flat-slab memory before the next fed front.
    upper.lastFedAt = world.time - 20.0
    upper.flatSlabKm = 100.0
    quiet_front = {**front, "flat": 0}
    module._feed_subduction_far_field(world, {upper: quiet_front}, 1.0)
    assert upper.flatSlab is False
    assert upper.flatSlabKm == 0.0
    assert upper.lastFedAt == pytest.approx(world.time)
    assert world.events == []
    assert len(feed_calls) == 2



def _subduction_local_belt_config():
    return {
        "belts": {
            "andean": {"baseKm": 120.0, "perKm": 1.0, "maxKm": 400.0},
            "laramide": {"baseKm": 200.0, "perKm": 0.6, "maxKm": 1800.0},
            "transpression": {"baseKm": 40.0, "perKm": 0.3, "maxKm": 150.0},
        },
        "beltSpeedRefCmYr": 5.0,
        "beltSpeedExponent": 0.5,
        "beltSpeedFactorMin": 0.6,
        "beltSpeedFactorMax": 1.6,
        "beltYoungSlabBoost": 0.5,
        "beltYoungSlabRefMyr": 60.0,
        "beltOldSlabMin": 0.75,
        "arcBeltMaxKm": 243.0,
        "cratonRimKm": 150.0,
        "cratonRimCost": 2.0,
        "cratonRimShare": 0.35,
        "weakCellCost": 0.4,
        "beltWorkShare": 0.35,
        "thickeningPerWorkKm": 0.05,
    }


@pytest.mark.parametrize(
    ("slab_age_myr", "expected_type"),
    [
        (60.0, OrogenyType["ANDEAN"]),
        (10.0, OrogenyType["LARAMIDE"]),
    ],
)
def test_contact_classifier_spreads_local_andean_and_laramide_belts(
    monkeypatch, slab_age_myr, expected_type
):
    module, _mesh, world, upper, _lower = _subduction_contact(
        monkeypatch, slab_age_myr
    )
    world.config.update(_subduction_local_belt_config())
    calls = []
    monkeypatch.setattr(module, "feed_far_field", lambda *_args: None)
    monkeypatch.setattr(module, "spread_belt", lambda *args: calls.append(args))

    assert classify_contact_boundaries(world, 1.0) == 1
    assert bool(upper.flatSlab) is (expected_type == OrogenyType["LARAMIDE"])
    assert len(calls) == 1

    plate, _mesh, cells, belt, _config = calls[0]
    assert plate is upper
    assert cells == [0]
    assert belt["type"] == expected_type
    assert len(belt["widths"]) == len(belt["works"]) == len(belt["shears"]) == 1

    km = 0.01 * 6371.0
    rate = km / 10.0
    speed_factor = min(1.6, max(0.6, (rate / 5.0) ** 0.5))
    steep = min(400.0, 120.0 + km * 0.4) * speed_factor
    shape = 1.0 + 0.5 * (1.0 - slab_age_myr / 60.0)
    if expected_type == OrogenyType["LARAMIDE"]:
        laramide = (200.0 + 0.6 * km) * speed_factor
        expected_width = min(1800.0 * 1.6, max(steep, laramide) * shape)
    else:
        expected_width = min(400.0 * 1.6, steep * shape)
    assert belt["widths"] == pytest.approx([expected_width])
    assert belt["works"] == pytest.approx([km * 0.4])
    assert belt["shears"] == pytest.approx([0.0])


def test_contact_classifier_keeps_arc_belts_narrow_and_silent_over_flat_slabs(
    monkeypatch,
):
    module, _mesh, world, upper, _lower = _subduction_contact(
        monkeypatch, slab_age_myr=60.0
    )
    world.config.update(_subduction_local_belt_config())
    upper.crust.type[0] = CrustType["ARC"]
    calls = []
    monkeypatch.setattr(module, "feed_far_field", lambda *_args: None)
    monkeypatch.setattr(module, "spread_belt", lambda *args: calls.append(args))

    assert classify_contact_boundaries(world, 1.0) == 1
    assert len(calls) == 1
    assert calls[0][2] == [0]
    assert calls[0][3]["type"] == OrogenyType["ARC"]
    assert calls[0][3]["widths"][0] <= world.config["arcBeltMaxKm"]

    flat_module, _mesh, flat_world, flat_upper, _flat_lower = _subduction_contact(
        monkeypatch, slab_age_myr=10.0
    )
    flat_world.config.update(_subduction_local_belt_config())
    flat_upper.crust.type[0] = CrustType["ARC"]
    flat_calls = []
    monkeypatch.setattr(flat_module, "feed_far_field", lambda *_args: None)
    monkeypatch.setattr(
        flat_module, "spread_belt", lambda *args: flat_calls.append(args)
    )
    assert classify_contact_boundaries(flat_world, 1.0) == 1
    assert flat_upper.flatSlab is True
    assert flat_calls == []


def test_contact_classifier_spreads_transpression_on_oblique_land_closing(
    monkeypatch,
):
    import orogen.v2.tectonics.boundaries as module

    mesh, world, upper, _lower, position, normal = _setup()
    world.config.update(_subduction_far_field_config())
    world.config.update(_subduction_local_belt_config())
    world.time = 10.0
    upper.crust.type[0] = CrustType["CONTINENT"]
    tangent = np.cross(position, normal)
    velocity = normal * 0.001 + tangent * 0.01
    upper.omega = list(np.cross(position, velocity))

    def contact(_world, cell, out):
        if cell == 0:
            out[:] = [*normal, 1.0]
            return 1
        return -1

    monkeypatch.setattr(module, "contact_at", contact)
    calls = []
    monkeypatch.setattr(module, "spread_belt", lambda *args: calls.append(args))
    monkeypatch.setattr(module, "feed_far_field", lambda *_args: None)

    assert classify_contact_boundaries(world, 1.0) == 1
    assert world.boundaryKind[0] == BoundaryKind["TRANSFORM"]
    assert len(calls) == 1
    plate, _mesh, cells, belt, _config = calls[0]
    assert plate is upper
    assert cells == [0]
    assert belt["type"] == OrogenyType["TRANSPRESSION"]
    assert belt["works"][0] == pytest.approx(0.001 * 6371.0)
    assert belt["widths"][0] > 0.0
    assert belt["shears"][0] != pytest.approx(0.0)


def test_local_belt_spreading_remains_disabled_for_partial_far_field_config(
    monkeypatch,
):
    module, _mesh, world, _upper, _lower = _subduction_contact(monkeypatch)
    calls = []
    monkeypatch.setattr(module, "spread_belt", lambda *args: calls.append(args))
    monkeypatch.setattr(module, "feed_far_field", lambda *_args: None)

    assert classify_contact_boundaries(world, 1.0) == 1
    assert calls == []
