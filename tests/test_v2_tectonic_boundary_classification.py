"""Tests for the partial world contact-classification pass."""
import numpy as np
import importlib

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.boundaries import classify_contact_boundaries
from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.world import BoundaryKind, TectonicWorld


def _setup():
    mesh = build_sim_mesh(48, 19)
    world = TectonicWorld(mesh, 19, {
        "boundaryActiveCmYr": 0.3, "obliqueLimit": 0.35,
        "closingCmYr": 0.2, "segmentPolarity": False,
    })
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
    monkeypatch.setattr(module, "contact_at", lambda w, cell, out: (out.__setitem__(slice(0, 4), [*n, 1]) or 1) if cell == 0 else -1)
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
    monkeypatch.setattr(module, "contact_at", lambda w, cell, out: (out.__setitem__(slice(0, 4), [*n, 1]) or 1) if cell == 0 else -1)
    count = classify_contact_boundaries(world, 2.0)
    assert count == 1
    assert world.boundaryKind[0] in (BoundaryKind["OVERRIDING"], BoundaryKind["SUBDUCTING"])
    assert world.convergence[0] > 0
    if world.boundaryKind[0] == BoundaryKind["OVERRIDING"]:
        assert a.crust.flags[0] & CrustFlag["TRENCH"]


def test_contact_classifier_marks_land_overlap_and_clears_interior_ocean_trench(monkeypatch):
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
