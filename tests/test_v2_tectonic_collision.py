"""Tests for continental collision geometry and terrane handling."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.collision import (
    accrete_arcs,
    enclosed_sea_between_km2,
    open_sea_between,
)
from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plate import Plate


class Index:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

    def nearest(self, x, y, z):
        point = np.asarray([x, y, z], dtype=float)
        return int(np.argmin(np.sum((self.points - point) ** 2, axis=1)))


class Mesh:
    def __init__(self, n=5):
        angles = np.arange(n, dtype=float) * 0.02
        points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(n)))
        self.xyz = points.astype(np.float32).ravel()
        self.num_cells = n
        self.edge_km = 100.0
        self.edge_rad = self.edge_km / 6371.0
        self.cell_area_km2 = 100.0
        self.index = Index(points)
        offsets = [0]
        adjacency = []
        for cell in range(n):
            neighbours = []
            if cell:
                neighbours.append(cell - 1)
            if cell + 1 < n:
                neighbours.append(cell + 1)
            adjacency.extend(neighbours)
            offsets.append(len(adjacency))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)


def sea_world():
    mesh = Mesh(5)
    upper, lower = Plate(0, 5), Plate(1, 5)
    upper.crust.create(0, CrustType["CONTINENT"], -10.0, 1)
    lower.crust.create(4, CrustType["CONTINENT"], -10.0, 2)
    upper.stats["continentalCells"] = 1
    lower.stats["continentalCells"] = 1
    return (
        SimpleNamespace(
            mesh=mesh,
            plates=[upper, lower],
            owner=np.asarray([0, -1, -1, -1, 1], dtype=np.int16),
            ownerCell=np.arange(5, dtype=np.int32),
            landClaims=np.zeros(5, dtype=np.uint8),
            config={"sutureGapKm": 400.0},
        ),
        upper,
        lower,
    )


def contact_world():
    mesh = Mesh(5)
    a, b = Plate(0, 5), Plate(1, 5)
    a.crust.create(0, CrustType["CONTINENT"], -20.0, 10)
    b.crust.create(0, CrustType["CONTINENT"], -20.0, 20)
    a.stats["continentalCells"] = 3
    b.stats["continentalCells"] = 1
    events = []

    def log(kind, text, data=None):
        events.append(
            {
                "kind": kind,
                "text": text,
                "data": {} if data is None else data,
            }
        )

    world = SimpleNamespace(
        mesh=mesh,
        plates=[a, b],
        time=0.0,
        events=events,
        budget={"recycledCells": 0, "underthrustCells": 0},
        arcScrapes={},
        config={},
        log=log,
    )
    return world, a, b


def test_enclosed_sea_between_returns_zero_when_all_cells_are_land():
    world, upper, lower = sea_world()
    for cell in range(1, world.mesh.num_cells):
        world.owner[cell] = upper.id
        world.ownerCell[cell] = cell
        upper.crust.create(cell, CrustType["CONTINENT"], -10.0, cell + 10)
    assert enclosed_sea_between_km2(world, upper, lower) == 0.0


def test_open_sea_between_counts_facing_but_not_directly_opposed_cells(monkeypatch):
    from orogen.v2.tectonics import collision

    world, upper, lower = sea_world()
    calls = {"count": 0}

    def shallow_angle(*_args):
        calls["count"] += 1
        return [1.0, 0.0, 0.0] if calls["count"] % 2 else [-0.7, math.sqrt(0.51), 0.0]

    monkeypatch.setattr(collision, "tangent_toward", shallow_angle)
    result = open_sea_between(world, upper, lower)
    assert result["km2"] > 0.0
    assert result["widestKm"] == 0.0


def test_recycle_skips_arc_recycled_event_for_tiny_fragment():
    from orogen.v2.tectonics.collision import recycle

    world, donor, _host = contact_world()
    donor.crust.create(0, CrustType["ARC"], -20.0, 30)
    world.budget = {"recycledCells": 0}
    world.events = []
    world.log = lambda kind, text, data=None: world.events.append({"kind": kind})
    world.mesh.cell_area_km2 = 100.0
    world.config = {"terraneMinAreaKm2": 1000.0}

    assert recycle(world, donor, [0]) == 1
    assert world.budget["recycledCells"] == 1
    assert world.events == []


def test_accrete_arcs_selects_one_donor_when_both_plates_have_arc_cells():
    world, a, b = contact_world()
    a.crust.type[0] = CrustType["ARC"]
    b.crust.type[0] = CrustType["ARC"]
    a.crust.remove(0)
    world.time = 15.0
    world.arcScrapes = {}
    world.budget = {"recycledCells": 0}
    world.events = []
    world.config = {
        "arcAccreteMinKm2": 50.0,
        "terraneMinAreaKm2": 100.0,
        "arcScrapeLogKm2": 1000.0,
    }

    result = accrete_arcs(world, a, b, [0])

    assert result == {"core": [], "accreted": True}
    assert b.crust.mask[0] == 0
    assert a.crust.type[0] == CrustType["ARC"]
    assert world.arcScrapes["1:0"]["cells"] == 1
    assert world.arcScrapes["1:0"].get("logged", False) is False
    assert world.events == []


def test_accrete_arcs_skips_transfer_when_donor_is_dead():
    world, donor, host = contact_world()
    donor.crust.type[0] = CrustType["ARC"]
    donor.alive = False
    world.time = 10.0
    world.arcScrapes = {}
    world.budget = {"recycledCells": 0}
    world.events = []
    world.config = {"arcAccreteMinKm2": 100.0, "terraneMinAreaKm2": 100.0}
    assert accrete_arcs(world, donor, host, [0]) == {"core": [], "accreted": False}
    assert world.arcScrapes == {}


def test_dock_if_terrane_docks_using_accretion_and_logs_event(monkeypatch):
    import importlib

    collision_module = importlib.import_module("orogen.v2.tectonics.collision")
    accretion_module = importlib.import_module(
        "orogen.v2.tectonics.ops.plate_accretion"
    )
    world, a, b = contact_world()
    world.time = 25.0
    world.config = {
        "terraneMaxAreaKm2": 200.0,
        "arcAccreteMinKm2": 50.0,
        "terraneMinAreaKm2": 50.0,
    }
    monkeypatch.setattr(
        accretion_module,
        "landmass_of",
        lambda plate, _mesh, _seeds: [0] if plate.id == 0 else [1],
    )
    calls = []
    monkeypatch.setattr(
        accretion_module,
        "accrete_terrane",
        lambda world, host, donor, mass, rate: (
            calls.append((host.id, donor.id, list(mass), rate))
            or {"cells": 1, "sealed": 2}
        ),
    )

    assert collision_module.dock_if_terrane(world, a, b, [0])
    assert calls and calls[0][:3] == (0, 1, [1])
    assert world.events[-1]["kind"] == "dock"


def test_dock_if_terrane_returns_after_scrap_without_dock_event(monkeypatch):
    import importlib

    collision_module = importlib.import_module("orogen.v2.tectonics.collision")
    accretion_module = importlib.import_module(
        "orogen.v2.tectonics.ops.plate_accretion"
    )
    world, a, b = contact_world()
    world.time = 25.0
    world.events = []
    world.config = {
        "terraneMaxAreaKm2": 200.0,
        "arcAccreteMinKm2": 50.0,
        "terraneMinAreaKm2": 50.0,
    }
    monkeypatch.setattr(
        accretion_module,
        "landmass_of",
        lambda plate, _mesh, _seeds: [0] if plate.id == 0 else [1],
    )
    monkeypatch.setattr(
        accretion_module,
        "accrete_terrane",
        lambda *_args: {"scrap": True},
    )
    assert collision_module.dock_if_terrane(world, a, b, [0])
    assert world.events == []


def test_dock_if_terrane_recycles_tiny_arc_piece_instead_of_accreting(monkeypatch):
    import importlib

    collision_module = importlib.import_module("orogen.v2.tectonics.collision")
    accretion_module = importlib.import_module(
        "orogen.v2.tectonics.ops.plate_accretion"
    )
    world, a, b = contact_world()
    b.crust.type[0] = CrustType["ARC"]
    world.time = 25.0
    world.events = []
    world.config = {
        "terraneMaxAreaKm2": 200.0,
        "arcAccreteMinKm2": 200.0,
        "terraneMinAreaKm2": 1000.0,
    }
    monkeypatch.setattr(
        accretion_module,
        "landmass_of",
        lambda plate, _mesh, _seeds: [0],
    )
    monkeypatch.setattr(
        accretion_module,
        "accrete_terrane",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must recycle arc")),
    )
    assert collision_module.dock_if_terrane(world, a, b, [0])
    assert world.budget["recycledCells"] == 1
    assert world.events == []


def test_dock_if_terrane_returns_false_when_one_mass_is_missing(monkeypatch):
    import importlib

    collision_module = importlib.import_module("orogen.v2.tectonics.collision")
    accretion_module = importlib.import_module(
        "orogen.v2.tectonics.ops.plate_accretion"
    )
    world, a, b = contact_world()
    world.config = {"terraneMaxAreaKm2": 200.0}
    monkeypatch.setattr(
        accretion_module,
        "landmass_of",
        lambda plate, _mesh, _seeds: [] if plate.id == 0 else [0],
    )
    assert not collision_module.dock_if_terrane(world, a, b, [0])


def collision_tracking_world():
    mesh = Mesh(5)
    upper, lower = Plate(0, 5), Plate(1, 5)
    for cell in range(5):
        upper.crust.create(cell, CrustType["CONTINENT"], -10.0, 10)
        lower.crust.create(cell, CrustType["CONTINENT"], -10.0, 20)
    upper.stats["continentalCells"] = 5
    lower.stats["continentalCells"] = 5
    upper.capCenter = [0.0, 1.0, 0.0]
    lower.capCenter = [0.0, -1.0, 0.0]
    lower.omega = [0.0, 0.0, 0.01]
    owner = np.asarray([0, 0, 0, 1, 1], dtype=np.int16)
    land_claims = np.zeros(5, dtype=np.uint8)
    land_claims[2] = 2
    collider = np.full(5, -1, dtype=np.int16)
    collider[2] = 1
    events = []

    def log(kind, text, data=None):
        events.append(
            {
                "kind": kind,
                "text": text,
                "data": {} if data is None else data,
                "time": 10.0,
            }
        )

    world = SimpleNamespace(
        mesh=mesh,
        plates=[upper, lower],
        owner=owner,
        ownerCell=np.arange(5, dtype=np.int32),
        landClaims=land_claims,
        collider=collider,
        collisions={},
        time=10.0,
        budget={"underthrustCells": 0, "recycledCells": 0},
        events=events,
        config={
            "collisionForgetMyr": 8.0,
            "collisionJitterFrontKm": 0.0,
            "sutureStallCmYr": 0.4,
            "underthrustShare": 0.0,
        },
        log=log,
    )
    return world, upper, lower


def test_track_collisions_records_front_shortening_and_contact_age(monkeypatch):
    from orogen.v2.tectonics import collision

    world, upper, lower = collision_tracking_world()
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])

    result = collision.track_collisions(world, 1.0)

    record = world.collisions["0:1"]
    assert result == {
        "tracked": 1,
        "started": 1,
        "underthrustCells": 0,
        "changed": False,
    }
    assert record["upper"] == upper.id
    assert record["lower"] == lower.id
    assert record["cells"] == 1
    assert record["contactMyr"] == 1.0
    assert record["closingCmYr"] > 0.0
    assert record["shorteningKm"] == pytest.approx(63.71)
    assert record["frontKm"] == pytest.approx(100.0 * math.sqrt(3.0) / 2.0)
    assert record["meanRateCmYr"] == pytest.approx(6.371)
    assert record["sutureCandidate"] is False
    assert world.events[-1]["kind"] == "collision"


def test_track_collisions_forgets_stale_records_before_touching(monkeypatch):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    world.landClaims[:] = 0
    world.collider[:] = -1
    world.collisions["0:1"] = {
        "sutured": False,
        "lastActive": 0.0,
        "upper": 0,
        "lower": 1,
    }
    world.collisions["0:2"] = {
        "sutured": True,
        "lastActive": 0.0,
        "upper": 0,
        "lower": 2,
    }

    result = collision.track_collisions(world, 1.0)

    assert world.collisions == {"0:2": world.collisions["0:2"]}
    assert result["tracked"] == 0
    assert result["started"] == 0


def test_track_collisions_docks_touching_terrane(monkeypatch):
    from orogen.v2.tectonics import collision

    world, upper, lower = collision_tracking_world()
    world.landClaims[:] = 0
    world.collider[:] = -1
    world.config["terraneMaxAreaKm2"] = 1.0
    monkeypatch.setattr(
        collision,
        "dock_if_terrane",
        lambda w, a, b, cells: a.id == upper.id and b.id == lower.id,
    )

    result = collision.track_collisions(world, 1.0)

    assert result["changed"] is True
    assert result["tracked"] == 0
    assert world.collisions == {}


def test_track_collisions_skips_fresh_rift_jitter(monkeypatch):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    world.config["riftJitterFrontKm"] = 1000.0
    monkeypatch.setattr(collision, "is_rift_jitter", lambda *_args: True)

    result = collision.track_collisions(world, 1.0)

    assert result["started"] == 0
    assert result["tracked"] == 0
    assert world.collisions == {}


def test_track_collisions_rejects_non_positive_or_non_finite_dt():
    from orogen.v2.tectonics.collision import track_collisions

    world, _upper, _lower = collision_tracking_world()
    for dt in (0.0, -1.0, float("inf"), float("nan")):
        try:
            track_collisions(world, dt)
        except ValueError as error:
            assert "finite positive" in str(error)
        else:
            raise AssertionError("invalid dt should be rejected")


def test_collect_contacts_detects_overlap_and_touching_pairs():
    from orogen.v2.tectonics.collision import collect_contacts

    world, a, b = contact_world()
    for plate in (a, b):
        for cell in range(1, 5):
            plate.crust.create(cell, CrustType["CONTINENT"], -10.0, plate.id + 10)
    world.owner = np.asarray([0, 0, 1, 1, -1], dtype=np.int16)
    world.ownerCell = np.arange(5, dtype=np.int32)
    world.landClaims = np.zeros(5, dtype=np.uint8)
    world.collider = np.full(5, -1, dtype=np.int16)
    world.landClaims[2] = 2
    world.collider[2] = 1
    world.owner[2] = 0

    contacts = collect_contacts(world)
    assert contacts["overlap"] == {"0:1": [2]}
    assert contacts["touching"] == {}

    world.landClaims[2] = 0
    world.collider[2] = -1
    world.owner[2] = 1
    contacts = collect_contacts(world)
    assert contacts["overlap"] == {}
    assert contacts["touching"]["0:1"] == [1, 2]


def test_collect_contacts_ignores_invalid_and_non_land_claims():
    from orogen.v2.tectonics.collision import collect_contacts
    from orogen.v2.tectonics.crust import CrustFlag

    world, a, b = contact_world()
    for plate in (a, b):
        for cell in range(1, 5):
            plate.crust.create(cell, CrustType["CONTINENT"], -10.0, plate.id + 1)
    world.owner = np.asarray([0, 0, 1, 1, -1], dtype=np.int16)
    world.ownerCell = np.arange(5, dtype=np.int32)
    world.landClaims = np.zeros(5, dtype=np.uint8)
    world.collider = np.full(5, -1, dtype=np.int16)

    # A missing, oceanic, rift-margin or background cell is not land contact.
    b.crust.remove(2)
    assert collect_contacts(world)["touching"] == {}
    b.crust.create(2, CrustType["OCEAN"], -10.0, 2)
    assert collect_contacts(world)["touching"] == {}
    b.crust.type[2] = CrustType["CONTINENT"]
    b.crust.flags[2] = CrustFlag["RIFT_MARGIN"]
    assert collect_contacts(world)["touching"] == {}
    b.crust.flags[2] = 0
    b.isBackground = True
    assert collect_contacts(world)["touching"] == {}

    b.isBackground = False
    world.landClaims[2] = 2
    world.collider[2] = 1
    world.owner[2] = -1
    assert collect_contacts(world)["overlap"] == {}


def test_forget_stale_collisions_keeps_recent_and_sutured_pairs():
    from orogen.v2.tectonics.collision import forget_stale_collisions

    world, _a, _b = contact_world()
    world.time = 20.0
    world.config = {"collisionForgetMyr": 8.0}
    world.collisions = {
        "old": {"lastActive": 1.0, "sutured": False},
        "recent": {"lastActive": 15.0, "sutured": False},
        "sutured": {"lastActive": 1.0, "sutured": True},
    }
    assert forget_stale_collisions(world) == 1
    assert set(world.collisions) == {"recent", "sutured"}


def test_young_rift_partners_require_reciprocal_recent_relationship():
    from orogen.v2.tectonics.collision import are_young_rift_partners

    world, a, b = contact_world()
    a.riftPartner, b.riftPartner = b.id, a.id
    a.born, b.born = 5.0, 8.0
    assert are_young_rift_partners(a, b, 20.0, {"riftPushMyr": 15.0})
    assert not are_young_rift_partners(a, b, 24.0, {"riftPushMyr": 15.0})
    b.riftPartner = -1
    assert not are_young_rift_partners(a, b, 10.0, {"riftPushMyr": 100.0})


def test_choose_sides_prefers_trench_flag_then_continental_area():
    from orogen.v2.tectonics.collision import choose_sides
    from orogen.v2.tectonics.crust import CrustFlag

    world, a, b = contact_world()
    a.crust.create(1, CrustType["CONTINENT"], -10.0, 1)
    b.crust.create(1, CrustType["CONTINENT"], -10.0, 2)
    a.crust.flags[1] = CrustFlag["TRENCH"]
    result = choose_sides(world, a, b, [0, 2])
    assert result == {"upper": a, "lower": b}

    a.crust.flags[1] = 0
    a.stats["continentalCells"] = 1
    b.stats["continentalCells"] = 3
    result = choose_sides(world, a, b, [0])
    assert result == {"upper": b, "lower": a}


def underthrust_world():
    mesh = Mesh(5)
    upper, lower = Plate(0, 5), Plate(1, 5)
    for cell in range(5):
        upper.crust.create(cell, CrustType["CONTINENT"], -10.0, 10)
        lower.crust.create(cell, CrustType["CONTINENT"], -20.0, 20)
    upper.stats["continentalCells"] = 5
    lower.stats["continentalCells"] = 5
    owner = np.asarray([0, 0, 0, 0, 1], dtype=np.int16)
    land_claims = np.zeros(5, dtype=np.uint8)
    land_claims[[1, 2, 3]] = 2
    world = SimpleNamespace(
        mesh=mesh,
        plates=[upper, lower],
        owner=owner,
        ownerCell=np.arange(5, dtype=np.int32),
        landClaims=land_claims,
        budget={"underthrustCells": 0},
        config={
            "underthrustShare": 1.0,
            "overlapCapKm": 1000.0,
            "underthrustThickenKm": 25.0,
            "underthrustMaxThickKm": 40.0,
        },
    )
    return world, upper, lower


def test_underthrust_consumes_lower_edge_and_roots_upper_plate():
    from orogen.v2.tectonics.collision import underthrust

    world, upper, lower = underthrust_world()
    record = {"frontKm": 100.0, "underthrustDue": 0.0}
    consumed = underthrust(world, upper, lower, [1, 2, 3], record, 2.0)

    assert consumed == 2
    assert lower.crust.mask[1] == 0
    assert lower.crust.mask[2] == 0
    assert upper.crust.thickKm[1] == pytest.approx(25.0)
    assert upper.crust.rootKm[2] == pytest.approx(25.0)
    assert record["underthrustDue"] == 0.0
    assert record["underthrustCells"] == 2
    assert world.budget["underthrustCells"] == 2
    assert lower.reshaped is True
    assert upper.dirty is True


def test_underthrust_skips_without_convergence_or_lower_side_and_caps_unpaid_due():
    from orogen.v2.tectonics.collision import underthrust

    world, upper, lower = underthrust_world()
    record = {"frontKm": 100.0, "underthrustDue": 0.0}
    assert underthrust(world, upper, lower, [1, 2, 3], record, 0.0) == 0
    world.owner[4] = upper.id
    assert underthrust(world, upper, lower, [1, 2, 3], record, 2.0) == 0
    assert record["underthrustDue"] == pytest.approx(2.0)
    assert world.budget["underthrustCells"] == 0


def test_underthrust_skips_missing_ocean_and_craton_cells():
    from orogen.v2.tectonics.collision import underthrust

    world, upper, lower = underthrust_world()
    lower.crust.remove(2)
    lower.crust.type[1] = CrustType["CRATON"]
    lower.crust.type[3] = CrustType["OCEAN"]
    record = {"frontKm": 100.0, "underthrustDue": 0.0}
    assert underthrust(world, upper, lower, [1, 2, 3], record, 2.0) == 0
    assert world.budget["underthrustCells"] == 0
    assert record["underthrustDue"] == pytest.approx(1.0)


def test_enclosed_sea_detects_a_small_pocket_but_excludes_world_ocean():
    from orogen.v2.tectonics.collision import enclosed_sea_between_km2

    mesh = Mesh(7)
    upper, lower = Plate(0, 7), Plate(1, 7)
    upper.crust.create(0, CrustType["CONTINENT"], -10.0, 1)
    lower.crust.create(2, CrustType["CONTINENT"], -10.0, 2)
    world = SimpleNamespace(
        mesh=mesh,
        plates=[upper, lower],
        owner=np.asarray([0, -1, 1, -1, -1, -1, -1], dtype=np.int16),
        ownerCell=np.arange(7, dtype=np.int32),
        landClaims=np.zeros(7, dtype=np.uint8),
    )
    assert enclosed_sea_between_km2(world, upper, lower) == pytest.approx(100.0)


def test_open_sea_between_measures_directly_opposed_coasts(monkeypatch):
    from orogen.v2.tectonics import collision

    world, upper, lower = sea_world()
    monkeypatch.setattr(
        collision,
        "tangent_toward",
        lambda _a, _b, source: (
            [1.0, 0.0, 0.0] if float(source[1]) < 0.05 else [-1.0, 0.0, 0.0]
        ),
    )
    result = open_sea_between(world, upper, lower)
    assert result["km2"] > 0.0
    assert result["widestKm"] > 0.0


def test_open_sea_between_returns_zero_without_coastal_land():
    from orogen.v2.tectonics.collision import open_sea_between

    world, upper, lower = sea_world()
    world.owner[:] = -1
    assert open_sea_between(world, upper, lower) == {"km2": 0.0, "widestKm": 0.0}


def test_is_rift_jitter_requires_small_front_and_both_margin_flags():
    from orogen.v2.tectonics.collision import is_rift_jitter
    from orogen.v2.tectonics.crust import CrustFlag

    world, a, b = contact_world()
    world.config = {"riftJitterFrontKm": 2.0}
    a.crust.flags[0] = CrustFlag["RIFT_MARGIN"]
    b.crust.flags[0] = CrustFlag["RIFT_MARGIN"]
    assert is_rift_jitter(world, a, b, [0])
    b.crust.flags[0] = 0
    assert not is_rift_jitter(world, a, b, [0])
    assert not is_rift_jitter(world, a, b, [0, 1, 2])


def test_recycle_logs_sizable_arc_and_retires_donor():
    from orogen.v2.tectonics.collision import recycle

    world, donor, _host = contact_world()
    donor.crust.create(0, CrustType["ARC"], -20.0, 30)
    world.mesh.cell_area_km2 = 100.0
    world.config = {"terraneMinAreaKm2": 100.0}
    world.budget = {"recycledCells": 0}
    world.events = []
    world.log = lambda kind, text, data=None: world.events.append(
        {"kind": kind, "text": text, "data": data}
    )
    assert recycle(world, donor, [0]) == 1
    assert world.events[0]["kind"] == "arc-recycled"
    assert donor.alive is False
    assert donor.omega == [0.0, 0.0, 0.0]
    assert donor.dynamics is None


def test_recycle_small_arc_recycles_only_components_below_area_threshold():
    from orogen.v2.tectonics.collision import recycle_small_arc

    world, donor, _host = contact_world()
    donor.crust.create(0, CrustType["ARC"], -20.0, 30)
    donor.crust.create(1, CrustType["ARC"], -20.0, 30)
    world.config = {
        "arcAccreteMinKm2": 250.0,
        "terraneMinAreaKm2": 1000.0,
    }
    assert recycle_small_arc(world, donor, 0)
    assert donor.crust.mask[0] == 0
    assert donor.crust.mask[1] == 0

    donor.crust.create(0, CrustType["ARC"], -20.0, 30)
    donor.crust.create(1, CrustType["ARC"], -20.0, 30)
    donor.crust.create(2, CrustType["ARC"], -20.0, 30)
    assert not recycle_small_arc(world, donor, 0)
    donor.crust.type[3] = CrustType["CONTINENT"]
    assert not recycle_small_arc(world, donor, 3)


def test_accrete_arcs_returns_core_when_both_contacts_are_continent():
    from orogen.v2.tectonics.collision import accrete_arcs

    world, a, b = contact_world()
    world.config = {}
    assert accrete_arcs(world, a, b, [0]) == {"core": [0], "accreted": False}


def test_accrete_arcs_prefers_smaller_arc_donor_and_logs_transfer(monkeypatch):
    import importlib

    from orogen.v2.tectonics import collision

    scrape_module = importlib.import_module("orogen.v2.tectonics.ops.scrape_arc")

    world, a, b = contact_world()
    a.crust.type[0] = CrustType["ARC"]
    b.crust.type[0] = CrustType["ARC"]
    a.stats["continentalCells"] = 1
    b.stats["continentalCells"] = 4
    world.config = {
        "arcAccreteMinKm2": 0.0,
        "terraneMinAreaKm2": 100.0,
        "arcScrapeLogKm2": 100.0,
    }
    world.mesh.cell_area_km2 = 100.0
    calls = []
    monkeypatch.setattr(
        scrape_module,
        "scrape_arc",
        lambda _world, host, donor, cells: (
            calls.append((host.id, donor.id, list(cells)))
            or {"moved": 1, "absorbed": 0}
        ),
    )
    world.log = lambda kind, text, data=None: world.events.append(
        {"kind": kind, "text": text, "data": data}
    )

    result = collision.accrete_arcs(world, a, b, [0])

    assert result == {"core": [], "accreted": True}
    assert calls == [(1, 0, [0])]
    assert world.arcScrapes["0:1"]["logged"] is True
    assert world.events[-1]["kind"] == "arc-scrape"


def test_dock_if_terrane_returns_false_for_two_large_continents(monkeypatch):
    import importlib

    collision_module = importlib.import_module("orogen.v2.tectonics.collision")
    accretion_module = importlib.import_module(
        "orogen.v2.tectonics.ops.plate_accretion"
    )
    world, a, b = contact_world()
    world.mesh.cell_area_km2 = 100.0
    world.config = {"terraneMaxAreaKm2": 50.0}
    monkeypatch.setattr(
        accretion_module,
        "landmass_of",
        lambda plate, _mesh, _seeds: [0],
    )
    assert not collision_module.dock_if_terrane(world, a, b, [0])


def test_track_collisions_skips_sutured_records_and_marks_ready_sutures(monkeypatch):
    from orogen.v2.tectonics import collision

    world, upper, lower = collision_tracking_world()
    world.collisions["0:1"] = {
        "upper": 0,
        "lower": 1,
        "sutured": True,
        "lastActive": 10.0,
        "startedAt": 0.0,
    }
    result = collision.track_collisions(world, 1.0)
    assert result["tracked"] == 0
    assert world.collisions["0:1"]["sutured"] is True

    world.collisions.clear()
    world.config.update(
        {
            "sutureShorteningKm": 0.0,
            "sutureMinContactKm": 0.0,
        }
    )
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])
    result = collision.track_collisions(world, 1.0)
    assert result["tracked"] == 1
    assert world.collisions["0:1"]["sutureCandidate"] is True


def test_track_collisions_removes_a_newly_detected_parting_sliver(monkeypatch):
    from orogen.v2.tectonics import collision

    world, _upper, lower = collision_tracking_world()
    lower.omega = [0.0, 0.0, -0.01]
    world.config["collisionJitterFrontKm"] = 100.0
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])

    result = collision.track_collisions(world, 1.0)

    assert result["tracked"] == 0
    assert world.collisions == {}


def test_track_collisions_uses_arc_result_before_creating_record(monkeypatch):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    world.config.update(
        {
            "arcAccreteMinKm2": 100.0,
            "terraneMinAreaKm2": 100.0,
            "arcScrapeLogKm2": 100.0,
        }
    )
    monkeypatch.setattr(
        collision,
        "accrete_arcs",
        lambda *_args: {"core": [], "accreted": True},
    )

    result = collision.track_collisions(world, 1.0)

    assert result["tracked"] == 0
    assert result["changed"] is True
    assert world.collisions == {}


def test_underthrust_stops_after_reaching_the_due_target():
    from orogen.v2.tectonics.collision import underthrust

    world, upper, lower = underthrust_world()
    record = {"frontKm": 100.0, "underthrustDue": 0.0}
    consumed = underthrust(world, upper, lower, [0, 1, 2, 3], record, 1.0)
    assert consumed == 1
    assert record["underthrustCells"] == 1


def test_accrete_arcs_recycles_tiny_fragment_without_attempting_scrape(monkeypatch):
    import importlib

    from orogen.v2.tectonics import collision

    scrape_module = importlib.import_module("orogen.v2.tectonics.ops.scrape_arc")
    world, donor, host = contact_world()
    donor.crust.type[0] = CrustType["ARC"]
    world.config = {
        "arcAccreteMinKm2": 400.0,
        "terraneMinAreaKm2": 1000.0,
        "arcScrapeLogKm2": 1000.0,
    }
    monkeypatch.setattr(
        scrape_module,
        "scrape_arc",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("a tiny arc fragment should be recycled")
        ),
    )
    result = collision.accrete_arcs(world, donor, host, [0])
    assert result == {"core": [], "accreted": True}
    assert donor.crust.mask[0] == 0


def test_track_collisions_skips_touching_pair_with_existing_record():
    from orogen.v2.tectonics.collision import track_collisions

    world, _upper, _lower = collision_tracking_world()
    world.landClaims[:] = 0
    world.collider[:] = -1
    world.owner[:] = [0, 0, 1, 1, -1]
    world.config["terraneMaxAreaKm2"] = 100.0
    world.collisions["0:1"] = {
        "upper": 0,
        "lower": 1,
        "sutured": False,
        "lastActive": world.time,
        "startedAt": world.time,
    }
    result = track_collisions(world, 1.0)
    assert result["tracked"] == 0
    assert result["started"] == 0
    assert world.collisions["0:1"]["lastActive"] == world.time


def test_track_collisions_ignores_overlap_when_plate_is_dead():
    from orogen.v2.tectonics.collision import track_collisions

    world, _upper, lower = collision_tracking_world()
    lower.alive = False
    result = track_collisions(world, 1.0)
    assert result["tracked"] == 0
    assert result["started"] == 0
    assert world.collisions == {}


def test_track_collisions_skips_young_conjugate_rift_partner_sliver(monkeypatch):
    from orogen.v2.tectonics import collision

    world, upper, lower = collision_tracking_world()
    upper.riftPartner, lower.riftPartner = lower.id, upper.id
    upper.born = lower.born = 0.0
    world.config.update(
        {
            "riftJitterFrontKm": 100.0,
            "riftPushMyr": 20.0,
        }
    )
    monkeypatch.setattr(collision, "is_rift_jitter", lambda *_args: False)
    assert collision.are_young_rift_partners(upper, lower, world.time, world.config)
    result = collision.track_collisions(world, 1.0)
    assert result["started"] == 0
    assert result["tracked"] == 0
    assert world.collisions == {}


def test_track_collisions_docks_new_overlap_and_removes_its_record(monkeypatch):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    world.config["terraneMaxAreaKm2"] = 100.0
    monkeypatch.setattr(collision, "dock_if_terrane", lambda *_args: True)
    result = collision.track_collisions(world, 1.0)
    assert result["started"] == 1
    assert result["changed"] is True
    assert world.collisions == {}


def test_track_collisions_skips_record_if_its_saved_upper_plate_died():
    from orogen.v2.tectonics.collision import track_collisions

    world, _upper, lower = collision_tracking_world()
    dead = Plate(2, 5)
    dead.alive = False
    world.plates.append(dead)
    world.collisions["0:1"] = {
        "upper": 2,
        "lower": 1,
        "sutured": False,
        "lastActive": world.time,
        "startedAt": world.time,
    }
    result = track_collisions(world, 1.0)
    assert result["tracked"] == 0
    assert result["underthrustCells"] == 0


def test_track_collisions_skips_ambiguous_neighbour_claims(monkeypatch):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    world.landClaims[1] = 2
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])
    result = collision.track_collisions(world, 1.0)
    assert result["tracked"] == 1
    assert world.collisions["0:1"]["frontKm"] >= 0.0


def test_track_collisions_spreads_plateau_and_foreland_belts(monkeypatch):
    from orogen.v2.tectonics import collision
    from orogen.v2.tectonics.crust import BeltSide, OrogenyType

    world, upper, lower = collision_tracking_world()
    world.config.update(
        {
            "sutureMinContactKm": 0.0,
            "belts": {
                "himalayan": {"baseKm": 100.0, "perKm": 0.5, "maxKm": 500.0},
                "foreland": {"baseKm": 80.0, "perKm": 0.2, "maxKm": 300.0},
                "ural": {"baseKm": 50.0, "perKm": 0.2, "maxKm": 200.0},
            },
            "beltSpeedFactorMin": 0.6,
            "beltSpeedFactorMax": 1.6,
            "beltSpeedRefCmYr": 5.0,
            "beltSpeedExponent": 0.5,
            "beltObliqueMin": 0.35,
            "apronKm": 100.0,
            "orogenyQuietMyr": 10.0,
            "beltWorkShare": 0.5,
            "thickeningPerWorkKm": 0.1,
            "weakCellCost": 0.5,
            "cratonRimKm": 200.0,
            "cratonRimCost": 2.0,
            "cratonRimShare": 0.25,
        }
    )
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])

    result = collision.track_collisions(world, 1.0)

    assert result["tracked"] == 1
    assert upper.crust.oroType[2] == OrogenyType["HIMALAYAN"]
    assert lower.crust.oroType[2] == OrogenyType["HIMALAYAN"]
    assert upper.crust.beltSide[2] == BeltSide["UPPER"]
    assert lower.crust.beltSide[2] == BeltSide["LOWER"]
    assert upper.crust.thickKm[2] > 0.0
    assert lower.crust.thickKm[2] > 0.0


def suture_gap_config():
    return {
        "sutureShorteningKm": 0.0,
        "sutureMinContactKm": 0.0,
        "sutureCheckMyr": 4.0,
        "sutureGapKm": 200.0,
        "sutureGapWidthKm": 100.0,
        "sutureStallGiveUpMyr": 100.0,
        "sutureGapStallMyr": 40.0,
        "sutureLockMaxMyr": 100.0,
        "sutureStackKm": 10000.0,
        "ringSeaKm2": 100.0,
    }


def test_track_collisions_checks_gap_and_blocks_weld_for_large_enclosed_sea(
    monkeypatch,
):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    world.config.update(suture_gap_config())
    calls = []
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])
    monkeypatch.setattr(
        collision,
        "open_sea_between",
        lambda *_args: calls.append("gap") or {"km2": 2000.0, "widestKm": 250.0},
    )
    monkeypatch.setattr(
        collision,
        "enclosed_sea_between_km2",
        lambda *_args: calls.append("ring") or 5000.0,
    )
    world.config["sutureStallGiveUpMyr"] = 0.0

    result = collision.track_collisions(world, 1.0)
    record = world.collisions["0:1"]

    assert result["tracked"] == 1
    assert calls == ["gap", "ring"]
    assert record["gapKm2"] == 2000.0
    assert record["gapWidestKm"] == 250.0
    assert record["gapBestKm2"] == 2000.0
    assert record["gapShrankAt"] == world.time
    assert record["checkedAt"] == world.time
    assert record["ringKm2"] == 5000.0
    assert record["weldCandidate"] is True
    assert record["weldBlockedByRing"] is True
    assert record["mergeEligible"] is False
    assert record["ringLogged"] is True
    assert world.events[-1]["kind"] == "collision"
    assert world.events[-1]["data"]["ringKm2"] == 5000.0


def test_track_collisions_throttles_gap_checks_until_check_interval(monkeypatch):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    world.config.update(suture_gap_config())
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])
    calls = []
    monkeypatch.setattr(
        collision,
        "open_sea_between",
        lambda *_args: calls.append(world.time) or {"km2": 1000.0, "widestKm": 700.0},
    )
    monkeypatch.setattr(collision, "enclosed_sea_between_km2", lambda *_args: 0.0)

    collision.track_collisions(world, 1.0)
    assert calls == [10.0]
    record = world.collisions["0:1"]
    assert record["checkedAt"] == 10.0

    world.time = 11.0
    collision.track_collisions(world, 1.0)
    assert calls == [10.0]
    assert record["checkedAt"] == 10.0


def test_track_collisions_records_improving_gap_only_after_ten_percent_shrink(
    monkeypatch,
):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    world.config.update(suture_gap_config())
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])
    gaps = iter(
        [
            {"km2": 1000.0, "widestKm": 700.0},
            {"km2": 950.0, "widestKm": 650.0},
            {"km2": 800.0, "widestKm": 600.0},
        ]
    )
    monkeypatch.setattr(collision, "open_sea_between", lambda *_args: next(gaps))
    monkeypatch.setattr(collision, "enclosed_sea_between_km2", lambda *_args: 0.0)

    collision.track_collisions(world, 1.0)
    record = world.collisions["0:1"]
    assert record["gapBestKm2"] == 1000.0
    assert record["gapShrankAt"] == 10.0

    world.time = 15.0
    collision.track_collisions(world, 1.0)
    assert record["gapBestKm2"] == 1000.0
    assert record["gapShrankAt"] == 10.0

    world.time = 20.0
    collision.track_collisions(world, 1.0)
    assert record["gapBestKm2"] == 800.0
    assert record["gapShrankAt"] == 20.0


def test_track_collisions_allows_overdue_merge_candidate_despite_large_ring(
    monkeypatch,
):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    cfg = suture_gap_config()
    cfg["sutureLockMaxMyr"] = 0.0
    world.config.update(cfg)
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])
    monkeypatch.setattr(
        collision,
        "open_sea_between",
        lambda *_args: {"km2": 1000.0, "widestKm": 900.0},
    )
    monkeypatch.setattr(collision, "enclosed_sea_between_km2", lambda *_args: 5000.0)

    collision.track_collisions(world, 1.0)
    record = world.collisions["0:1"]
    assert record["weldCandidate"] is True
    assert record["weldBlockedByRing"] is False
    assert record["mergeEligible"] is True
    assert record["ringLogged"] is True
    assert world.events[-1]["kind"] == "collision"
    assert world.events[-1]["data"]["ringKm2"] == 5000.0


def test_track_collisions_executes_merge_and_logs_suture(monkeypatch):
    from orogen.v2.tectonics import collision
    from orogen.v2.tectonics.world import EventKind

    world, upper, lower = collision_tracking_world()
    world.config.update(suture_gap_config())
    world.config.update(
        {
            "sutureShorteningKm": 0.0,
            "belts": {
                "himalayan": {"baseKm": 100.0, "perKm": 0.5, "maxKm": 500.0},
                "foreland": {"baseKm": 80.0, "perKm": 0.2, "maxKm": 300.0},
                "ural": {"baseKm": 50.0, "perKm": 0.2, "maxKm": 200.0},
            },
            "beltSpeedFactorMin": 0.6,
            "beltSpeedFactorMax": 1.6,
            "beltSpeedRefCmYr": 5.0,
            "beltSpeedExponent": 0.5,
            "beltObliqueMin": 0.35,
            "apronKm": 100.0,
            "orogenyQuietMyr": 10.0,
            "beltWorkShare": 0.5,
            "thickeningPerWorkKm": 0.1,
            "weakCellCost": 0.5,
            "cratonRimKm": 200.0,
            "cratonRimCost": 2.0,
            "cratonRimShare": 0.25,
            "indentFalloffKm": 1500.0,
            "indentGain": 1.5,
            "escapeShare": 0.75,
            "sutureHalfWidthKm": 80.0,
            "sealStraitKm": 260.0,
            "inlandSeaMaxKm2": 500000.0,
        }
    )
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])
    monkeypatch.setattr(collision, "spread_belt", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        collision,
        "open_sea_between",
        lambda *_args: {"km2": 0.0, "widestKm": 0.0},
    )
    monkeypatch.setattr(collision, "enclosed_sea_between_km2", lambda *_args: 0.0)
    calls = []

    def fake_merge(w, a, b, cells, record):
        calls.append((w, a, b, list(cells), record))
        b.alive = False
        return {
            "landLostCells": 2,
            "maxIndentKm": 300.0,
            "meanIndentKm": 120.0,
            "overlapCells": len(cells),
            "sealedCells": 1,
        }

    monkeypatch.setattr(collision, "merge_plates", fake_merge)

    result = collision.track_collisions(world, 1.0)

    record = world.collisions["0:1"]
    assert result["tracked"] == 1
    assert result["changed"] is True
    assert len(calls) == 1
    assert calls[0][:4] == (world, upper, lower, [2])
    assert calls[0][4] is record
    assert record["sutured"] is True
    assert record["sutureReason"] == "shortening"
    assert lower.alive is False
    assert world.events[-1]["kind"] == EventKind["SUTURE"]
    assert world.events[-1]["data"]["maxIndentKm"] == 300.0
    assert world.events[-1]["data"]["sealedCells"] == 1


def test_track_collisions_keeps_merge_candidate_pending_without_merge_config(
    monkeypatch,
):
    from orogen.v2.tectonics import collision

    world, _upper, _lower = collision_tracking_world()
    world.config.update(suture_gap_config())
    world.config["sutureLockMaxMyr"] = 0.0
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])
    monkeypatch.setattr(
        collision,
        "open_sea_between",
        lambda *_args: {"km2": 0.0, "widestKm": 0.0},
    )
    monkeypatch.setattr(collision, "enclosed_sea_between_km2", lambda *_args: 0.0)
    calls = []
    monkeypatch.setattr(collision, "merge_plates", lambda *_args: calls.append(True))

    collision.track_collisions(world, 1.0)

    record = world.collisions["0:1"]
    assert record["mergeEligible"] is True
    assert record.get("sutured", False) is False
    assert calls == []


def test_track_collisions_feeds_far_field_from_both_collision_fronts(monkeypatch):
    from orogen.v2.tectonics import collision

    world, upper, lower = collision_tracking_world()
    world.config["farCollisionShare"] = 0.5
    monkeypatch.setattr(
        collision, "centroid_of_cells", lambda _xyz, _cells: [1.0, 0.0, 0.0]
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])
    calls = []
    monkeypatch.setattr(
        collision,
        "feed_far_field",
        lambda w, plate, key, cells, works, front, share: calls.append(
            (w, plate, key, list(cells), works, front, share)
        ),
    )

    result = collision.track_collisions(world, 1.0)

    assert result["tracked"] == 1
    assert len(calls) == 2
    assert calls[0][0:3] == (world, upper, "c0:1:u")
    assert calls[1][0:3] == (world, lower, "c0:1:l")
    assert calls[0][3] == [2]
    assert calls[1][3] == [2]
    assert calls[0][4] == calls[1][4] == pytest.approx(63.71)
    assert calls[0][5] == calls[1][5] == world.collisions["0:1"]["frontKm"]
    assert calls[0][6] == calls[1][6] == 0.5


def test_underthrust_consumes_lower_cells_without_upper_land_to_thicken():
    from orogen.v2.tectonics.collision import underthrust

    world, upper, lower = underthrust_world()
    upper.crust.remove(1)
    upper.crust.type[2] = CrustType["OCEAN"]
    record = {"frontKm": 100.0, "underthrustDue": 2.0}
    consumed = underthrust(world, upper, lower, [1, 2, 3], record, 2.0)
    assert consumed == 2
    assert lower.crust.mask[1] == 0
    assert lower.crust.mask[2] == 0
    assert upper.crust.mask[3] == 1
    # The oceanic upper cell is consumed below but must not be thickened.
    assert upper.crust.thickKm[2] == pytest.approx(0.0)
    assert upper.crust.rootKm[2] == pytest.approx(0.0)


def test_collision_front_cells_returns_unique_valid_land_contacts():
    from orogen.v2.tectonics.collision import _collision_front_cells

    world, upper, lower = collision_tracking_world()

    upper_cells, lower_cells = _collision_front_cells(world, upper, lower, [2, 2])
    assert upper_cells == [2]
    assert lower_cells == [2]

    lower.crust.type[3] = CrustType["OCEAN"]
    upper_cells, lower_cells = _collision_front_cells(world, upper, lower, [2])
    assert upper_cells == [2]
    assert lower_cells == []

    lower.crust.type[3] = CrustType["CONTINENT"]
    world.landClaims[1] = 2
    world.landClaims[3] = 2
    assert _collision_front_cells(world, upper, lower, [2]) == ([], [])


def test_step_world_runs_real_collision_tracker_on_deterministic_contact(monkeypatch):
    """Exercise the stepper-to-collision seam without replacing the tracker."""
    from orogen.v2.tectonics import collision, stepper

    world, upper, lower = collision_tracking_world()
    world.step = 0

    # Keep unrelated stages inert: the contact geometry and collision tracker
    # remain real, while this test isolates their integration in step_world.
    monkeypatch.setattr(stepper, "closing_rate", lambda _world: 0.0)
    monkeypatch.setattr(stepper, "advance_plates", lambda *_args: None)
    monkeypatch.setattr(stepper, "resolve", lambda _world: {})
    monkeypatch.setattr(stepper, "classify_contact_boundaries", lambda _world, _dt: 0)
    monkeypatch.setattr(stepper, "update_plumes", lambda _world, _dt: None)
    monkeypatch.setattr(stepper, "open_back_arcs", lambda _world, _dt: 0)
    monkeypatch.setattr(
        collision,
        "centroid_of_cells",
        lambda _xyz, _cells: [1.0, 0.0, 0.0],
    )
    monkeypatch.setattr(collision, "tangent_toward", lambda *_args: [0.0, 1.0, 0.0])

    result = stepper.step_world(
        world,
        1.0,
        update_motion=False,
        update_cycle=False,
    )

    record = world.collisions["0:1"]
    assert result["collisions"] == {
        "tracked": 1,
        "started": 1,
        "underthrustCells": 0,
        "changed": False,
    }
    assert record["upper"] == upper.id
    assert record["lower"] == lower.id
    assert record["shorteningKm"] == pytest.approx(63.71)
    assert [event["kind"] for event in world.events] == ["collision"]
