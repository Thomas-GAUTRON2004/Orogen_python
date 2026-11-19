import math

import numpy as np
import pytest

from orogen.v2.core.sphere_geometry import PLANET_RADIUS_KM
from orogen.v2.tectonics.backarc import (
    open_back_arcs,
    rollback_cm_yr,
    steer_back_arc_strips,
)
from orogen.v2.tectonics.config import CM_PER_YR_TO_RAD_PER_MYR
from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.world import BoundaryKind, EventKind, TectonicWorld


class Index:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)

    def nearest(self, x, y, z):
        point = np.asarray([x, y, z], dtype=np.float64)
        return int(np.argmax(self.points @ point))


class Mesh:
    def __init__(self, points, neighbours, edge_km=100.0, cell_area_km2=10000.0):
        self.num_cells = len(points)
        self.xyz = np.asarray(points, dtype=np.float32).reshape(-1)
        self.edge_km = float(edge_km)
        self.edge_rad = self.edge_km / PLANET_RADIUS_KM
        self.cell_area_km2 = float(cell_area_km2)
        offsets = [0]
        adjacency = []
        for row in neighbours:
            adjacency.extend(row)
            offsets.append(len(adjacency))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)
        self.index = Index(np.asarray(points, dtype=np.float64))


def backarc_config():
    return {
        "rollbackMinAgeMyr": 70.0,
        "rollbackFullAgeMyr": 150.0,
        "rollbackMaxCmYr": 3.0,
        "backArcRatePerMyr": 1.0,
        "backArcMinLengthKm": 800.0,
        "backArcWidthKm": 350.0,
        "backArcContinentalWidthKm": 650.0,
        "backArcContinentalShare": 1.0,
        "backArcMinLandKm2": 300000.0,
        "backArcActiveMyr": 30.0,
        "backArcSeamKm": 400.0,
        "backArcTenureMyr": 60.0,
        "riftPushMyr": 20.0,
    }


def make_line_mesh(count=5, edge_km=100.0):
    step = edge_km / 6371.0
    points = [[math.cos(i * step), math.sin(i * step), 0.0] for i in range(count)]
    neighbours = []
    for cell in range(count):
        row = []
        if cell:
            row.append(cell - 1)
        if cell + 1 < count:
            row.append(cell + 1)
        neighbours.append(row)
    return Mesh(points, neighbours, edge_km=edge_km)


def make_margin_world(*, patch_rng=True):
    rows, cols = 18, 15
    edge_km = 100.0
    step = edge_km / 6371.0
    points = []
    neighbours = []
    for row in range(rows):
        latitude = row * step
        for col in range(cols):
            longitude = (col - (cols - 1) / 2.0) * step
            points.append(
                [
                    math.cos(latitude) * math.cos(longitude),
                    math.cos(latitude) * math.sin(longitude),
                    math.sin(latitude),
                ]
            )
            cell = row * cols + col
            adjacent = []
            if row:
                adjacent.append(cell - cols)
            if col:
                adjacent.append(cell - 1)
            if col + 1 < cols:
                adjacent.append(cell + 1)
            if row + 1 < rows:
                adjacent.append(cell + cols)
            neighbours.append(adjacent)

    mesh = Mesh(points, neighbours, edge_km=edge_km)
    world = TectonicWorld(mesh, 73, backarc_config())
    world.time = 100.0
    parent = world.add_plate("West Arc")
    slab = world.add_plate("Old Ocean")
    other = mesh.num_cells - 1
    for cell in range(mesh.num_cells):
        if cell == other:
            continue
        row = cell // cols
        crust_type = CrustType["OCEAN"] if row == 0 else CrustType["CONTINENT"]
        parent.crust.create(cell, crust_type, 0.0, cell + 1)
    slab.crust.create(other, CrustType["OCEAN"], 0.0, 999)
    parent.refresh_extent(mesh.xyz)
    slab.refresh_extent(mesh.xyz)
    world.owner.fill(parent.id)
    world.owner[other] = slab.id
    world.ownerCell[:] = np.arange(mesh.num_cells, dtype=np.int32)
    world.boundaryKind.fill(BoundaryKind["NONE"])
    world.boundaryKind[1 : cols - 1] = BoundaryKind["OVERRIDING"]
    if patch_rng:
        world.rngBackArc = lambda: 0.0
    return world, parent, slab, other


def test_rollback_rate_is_age_weighted_and_capped():
    config = backarc_config()
    assert rollback_cm_yr(60.0, config) == 0.0
    assert rollback_cm_yr(70.0, config) == 0.0
    assert rollback_cm_yr(110.0, config) == pytest.approx(1.5)
    assert rollback_cm_yr(150.0, config) == pytest.approx(3.0)
    assert rollback_cm_yr(250.0, config) == pytest.approx(3.0)


def test_rollback_rate_rejects_degenerate_age_interval():
    config = backarc_config()
    config["rollbackFullAgeMyr"] = config["rollbackMinAgeMyr"]
    with pytest.raises(ValueError, match="must exceed"):
        rollback_cm_yr(100.0, config)


def test_open_back_arc_splits_a_coherent_continental_margin(monkeypatch):
    import orogen.v2.tectonics.backarc as backarc

    world, parent, _slab, other = make_margin_world()

    def contact(_world, _cell, normal):
        normal[:] = [0.0, 1.0, 0.0, 1.0]
        return other

    monkeypatch.setattr(backarc, "contact_at", contact)

    assert open_back_arcs(world, 1.0) == 1
    fresh = world.plates[-1]
    assert fresh.name == "West Isles"
    assert fresh.backArcOf == parent.id
    assert fresh.rollbackCmYr == pytest.approx(1.125)
    assert fresh.lastBackArcAt == world.time
    assert parent.lastBackArcAt == world.time
    assert fresh.crust.count == 105
    assert parent.crust.count == world.mesh.num_cells - 1 - fresh.crust.count
    assert fresh.riftPartner == parent.id
    assert parent.riftPartner == fresh.id
    assert np.any(fresh.crust.flags & CrustFlag["RIFT_MARGIN"])
    assert np.any(parent.crust.flags & CrustFlag["RIFT_MARGIN"])
    assert world.events[-1]["kind"] == EventKind["BACKARC_OPEN"]
    assert world.events[-1]["data"]["continental"] is True


@pytest.mark.parametrize("slab_age", [20.0, 50.0, 69.9])
def test_open_back_arc_ignores_young_slabs(monkeypatch, slab_age):
    import orogen.v2.tectonics.backarc as backarc

    world, _parent, slab, other = make_margin_world()
    slab.crust.formedAt[other] = world.time - slab_age

    def contact(_world, _cell, normal):
        normal[:] = [0.0, 1.0, 0.0, 1.0]
        return other

    monkeypatch.setattr(backarc, "contact_at", contact)

    assert open_back_arcs(world, 1.0) == 0
    assert len(world.plates) == 2


def test_open_back_arc_requires_the_overriding_plate_to_lag_rollback(monkeypatch):
    import orogen.v2.tectonics.backarc as backarc

    world, parent, _slab, other = make_margin_world()
    parent.omega = [0.0, 0.0, 2.0 * CM_PER_YR_TO_RAD_PER_MYR]

    def contact(_world, _cell, normal):
        normal[:] = [0.0, 1.0, 0.0, 1.0]
        return other

    monkeypatch.setattr(backarc, "contact_at", contact)

    assert open_back_arcs(world, 1.0) == 0
    assert len(world.plates) == 2


def test_finished_back_arc_rejoins_and_bridges_its_empty_seam():
    mesh = make_line_mesh()
    world = TectonicWorld(mesh, 11, backarc_config())
    world.time = 30.0
    world.config["backArcRatePerMyr"] = 0.0
    parent = world.add_plate("Parent")
    strip = world.add_plate("Parent Arc")
    for cell in (0, 1):
        parent.crust.create(cell, CrustType["CONTINENT"], -10.0, cell + 1)
    for cell in (3, 4):
        strip.crust.create(cell, CrustType["CONTINENT"], -5.0, cell + 10)
    parent.refresh_extent(mesh.xyz)
    strip.refresh_extent(mesh.xyz)
    strip.backArcOf = parent.id
    strip.born = 0.0
    parent.riftPartner = strip.id

    assert open_back_arcs(world, 1.0) == 1
    assert strip.alive is False
    assert strip.backArcOf == -1
    assert strip.crust.count == 0
    assert parent.crust.mask.tolist() == [1, 1, 1, 1, 1]
    assert parent.crust.type[2] == CrustType["OCEAN"]
    assert parent.riftPartner == -1
    assert world.events[-1]["kind"] == EventKind["BACKARC_CLOSE"]
    assert world.events[-1]["data"]["rejoined"] is True
    assert world.events[-1]["data"]["seamCells"] == 1


def test_finished_back_arc_becomes_independent_when_parent_is_gone():
    mesh = make_line_mesh()
    world = TectonicWorld(mesh, 12, backarc_config())
    world.time = 30.0
    world.config["backArcRatePerMyr"] = 0.0
    parent = world.add_plate("Parent")
    strip = world.add_plate("Parent Arc")
    parent.alive = False
    strip.backArcOf = parent.id
    strip.born = 0.0

    assert open_back_arcs(world, 1.0) == 0
    assert strip.backArcOf == -1
    assert strip.alive is True
    assert world.events[-1]["kind"] == EventKind["BACKARC_CLOSE"]
    assert world.events[-1]["data"]["rejoined"] is False


def test_steer_back_arc_strip_adds_parent_velocity_and_rollback():
    mesh = make_line_mesh()
    world = TectonicWorld(mesh, 13, backarc_config())
    world.time = 10.0
    parent = world.add_plate("Parent")
    strip = world.add_plate("Parent Arc")
    parent.omega = [0.01, 0.02, 0.03]
    strip.backArcOf = parent.id
    strip.backArcPushDir = [0.0, 1.0, 0.0]
    strip.capCenter = [0.0, 0.0, 1.0]
    strip.rollbackCmYr = 2.0
    strip.born = 0.0

    assert steer_back_arc_strips(world) == 1
    rate = 2.0 * CM_PER_YR_TO_RAD_PER_MYR
    assert strip.omega == pytest.approx([0.01 - rate, 0.02, 0.03])
    added = np.asarray(strip.omega) - np.asarray(parent.omega)
    centre = np.asarray(strip.to_world(*strip.capCenter))
    tangent = np.asarray(strip.to_world(*strip.backArcPushDir))
    tangent -= np.dot(tangent, centre) * centre
    velocity = np.cross(added, centre)
    assert np.dot(added, centre) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(velocity, tangent) > 0.0

    world.time = 30.0
    assert steer_back_arc_strips(world) == 0


def test_rollback_fronts_skips_invalid_overriding_contacts(monkeypatch):
    import orogen.v2.tectonics.backarc as backarc

    world, parent, slab, other = make_margin_world()

    def contact(_world, _cell, normal):
        normal[:] = [0.0, 1.0, 0.0, 1.0]
        return other

    monkeypatch.setattr(backarc, "contact_at", contact)
    world.boundaryKind.fill(BoundaryKind["NONE"])
    assert backarc._rollback_fronts(world) == {}

    world.boundaryKind[1] = BoundaryKind["OVERRIDING"]
    world.owner[1] = -1
    assert backarc._rollback_fronts(world) == {}
    world.owner[1] = parent.id

    for state in ("dead", "background", "reshaped"):
        if state == "dead":
            parent.alive = False
        elif state == "background":
            parent.isBackground = True
        else:
            parent.mark_reshaped()
        assert backarc._rollback_fronts(world) == {}
        parent.alive = True
        parent.isBackground = False
        parent.reshaped = False

    monkeypatch.setattr(backarc, "contact_at", lambda *_args: -1)
    assert backarc._rollback_fronts(world) == {}
    monkeypatch.setattr(backarc, "contact_at", contact)
    world.owner[other] = -1
    assert backarc._rollback_fronts(world) == {}
    world.owner[other] = slab.id


@pytest.mark.parametrize(
    "invalid_slab", ["dead", "background", "reshaped", "empty", "continental"]
)
def test_rollback_fronts_skips_non_ocean_slab_cells(monkeypatch, invalid_slab):
    import orogen.v2.tectonics.backarc as backarc

    world, _parent, slab, other = make_margin_world()

    def contact(_world, _cell, normal):
        normal[:] = [0.0, 1.0, 0.0, 1.0]
        return other

    monkeypatch.setattr(backarc, "contact_at", contact)
    if invalid_slab == "dead":
        slab.alive = False
    elif invalid_slab == "background":
        slab.isBackground = True
    elif invalid_slab == "reshaped":
        slab.mark_reshaped()
    elif invalid_slab == "empty":
        slab.crust.remove(other)
    else:
        slab.crust.type[other] = CrustType["CONTINENT"]

    assert backarc._rollback_fronts(world) == {}


def test_rollback_fronts_groups_eligible_cells_and_rejects_young_or_fast_contacts(
    monkeypatch,
):
    import orogen.v2.tectonics.backarc as backarc

    world, parent, slab, other = make_margin_world()
    calls = 0

    def contact(_world, _cell, normal):
        nonlocal calls
        calls += 1
        normal[:] = [0.0, 1.0, 0.0, 1.0]
        return other

    monkeypatch.setattr(backarc, "contact_at", contact)
    slab.crust.formedAt[other] = world.time - 69.0
    assert backarc._rollback_fronts(world) == {}

    slab.crust.formedAt[other] = world.time - 100.0
    parent.omega = [0.0, 0.0, 2.0 * CM_PER_YR_TO_RAD_PER_MYR]
    assert backarc._rollback_fronts(world) == {}

    parent.omega = [0.0, 0.0, 0.0]
    fronts = backarc._rollback_fronts(world)
    assert len(fronts) == 1
    front = fronts[(parent.id, slab.id)]
    assert front["cells"]
    assert front["ages"] == pytest.approx(100.0 * len(front["cells"]))
    assert front["direction"][1] == pytest.approx(float(len(front["cells"])))


def test_open_back_arcs_handles_missing_config_nonpositive_dt_and_disabled_rate():
    mesh = make_line_mesh()
    world = TectonicWorld(mesh, 21, {})
    assert open_back_arcs(world, 1.0) == 0

    world.config = {"backArcActiveMyr": 30.0, "backArcSeamKm": 400.0}
    assert open_back_arcs(world, 0.0) == 0

    world.config = backarc_config()
    world.config["backArcRatePerMyr"] = 0.0
    assert open_back_arcs(world, 1.0) == 0


@pytest.mark.parametrize(
    "reason",
    ["dead", "reshaped", "tenure", "short", "chance", "no_strip", "continental_chance"],
)
def test_try_open_back_arc_rejects_ineligible_fronts(monkeypatch, reason):
    import orogen.v2.tectonics.backarc as backarc

    world, parent, _slab, _other = make_margin_world()
    front = {
        "plate": parent,
        "cells": list(range(1, 12)),
        "ages": 100.0 * 11,
        "direction": [0.0, 1.0, 0.0],
    }
    if reason == "dead":
        parent.alive = False
    elif reason == "reshaped":
        parent.mark_reshaped()
    elif reason == "tenure":
        parent.lastBackArcAt = world.time
    elif reason == "short":
        front["cells"] = [1]
    elif reason == "chance":
        world.rngBackArc = lambda: 2.0
    elif reason == "no_strip":
        monkeypatch.setattr(backarc, "_strip_behind", lambda *_args: None)
    else:
        monkeypatch.setattr(
            backarc,
            "_strip_behind",
            lambda *_args: {"cells": [1], "continental": True},
        )
        world.config["backArcContinentalShare"] = 0.0
        world.rngBackArc = lambda: 0.5

    assert backarc._try_open_back_arc(world, front, 1.0) is False
    assert len(world.plates) == 2


def test_judge_strip_rejects_cratons_majority_slivers_and_disconnected_land():
    import orogen.v2.tectonics.backarc as backarc

    world, parent, _slab, _other = make_margin_world()
    selected = np.zeros(world.mesh.num_cells, dtype=np.uint8)

    selected[0] = 1
    parent.crust.type[0] = CrustType["CRATON"]
    assert backarc._judge_strip(world, parent, selected) is None
    parent.crust.type[0] = CrustType["OCEAN"]

    selected[:] = 0
    selected[:140] = 1
    assert backarc._judge_strip(world, parent, selected) is None

    selected[:] = 0
    selected[np.arange(16, 256, 8)[:30]] = 1
    assert backarc._judge_strip(world, parent, selected) is None


def test_judge_strip_rejects_a_disconnected_remainder_and_accepts_ocean_arc():
    import orogen.v2.tectonics.backarc as backarc

    world, parent, _slab, _other = make_margin_world()
    selected = np.zeros(world.mesh.num_cells, dtype=np.uint8)
    selected[np.arange(7, world.mesh.num_cells, 15)] = 1
    world.config["backArcMinLandKm2"] = 0.0
    assert backarc._judge_strip(world, parent, selected) is None

    selected[:] = 0
    selected[:14] = 1
    parent.crust.type[:14] = CrustType["ARC"]
    result = backarc._judge_strip(world, parent, selected)
    assert result == {"cells": list(range(14)), "continental": False}


def test_strip_name_is_unique_and_strips_existing_suffixes():
    import orogen.v2.tectonics.backarc as backarc

    world, parent, _slab, _other = make_margin_world()
    assert backarc._strip_name(world, parent, True) == "West Isles"
    world.add_plate("West Isles")
    assert backarc._strip_name(world, parent, True) == "West Isles 2"
    world.add_plate("West Isles 2")
    assert backarc._strip_name(world, parent, True) == "West Isles 3"
    assert backarc._strip_name(world, parent, False) == "West Arc 2"


def test_steer_back_arc_strips_skips_inactive_or_invalid_strips():
    mesh = make_line_mesh()
    world = TectonicWorld(mesh, 22, {})
    assert steer_back_arc_strips(world) == 0

    world.config = backarc_config()
    parent = world.add_plate("Parent")
    strips = [world.add_plate(f"Strip {index}") for index in range(5)]
    strips[0].backArcOf = -1
    strips[1].backArcOf = parent.id
    strips[1].backArcPushDir = None
    strips[2].backArcOf = 99
    strips[2].backArcPushDir = [0.0, 1.0, 0.0]
    strips[3].backArcOf = parent.id
    strips[3].backArcPushDir = [0.0, 1.0, 0.0]
    strips[3].born = -30.0
    strips[4].backArcOf = parent.id
    strips[4].backArcPushDir = [0.0, 1.0, 0.0]
    parent.alive = False
    assert steer_back_arc_strips(world) == 0


def test_open_back_arc_uses_real_world_rng_and_contact_geometry():
    import orogen.v2.tectonics.backarc as backarc

    world, parent, slab, _other = make_margin_world(patch_rng=False)
    world.config["backArcRatePerMyr"] = 0.0
    cols = 15
    for col in range(1, 11):
        cell = cols + col
        parent.crust.remove(cell)
        slab.crust.create(cell, CrustType["OCEAN"], 0.0, 2000 + col)
        world.owner[cell] = slab.id
        world.ownerCell[cell] = cell

    fronts = backarc._rollback_fronts(world)
    assert (parent.id, slab.id) in fronts
    assert len(fronts[(parent.id, slab.id)]["cells"]) >= 8
    assert open_back_arcs(world, 1.0) == 0
    assert len(world.plates) == 2


def test_split_strip_clears_expired_reciprocal_partner():
    import orogen.v2.tectonics.backarc as backarc

    world, parent, _slab, _other = make_margin_world()
    old_partner = world.add_plate("Old Rift Partner")
    parent.riftPartner = old_partner.id
    old_partner.riftPartner = parent.id
    old_partner.riftPushDir = [1.0, 0.0, 0.0]
    parent.born = 0.0
    old_partner.born = 0.0

    fresh = backarc._split_strip(
        world,
        parent,
        {"cells": [1], "continental": False},
        [0.0, 1.0, 0.0],
        1.0,
    )

    assert parent.riftPartner == fresh.id
    assert fresh.riftPartner == parent.id
    assert old_partner.riftPartner == -1
    assert old_partner.riftPushDir is None
