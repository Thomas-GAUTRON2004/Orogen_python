"""Focused edge tests for remaining Python V2 coverage gaps."""

from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import backarc, boundaries, config, initial_world, scenarios
from orogen.v2.tectonics.backarc import (
    _bridge_pieces,
    _judge_strip,
    _split_strip,
    _strip_behind,
)
from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType
from orogen.v2.tectonics.world import EventKind, TectonicWorld


def _line_graph(count=5, *, cell_area_km2=200_000.0):
    points = np.column_stack(
        (
            np.cos(np.arange(count) * 0.01),
            np.sin(np.arange(count) * 0.01),
            np.zeros(count),
        )
    ).astype(np.float32)
    neighbours = []
    for cell in range(count):
        row = []
        if cell:
            row.append(cell - 1)
        if cell + 1 < count:
            row.append(cell + 1)
        neighbours.append(row)
    offsets = [0]
    adjacency = []
    for row in neighbours:
        adjacency.extend(row)
        offsets.append(len(adjacency))
    return SimpleNamespace(
        num_cells=count,
        xyz=points.reshape(-1),
        adj_offset=np.asarray(offsets, dtype=np.int32),
        adj_list=np.asarray(adjacency, dtype=np.int32),
        edge_km=100.0,
        edge_rad=100.0 / 6371.0,
        cell_area_km2=cell_area_km2,
    )


def _backarc_config():
    return {
        "backArcContinentalWidthKm": 650.0,
        "backArcWidthKm": 350.0,
        "backArcMinLandKm2": 300_000.0,
        "backArcActiveMyr": 30.0,
        "backArcSeamKm": 400.0,
        "backArcTenureMyr": 60.0,
        "backArcRatePerMyr": 1.0,
        "backArcMinLengthKm": 800.0,
        "backArcContinentalShare": 1.0,
        "rollbackMinAgeMyr": 70.0,
        "rollbackFullAgeMyr": 150.0,
        "rollbackMaxCmYr": 3.0,
        "riftPushMyr": 20.0,
    }


def _crust(types):
    result = CrustRaster(len(types))
    for cell, crust_type in enumerate(types):
        result.create(cell, crust_type, -10.0, cell + 1)
    return result


def test_initial_world_private_mesh_and_distance_helpers_validate_inputs():
    mesh = build_sim_mesh(48, 91)
    with pytest.raises(ValueError, match="xyz does not match"):
        initial_world._mesh_xyz(SimpleNamespace(xyz=np.zeros(3), num_cells=2))
    with pytest.raises(ValueError, match="source mask"):
        initial_world._distance_hops(mesh, np.zeros(2, dtype=bool))
    empty = initial_world._distance_hops(mesh, np.zeros(mesh.num_cells, dtype=bool))
    assert np.all(empty == -1)

    with pytest.raises(ValueError, match="source mask"):
        initial_world._distance_km(mesh, np.zeros(2, dtype=bool))
    distances, strengths = initial_world._distance_km(
        mesh, np.zeros(mesh.num_cells, dtype=bool)
    )
    assert np.all(np.isinf(distances))
    assert np.all(strengths == 0)

    sources = np.zeros(mesh.num_cells, dtype=bool)
    sources[0] = True
    distances, strengths = initial_world._distance_km(
        mesh, sources, can_enter=lambda _cell: False
    )
    assert distances[0] == 0
    assert np.all(np.isinf(distances[1:]))
    assert strengths[0] == 1


def test_initial_world_craton_picker_falls_back_when_core_is_too_small():
    xyz = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (5, 1))
    cells = np.arange(5, dtype=np.int32)
    picked = initial_world._pick_craton_centres(
        xyz, cells, xyz[0], 0.0, 4, 10_000.0, 1.0, lambda: 0.5
    )
    assert picked.shape == (4,)
    assert len(np.unique(picked)) == 4


def test_initial_world_seams_distinguish_major_inner_and_ring_edges():
    mesh = build_sim_mesh(48, 92)
    xyz = np.asarray(mesh.xyz).reshape(-1, 3)
    first = 0
    second = int(mesh.adj_list[mesh.adj_offset[first]])
    land_cells = np.asarray([first, second], dtype=np.int32)
    mask = np.zeros(mesh.num_cells, dtype=bool)
    mask[land_cells] = True
    centre = xyz[first]

    major = initial_world._seams(
        mesh, mask, land_cells, np.asarray([1, 2]), centre, 0.0, np.asarray([0, 1])
    )
    assert major[first] == initial_world.StartSeam["MAJOR"]
    inner = initial_world._seams(
        mesh, mask, land_cells, np.asarray([1, 2]), centre, 0.0, np.asarray([0, 0])
    )
    assert inner[first] == initial_world.StartSeam["INNER"]

    ring_mask = np.zeros(mesh.num_cells, dtype=bool)
    ring_mask[first] = True
    ring = initial_world._seams(
        mesh,
        ring_mask,
        np.asarray([first], dtype=np.int32),
        np.asarray([1]),
        centre,
        0.0,
        np.asarray([0]),
    )
    assert ring[first] == initial_world.StartSeam["RING"]


def test_paint_old_ranges_skips_invalid_and_empty_range_definitions():
    mesh = build_sim_mesh(48, 93)
    world = TectonicWorld(mesh, 93)
    plate = world.add_plate("Test")
    plate.crust.create_many(
        np.arange(mesh.num_cells, dtype=np.int32), CrustType["CONTINENT"], -100.0
    )
    land = np.ones(mesh.num_cells, dtype=bool)
    seams = np.zeros(mesh.num_cells, dtype=np.uint8)

    world.config["startRanges"] = None
    initial_world._paint_old_ranges(world, plate, land, seams)
    world.config["startRanges"] = {
        "inner": [0.0, 10.0, 20.0, 0.5],
        "ring": [100.0, 10.0, 20.0],
        "major": [100.0, 0.0, 20.0, 0.5],
    }
    initial_world._paint_old_ranges(world, plate, land, seams)
    assert np.count_nonzero(plate.crust.oroType) == 0


def test_paint_old_ranges_spreads_a_real_inherited_seam():
    mesh = build_sim_mesh(96, 94)
    world = TectonicWorld(mesh, 94)
    plate = world.add_plate("Test")
    plate.crust.create_many(
        np.arange(mesh.num_cells, dtype=np.int32), CrustType["CONTINENT"], -100.0
    )
    land = np.ones(mesh.num_cells, dtype=bool)
    seams = np.zeros(mesh.num_cells, dtype=np.uint8)
    seams[0] = initial_world.StartSeam["INNER"]
    world.config["startRanges"] = {
        "inner": [2500.0, 100.0, 50.0, 0.5],
        "ring": [],
        "major": [],
    }
    initial_world._paint_old_ranges(world, plate, land, seams)
    assert plate.crust.oroType[0] != 0
    assert np.count_nonzero(plate.crust.beltReach) > 0


def test_start_world_handles_small_mesh_without_apron_or_active_trench(monkeypatch):
    tiny = SimpleNamespace(num_cells=3, xyz=np.zeros(9))
    monkeypatch.setattr(
        initial_world,
        "TectonicWorld",
        lambda *_args, **_kwargs: SimpleNamespace(config={"continentalFraction": 0.29}),
    )
    with pytest.raises(ValueError, match="at least four"):
        initial_world.build_start_world(tiny, 95)
    monkeypatch.undo()

    mesh = build_sim_mesh(64, 96)

    def land_only_hops(_mesh, sources, **_kwargs):
        source_mask = np.asarray(sources, dtype=bool)
        if not np.any(source_mask):
            return np.full(mesh.num_cells, -1, dtype=np.int16)
        return np.where(source_mask, 0, -1).astype(np.int16)

    monkeypatch.setattr(initial_world, "_distance_hops", land_only_hops)
    result = initial_world.build_start_world(
        mesh,
        96,
        {
            "rimTrenchCoverage": 0.0,
            "farStartWorkKm": 0.0,
            "startRanges": {},
        },
    )
    continent, _ocean = result["world"].plates
    assert result["start"]["apronCells"] == 0
    assert result["start"]["trench"].size == 0
    assert not np.any(continent.crust.flags & CrustFlag["TRENCH"])


def test_scenario_validation_and_oceanic_empty_hemisphere_fallback(monkeypatch):
    with pytest.raises(ValueError, match="xyz does not match"):
        scenarios._xyz(SimpleNamespace(xyz=np.zeros(3), num_cells=2))

    mesh = build_sim_mesh(48, 97)
    monkeypatch.setattr(
        scenarios,
        "_xyz",
        lambda sim_mesh: np.tile([[1.0, 0.0, 0.0]], (sim_mesh.num_cells, 1)),
    )
    monkeypatch.setattr(scenarios, "random_point", lambda _rng: [1.0, 0.0, 0.0])
    result = scenarios.build_oceanic_scene(mesh, 97)
    assert (
        result["start"]["plateACells"] + result["start"]["plateBCells"]
        == mesh.num_cells
    )
    assert result["start"]["plateACells"] > 0
    assert result["start"]["plateBCells"] > 0


def test_backarc_strip_judgement_covers_rejections_and_success():
    mesh = _line_graph()
    config_values = _backarc_config()
    world = SimpleNamespace(mesh=mesh, config=config_values)

    all_land = _crust([CrustType["CONTINENT"]] * 5)
    assert _judge_strip(world, SimpleNamespace(crust=all_land), np.zeros(5)) is None
    assert (
        _judge_strip(
            world,
            SimpleNamespace(crust=all_land),
            np.asarray([1, 1, 1, 1, 1], dtype=np.uint8),
        )
        is None
    )

    craton = _crust(
        [
            CrustType["CONTINENT"],
            CrustType["CRATON"],
            CrustType["CONTINENT"],
            CrustType["CONTINENT"],
            CrustType["CONTINENT"],
        ]
    )
    assert (
        _judge_strip(
            world,
            SimpleNamespace(crust=craton),
            np.asarray([0, 1, 0, 0, 0], dtype=np.uint8),
        )
        is None
    )

    small = _crust([CrustType["CONTINENT"]] * 5)
    low_area_world = SimpleNamespace(
        mesh=_line_graph(cell_area_km2=100.0), config=config_values
    )
    assert (
        _judge_strip(
            low_area_world,
            SimpleNamespace(crust=small),
            np.asarray([1, 0, 0, 0, 0], dtype=np.uint8),
        )
        is None
    )

    detached_land = _crust([CrustType["CONTINENT"]] * 5)
    assert (
        _judge_strip(
            world,
            SimpleNamespace(crust=detached_land),
            np.asarray([1, 0, 1, 0, 0], dtype=np.uint8),
        )
        is None
    )

    split_remainder = _crust(
        [
            CrustType["CONTINENT"],
            CrustType["OCEAN"],
            CrustType["OCEAN"],
            CrustType["OCEAN"],
            CrustType["CONTINENT"],
        ]
    )
    assert (
        _judge_strip(
            world,
            SimpleNamespace(crust=split_remainder),
            np.asarray([0, 1, 0, 0, 0], dtype=np.uint8),
        )
        is None
    )

    accepted = _judge_strip(
        world,
        SimpleNamespace(crust=all_land),
        np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
    )
    assert accepted == {"cells": [0, 1], "continental": True}


def test_strip_behind_retries_the_narrower_oceanic_band(monkeypatch):
    mesh = _line_graph()
    plate = SimpleNamespace(crust=_crust([CrustType["OCEAN"]] * 5))
    world = SimpleNamespace(mesh=mesh, config=_backarc_config())
    widths = []
    bands = [
        np.asarray([1, 0, 0, 0, 0], dtype=np.uint8),
        np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
    ]
    monkeypatch.setattr(
        backarc,
        "_band_behind",
        lambda _world, _plate, _segment, width: (widths.append(width), bands.pop(0))[1],
    )
    accepted = {"cells": [0], "continental": False}
    monkeypatch.setattr(backarc, "_judge_strip", lambda *_args: accepted)
    assert _strip_behind(world, plate, [0]) is accepted
    assert widths == [650.0, 350.0]


def test_bridge_pieces_handles_single_piece_and_fills_short_seam():
    mesh = _line_graph()
    crust = _crust(
        [
            CrustType["CONTINENT"],
            CrustType["CONTINENT"],
            CrustType["OCEAN"],
            CrustType["CONTINENT"],
            CrustType["CONTINENT"],
        ]
    )
    crust.remove(2)
    plate = SimpleNamespace(crust=crust, reshaped=False)
    plate.mark_reshaped = lambda: setattr(plate, "reshaped", True)
    assert _bridge_pieces(plate, mesh, 400.0) > 0
    assert plate.reshaped is True
    assert crust.mask[2]
    assert crust.type[2] == CrustType["OCEAN"]

    single = SimpleNamespace(crust=_crust([CrustType["CONTINENT"]] * 5))
    single.mark_reshaped = lambda: pytest.fail("single piece should not be reshaped")
    assert _bridge_pieces(single, mesh, 400.0) == 0


def test_split_strip_preserves_an_active_parent_rift_partnership():
    mesh = build_sim_mesh(48, 98)
    world = TectonicWorld(mesh, 98, _backarc_config())
    world.time = 10.0
    world.config["riftPushMyr"] = 20.0
    parent = world.add_plate("Parent")
    partner = world.add_plate("Partner")
    parent.born = 0.0
    partner.born = 5.0
    parent.riftPartner = partner.id
    partner.riftPartner = parent.id
    parent.crust.create(0, CrustType["CONTINENT"], -10.0, 1)
    partner.crust.create(1, CrustType["CONTINENT"], -10.0, 2)
    fresh = _split_strip(
        world, parent, {"cells": [0], "continental": True}, [0.0, 1.0, 0.0], 1.0
    )
    assert fresh.backArcOf == parent.id
    assert parent.riftPartner == partner.id
    assert partner.riftPartner == parent.id


def test_backarc_open_guards_and_orphan_strip_cleanup():
    mesh = build_sim_mesh(32, 99)
    world = TectonicWorld(mesh, 99, _backarc_config())
    assert backarc.open_back_arcs(world, 0.0) == 0
    world.config.pop("backArcRatePerMyr")
    assert backarc.open_back_arcs(world, 1.0) == 0

    world = TectonicWorld(mesh, 100, _backarc_config())
    world.time = 100.0
    world.config["backArcRatePerMyr"] = 0.0
    strip = world.add_plate("Orphan Arc")
    strip.backArcOf = 99
    strip.born = 0.0
    assert backarc.open_back_arcs(world, 1.0) == 0
    assert strip.backArcOf == -1
    assert world.events[-1]["kind"] == EventKind["BACKARC_CLOSE"]
    assert world.events[-1]["data"]["rejoined"] is False


def _boundary_world(seed=101):
    mesh = build_sim_mesh(64, seed)
    world = TectonicWorld(mesh, seed)
    plate = world.add_plate("Boundary test")
    plate.crust.create(0, CrustType["CONTINENT"], -100.0, 1)
    plate.crust.create(1, CrustType["ARC"], -50.0, 1)
    return mesh, world, plate


def test_boundary_far_field_feeding_handles_quiet_reset_empty_front_and_steepening(
    monkeypatch,
):
    mesh, world, plate = _boundary_world()
    world.time = 100.0
    plate.lastFedAt = 0.0
    plate.flatSlab = True
    plate.flatSlabKm = 500.0
    monkeypatch.setattr(boundaries, "near_land", lambda *_args: False)
    recorded = []
    monkeypatch.setattr(
        boundaries, "feed_far_field", lambda *args: recorded.append(args)
    )

    boundaries._feed_subduction_far_field(world, {}, 1.0)
    assert plate.flatSlab is True

    front = {"cells": [0], "flat": 0.0, "km": 50.0, "kms": [50.0], "headOn": [1.0]}
    boundaries._feed_subduction_far_field(world, {plate: front}, 1.0)
    assert plate.flatSlab is False
    assert plate.flatSlabKm == 0.0

    plate.flatSlab = True
    plate.flatSlabKm = 123.0
    plate.lastFedAt = world.time
    boundaries._feed_subduction_far_field(world, {plate: front}, 1.0)
    assert plate.flatSlab is False
    assert world.events[-1]["kind"] == EventKind["SLAB_STEEP"]
    assert recorded

    plate.flatSlab = True
    plate.flatSlabKm = 123.0
    plate.lastFedAt = world.time
    ocean_only = {"cells": [2], "flat": 0.0, "km": 10.0, "kms": [10.0], "headOn": [1.0]}
    plate.crust.create(2, CrustType["OCEAN"], -100.0, 1)
    boundaries._feed_subduction_far_field(world, {plate: ocean_only}, 1.0)
    assert plate.flatSlabKm == 0.0


def test_boundary_belt_spreaders_cover_filtered_and_arc_paths(monkeypatch):
    mesh, world, plate = _boundary_world(102)
    calls = []
    monkeypatch.setattr(boundaries, "near_land", lambda *_args: False)
    monkeypatch.setattr(boundaries, "spread_belt", lambda *args: calls.append(args))
    empty_source = {
        "cells": [2],
        "kms": [20.0],
        "headOn": [1.0],
        "ages": [100.0],
        "shears": [0.0],
    }
    plate.crust.create(2, CrustType["OCEAN"], -100.0, 1)
    boundaries._spread_subduction_belts(world, {plate: empty_source}, 1.0)
    assert not calls

    arc_front = {
        "cells": [1],
        "kms": [20.0],
        "headOn": [1.0],
        "ages": [100.0],
        "shears": [5.0],
    }
    boundaries._spread_subduction_belts(world, {plate: arc_front}, 1.0)
    assert calls[-1][2] == [1]
    assert calls[-1][3]["type"] != 0

    boundaries._spread_transpression_belts(
        world, {plate: {"cells": [], "kms": [], "shears": []}}, 1.0
    )
    boundaries._spread_transpression_belts(
        world, {plate: {"cells": [0], "kms": [25.0], "shears": [10.0]}}, 1.0
    )
    assert calls[-1][2] == [0]


def test_config_validation_directly_covers_optional_and_numeric_guards(monkeypatch):
    assert config.validate_tectonics_config({}) is None
    monkeypatch.setattr(config, "_CONFIG_SCHEMA", {})
    with pytest.raises(TypeError, match="real number"):
        config.validate_tectonics_config({"maxStepMyr": True})
    with pytest.raises(ValueError, match="finite"):
        config.validate_tectonics_config({"maxStepMyr": float("nan")})
    with pytest.raises(TypeError, match="must be a mapping"):
        config.validate_tectonics_config({"belts": []})
    assert config.validate_tectonics_config({"belts": None}) is None



def test_kinematics_skips_dead_plates_and_advances_world_clock():
    from types import SimpleNamespace

    from orogen.v2.tectonics.kinematics import advance_plates

    class FakePlate:
        def __init__(self, alive):
            self.alive = alive
            self.quat = [1.0, 0.0, 0.0, 0.0]
            self.omega = [0.0, 0.0, 0.0]
            self.calls = 0

        def set_quat(self, value):
            self.calls += 1
            self.quat = value

    active, retired = FakePlate(True), FakePlate(False)
    world = SimpleNamespace(plates=[active, retired], time=2.0, step=4)
    advance_plates(world, 0.5)
    assert active.calls == 1
    assert retired.calls == 0
    assert world.time == 2.5
    assert world.step == 5


def test_spread_belt_front_update_guards_and_preserves_shear():
    from orogen.v2.tectonics.spread_belt import _update_belt_front

    crust = CrustRaster(1)
    crust.create(0, CrustType["CONTINENT"], -10.0, 1)
    crust.beltReach[0] = 20.0
    crust.beltAt[0] = 5.0
    crust.beltU[0] = 0.4
    crust.beltShear[0] = 17

    belt = {"keepShear": True, "shear": -40.0}
    _update_belt_front(crust, 0, belt, 0, 2, 1, 0.2, 50.0, 5.0)
    assert crust.beltReach[0] == 20.0  # mismatched orogeny type is ignored

    crust.oroType[0] = 2
    _update_belt_front(crust, 0, belt, 0, 2, 1, 0.2, 50.0, 5.0)
    assert crust.beltReach[0] == 50.0
    assert crust.beltShear[0] == 17  # keepShear leaves the inherited value intact

    belt["keepShear"] = False
    _update_belt_front(crust, 0, belt, 0, 2, 1, 0.1, 60.0, 5.0)
    assert crust.beltShear[0] == -40


def test_backarc_band_skips_uncrusted_cells_and_ignores_empty_segments():
    mesh = _line_graph()
    world = SimpleNamespace(mesh=mesh, ownerCell=np.arange(mesh.num_cells))
    crust = CrustRaster(5)
    crust.create(0, CrustType["CONTINENT"], -10.0, 1)
    crust.create(2, CrustType["CONTINENT"], -10.0, 3)
    plate = SimpleNamespace(crust=crust)
    band = backarc._band_behind(world, plate, [0], 500.0)
    assert band.tolist() == [1, 0, 0, 0, 0]
    assert not np.any(backarc._band_behind(world, plate, [], 500.0))


def test_merge_hole_repair_stops_after_a_pass_without_fillable_holes():
    from orogen.v2.tectonics.ops.merge_plates import _fill_inverse_warp_holes

    crust = _crust([CrustType["CONTINENT"]] * 5)
    origin = np.ones(5, dtype=np.int8)
    mesh = _line_graph()
    _fill_inverse_warp_holes(
        crust, origin, mesh.adj_offset, mesh.adj_list, mesh.num_cells
    )
    assert crust.count == 5
    assert origin.tolist() == [1, 1, 1, 1, 1]
