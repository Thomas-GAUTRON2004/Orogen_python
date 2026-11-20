from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import initial_world, scenarios
from orogen.v2.tectonics.crust import CrustFlag, CrustType


def _coverage(world):
    masks = np.stack([plate.crust.mask.astype(bool) for plate in world.plates])
    return masks.sum(axis=0)


def test_mesh_xyz_rejects_inconsistent_cell_count():
    mesh = SimpleNamespace(xyz=np.zeros((4, 3)), num_cells=3)
    with pytest.raises(ValueError, match="does not match num_cells"):
        initial_world._mesh_xyz(mesh)
    with pytest.raises(ValueError, match="does not match num_cells"):
        scenarios._xyz(mesh)


def test_distance_hops_empty_sources_and_invalid_shape():
    mesh = build_sim_mesh(48, 901)
    empty = initial_world._distance_hops(mesh, np.zeros(mesh.num_cells, dtype=bool))
    assert np.all(empty == -1)
    with pytest.raises(ValueError, match="one value per mesh cell"):
        initial_world._distance_hops(mesh, np.zeros(mesh.num_cells - 1, dtype=bool))


def test_land_cells_clamps_requested_count():
    mesh = build_sim_mesh(48, 902)
    centre = np.asarray(mesh.xyz[:3], dtype=np.float64)
    cells, radius = initial_world._land_cells(mesh, centre, mesh.num_cells + 10)
    assert cells.size == mesh.num_cells - 1
    assert np.isfinite(radius)
    one, _ = initial_world._land_cells(mesh, centre, 0)
    assert one.size == 1


def test_pick_craton_centres_falls_back_to_full_land_candidate_set():
    mesh = build_sim_mesh(48, 903)
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape((-1, 3))
    land = np.array([0, 1], dtype=np.int32)
    centre = xyz[0]
    picked = initial_world._pick_craton_centres(
        xyz, land, centre, 0.0, 2, 1.0e9, float(mesh.edge_km), lambda: 0.5
    )
    assert picked.shape == (2,)
    assert set(picked.tolist()).issubset(set(land.tolist()))


def test_assign_cratons_clamps_share_and_always_selects_centres():
    mesh = build_sim_mesh(48, 904)
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape((-1, 3))
    land = np.array([0, 1, 2, 3], dtype=np.int32)
    centres = np.array([0, 3], dtype=np.int32)
    owners, selected, local = initial_world._assign_cratons(xyz, land, centres, 0.0)
    assert owners.shape == selected.shape == local.shape == (4,)
    assert selected[0] and selected[3]
    _, all_selected, _ = initial_world._assign_cratons(xyz, land, centres, 2.0)
    assert np.all(all_selected)


def test_distance_km_handles_empty_sources_limits_and_entry_filter():
    mesh = build_sim_mesh(48, 905)
    empty, strength = initial_world._distance_km(
        mesh, np.zeros(mesh.num_cells, dtype=bool)
    )
    assert np.all(np.isinf(empty))
    assert not np.any(strength)
    sources = np.zeros(mesh.num_cells, dtype=bool)
    sources[0] = True
    blocked, _ = initial_world._distance_km(mesh, sources, max_km=0.0)
    assert blocked[0] == 0.0
    assert np.count_nonzero(np.isfinite(blocked)) == 1
    allowed = {0, int(mesh.adj_list[int(mesh.adj_offset[0])])}
    restricted, restricted_strength = initial_world._distance_km(
        mesh, sources, can_enter=lambda cell: cell in allowed
    )
    assert np.isfinite(restricted[0])
    assert np.count_nonzero(np.isfinite(restricted)) <= 2
    assert restricted_strength[0] == 1.0


def test_paint_old_ranges_ignores_invalid_and_incomplete_settings():
    mesh = build_sim_mesh(48, 906)
    result = initial_world.build_start_world(mesh, 19)
    world = result["world"]
    continent = world.plates[0]
    land = continent.crust.mask.astype(bool) & (continent.crust.type != 0)
    before = continent.crust.rootKm.copy()
    world.config["startRanges"] = None
    initial_world._paint_old_ranges(world, continent, land, result["start"]["seams"])
    assert np.array_equal(continent.crust.rootKm, before)
    world.config["startRanges"] = {
        "inner": [100.0, 50.0],
        "ring": [0.0, 10.0, 20.0, 0.5],
    }
    initial_world._paint_old_ranges(world, continent, land, result["start"]["seams"])
    assert np.array_equal(continent.crust.rootKm, before)


def test_start_world_without_trench_coverage_and_without_old_ranges():
    mesh = build_sim_mesh(96, 907)
    result = initial_world.build_start_world(
        mesh,
        20,
        {
            "rimTrenchCoverage": 0.0,
            "startRanges": {},
            "farStartWorkKm": 0.0,
        },
    )
    assert result["start"]["trench"].size == 0
    assert np.count_nonzero(result["world"].plates[0].crust.flags) >= 0


def test_start_world_rejects_mesh_with_too_few_cells(monkeypatch):
    mesh = SimpleNamespace(num_cells=3)
    monkeypatch.setattr(
        initial_world,
        "TectonicWorld",
        lambda *args, **kwargs: SimpleNamespace(config={}),
    )
    with pytest.raises(ValueError, match="at least four"):
        initial_world.build_start_world(mesh, 1)


def test_scenario_builders_reject_invalid_fraction_and_normalize_dispatch():
    mesh = build_sim_mesh(48, 909)
    for fraction in (0.0, 1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="fraction must be finite"):
            scenarios.build_continental_scene(mesh, fraction=fraction)
    assert (
        scenarios.build_scenario_scene(mesh, 4, "  CONTINENTAL ")["start"]["scenario"]
        == "continental"
    )


def test_oceanic_scene_falls_back_when_random_pole_has_one_hemisphere(monkeypatch):
    mesh = build_sim_mesh(48, 910)
    monkeypatch.setattr(scenarios, "random_point", lambda _rng: [0.0, 0.0, 1.0])
    monkeypatch.setattr(
        scenarios,
        "_xyz",
        lambda _mesh: np.tile([0.0, 0.0, 1.0], (mesh.num_cells, 1)),
    )
    result = scenarios.build_oceanic_scene(mesh, 5)
    assert (
        result["start"]["plateACells"] + result["start"]["plateBCells"]
        == mesh.num_cells
    )
    assert result["start"]["plateACells"] > 0
    assert result["start"]["plateBCells"] > 0


def test_scene_kit_builds_continent_apron_and_background_partition():
    from orogen.v2.core.fast_noise import FastNoise
    from orogen.v2.tectonics.scenarios import (
        add_apron_and_trench,
        add_continent,
        fill_background,
        scene_canvas,
    )
    from orogen.v2.tectonics.world import TectonicWorld

    mesh = build_sim_mesh(192, 911)
    world = TectonicWorld(mesh, 912)
    continent_plate = world.add_plate("Scene continent")
    canvas = scene_canvas(mesh, world.config)
    continent = add_continent(
        canvas,
        FastNoise(913),
        plate=continent_plate,
        centre=(12.0, 25.0),
        radius_km=8500.0,
        craton_km=2800.0,
    )
    assert np.count_nonzero(continent["isLand"]) > 0
    assert np.count_nonzero(continent["isLand"]) < mesh.num_cells
    add_apron_and_trench(canvas, continent_plate, continent, "all", world.config)
    ocean = fill_background(canvas, world, world.config)

    assert ocean.isBackground
    assert np.all(_coverage(world) == 1)
    assert np.count_nonzero(continent_plate.crust.type == CrustType["CRATON"]) > 0
    assert np.count_nonzero(continent_plate.crust.flags & CrustFlag["APRON"]) > 0
    assert np.count_nonzero(continent_plate.crust.flags & CrustFlag["TRENCH"]) > 0


def test_scene_kit_sectors_ocean_predicate_and_coast_trench():
    from orogen.v2.core.fast_noise import FastNoise
    from orogen.v2.tectonics.scenarios import (
        _in_sectors,
        add_continent,
        add_ocean,
        fill_background,
        flag_coast_trench,
        scene_canvas,
        scene_terrane,
    )
    from orogen.v2.tectonics.world import TectonicWorld

    assert _in_sectors(5.0, "all")
    assert _in_sectors(355.0, [[350.0, 20.0]])
    assert not _in_sectors(180.0, [[350.0, 20.0]])

    mesh = build_sim_mesh(96, 914)
    world = TectonicWorld(mesh, 915)
    plate = world.add_plate("Scene land")
    canvas = scene_canvas(mesh, world.config)
    assert scene_terrane(canvas) == 1
    continent = add_continent(
        canvas,
        FastNoise(916),
        plate=plate,
        centre=(0.0, 0.0),
        radius_km=9000.0,
    )
    flag_coast_trench(mesh, plate, continent["centre"], "all", world.config)
    add_ocean(canvas, plate, lambda x, y, z: z > 0.0, -150.0)
    ocean = fill_background(canvas, world, world.config)

    assert ocean.isBackground
    assert np.all(_coverage(world) == 1)
    assert np.all(canvas["taken"] == 1)
    assert np.count_nonzero(plate.crust.flags & CrustFlag["TRENCH"]) > 0


def test_scene_kit_rejects_invalid_continent_geometry_and_skips_taken_cells():
    from orogen.v2.core.fast_noise import FastNoise
    from orogen.v2.tectonics.scenarios import add_continent, scene_canvas
    from orogen.v2.tectonics.world import TectonicWorld

    mesh = build_sim_mesh(48, 920)
    world = TectonicWorld(mesh, 921)
    plate = world.add_plate("Validation")
    canvas = scene_canvas(mesh, world.config)

    with pytest.raises(ValueError, match="radius_km"):
        add_continent(canvas, FastNoise(922), plate=plate, centre=(0, 0), radius_km=0)
    with pytest.raises(ValueError, match="wobble"):
        add_continent(
            canvas,
            FastNoise(922),
            plate=plate,
            centre=(0, 0),
            radius_km=1000,
            wobble=-1,
        )
    with pytest.raises(ValueError, match="craton_km"):
        add_continent(
            canvas,
            FastNoise(922),
            plate=plate,
            centre=(0, 0),
            radius_km=1000,
            craton_km=1001,
        )

    canvas["taken"][:] = 1
    result = add_continent(
        canvas, FastNoise(922), plate=plate, centre=(0, 0), radius_km=1000
    )
    assert not np.any(result["isLand"])


def test_scene_kit_unselected_rim_and_coast_sectors(monkeypatch):
    from orogen.v2.tectonics.scenarios import (
        add_apron_and_trench,
        flag_coast_trench,
        scene_canvas,
    )
    from orogen.v2.tectonics.world import TectonicWorld

    mesh = build_sim_mesh(48, 923)
    world = TectonicWorld(mesh, 924)
    plate = world.add_plate("Sector fixture")
    canvas = scene_canvas(mesh, world.config)
    centre = [1.0, 0.0, 0.0]
    continent = {"centre": centre, "isLand": np.zeros(mesh.num_cells, dtype=np.uint8)}
    in_plate = np.zeros(mesh.num_cells, dtype=bool)
    in_plate[1] = True
    monkeypatch.setattr(
        scenarios,
        "_plate_outline_with_apron",
        lambda *_args: (in_plate, np.asarray([1], dtype=np.int32)),
    )
    add_apron_and_trench(canvas, plate, continent, [], world.config)
    assert plate.crust.mask[1]
    assert not (plate.crust.flags[1] & CrustFlag["TRENCH"])

    plate.crust.create(0, CrustType["CONTINENT"], -1000.0)
    flag_coast_trench(mesh, plate, centre, [], world.config)
    assert not np.any(plate.crust.flags & CrustFlag["TRENCH"])
