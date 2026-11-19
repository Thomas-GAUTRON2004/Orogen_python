import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import (
    CrustFlag,
    CrustType,
    build_continental_scene,
    build_mixed_scene,
    build_oceanic_scene,
    build_scenario_scene,
    build_start_world,
)


def _coverage(world):
    masks = np.stack([plate.crust.mask.astype(bool) for plate in world.plates])
    return masks.sum(axis=0)


def test_start_world_is_a_complete_partition_with_initial_geology():
    mesh = build_sim_mesh(320, 712)
    result = build_start_world(mesh, 941)
    world, start = result["world"], result["start"]
    continent, ocean = world.plates

    assert np.all(_coverage(world) == 1)
    assert ocean.isBackground
    assert ocean.crust.count > 0
    assert continent.crust.count > 0
    assert np.count_nonzero(continent.crust.type == CrustType["CRATON"]) > 0
    assert np.count_nonzero(continent.crust.type == CrustType["CONTINENT"]) > 0
    assert np.all(ocean.crust.type[ocean.crust.mask.astype(bool)] == CrustType["OCEAN"])
    assert np.all(continent.crust.formedAt[continent.crust.mask.astype(bool)] <= 0)
    apron = (continent.crust.flags & CrustFlag["APRON"]) != 0
    trench = (continent.crust.flags & CrustFlag["TRENCH"]) != 0
    assert np.count_nonzero(apron) == start["apronCells"]
    assert np.all(continent.crust.type[apron] == CrustType["OCEAN"])
    assert np.all(apron[trench])
    assert np.all(start["seams"] <= 3)
    assert start["landCells"] == round(
        mesh.num_cells * world.config["continentalFraction"]
    )
    assert len(start["cratons"]) >= 1
    assert world.events[-1]["kind"] == "start"
    assert world.startSeamKm["major"].shape == (mesh.num_cells,)
    assert world.startSeamKm["ring"].shape == (mesh.num_cells,)
    land_mask = continent.crust.mask.astype(bool) & (
        continent.crust.type != CrustType["OCEAN"]
    )
    major_distance = world.startSeamKm["major"]
    ring_distance = world.startSeamKm["ring"]
    assert np.all(np.isinf(major_distance[~land_mask]))
    assert np.all(np.isinf(ring_distance[~land_mask]))
    assert np.all(np.isfinite(major_distance[major_distance < np.inf]))
    assert np.all(world.startSeamKm["majorRange"][major_distance < np.inf] == 1.0)
    assert np.all(world.startSeamKm["majorRange"][major_distance == np.inf] == 0.0)
    # Pre-history seams must leave actual belt and far-field records, not only
    # labels in the initial-world metadata.
    assert np.count_nonzero(continent.crust.oroType) > 0
    assert np.count_nonzero(continent.crust.farWork > 0) > 0
    assert np.all(continent.crust.farAt[continent.crust.farWork > 0] <= 0)


def test_start_world_is_deterministic_for_seed_and_configuration():
    mesh = build_sim_mesh(192, 81)
    left = build_start_world(mesh, 5501)
    right = build_start_world(mesh, 5501)

    assert left["start"]["centre"] == right["start"]["centre"]
    assert left["start"]["cratons"] == right["start"]["cratons"]
    for a, b in zip(left["world"].plates, right["world"].plates, strict=True):
        assert np.array_equal(a.crust.mask, b.crust.mask)
        assert np.array_equal(a.crust.type, b.crust.type)
        assert np.array_equal(a.crust.flags, b.crust.flags)
        assert np.array_equal(a.crust.terrane, b.crust.terrane)


def test_start_world_rejects_invalid_fraction():
    mesh = build_sim_mesh(24, 18)
    with pytest.raises(ValueError):
        build_start_world(mesh, 1, {"continentalFraction": 1.0})


def test_continental_oceanic_and_mixed_scenario_contracts():
    mesh = build_sim_mesh(160, 42)
    continental = build_continental_scene(mesh, 7)
    oceanic = build_oceanic_scene(mesh, 8)
    mixed = build_mixed_scene(mesh, 9)
    assert np.all(_coverage(continental["world"]) == 1)
    assert np.all(_coverage(oceanic["world"]) == 1)
    assert np.all(_coverage(mixed["world"]) == 1)
    assert continental["start"]["scenario"] == "continental"
    assert oceanic["start"]["scenario"] == "oceanic"
    assert mixed["start"]["scenario"] == "mixed"
    for plate in oceanic["world"].plates:
        assert np.all(
            plate.crust.type[plate.crust.mask.astype(bool)] == CrustType["OCEAN"]
        )


def test_scenario_dispatcher_and_invalid_name():
    mesh = build_sim_mesh(96, 23)
    assert (
        build_scenario_scene(mesh, 3, "continental")["start"]["scenario"]
        == "continental"
    )
    assert build_scenario_scene(mesh, 3, "oceanic")["start"]["scenario"] == "oceanic"
    assert build_scenario_scene(mesh, 3, "mixed")["start"]["scenario"] == "mixed"
    with pytest.raises(ValueError, match="scenario must be one of"):
        build_scenario_scene(mesh, 3, "unknown")


def test_scenario_builders_apply_configuration_overrides():
    mesh = build_sim_mesh(160, 42)
    continental = build_continental_scene(
        mesh, 17, fraction=0.25, overrides={"initialLandAgeMyr": 1234.0}
    )
    oceanic = build_oceanic_scene(mesh, 18, overrides={"initialOceanAgeMyr": 87.0})
    dispatched = build_scenario_scene(
        mesh,
        19,
        scenario="oceanic",
        overrides={"initialOceanAgeMyr": 63.0},
    )

    continent = continental["world"].plates[0]
    land = continent.crust.mask.astype(bool) & (
        continent.crust.type != CrustType["OCEAN"]
    )
    assert np.all(continent.crust.formedAt[land] == -1234.0)
    assert continental["start"]["landCells"] == round(mesh.num_cells * 0.25)

    for result, expected_age in ((oceanic, -87.0), (dispatched, -63.0)):
        for plate in result["world"].plates:
            mask = plate.crust.mask.astype(bool)
            assert np.all(plate.crust.formedAt[mask] == expected_age)
