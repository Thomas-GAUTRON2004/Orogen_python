from importlib import import_module

import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh


_SCENARIOS = (
    pytest.param(
        "orogen.v2.tectonics.event_scenes", "build_event_scene", "wings",
        id="event-wings",
    ),
    pytest.param(
        "orogen.v2.tectonics.budget_scenes", "build_budget_scene", "erosion",
        id="budget-erosion",
    ),
    pytest.param(
        "orogen.v2.tectonics.force_scenes", "build_force_scene", "one-sided",
        id="force-one-sided",
    ),
    pytest.param(
        "orogen.v2.tectonics.margin_scenes", "build_margin_scene", "margins",
        id="margin-margins",
    ),
    pytest.param(
        "orogen.v2.tectonics.rift_scenes", "build_rift_scene", "plume",
        id="rift-plume",
    ),
    pytest.param(
        "orogen.v2.tectonics.terrain_scenes", "build_terrain_scene", "andes",
        id="terrain-andes",
    ),
    pytest.param(
        "orogen.v2.tectonics.hotspot_scenes", "build_hotspot_scene", "hawaii",
        id="hotspot-hawaii",
    ),
    pytest.param(
        "orogen.v2.tectonics.collision_scenes", "build_collision_scene", "zipper",
        id="collision-zipper",
    ),
)


@pytest.mark.parametrize(
    ("module_path", "builder_name", "scenario"), _SCENARIOS
)
def test_scenario_family_builders_reproduce_the_same_seeded_world(
    module_path, builder_name, scenario
):
    builder = getattr(import_module(module_path), builder_name)
    mesh = build_sim_mesh(512, 7701)
    first = builder(mesh, 7702, scenario)
    second = builder(mesh, 7702, scenario)

    assert first.config == second.config
    assert first.time == second.time
    assert first.events == second.events
    assert first.plumes == second.plumes
    assert [plate.name for plate in first.plates] == [
        plate.name for plate in second.plates
    ]
    for left, right in zip(first.plates, second.plates, strict=True):
        assert left.alive is right.alive
        assert left.isBackground is right.isBackground
        assert left.born == right.born
        np.testing.assert_array_equal(left.quat, right.quat)
        np.testing.assert_array_equal(left.omega, right.omega)
        for left_field, right_field in zip(
            left.crust.fields, right.crust.fields, strict=True
        ):
            np.testing.assert_array_equal(left_field, right_field)
