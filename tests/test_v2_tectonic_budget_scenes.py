import numpy as np
import pytest
from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import mechanisms_on, only_mechanisms
from orogen.v2.tectonics.budget_scenes import BUDGET_SCENES, build_budget_scene
from orogen.v2.tectonics.config import TECTONICS


def _coverage(world):
    return np.stack([plate.crust.mask.astype(bool) for plate in world.plates]).sum(
        axis=0
    )


@pytest.mark.parametrize("scene_id", ["erosion", "india", "arcs"])
def test_budget_scene_builders_create_complete_partitions(scene_id):
    mesh = build_sim_mesh(512, 1801)
    world = build_budget_scene(mesh, 1802, scene_id)
    assert np.all(_coverage(world) == 1)
    assert all(plate.crust.count > 0 for plate in world.plates)
    scene = next(item for item in BUDGET_SCENES if item["id"] == scene_id)
    assert callable(scene["drive"])


def test_budget_scene_history_and_mechanism_overrides():
    mesh = build_sim_mesh(128, 1803)
    history = build_budget_scene(mesh, 1804, "history")
    assert np.all(_coverage(history) == 1)
    assert {scene["id"] for scene in BUDGET_SCENES} == {
        "erosion",
        "india",
        "arcs",
        "history",
    }

    enabled = only_mechanisms("underthrust")
    restored = mechanisms_on("underthrust")
    assert enabled["riftRatePerMyr"] == 0
    assert "underthrustShare" not in enabled
    assert restored == {"underthrustShare": TECTONICS["underthrustShare"]}


def test_budget_scene_dispatcher_normalizes_names_and_rejects_unknown_scene():
    mesh = build_sim_mesh(48, 1805)
    world = build_budget_scene(mesh, 1806, "  INDIA ")
    assert world.plates[0].name == "Asia"
    with pytest.raises(ValueError, match="unknown budget scenario"):
        build_budget_scene(mesh, 1806, "unknown")
