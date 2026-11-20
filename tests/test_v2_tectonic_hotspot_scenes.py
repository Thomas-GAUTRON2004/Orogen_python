import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.hotspot_scenes import (
    DECCAN,
    HAWAII,
    HOTSPOT_SCENES,
    build_hotspot_scene,
)


def _coverage(world):
    return np.stack([plate.crust.mask.astype(bool) for plate in world.plates]).sum(
        axis=0
    )


@pytest.mark.parametrize("scene_id", ["hawaii", "deccan"])
def test_hotspot_kinematic_scenes_build_complete_worlds_and_prescribed_motion(scene_id):
    mesh = build_sim_mesh(512, 1901)
    world = build_hotspot_scene(mesh, 1902, scene_id)
    assert np.all(_coverage(world) == 1)
    scene = next(item for item in HOTSPOT_SCENES if item["id"] == scene_id)
    assert callable(scene["drive"])
    scene["drive"](world)
    assert np.linalg.norm(world.plates[0].omega) > 0.0
    assert len(world.plumes) == 1
    assert world.plumes[0]["head"] is (scene_id == "deccan")


def test_hotspot_history_dispatch_and_reference_parameters():
    mesh = build_sim_mesh(128, 1903)
    world = build_hotspot_scene(mesh, 1904, "history")
    assert np.all(_coverage(world) == 1)
    assert HAWAII["bendMyr"] == 45.0
    assert DECCAN["arrivesMyr"] == 12.0
    assert {scene["id"] for scene in HOTSPOT_SCENES} == {
        "hawaii",
        "deccan",
        "history",
    }
    with pytest.raises(ValueError, match="unknown hotspot scenario"):
        build_hotspot_scene(mesh, 1904, "unknown")
