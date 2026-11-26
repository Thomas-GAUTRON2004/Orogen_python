import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import CrustFlag
from orogen.v2.tectonics.collision_scenes import (
    COLLISION_SCENES,
    build_collision_scene,
)


def _coverage(world):
    return np.stack([plate.crust.mask.astype(bool) for plate in world.plates]).sum(
        axis=0
    )


def test_collision_scene_catalogue_builds_the_zipper_partition():
    mesh = build_sim_mesh(512, 1601)
    world = build_collision_scene(mesh, 1602)
    assert [plate.name for plate in world.plates[:2]] == ["Borea", "Austra"]
    assert np.all(_coverage(world) == 1)
    assert all(plate.crust.count > 0 for plate in world.plates)
    assert np.count_nonzero(world.plates[0].crust.flags & CrustFlag["TRENCH"]) > 0
    assert COLLISION_SCENES[0]["id"] == "zipper"
    assert callable(COLLISION_SCENES[0]["build"])


def test_collision_scene_dispatcher_normalizes_names_and_rejects_unknown_scene():
    mesh = build_sim_mesh(48, 1603)
    world = build_collision_scene(mesh, 1604, " ZIPPER ")
    assert world.plates[0].name == "Borea"
    with pytest.raises(ValueError, match="unknown collision scenario"):
        build_collision_scene(mesh, 1604, "unknown")
