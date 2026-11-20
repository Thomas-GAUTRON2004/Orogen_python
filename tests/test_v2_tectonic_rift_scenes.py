import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.rift_scenes import RIFT_SCENES, build_rift_scene


def _coverage(world):
    return np.stack([plate.crust.mask.astype(bool) for plate in world.plates]).sum(
        axis=0
    )


def test_plume_rift_scene_has_scheduled_plume_and_complete_partition():
    mesh = build_sim_mesh(512, 2001)
    world = build_rift_scene(mesh, 2002, "plume")
    assert [plate.name for plate in world.plates[:1]] == ["Pangaea"]
    assert np.all(_coverage(world) == 1)
    assert len(world.plumes) == 1
    assert world.plumes[0]["time"] == 20.0
    assert world.plumes[0]["plate"] == world.plates[0].id


def test_founder_and_breakup_rift_scenes_are_complete_partitions():
    mesh = build_sim_mesh(512, 2003)
    founder = build_rift_scene(mesh, 2004, "founder")
    assert [plate.name for plate in founder.plates[:2]] == ["Westland", "Eastland"]
    assert np.all(_coverage(founder) == 1)
    assert math_is_inf(founder.config["oceanPlateMinAreaKm2"])

    breakup = build_rift_scene(build_sim_mesh(128, 2005), 2006, "breakup")
    assert np.all(_coverage(breakup) == 1)
    assert {scene["id"] for scene in RIFT_SCENES} == {
        "plume",
        "breakup",
        "founder",
    }


def math_is_inf(value):
    return bool(np.isinf(value))


def test_rift_scene_dispatcher_normalizes_names_and_rejects_unknown_scene():
    mesh = build_sim_mesh(48, 2007)
    world = build_rift_scene(mesh, 2008, "  PLUME ")
    assert world.plates[0].name == "Pangaea"
    with pytest.raises(ValueError, match="unknown rift scenario"):
        build_rift_scene(mesh, 2008, "unknown")
