import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import CrustFlag, CrustType
from orogen.v2.tectonics.event_scenes import EVENT_SCENES, build_event_scene


def _coverage(world):
    return np.stack([plate.crust.mask.astype(bool) for plate in world.plates]).sum(
        axis=0
    )


@pytest.mark.parametrize("scene_id", [scene["id"] for scene in EVENT_SCENES])
def test_event_scene_catalogue_builders_make_complete_partitions(scene_id):
    mesh = build_sim_mesh(512, 1501)
    world = build_event_scene(mesh, 1502, scene_id)
    assert np.all(_coverage(world) == 1)
    assert all(plate.crust.count > 0 for plate in world.plates)


def test_event_scene_catalogue_has_expected_ids_and_arc_material():
    assert {scene["id"] for scene in EVENT_SCENES} == {
        "wings",
        "terrane",
        "arc",
        "arc-collision",
    }
    assert all(callable(scene["build"]) for scene in EVENT_SCENES)
    mesh = build_sim_mesh(512, 1503)
    oceanic_arc = build_event_scene(mesh, 1504, "arc")
    assert oceanic_arc.plates[1].name == "Oceanica"
    assert np.count_nonzero(oceanic_arc.plates[1].crust.type == CrustType["ARC"]) > 0

    collision = build_event_scene(mesh, 1504, "arc-collision")
    assert all(
        np.count_nonzero(plate.crust.flags & CrustFlag["TRENCH"]) > 0
        for plate in collision.plates[:2]
    )
    assert all(
        np.count_nonzero(plate.crust.type == CrustType["ARC"]) > 0
        for plate in collision.plates[:2]
    )


def test_event_scene_dispatcher_normalizes_names_and_rejects_unknown_scene():
    mesh = build_sim_mesh(48, 1505)
    world = build_event_scene(mesh, 1506, "  WINGS ")
    assert world.plates[0].name == "Solo"
    with pytest.raises(ValueError, match="unknown event scenario"):
        build_event_scene(mesh, 1506, "unknown")
