import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import CrustFlag
from orogen.v2.tectonics.margin_scenes import (
    MARGIN_SCENES,
    build_margin_scene,
    heading,
    omega_for,
)


def _coverage(world):
    return np.stack([plate.crust.mask.astype(bool) for plate in world.plates]).sum(
        axis=0
    )


@pytest.mark.parametrize(
    "scene_id",
    ["margins", "laramide", "mariana", "japan"],
)
def test_margin_scene_builders_create_complete_plate_partitions(scene_id):
    mesh = build_sim_mesh(512, 1701)
    world = build_margin_scene(mesh, 1702, scene_id)
    assert np.all(_coverage(world) == 1)
    assert all(plate.crust.count > 0 for plate in world.plates)
    scene = next(item for item in MARGIN_SCENES if item["id"] == scene_id)
    assert callable(scene["drive"])
    scene["drive"](world)


def test_margin_scene_history_dispatcher_and_motion_helpers():
    mesh = build_sim_mesh(128, 1703)
    world = build_margin_scene(mesh, 1704, "history")
    assert np.all(_coverage(world) == 1)

    point = np.asarray([1.0, 0.0, 0.0])
    direction = heading(point, 90.0)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    omega = omega_for(point, direction, 5.0)
    assert np.isfinite(omega).all()
    with pytest.raises(ValueError, match="unknown margin scenario"):
        build_margin_scene(mesh, 1704, "unknown")


def test_margin_scenes_expose_trench_and_arc_metadata():
    mesh = build_sim_mesh(512, 1705)
    margins = build_margin_scene(mesh, 1706, "margins")
    assert np.count_nonzero(margins.plates[0].crust.flags & CrustFlag["TRENCH"]) > 0
    mariana = build_margin_scene(mesh, 1706, "mariana")
    assert mariana.plates[1].arcTerraneId == 900
