import math

import numpy as np
import pytest

from orogen.v2.core.mesh_bfs import hop_distance
from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import CrustFlag, CrustType
from orogen.v2.tectonics.margin_scenes import (
    ARC_TERRANE_ID,
    MARIANA_ARC_WIDTH_KM,
    MARGIN_SCENES,
    _arc_behind,
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
    plate = mariana.plates[1]
    assert plate.arcTerraneId == ARC_TERRANE_ID
    trench = (plate.crust.flags & CrustFlag["TRENCH"]) != 0
    arc = (plate.crust.type == CrustType["ARC"]) & (
        plate.crust.terrane == ARC_TERRANE_ID
    )
    hops = max(1, math.floor(MARIANA_ARC_WIDTH_KM / mesh.edge_km + 0.5))
    distance = hop_distance(
        mesh,
        lambda cell: bool(trench[cell]),
        max_hops=hops,
        can_enter=lambda cell: bool(plate.crust.mask[cell]),
    )
    assert np.count_nonzero(arc) > 0
    assert np.array_equal(arc, distance == hops)


def test_arc_behind_paints_only_the_exact_requested_ring():
    from types import SimpleNamespace

    mesh = SimpleNamespace(
        adj_offset=np.asarray([0, 1, 3, 5, 6], dtype=np.int32),
        adj_list=np.asarray([1, 0, 2, 1, 3, 2], dtype=np.int32),
    )
    plate = SimpleNamespace(
        crust=SimpleNamespace(
            mask=np.ones(4, dtype=bool),
            type=np.zeros(4, dtype=np.uint8),
            terrane=np.zeros(4, dtype=np.uint16),
        ),
        arcTerraneId=-1,
    )

    _arc_behind(mesh, plate, [0], 2)

    assert plate.crust.type.tolist() == [0, 0, CrustType["ARC"], 0]
    assert plate.crust.terrane.tolist() == [0, 0, ARC_TERRANE_ID, 0]
    assert plate.arcTerraneId == ARC_TERRANE_ID
