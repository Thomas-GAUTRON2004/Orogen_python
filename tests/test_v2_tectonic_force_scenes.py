import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import CrustFlag, CrustType
from orogen.v2.tectonics.force_scenes import (
    FORCE_SCENES,
    build_force_scene,
)


def _coverage(world):
    return np.stack([plate.crust.mask.astype(bool) for plate in world.plates]).sum(
        axis=0
    )


@pytest.mark.parametrize("scene_id", [scene["id"] for scene in FORCE_SCENES])
def test_force_scene_catalogue_builders_make_complete_deterministic_partitions(
    scene_id,
):
    mesh = build_sim_mesh(128, 1401)
    first = build_force_scene(mesh, 1402, scene_id)
    second = build_force_scene(mesh, 1402, scene_id)

    assert np.all(_coverage(first) == 1)
    assert [plate.name for plate in first.plates] == [
        plate.name for plate in second.plates
    ]
    for left, right in zip(first.plates, second.plates, strict=True):
        assert np.array_equal(left.crust.mask, right.crust.mask)
        assert np.array_equal(left.crust.type, right.crust.type)
        assert np.array_equal(left.crust.flags, right.crust.flags)


def test_force_scene_catalogue_contracts_and_trench_geometry():
    by_id = {scene["id"]: scene for scene in FORCE_SCENES}
    assert set(by_id) == {"one-sided", "tug", "wilson", "india", "start"}
    assert all(callable(scene["build"]) for scene in FORCE_SCENES)
    assert all(scene["checklist"] and scene["expect"] for scene in FORCE_SCENES)

    mesh = build_sim_mesh(192, 1403)
    one_sided = build_force_scene(mesh, 1404, "one-sided")
    assert one_sided.plates[0].name == "Solo"
    assert np.count_nonzero(one_sided.plates[0].crust.flags & CrustFlag["TRENCH"]) > 0

    tug = build_force_scene(mesh, 1404, "tug")
    assert [plate.name for plate in tug.plates[:2]] == ["Balanced", "Lopsided"]
    assert all(
        np.count_nonzero(plate.crust.flags & CrustFlag["TRENCH"]) > 0
        for plate in tug.plates[:2]
    )

    wilson = build_force_scene(mesh, 1404, "wilson")
    assert [plate.name for plate in wilson.plates[:3]] == ["West", "East", "Far"]
    assert np.count_nonzero(wilson.plates[0].crust.flags & CrustFlag["TRENCH"]) > 0
    assert np.count_nonzero(wilson.plates[1].crust.flags & CrustFlag["TRENCH"]) > 0

    india = build_force_scene(mesh, 1404, "india")
    assert [plate.name for plate in india.plates[:2]] == ["Asia", "India"]
    assert np.count_nonzero(india.plates[1].crust.type == CrustType["CONTINENT"]) > 0

    start = build_force_scene(mesh, 1404, "start")
    assert start.plates[0].crust.count > 0


def test_force_scene_dispatcher_normalizes_names_and_rejects_unknown_scene():
    mesh = build_sim_mesh(48, 1405)
    world = build_force_scene(mesh, 1406, "  ONE-SIDED ")
    assert world.plates[0].name == "Solo"
    with pytest.raises(ValueError, match="unknown force scenario"):
        build_force_scene(mesh, 1406, "unknown")
