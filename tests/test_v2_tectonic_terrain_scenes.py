import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import CrustFlag, CrustType
from orogen.v2.tectonics.terrain_scenes import (
    FAR_TERRANES,
    TERRAIN_SCENES,
    build_terrain_scene,
    flagged,
)


def _coverage(world):
    return np.stack([plate.crust.mask.astype(bool) for plate in world.plates]).sum(
        axis=0
    )


@pytest.mark.parametrize(
    "scene_id",
    [
        "andes",
        "laramide",
        "india",
        "mariana",
        "japan",
        "hawaii",
        "traps",
        "rift",
        "cratons",
    ],
)
def test_terrain_scenes_build_complete_land_and_ocean_partitions(scene_id):
    mesh = build_sim_mesh(512, 2101)
    world = build_terrain_scene(mesh, 2102, scene_id)
    assert np.all(_coverage(world) == 1)
    assert all(plate.crust.count > 0 for plate in world.plates)


def test_farfield_scene_preserves_named_terrane_features_and_partition():
    mesh = build_sim_mesh(4096, 2103)
    world = build_terrain_scene(mesh, 2104, "farfield")
    assert np.all(_coverage(world) == 1)
    assert set(FAR_TERRANES.values()) == {901, 902, 903, 904, 905}
    asia = world.plates[0]
    assert np.count_nonzero(asia.crust.terrane == FAR_TERRANES["tarim"]) > 0
    assert np.count_nonzero(asia.crust.terrane == FAR_TERRANES["siberia"]) > 0


def test_terrain_history_and_cycle_apply_whole_planet_pacing():
    mesh = build_sim_mesh(128, 2105)
    history = build_terrain_scene(mesh, 2106, "history")
    cycle = build_terrain_scene(mesh, 2106, "cycle")
    assert np.all(_coverage(history) == 1)
    assert np.all(_coverage(cycle) == 1)
    assert history.config["maxCellsPerStep"] == 2.25
    assert cycle.config["slowRulesEveryMyr"] == 3


def test_flagged_focus_returns_centroid_or_none_and_scene_dispatch_rejects_unknown():
    mesh = build_sim_mesh(48, 2107)
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape((-1, 3))
    frame = {
        "owner": np.asarray([0] + [-1] * (mesh.num_cells - 1)),
        "type": np.asarray([CrustType["CONTINENT"]] + [CrustType["OCEAN"]] * (mesh.num_cells - 1)),
        "flags": np.asarray([CrustFlag["LIP"]] + [0] * (mesh.num_cells - 1)),
    }
    focus = flagged(CrustFlag["LIP"])
    result = focus(frame, mesh)
    assert result is not None
    assert np.isfinite(result).all()
    frame["flags"][:] = 0
    assert focus(frame, mesh) is None

    assert {scene["id"] for scene in TERRAIN_SCENES} == {
        "andes",
        "laramide",
        "india",
        "mariana",
        "japan",
        "hawaii",
        "traps",
        "rift",
        "cratons",
        "farfield",
        "history",
        "cycle",
    }
    with pytest.raises(ValueError, match="unknown terrain scenario"):
        build_terrain_scene(mesh, 2108, "unknown")
