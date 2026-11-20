import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics import (
    CrustFlag,
    CrustType,
    build_continental_scene,
    build_mixed_scene,
    build_oceanic_scene,
    build_scenario_scene,
    build_start_world,
)
from orogen.v2.tectonics.config import TECTONICS

_CONFIG_KEYS = (
    "cratonCount",
    "continentalFraction",
    "cratonShare",
    "cratonMinAreaScale",
    "cratonSizeRatio",
    "cratonSizeSkew",
    "cratonMaxDiameterKm",
    "cratonMinGapKm",
    "cratonMinCoastKm",
    "cratonEdgeNoiseKm",
    "cratonMaxAspect",
    "cratonLobes",
    "cratonRiftEdges",
    "cratonRiftDepth",
    "cratonRiftSteps",
    "cratonRiftStepKm",
    "cratonRiftNoise",
    "cratonMinCompact",
    "cratonSiblingChance",
    "clusterBlocks",
    "clusterElongation",
    "clusterApart",
    "clusterGapVariety",
    "clusterLooseness",
    "startMaxLatitudeDeg",
    "bridgeHalfWidthKm",
    "coastReliefKm",
    "coastUnionSoftnessKm",
    "terraneBorderWanderKm",
    "cratonGrainNoiseKm",
    "sutureHalfWidthKm",
    "startCoreKm",
    "startGrowthBias",
    "startRingKm",
    "startSeamWanderKm",
    "startCoastDetailKm",
    "startArcKm",
    "startArcRadiusKm",
    "startStepKm",
    "startStepOffsetKm",
    "rimTrenchCoverage",
    "spanAllowanceDeg",
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/reference/v2/initial-world-layout-runner.mjs"


def _run_initial_world_reference(node: str, mesh, seed: int, config: dict):
    payload = {
        "seed": seed,
        "config": {key: copy.deepcopy(config[key]) for key in _CONFIG_KEYS},
        "mesh": {
            "numCells": int(mesh.num_cells),
            "xyz": np.asarray(mesh.xyz, dtype=np.float32).reshape(-1).tolist(),
            "adjOffset": np.asarray(mesh.adj_offset, dtype=np.int32).tolist(),
            "adjList": np.asarray(mesh.adj_list, dtype=np.int32).tolist(),
        },
    }
    result = subprocess.run(
        [node, str(RUNNER)],
        input=json.dumps(payload, separators=(",", ":"), allow_nan=False),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"upstream initial-world JS runner failed:\\n{result.stderr}\\n{result.stdout}"
    )
    return json.loads(result.stdout)


def _coverage(world):
    masks = np.stack([plate.crust.mask.astype(bool) for plate in world.plates])
    return masks.sum(axis=0)


def test_start_world_is_a_complete_partition_with_initial_geology():
    mesh = build_sim_mesh(320, 712)
    result = build_start_world(mesh, 941)
    world, start = result["world"], result["start"]
    continent, ocean = world.plates

    assert np.all(_coverage(world) == 1)
    assert ocean.isBackground
    assert ocean.crust.count > 0
    assert continent.crust.count > 0
    assert np.count_nonzero(continent.crust.type == CrustType["CRATON"]) > 0
    assert np.count_nonzero(continent.crust.type == CrustType["CONTINENT"]) > 0
    assert np.all(ocean.crust.type[ocean.crust.mask.astype(bool)] == CrustType["OCEAN"])
    assert np.all(continent.crust.formedAt[continent.crust.mask.astype(bool)] <= 0)
    apron = (continent.crust.flags & CrustFlag["APRON"]) != 0
    trench = (continent.crust.flags & CrustFlag["TRENCH"]) != 0
    assert np.count_nonzero(apron) == start["apronCells"]
    assert np.all(continent.crust.type[apron] == CrustType["OCEAN"])
    assert np.all(apron[trench])
    assert np.all(start["seams"] <= 3)
    assert start["landCells"] == round(
        mesh.num_cells * world.config["continentalFraction"]
    )
    assert len(start["cratons"]) >= 1
    assert world.events[-1]["kind"] == "start"
    assert world.startSeamKm["major"].shape == (mesh.num_cells,)
    assert world.startSeamKm["ring"].shape == (mesh.num_cells,)
    land_mask = continent.crust.mask.astype(bool) & (
        continent.crust.type != CrustType["OCEAN"]
    )
    major_distance = world.startSeamKm["major"]
    ring_distance = world.startSeamKm["ring"]
    assert np.all(np.isinf(major_distance[~land_mask]))
    assert np.all(np.isinf(ring_distance[~land_mask]))
    assert np.all(np.isfinite(major_distance[major_distance < np.inf]))
    assert np.all(world.startSeamKm["majorRange"][major_distance < np.inf] == 1.0)
    assert np.all(world.startSeamKm["majorRange"][major_distance == np.inf] == 0.0)
    # Pre-history seams must leave actual belt and far-field records, not only
    # labels in the initial-world metadata.
    assert np.count_nonzero(continent.crust.oroType) > 0
    assert np.count_nonzero(continent.crust.farWork > 0) > 0
    assert np.all(continent.crust.farAt[continent.crust.farWork > 0] <= 0)


def test_start_world_is_deterministic_for_seed_and_configuration():
    mesh = build_sim_mesh(192, 81)
    left = build_start_world(mesh, 5501)
    right = build_start_world(mesh, 5501)

    assert left["start"]["centre"] == right["start"]["centre"]
    assert left["start"]["cratons"] == right["start"]["cratons"]
    for a, b in zip(left["world"].plates, right["world"].plates, strict=True):
        assert np.array_equal(a.crust.mask, b.crust.mask)
        assert np.array_equal(a.crust.type, b.crust.type)
        assert np.array_equal(a.crust.flags, b.crust.flags)
        assert np.array_equal(a.crust.terrane, b.crust.terrane)


def test_start_world_rejects_invalid_fraction():
    mesh = build_sim_mesh(24, 18)
    with pytest.raises(ValueError):
        build_start_world(mesh, 1, {"continentalFraction": 1.0})


def test_continental_oceanic_and_mixed_scenario_contracts():
    mesh = build_sim_mesh(160, 42)
    continental = build_continental_scene(mesh, 7)
    oceanic = build_oceanic_scene(mesh, 8)
    mixed = build_mixed_scene(mesh, 9)
    assert np.all(_coverage(continental["world"]) == 1)
    assert np.all(_coverage(oceanic["world"]) == 1)
    assert np.all(_coverage(mixed["world"]) == 1)
    assert continental["start"]["scenario"] == "continental"
    assert oceanic["start"]["scenario"] == "oceanic"
    assert mixed["start"]["scenario"] == "mixed"
    for plate in oceanic["world"].plates:
        assert np.all(
            plate.crust.type[plate.crust.mask.astype(bool)] == CrustType["OCEAN"]
        )


def test_scenario_dispatcher_and_invalid_name():
    mesh = build_sim_mesh(96, 23)
    assert (
        build_scenario_scene(mesh, 3, "continental")["start"]["scenario"]
        == "continental"
    )
    assert build_scenario_scene(mesh, 3, "oceanic")["start"]["scenario"] == "oceanic"
    assert build_scenario_scene(mesh, 3, "mixed")["start"]["scenario"] == "mixed"
    with pytest.raises(ValueError, match="scenario must be one of"):
        build_scenario_scene(mesh, 3, "unknown")


def test_scenario_builders_apply_configuration_overrides():
    mesh = build_sim_mesh(160, 42)
    continental = build_continental_scene(
        mesh, 17, fraction=0.25, overrides={"initialLandAgeMyr": 1234.0}
    )
    oceanic = build_oceanic_scene(mesh, 18, overrides={"initialOceanAgeMyr": 87.0})
    dispatched = build_scenario_scene(
        mesh,
        19,
        scenario="oceanic",
        overrides={"initialOceanAgeMyr": 63.0},
    )

    continent = continental["world"].plates[0]
    land = continent.crust.mask.astype(bool) & (
        continent.crust.type != CrustType["OCEAN"]
    )
    assert np.all(continent.crust.formedAt[land] == -1234.0)
    assert continental["start"]["landCells"] == round(mesh.num_cells * 0.25)

    for result, expected_age in ((oceanic, -87.0), (dispatched, -63.0)):
        for plate in result["world"].plates:
            mask = plate.crust.mask.astype(bool)
            assert np.all(plate.crust.formedAt[mask] == expected_age)


def test_production_resolution_start_world_uses_packed_landmass_path():
    mesh = build_sim_mesh(512, 911)
    seed = 912
    node = shutil.which("node")
    expected = None
    if node is not None:
        expected = _run_initial_world_reference(node, mesh, seed, TECTONICS)
    result = build_start_world(mesh, seed)
    world, start = result["world"], result["start"]

    assert start["attempts"] >= 1
    assert len(start["cratons"]) == world.config["cratonCount"]
    assert start["fractionMet"]
    assert (
        abs(start["continentalFraction"] - world.config["continentalFraction"]) <= 0.004
    )
    assert len(world.plates) == 2
    assert np.all(_coverage(world) == 1)
    assert np.any(world.plates[0].crust.type == CrustType["CRATON"])

    if expected is not None:
        assert start["attempts"] == expected["attempts"]
        assert start["cratonAreaScale"] == pytest.approx(expected["areaScale"], abs=0)
        assert start["fractionMet"] is expected["fractionMet"]
        assert start["elongation"] == pytest.approx(
            expected["layout"]["elongation"], abs=2e-12
        )
        assert start["centre"] == pytest.approx(expected["middle"], abs=2e-12)
        assert start["rimCentre"] == pytest.approx(
            expected["land"]["rimCentre"], abs=2e-12
        )
        assert start["skirtKm"] == pytest.approx(expected["land"]["skirtKm"], abs=2e-8)
        assert len(start["cratons"]) == len(expected["layout"]["cratons"])
        for py_craton, js_craton in zip(
            start["cratons"], expected["layout"]["cratons"], strict=True
        ):
            for key in ("id", "letter", "cluster"):
                assert py_craton[key] == js_craton[key]
            assert py_craton["centre"] == pytest.approx(js_craton["centre"], abs=2e-12)
            for key in ("radius", "reach", "inner", "grain"):
                assert py_craton[key] == pytest.approx(js_craton[key], abs=2e-12)
        continent = world.plates[0].crust
        for field in ("type", "terrane"):
            np.testing.assert_array_equal(
                getattr(continent, field), expected["land"][field]
            )
        for field in ("seams", "owner"):
            np.testing.assert_array_equal(
                start[field],
                expected["land"]["suture" if field == "seams" else "owner"],
            )


def test_rigid_drift_scenario_builds_four_moving_continents_and_ocean():
    from orogen.v2.tectonics import build_rigid_drift_world, set_rigid_drift_time

    mesh = build_sim_mesh(128, 1201)
    world = build_rigid_drift_world(mesh, 1202)

    assert len(world.plates) == 5
    assert np.all(_coverage(world) == 1)
    assert world.plates[-1].isBackground
    assert all(hasattr(plate, "scene") for plate in world.plates)
    initial = [plate.quat.copy() for plate in world.plates]
    set_rigid_drift_time(world, 25.0)
    assert world.time == 25.0
    assert any(
        not np.allclose(before, plate.quat)
        for before, plate in zip(initial, world.plates, strict=True)
    )


def test_wilson_scenario_reverses_western_plate_at_configured_time():
    from orogen.v2.tectonics import build_wilson_world, drive_wilson

    mesh = build_sim_mesh(128, 1203)
    world = build_wilson_world(mesh, 1204)

    assert len(world.plates) == 4
    assert np.all(_coverage(world) == 1)
    assert world.plates[0].name == "West"
    assert world.plates[1].name == "East"
    assert np.count_nonzero(world.plates[0].crust.flags & CrustFlag["RIFT_MARGIN"]) > 0

    drive_wilson(world)
    west_pole = np.asarray(world.plates[0].scene["motion"]["pole"])
    assert float(np.dot(world.plates[0].omega, west_pole)) < 0.0
    world.time = 100.0
    drive_wilson(world)
    assert float(np.dot(world.plates[0].omega, west_pole)) > 0.0
    assert float(np.dot(world.plates[1].omega, west_pole)) > 0.0


def test_nudge_scenario_returns_normalized_heading_and_drive_callback():
    from orogen.v2.tectonics import build_nudge_scene

    mesh = build_sim_mesh(96, 1205)
    result = build_nudge_scene(mesh, 1206)
    world = result["world"]
    assert np.isclose(np.linalg.norm(result["heading"]), 1.0)
    assert np.isclose(np.linalg.norm(result["pole"]), 1.0)
    result["drive"](world)
    continent = world.plates[result["start"]["continentId"]]
    assert np.isclose(np.linalg.norm(continent.omega), 2.0 * 10.0 / 6371.0)
