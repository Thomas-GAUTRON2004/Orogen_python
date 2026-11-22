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
from orogen.v2.tectonics.initial_world import _plate_outline_with_apron

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
    "apronKm",
    "apronSmoothKm",
    "slabMatureKm",
    "riftTenureMyr",
    "initialOceanAgeMyr",
    "initialLandAgeMyr",
    "farStartWorkKm",
    "farReachKm",
    "farFrontRefKm",
    "farMaxKm",
    "farCratonCost",
    "farWeakCost",
    "farCratonTake",
    "farWeakTake",
    "farFoldMyr",
    "startRangeVary",
    "startRangeVaryKm",
    "orogenyQuietMyr",
    "weakCellCost",
    "cratonRimKm",
    "cratonRimCost",
    "cratonRimShare",
    "beltWorkShare",
    "thickeningPerWorkKm",
    "startRanges",
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
    reference = json.loads(result.stdout)
    # JSON.stringify serializes JavaScript Infinity values as null. Restore
    # that transport representation for unreachable seam distances so the
    # differential comparison preserves the reference's numeric semantics.
    for field in ("major", "ring"):
        distances = reference["world"]["startSeamKm"][field]
        reference["world"]["startSeamKm"][field] = [
            float("inf") if value is None else value for value in distances
        ]
    return reference


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


@pytest.mark.parametrize(
    ("mesh_cells", "seed"),
    ((512, 912), (768, 777)),
)
def test_production_resolution_start_world_matches_javascript_reference(
    mesh_cells: int, seed: int
):
    mesh = build_sim_mesh(mesh_cells, 911)
    node = shutil.which("node")
    expected = None
    if node is not None:
        expected = _run_initial_world_reference(node, mesh, seed, TECTONICS)
        assert expected is not None, (
            "JavaScript reference did not produce a start-world layout"
        )
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

        # Compare the JavaScript plate/crust assembly after the landmass is
        # wrapped in the smoothed ocean apron and active/passive rim sectors.
        reference_world = expected["world"]
        continent, ocean = world.plates
        python_in_plate, _python_rim = _plate_outline_with_apron(
            mesh,
            np.asarray(expected["land"]["type"]) != CrustType["OCEAN"],
            world.config,
        )
        np.testing.assert_array_equal(
            python_in_plate,
            reference_world["continentMask"],
            err_msg=(
                "Python/JavaScript apron mismatch: "
                f"Python cells={int(np.count_nonzero(python_in_plate))}, "
                f"JavaScript cells={int(np.count_nonzero(reference_world['continentMask']))}, "
                f"Python hops=({world.config['apronKm'] / mesh.edge_km:.6f}, "
                f"{world.config['apronSmoothKm'] / mesh.edge_km:.6f}), "
                f"JavaScript diagnostics={reference_world['apronDiagnostics']}"
            ),
        )
        np.testing.assert_array_equal(
            continent.crust.mask, reference_world["continentMask"]
        )
        np.testing.assert_array_equal(ocean.crust.mask, reference_world["oceanMask"])
        for field in ("type", "flags", "terrane", "slabKm", "formedAt"):
            np.testing.assert_array_equal(
                getattr(continent.crust, field), reference_world[field]
            )
        np.testing.assert_array_equal(
            ocean.crust.formedAt, reference_world["oceanFormedAt"]
        )
        for field in ("major", "majorRange", "ring"):
            np.testing.assert_array_equal(
                world.startSeamKm[field], reference_world["startSeamKm"][field]
            )

        # Compare the inherited pre-history ranges and far-field assembly pushes
        # against the pinned JavaScript start-world implementation.
        initial_record_fields = (
            "oroType",
            "oroWork",
            "oroStart",
            "oroLast",
            "beltU",
            "beltReach",
            "beltAt",
            "beltSide",
            "beltShear",
            "prevType",
            "prevLast",
            "prevU",
            "prevReach",
            "prevSide",
            "prevShear",
            "prevRugged",
            "thickKm",
            "rootKm",
            "upliftKm",
            "rugged",
            "farWork",
            "farDist",
            "farFold",
            "farAt",
            "farDx",
            "farDy",
            "farDz",
        )
        integer_record_fields = {
            "oroType",
            "beltSide",
            "beltShear",
            "prevType",
            "prevSide",
            "prevShear",
            "farDx",
            "farDy",
            "farDz",
        }
        for field in initial_record_fields:
            actual = getattr(continent.crust, field)
            reference = reference_world["initialRecords"][field]
            if field in integer_record_fields:
                np.testing.assert_array_equal(actual, reference, err_msg=field)
            else:
                np.testing.assert_allclose(
                    actual,
                    reference,
                    rtol=2e-5,
                    atol=2e-5,
                    err_msg=field,
                )
        assert continent.name == reference_world["continentName"]
        assert continent.born == pytest.approx(reference_world["born"], abs=0)
        assert start["rim"].tolist() == reference_world["rim"]
        assert start["trench"].tolist() == reference_world["trench"]
        assert start["passiveFrom"] == pytest.approx(
            reference_world["passiveFrom"], abs=2e-12
        )
        assert start["passiveTo"] == pytest.approx(
            reference_world["passiveTo"], abs=2e-12
        )
        assert start["reference"] == pytest.approx(
            reference_world["reference"], abs=2e-12
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


def test_public_camel_case_builders_are_snake_case_aliases():
    import orogen.v2.tectonics as tectonics

    aliases = (
        ("buildBudgetScene", "build_budget_scene"),
        ("buildCollisionScene", "build_collision_scene"),
        ("buildEventScene", "build_event_scene"),
        ("buildForceScene", "build_force_scene"),
        ("buildHotspotScene", "build_hotspot_scene"),
        ("buildMarginScene", "build_margin_scene"),
        ("buildRiftScene", "build_rift_scene"),
        ("buildTerrainScene", "build_terrain_scene"),
        ("buildStartWorld", "build_start_world"),
        ("buildContinentalScene", "build_continental_scene"),
        ("buildOceanicScene", "build_oceanic_scene"),
        ("buildMixedScene", "build_mixed_scene"),
        ("buildScenarioScene", "build_scenario_scene"),
        ("buildRigidDriftWorld", "build_rigid_drift_world"),
        ("setRigidDriftTime", "set_rigid_drift_time"),
        ("buildWilsonWorld", "build_wilson_world"),
        ("driveWilson", "drive_wilson"),
        ("buildNudgeScene", "build_nudge_scene"),
        ("headingForMargin", "heading_for_margin"),
        ("omegaForMargin", "omega_for_margin"),
    )
    for camel_case, snake_case in aliases:
        assert getattr(tectonics, camel_case) is getattr(tectonics, snake_case)
