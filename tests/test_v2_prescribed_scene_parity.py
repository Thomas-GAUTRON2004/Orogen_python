"""Differential tests for prescribed-motion scenes against the pinned JS engine.

The matrix in ``prescribed-scenes-cases.json`` covers several meshes and seeds. Each
case is built by the Python scenes and by the upstream JavaScript scenes (submodule
``external/planet_heightmap_generation``, ed466f9), then compared field by field.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import os
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.scenarios import (
    build_nudge_scene,
    build_rigid_drift_world,
    build_wilson_world,
    drive_wilson,
    set_rigid_drift_time,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference" / "v2" / "scenes"
MATRIX = json.loads(
    (REFERENCE / "prescribed-scenes-cases.json").read_text(encoding="utf-8")
)
CRUST_FIELDS = ("mask", "type", "flags", "formedAt", "terrane")


def _mesh_for(entry):
    return build_sim_mesh(entry["cells"], entry["seed"])


def _mesh_arrays(mesh):
    return {
        "numCells": int(mesh.num_cells),
        "xyz": np.asarray(mesh.xyz, dtype=np.float32).reshape(-1).tolist(),
        "adjOffset": np.asarray(mesh.adj_offset, dtype=np.int32).tolist(),
        "adjList": np.asarray(mesh.adj_list, dtype=np.int32).tolist(),
    }


@pytest.fixture(scope="module")
def js_reference():
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("node is not installed")
        pytest.skip("node is not installed")
    payload = {
        "meshes": [
            {"mesh": _mesh_arrays(_mesh_for(entry)), "cases": entry["cases"]}
            for entry in MATRIX["meshes"]
        ]
    }
    runner = REFERENCE / "dump_prescribed_scenes.mjs"
    result = subprocess.run(
        [node, str(runner)],
        input=json.dumps(payload, separators=(",", ":"), allow_nan=False),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    assert result.returncode == 0, (
        f"upstream prescribed-scene JS runner failed:\n{result.stderr}\n{result.stdout}"
    )
    return json.loads(result.stdout)


def _assert_crust_matches(plate, expected):
    for field in CRUST_FIELDS:
        np.testing.assert_array_equal(
            getattr(plate.crust, field),
            expected["crust"][field],
            err_msg=f"{plate.name}.crust.{field}",
        )


def _assert_plates_match(world, expected):
    assert len(world.plates) == len(expected)
    for actual, reference in zip(world.plates, expected, strict=True):
        assert actual.id == reference["id"]
        assert actual.name == reference["name"]
        assert bool(actual.isBackground) is reference["isBackground"]
        _assert_crust_matches(actual, reference)
        # Compare the actual kinematic state too, not just plate metadata and crust.
        # These arrays are the state consumed by the simulation after scene setup.
        _assert_vector(actual.omega, reference["omega"], atol=2e-12)
        _assert_vector(actual.quat, reference["quat"], atol=2e-9)


def _assert_vector(actual, expected, *, atol=2e-10):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0,
        atol=atol,
    )


def _cases(scene):
    return [
        (mesh_index, case_index)
        for mesh_index, entry in enumerate(MATRIX["meshes"])
        for case_index in range(len(entry["cases"][scene]))
    ]


def _ids(pairs):
    return [f"mesh{m}-case{c}" for m, c in pairs]


@pytest.mark.parametrize(
    "indices", _cases("rigidDrift"), ids=_ids(_cases("rigidDrift"))
)
def test_rigid_drift_scene_matches_upstream_state_and_absolute_poses(
    indices, js_reference
):
    mesh_index, case_index = indices
    mesh = _mesh_for(MATRIX["meshes"][mesh_index])
    test_case = MATRIX["meshes"][mesh_index]["cases"]["rigidDrift"][case_index]
    expected = js_reference[mesh_index]["rigidDrift"][case_index]
    world = build_rigid_drift_world(mesh, test_case["seed"])

    _assert_plates_match(world, expected["plates"])
    for plate, reference in zip(world.plates, expected["plates"], strict=True):
        scene = plate.scene
        _assert_vector(scene["q0"], reference["q0"])
        _assert_vector(scene["centre"], reference["centre"])
        _assert_vector(scene["pole"], reference["pole"])
        definition = scene["def"]
        for key in ("name", "poleOffsetDeg", "speedCmYr"):
            assert definition[key] == reference["def"][key]
        _assert_vector(definition["centre"], reference["def"]["centre"])
        _assert_vector(plate.omega, reference["omega"])

    assert len(expected["times"]) == len(test_case["timesMyr"])
    for time, reference in zip(test_case["timesMyr"], expected["times"], strict=True):
        set_rigid_drift_time(world, time)
        assert world.time == pytest.approx(reference["time"], abs=0)
        for plate, expected_quat in zip(world.plates, reference["quats"], strict=True):
            _assert_vector(plate.quat, expected_quat, atol=2e-9)


@pytest.mark.parametrize("indices", _cases("wilson"), ids=_ids(_cases("wilson")))
def test_wilson_scene_matches_upstream_crust_and_time_dependent_motion(
    indices, js_reference
):
    mesh_index, case_index = indices
    mesh = _mesh_for(MATRIX["meshes"][mesh_index])
    test_case = MATRIX["meshes"][mesh_index]["cases"]["wilson"][case_index]
    expected = js_reference[mesh_index]["wilson"][case_index]
    world = build_wilson_world(mesh, test_case["seed"])

    _assert_plates_match(world, expected["plates"])
    for plate, reference in zip(world.plates, expected["plates"], strict=True):
        motion = plate.scene["motion"]
        _assert_vector(motion["pole"], reference["motion"]["pole"])
        assert motion["speedCmYr"] == reference["motion"]["speedCmYr"]

    assert len(expected["times"]) == len(test_case["timesMyr"])
    for time, reference in zip(test_case["timesMyr"], expected["times"], strict=True):
        world.time = time
        drive_wilson(world)
        assert world.time == pytest.approx(reference["time"], abs=0)
        for plate, expected_omega in zip(
            world.plates, reference["omegas"], strict=True
        ):
            _assert_vector(plate.omega, expected_omega, atol=2e-12)


@pytest.mark.parametrize("indices", _cases("nudge"), ids=_ids(_cases("nudge")))
def test_nudge_scene_matches_upstream_start_state_and_prescribed_motion(
    indices, js_reference
):
    mesh_index, case_index = indices
    mesh = _mesh_for(MATRIX["meshes"][mesh_index])
    test_case = MATRIX["meshes"][mesh_index]["cases"]["nudge"][case_index]
    expected = js_reference[mesh_index]["nudge"][case_index]
    result = build_nudge_scene(mesh, test_case["seed"])
    world, start = result["world"], result["start"]
    result["drive"](world)

    _assert_plates_match(world, expected["plates"])
    assert start["continentId"] == expected["start"]["continentId"]
    for key in ("centre", "reference"):
        _assert_vector(start[key], expected["start"][key])
    for key in ("passiveFrom", "passiveTo"):
        assert start[key] == pytest.approx(expected["start"][key], abs=2e-10)

    assert len(start["cratons"]) == len(expected["start"]["cratons"])
    for actual, reference in zip(
        start["cratons"], expected["start"]["cratons"], strict=True
    ):
        for key in ("id", "letter", "cluster"):
            assert actual[key] == reference[key]
        _assert_vector(actual["centre"], reference["centre"])
        for key in ("radius", "reach", "inner"):
            assert actual[key] == pytest.approx(reference[key], abs=2e-10)
        _assert_vector(actual["grain"], reference["grain"])

    _assert_vector(result["heading"], expected["heading"])
    _assert_vector(result["pole"], expected["pole"])
    continent = world.plates[int(start["continentId"])]
    _assert_vector(continent.omega, expected["drivenOmega"], atol=2e-12)
