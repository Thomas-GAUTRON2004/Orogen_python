"""Differential checks against the pinned upstream Orogen V2 JavaScript source."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from _differential import run_js_reference

from orogen.v2.core.rotation import (
    quat_from_axis_angle,
    quat_integrate,
    quat_mul,
    quat_normalize,
)
from orogen.v2.tectonics.crust import (
    CELL_FIELDS,
    CrustFlag,
    CrustRaster,
    CrustType,
    OrogenyType,
)
from orogen.v2.tectonics.plate import Plate, frame_rotation

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/differential-runner.mjs"


@pytest.fixture(scope="module")
def differential_payload():
    extent_cases = {
        "empty": ([], [1, 0, 0, 0, 1, 0], [0, 0]),
        "antipodal": (
            [0, 1],
            [1, 0, 0, -1, 0, 0],
            [CrustType["CRATON"], CrustType["OCEAN"]],
        ),
        "mixed": (
            [0, 1, 2],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0],
            [
                CrustType["CRATON"],
                CrustType["OCEAN"],
                CrustType["CONTINENT"],
                CrustType["OCEAN"],
            ],
        ),
    }
    return {
        "seedTexts": [
            "",
            "0",
            "42",
            "-17",
            "3.14159",
            "0x10",
            "hello",
            "Orogen 3.0.1",
            "é",
        ],
        "random": {"seed": 123456, "count": 12},
        "quaternion": {
            "q1Axis": [0, 0, 1],
            "q1Angle": math.pi / 3,
            "q2Axis": [1, 0, 0],
            "q2Angle": -math.pi / 7,
            "omega": [0.001, -0.002, 0.003],
            "dt": 12.5,
        },
        "crust": {
            "size": 6,
            "cell": 2,
            "type": CrustType["CONTINENT"],
            "longitude": -120.25,
            "terrane": 17,
            "trenchDepth": 35.5,
            "subduction": 2.75,
            "orogeny": [
                [OrogenyType["ANDEAN"], 8.125, 20, 15],
                [OrogenyType["HIMALAYAN"], 1.5, 25, 15],
            ],
        },
        "plate": {
            "id": 3,
            "numCells": 6,
            "radius": 4.5,
            "worldPoint": [0.25, -0.5, 0.75],
            "xyz": [1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1],
            "crustCells": [
                [0, CrustType["CRATON"], -10, 1],
                [1, CrustType["OCEAN"], -5, 2],
            ],
            "otherId": 4,
            "otherNumCells": 6,
            "capEdgeRad": 0.02,
        },
        "crustBranches": {
            "size": 5,
            "baseType": CrustType["CONTINENT"],
            "longitude": 0,
            "terraneStart": 1,
            "operations": [
                [0, OrogenyType["INTRAPLATE"], 2, 10, 5],
                [0, OrogenyType["ANDEAN"], 3, 12, 5],
                [1, OrogenyType["HIMALAYAN"], 4, 10, 5],
                [1, OrogenyType["ARC"], 1, 11, 5],
                [2, OrogenyType["ANDEAN"], 4, 10, 5],
                [2, OrogenyType["RIFT"], 1, 30, 5],
                [3, OrogenyType["TRANSPRESSION"], 4, 10, 5],
                [3, OrogenyType["URAL"], 1, 11, 5],
                [4, OrogenyType["ANDEAN"], 4, 10, 5],
                [4, OrogenyType["HIMALAYAN"], 1, 11, 5],
            ],
        },
        "extentPlate": {"id": 8, "radius": 2, "edgeRad": 0.015},
        "extentCases": [
            {
                "name": name,
                "occupied": occupied,
                "xyz": xyz,
                "types": types,
                "numCells": len(occupied),
                "longitude": -1,
                "terraneStart": 1,
            }
            for name, (occupied, xyz, types) in extent_cases.items()
        ],
        "kinematics": {
            "origin": [35, -120],
            "velocityPoint": [-12, 77],
            "eulerSpeed": 4.5,
            "poseAxis": [0, 1, 0],
            "poseAngle": 0.2,
            "poseDt": 3.25,
            "worldTime": 7,
            "worldStep": 2,
            "advanceDt": 1.75,
        },
        "occupancy": {
            "numCells": 4,
            "xyz": [1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0],
            "edgeKm": 500.0,
            "terrane": 19,
        },
    }


@pytest.fixture(scope="module")
def original(differential_payload):
    return run_js_reference(RUNNER, differential_payload)


def _assert_close(actual, expected, atol=2e-7):
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=atol)



\n\n\n
def test_plate_frame_transform_and_extent_match_original(
    original, differential_payload
):
    q_input = differential_payload["quaternion"]
    plate_input = differential_payload["plate"]
    q1 = quat_from_axis_angle(*q_input["q1Axis"], q_input["q1Angle"])
    q2 = quat_from_axis_angle(*q_input["q2Axis"], q_input["q2Angle"])
    integrated = quat_integrate(
        quat_normalize(quat_mul(q1, q2)), q_input["omega"], q_input["dt"]
    )
    plate = Plate(plate_input["id"], plate_input["numCells"], plate_input["radius"])
    plate.set_quat(integrated)
    world = plate.to_world(*plate_input["worldPoint"])
    local = plate.to_plate_frame(*world)
    xyz = np.asarray(plate_input["xyz"], dtype=np.float32)
    for cell, crust_type, longitude, terrane in plate_input["crustCells"]:
        plate.crust.create(cell, crust_type, longitude, terrane)
    plate.refresh_extent(xyz)
    other = Plate(plate_input["otherId"], plate_input["otherNumCells"])
    other.set_quat(q1)
    rotation = frame_rotation(plate, other)
    expected = original["plate"]
    _assert_close(world, expected["world"])
    _assert_close(local, expected["local"])
    _assert_close(plate.mat.reshape(-1), expected["matrix"])
    assert plate.stats == expected["stats"]
    _assert_close(plate.capCenter, expected["capCenter"])
    assert plate.capCos == pytest.approx(expected["capCos"], abs=1e-7)
    _assert_close(
        plate.cap_in_world(type("Mesh", (), {"edge_rad": plate_input["capEdgeRad"]})()),
        expected["cap"],
    )
    _assert_close(rotation.reshape(-1), expected["frameRotation"])

\n