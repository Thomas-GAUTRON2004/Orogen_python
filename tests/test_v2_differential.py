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




def test_crust_raster_state_and_snapshot_match_original(original, differential_payload):
    scenario = differential_payload["crust"]
    crust = CrustRaster(scenario["size"])
    crust.create(
        scenario["cell"], scenario["type"], scenario["longitude"], scenario["terrane"]
    )
    crust.arm_trench(scenario["cell"], scenario["trenchDepth"])
    crust.record_subduction(scenario["cell"], scenario["subduction"])
    for orogeny_type, work, start, last in scenario["orogeny"]:
        crust.record_orogeny(scenario["cell"], orogeny_type, work, start, last)
    expected = original["crust"]
    assert expected["fields"] == CELL_FIELDS
    assert crust.count == expected["count"]
    assert crust.mask.tolist() == expected["mask"]
    cell = scenario["cell"]
    assert crust.flags[cell] == expected["flags"] == CrustFlag["TRENCH"]
    assert crust.slabKm[cell] == pytest.approx(expected["slabKm"], abs=0)
    assert crust.oroType[cell] == expected["oroType"]
    assert crust.oroWork[cell] == pytest.approx(expected["oroWork"], abs=0)
    assert crust.oroStart[cell] == pytest.approx(expected["oroStart"], abs=0)
    assert crust.oroLast[cell] == pytest.approx(expected["oroLast"], abs=0)
    assert crust.snapshot(cell) == expected["snapshot"]


def test_crust_orogeny_branch_transitions_match_original(
    original, differential_payload
):
    scenario = differential_payload["crustBranches"]
    crust = CrustRaster(scenario["size"])
    for i in range(scenario["size"]):
        crust.create(
            i, scenario["baseType"], scenario["longitude"], scenario["terraneStart"] + i
        )
    for cell, orogeny_type, work, start, last in scenario["operations"]:
        crust.record_orogeny(cell, orogeny_type, work, start, last)
    assert [crust.snapshot(i) for i in range(scenario["size"])] == original[
        "crustBranches"
    ]


def test_plate_extent_empty_antipodal_and_mixed_match_original(
    original, differential_payload
):
    expected = original["extents"]
    for case in differential_payload["extentCases"]:
        name = case["name"]
        occupied, xyz, types = case["occupied"], case["xyz"], case["types"]
        extent_plate = differential_payload["extentPlate"]
        q_input = differential_payload["quaternion"]
        plate = Plate(extent_plate["id"], case["numCells"], extent_plate["radius"])
        plate.set_quat(quat_from_axis_angle(*q_input["q1Axis"], q_input["q1Angle"]))
        for i in occupied:
            plate.crust.create(i, types[i], case["longitude"], case["terraneStart"] + i)
        plate.dirty = True
        plate.refresh_extent(np.asarray(xyz, dtype=np.float32))
        got = expected[name]
        assert plate.stats == got["stats"]
        assert plate.crust.count == got["count"]
        assert plate.dirty is got["dirty"]
        _assert_close(plate.capCenter, got["capCenter"], atol=2e-7)
        assert plate.capCos == pytest.approx(got["capCos"], abs=2e-7)
        _assert_close(
            plate.cap_in_world(
                type(
                    "Mesh",
                    (),
                    {"edge_rad": differential_payload["extentPlate"]["edgeRad"]},
                )()
            ),
            got["cap"],
            atol=2e-7,
        )


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


def test_plate_kinematics_match_original(original, differential_payload):
    from orogen.v2.core.rotation import quat_from_axis_angle
    from orogen.v2.tectonics.kinematics import (
        advance_plates,
        lat_lon_to_xyz,
        omega_from_euler_pole,
        pose_at_constant_omega,
        speed_cm_yr_at,
        surface_velocity,
        xyz_to_lat_lon,
    )

    expected = original["kinematics"]
    kin_input = differential_payload["kinematics"]
    xyz = lat_lon_to_xyz(*kin_input["origin"])
    velocity_point = lat_lon_to_xyz(*kin_input["velocityPoint"])
    _assert_close(xyz, expected["xyz"], atol=1e-15)
    ll = xyz_to_lat_lon(*velocity_point)
    assert ll["lat"] == pytest.approx(expected["latlon"]["lat"], abs=1e-13)
    assert ll["lon"] == pytest.approx(expected["latlon"]["lon"], abs=1e-13)
    omega = omega_from_euler_pole(xyz, kin_input["eulerSpeed"])
    _assert_close(omega, expected["omega"], atol=1e-18)
    _assert_close(
        surface_velocity(omega, *velocity_point), expected["velocity"], atol=1e-18
    )
    assert speed_cm_yr_at(omega, *velocity_point) == pytest.approx(
        expected["speed"], abs=1e-13
    )
    q0 = quat_from_axis_angle(*kin_input["poseAxis"], kin_input["poseAngle"])
    _assert_close(
        pose_at_constant_omega(q0, omega, kin_input["poseDt"]),
        expected["pose"],
        atol=1e-15,
    )

    class World:
        pass

    class P:
        def __init__(self, alive, quat, omega):
            self.alive, self.quat, self.omega = alive, quat, omega

        def set_quat(self, q):
            self.quat = q

    dead = P(False, q0, [1, 2, 3])
    world = World()
    world.plates = [P(True, q0, omega), dead]
    world.time = kin_input["worldTime"]
    world.step = kin_input["worldStep"]
    advance_plates(world, kin_input["advanceDt"])
    _assert_close(world.plates[0].quat, expected["advanced"], atol=1e-15)
    _assert_close(world.plates[1].quat, expected["dead"], atol=0)
    assert world.time == expected["time"] and world.step == expected["step"]

\n