"""Differential checks against the pinned upstream Orogen V2 JavaScript source."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from orogen.v2.core.rotation import (
    quat_from_axis_angle,
    quat_integrate,
    quat_mul,
    quat_normalize,
    quat_to_euler_pole,
    quat_to_mat3,
)
from orogen.v2.core.seed import is_word_seed, make_rng, seed_of
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
def original():
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail(
                "Node.js is required for upstream JavaScript differential tests"
            )
        pytest.skip("Node.js is required for upstream JavaScript differential tests")
    result = subprocess.run(
        [node, str(RUNNER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"upstream JS runner failed:\\n{result.stderr}\\n{result.stdout}"
    )
    return json.loads(result.stdout)


def _assert_close(actual, expected, atol=2e-7):
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=atol)


def test_seed_parsing_and_park_miller_match_original(original):
    texts = ["", "0", "42", "-17", "3.14159", "0x10", "hello", "Orogen 3.0.1", "é"]
    expected = original["seeds"]
    assert [x["text"] for x in expected] == texts
    for item in expected:
        assert is_word_seed(item["text"]) is item["word"]
        assert seed_of(item["text"]) == item["seed"]
    py_rng = make_rng(123456)
    _assert_close([py_rng() for _ in range(12)], original["random"], atol=0)


def test_quaternion_operations_match_original(original):
    q1 = quat_from_axis_angle(0, 0, 1, math.pi / 3)
    q2 = quat_from_axis_angle(1, 0, 0, -math.pi / 7)
    q = quat_normalize(quat_mul(q1, q2))
    omega = [0.001, -0.002, 0.003]
    integrated = quat_integrate(q, omega, 12.5)
    _assert_close(q, original["quaternion"], atol=1e-15)
    _assert_close(integrated, original["integrated"], atol=1e-15)
    _assert_close(quat_to_mat3(integrated), original["matrix"], atol=1e-15)
    pole = quat_to_euler_pole(integrated)
    for key in ("lat", "lon", "angle"):
        assert pole[key] == pytest.approx(original["euler"][key], abs=1e-13)


def test_crust_raster_state_and_snapshot_match_original(original):
    crust = CrustRaster(6)
    crust.create(2, CrustType["CONTINENT"], -120.25, 17)
    crust.arm_trench(2, 35.5)
    crust.record_subduction(2, 2.75)
    crust.record_orogeny(2, OrogenyType["ANDEAN"], 8.125, 20, 15)
    crust.record_orogeny(2, OrogenyType["HIMALAYAN"], 1.5, 25, 15)
    expected = original["crust"]
    assert expected["fields"] == CELL_FIELDS
    assert crust.count == expected["count"]
    assert crust.mask.tolist() == expected["mask"]
    assert crust.flags[2] == expected["flags"] == CrustFlag["TRENCH"]
    assert crust.slabKm[2] == pytest.approx(expected["slabKm"], abs=0)
    assert crust.oroType[2] == expected["oroType"]
    assert crust.oroWork[2] == pytest.approx(expected["oroWork"], abs=0)
    assert crust.oroStart[2] == pytest.approx(expected["oroStart"], abs=0)
    assert crust.oroLast[2] == pytest.approx(expected["oroLast"], abs=0)
    assert crust.snapshot(2) == expected["snapshot"]


def test_crust_orogeny_branch_transitions_match_original(original):
    crust = CrustRaster(5)
    for i in range(5):
        crust.create(i, CrustType["CONTINENT"], 0, i + 1)
    crust.record_orogeny(0, OrogenyType["INTRAPLATE"], 2, 10, 5)
    crust.record_orogeny(0, OrogenyType["ANDEAN"], 3, 12, 5)
    crust.record_orogeny(1, OrogenyType["HIMALAYAN"], 4, 10, 5)
    crust.record_orogeny(1, OrogenyType["ARC"], 1, 11, 5)
    crust.record_orogeny(2, OrogenyType["ANDEAN"], 4, 10, 5)
    crust.record_orogeny(2, OrogenyType["RIFT"], 1, 30, 5)
    crust.record_orogeny(3, OrogenyType["TRANSPRESSION"], 4, 10, 5)
    crust.record_orogeny(3, OrogenyType["URAL"], 1, 11, 5)
    crust.record_orogeny(4, OrogenyType["ANDEAN"], 4, 10, 5)
    crust.record_orogeny(4, OrogenyType["HIMALAYAN"], 1, 11, 5)
    assert [crust.snapshot(i) for i in range(5)] == original["crustBranches"]


def test_plate_extent_empty_antipodal_and_mixed_match_original(original):
    expected = original["extents"]
    cases = {
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
    for name, (occupied, xyz, types) in cases.items():
        plate = Plate(8, len(types), 2)
        plate.set_quat(quat_from_axis_angle(0, 0, 1, math.pi / 3))
        for i in occupied:
            plate.crust.create(i, types[i], -1, i + 1)
        plate.dirty = True
        plate.refresh_extent(np.asarray(xyz, dtype=np.float32))
        got = expected[name]
        assert plate.stats == got["stats"]
        assert plate.crust.count == got["count"]
        assert plate.dirty is got["dirty"]
        _assert_close(plate.capCenter, got["capCenter"], atol=2e-7)
        assert plate.capCos == pytest.approx(got["capCos"], abs=2e-7)
        _assert_close(
            plate.cap_in_world(type("Mesh", (), {"edge_rad": 0.015})()),
            got["cap"],
            atol=2e-7,
        )


def test_plate_frame_transform_and_extent_match_original(original):
    q1 = quat_from_axis_angle(0, 0, 1, math.pi / 3)
    q2 = quat_from_axis_angle(1, 0, 0, -math.pi / 7)
    integrated = quat_integrate(
        quat_normalize(quat_mul(q1, q2)), [0.001, -0.002, 0.003], 12.5
    )
    plate = Plate(3, 6, 4.5)
    plate.set_quat(integrated)
    world = plate.to_world(0.25, -0.5, 0.75)
    local = plate.to_plate_frame(*world)
    xyz = np.array(
        [1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1], dtype=np.float32
    )
    plate.crust.create(0, CrustType["CRATON"], -10, 1)
    plate.crust.create(1, CrustType["OCEAN"], -5, 2)
    plate.refresh_extent(xyz)
    other = Plate(4, 6)
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
        plate.cap_in_world(type("Mesh", (), {"edge_rad": 0.02})()), expected["cap"]
    )
    _assert_close(rotation.reshape(-1), expected["frameRotation"])


def test_plate_kinematics_match_original(original):
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
    xyz = lat_lon_to_xyz(35, -120)
    _assert_close(xyz, expected["xyz"], atol=1e-15)
    ll = xyz_to_lat_lon(*lat_lon_to_xyz(-12, 77))
    assert ll["lat"] == pytest.approx(expected["latlon"]["lat"], abs=1e-13)
    assert ll["lon"] == pytest.approx(expected["latlon"]["lon"], abs=1e-13)
    omega = omega_from_euler_pole(xyz, 4.5)
    _assert_close(omega, expected["omega"], atol=1e-18)
    point = lat_lon_to_xyz(-12, 77)
    _assert_close(surface_velocity(omega, *point), expected["velocity"], atol=1e-18)
    assert speed_cm_yr_at(omega, *point) == pytest.approx(expected["speed"], abs=1e-13)
    q0 = quat_from_axis_angle(0, 1, 0, 0.2)
    _assert_close(pose_at_constant_omega(q0, omega, 3.25), expected["pose"], atol=1e-15)

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
    world.time = 7
    world.step = 2
    advance_plates(world, 1.75)
    _assert_close(world.plates[0].quat, expected["advanced"], atol=1e-15)
    _assert_close(world.plates[1].quat, expected["dead"], atol=0)
    assert world.time == expected["time"] and world.step == expected["step"]


def test_tectonic_occupancy_bins_match_original(original):
    from types import SimpleNamespace

    from orogen.v2.tectonics.occupancy import bin_index, mark_bin, may_have_crust_near

    mesh = SimpleNamespace(
        num_cells=4,
        xyz=np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0], dtype=np.float32),
        edge_km=500.0,
    )
    bins = bin_index(mesh)
    plate = SimpleNamespace(bins=None)
    mark_bin(plate, int(bins.bin_of[0]), 19, bins.count)
    expected = original["occupancy"]
    assert bins.count == expected["count"]
    assert bins.bin_of.tolist() == expected["binOf"]
    assert [len(x) for x in bins.neighbours] == expected["counts"]
    assert [may_have_crust_near(plate, i, 19, bins) for i in range(4)] == expected[
        "near"
    ]
    assert (
        may_have_crust_near(SimpleNamespace(bins=None), 0, 19, bins)
        is expected["absent"]
    )
    assert bin_index(mesh) is bins
