"""Behavioral coverage for the prescribed-motion scene builders.

These tests cover the Python scene contracts. They are not a substitute for a
JavaScript differential fixture; that requires executing the pinned submodule.
"""


from __future__ import annotations

import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.kinematics import CM_PER_YR_TO_RAD_PER_MYR
from orogen.v2.tectonics.scenarios import (
    EAST_CM_YR,
    NUDGE_CM_YR,
    WEST_CM_YR,
    WEST_REVERSES_MYR,
    build_nudge_scene,
    build_rigid_drift_world,
    build_wilson_world,
    drive_wilson,
    set_rigid_drift_time,
)

SEED = 941


@pytest.fixture(scope="module")
def sim_mesh():
    return build_sim_mesh(768, SEED)


def _omega_norm(plate):
    return float(np.linalg.norm(np.asarray(plate.omega, dtype=float)))


def test_rigid_drift_prescribes_expected_plate_speeds_and_absolute_poses(sim_mesh):
    world = build_rigid_drift_world(sim_mesh, SEED)
    by_name = {plate.name: plate for plate in world.plates}

    assert list(by_name) == [
        "Drifter",
        "Spinner",
        "Hinge",
        "Anchor",
        "Pre-history ocean",
    ]
    assert _omega_norm(by_name["Drifter"]) == pytest.approx(
        5.0 * CM_PER_YR_TO_RAD_PER_MYR
    )
    assert _omega_norm(by_name["Spinner"]) == pytest.approx(
        4.0 * CM_PER_YR_TO_RAD_PER_MYR
    )
    assert _omega_norm(by_name["Hinge"]) == pytest.approx(
        3.0 * CM_PER_YR_TO_RAD_PER_MYR
    )
    assert _omega_norm(by_name["Anchor"]) == 0
    assert _omega_norm(by_name["Pre-history ocean"]) == 0
    assert by_name["Pre-history ocean"].isBackground

    initial = {name: plate.mat.copy() for name, plate in by_name.items()}
    set_rigid_drift_time(world, 125.0)
    assert world.time == 125.0
    assert not np.allclose(by_name["Drifter"].mat, initial["Drifter"])
    assert not np.allclose(by_name["Spinner"].mat, initial["Spinner"])
    assert not np.allclose(by_name["Hinge"].mat, initial["Hinge"])
    assert np.allclose(by_name["Anchor"].mat, initial["Anchor"])
    assert np.allclose(
        by_name["Pre-history ocean"].mat, initial["Pre-history ocean"]
    )

    # Absolute-time evaluation must be reversible, not incremental.
    at_125 = by_name["Drifter"].mat.copy()
    set_rigid_drift_time(world, 0.0)
    assert world.time == 0.0
    assert np.allclose(by_name["Drifter"].mat, initial["Drifter"])
    set_rigid_drift_time(world, 125.0)
    assert np.allclose(by_name["Drifter"].mat, at_125)


def test_wilson_cycle_reverses_west_and_keeps_east_motion(sim_mesh):
    world = build_wilson_world(sim_mesh, SEED)
    by_name = {plate.name: plate for plate in world.plates}

    assert list(by_name) == ["West", "East", "Far", "Pre-history ocean"]
    assert by_name["Pre-history ocean"].isBackground

    world.time = WEST_REVERSES_MYR - 1e-6
    drive_wilson(world)
    west_before = np.asarray(by_name["West"].omega)
    east_before = np.asarray(by_name["East"].omega)
    pole = np.asarray(by_name["West"].scene["motion"]["pole"])
    assert np.dot(west_before, pole) < 0
    assert np.linalg.norm(west_before) == pytest.approx(
        WEST_CM_YR * CM_PER_YR_TO_RAD_PER_MYR
    )
    assert np.linalg.norm(east_before) == pytest.approx(
        EAST_CM_YR * CM_PER_YR_TO_RAD_PER_MYR
    )

    world.time = WEST_REVERSES_MYR
    drive_wilson(world)
    west_after = np.asarray(by_name["West"].omega)
    east_after = np.asarray(by_name["East"].omega)
    assert np.dot(west_after, pole) > 0
    assert np.allclose(west_after, -west_before)
    assert np.allclose(east_after, east_before)
    assert _omega_norm(by_name["Far"]) == 0
    assert _omega_norm(by_name["Pre-history ocean"]) == 0


def test_nudge_drives_only_the_generated_continent_at_prescribed_speed(sim_mesh):
    result = build_nudge_scene(sim_mesh, SEED)
    world = result["world"]
    continent = world.plates[int(result["start"]["continentId"])]
    before = [np.asarray(plate.omega).copy() for plate in world.plates]

    assert np.linalg.norm(result["heading"]) == pytest.approx(1.0)
    assert np.linalg.norm(result["pole"]) == pytest.approx(1.0)
    assert abs(float(np.dot(result["pole"], result["start"]["centre"]))) < 1e-10

    result["drive"](world)
    assert _omega_norm(continent) == pytest.approx(
        NUDGE_CM_YR * CM_PER_YR_TO_RAD_PER_MYR
    )
    for index, plate in enumerate(world.plates):
        if plate is not continent:
            assert np.allclose(plate.omega, before[index])
