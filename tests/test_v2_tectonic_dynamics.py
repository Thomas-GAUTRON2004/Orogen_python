from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics import (
    Plate, balanced_rotation, clamp_speed, drag_tensor, soften_speed, solve3,
)
from orogen.v2.tectonics.config import CM_PER_YR_TO_RAD_PER_MYR
from orogen.v2.tectonics.crust import CrustType


def fixture():
    plate = Plate(0, 2)
    plate.crust.create(0, CrustType["CONTINENT"], 0)
    plate.crust.create(1, CrustType["CRATON"], 0)
    mesh = SimpleNamespace(
        xyz=np.array([1, 0, 0, 0, 1, 0], dtype=np.float64),
        cell_area_km2=10000.0,
        num_cells=2,
    )
    config = {
        "mobility": 1.0,
        "dragOcean": 1.0,
        "dragContinent": 5.0,
        "dragCraton": 8.0,
        "slabResistance": 2.0,
        "collisionResistance": 3.0,
        "collisionSeparateShare": 0.05,
        "collisionShearShare": 0.5,
        "continentalPlateShare": 0.15,
        "continentalSoftCmYr": 7.0,
        "oceanicSoftCmYr": 12.0,
        "maxSpeedContinental": 10.0,
        "maxSpeedOceanic": 25.0,
    }
    return plate, mesh, config


def test_solve3_matches_linear_system_and_singular_fallback():
    matrix = [4, 1, 0, 1, 3, 1, 0, 1, 2]
    result = solve3(matrix, [1, 2, 3])
    assert result == pytest.approx(np.linalg.solve(np.asarray(matrix).reshape(3, 3), [1, 2, 3]))
    assert solve3([1, 2, 3, 2, 4, 6, 0, 0, 0], [1, 2, 3]) == [0, 0, 0]
    with pytest.raises(ValueError):
        solve3([1], [1, 2, 3])


def test_drag_tensor_weights_ocean_continent_craton_and_rotates_frame():
    plate, mesh, config = fixture()
    tensor = drag_tensor(plate, mesh, config)
    assert tensor == pytest.approx(np.diag([8.0, 5.0, 13.0]))
    plate.mat = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    rotated = drag_tensor(plate, mesh, config)
    assert rotated == pytest.approx(np.diag([5.0, 8.0, 13.0]))
    with pytest.raises(ValueError):
        drag_tensor(plate, SimpleNamespace(xyz=np.zeros(3), cell_area_km2=10000), config)
    with pytest.raises(ValueError):
        drag_tensor(plate, mesh, {**config, "mobility": 0})


def test_balanced_rotation_includes_slab_resistance_and_both_collision_directions():
    plate, mesh, config = fixture()
    plate.mat = np.eye(3)
    load = {
        "torque": [1.0, 2.0, 4.0],
        "slabs": [{"r": [1, 0, 0], "n": [0, 1, 0], "other": 1, "length": 1.0}],
        "contacts": [
            {"r": [1, 0, 0], "n": [0, 1, 0], "other": 1, "weight": 1.0},
            {"r": [0, 1, 0], "n": [0, 0, 1], "other": 1, "weight": 2.0},
        ],
    }
    omega = {0: [0, 0, 2], 1: [1, 0, 0]}
    result = balanced_rotation(plate, mesh, load, omega, config)
    assert len(result) == 3
    assert all(np.isfinite(result))
    without_couplings = balanced_rotation(
        plate, mesh, {"torque": [1, 2, 4]}, omega, config
    )
    assert result != pytest.approx(without_couplings)


def test_speed_softening_and_hard_clamping_for_ocean_and_continent():
    plate, mesh, config = fixture()
    plate.stats = {"cells": 2, "continentalCells": 2}
    plate.capCos = -1.0
    centre = [0, 0, 1]
    omega = [0.0, 0.0, 1.0]
    softened = soften_speed(omega, plate, centre, config)
    assert 0 < np.linalg.norm(softened) < 1
    capped, did_cap = clamp_speed(omega, plate, centre, config)
    assert did_cap is True
    assert np.linalg.norm(capped) == pytest.approx(
        config["maxSpeedContinental"] * CM_PER_YR_TO_RAD_PER_MYR
    )
    plate.stats = {"cells": 2, "continentalCells": 0}
    ocean_soft = soften_speed(omega, plate, centre, config)
    assert 0 < np.linalg.norm(ocean_soft) < 1
    under_limit, did_cap = clamp_speed([0.0, 0.0, 1e-4], plate, centre, config)
    assert under_limit == [0.0, 0.0, 1e-4]
    assert did_cap is False
    unchanged, did_cap = clamp_speed([1e-15, 0, 0], plate, centre, config)
    assert unchanged == [1e-15, 0, 0]
    assert did_cap is False
    no_soft = soften_speed(omega, plate, centre, {**config, "oceanicSoftCmYr": 0})
    assert no_soft == omega
