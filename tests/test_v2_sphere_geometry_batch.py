"""Batch spherical geometry and optional Numba acceleration parity tests."""
import math

import numpy as np
import pytest

import orogen.v2.core.sphere_geometry as geometry
from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE


def _sample_points():
    values = np.asarray([
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.2, -0.3, 0.93),
        (-0.4, 0.5, -0.77),
    ], dtype=np.float64)
    values /= np.linalg.norm(values, axis=1)[:, None]
    return values


def test_batch_spherical_geometry_matches_scalar_operations():
    points = _sample_points()
    bearings = np.linspace(-math.pi, math.pi, len(points))
    distances = np.linspace(0.001, 0.2, len(points))

    actual_targets = geometry.destination_many(points, bearings, distances)
    expected_targets = np.asarray([
        geometry.destination(points[i], bearings[i], distances[i])
        for i in range(len(points))
    ])
    np.testing.assert_allclose(actual_targets, expected_targets, rtol=0, atol=2e-15)

    actual_bearings = geometry.bearing_to_many(points, actual_targets)
    expected_bearings = np.asarray([
        geometry.bearing_to(points[i], expected_targets[i])
        for i in range(len(points))
    ])
    np.testing.assert_allclose(actual_bearings, expected_bearings, rtol=0, atol=2e-15)


def test_batch_spherical_geometry_accepts_scalar_parameters_and_empty_inputs():
    points = _sample_points()
    expected = np.asarray([
        geometry.destination(point, 0.71, 0.013) for point in points
    ])
    np.testing.assert_allclose(
        geometry.destination_many(points, 0.71, 0.013),
        expected, rtol=0, atol=2e-15,
    )
    assert geometry.destination_many(np.empty((0, 3)), 0.5, 0.1).shape == (0, 3)
    assert geometry.bearing_to_many(np.empty((0, 3)), np.empty((0, 3))).shape == (0,)


def test_batch_spherical_geometry_fallback_matches_scalar(monkeypatch):
    points = _sample_points()
    monkeypatch.setattr(geometry, "NUMBA_AVAILABLE", False)
    actual = geometry.destination_many(points, 0.71, 0.013)
    expected = np.asarray([
        geometry.destination(point, 0.71, 0.013) for point in points
    ])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    assert len(geometry.bearing_to_many(points, actual)) == len(points)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="install orogen[speed] to test Numba kernels")
def test_crust_numba_snapshot_reset_round_trip():
    from orogen.v2.tectonics.crust import CrustRaster, CrustType

    raster = CrustRaster(3)
    raster.create(1, CrustType["CONTINENT"], -12.5, 655)
    raster.flags[1] = 255
    raster.farDx[1] = -127
    raster.coverKm[1] = 1.25
    snapshot = raster.snapshot(1)
    assert len(snapshot) == len(raster.fields)
    raster.create(1, CrustType["OCEAN"], 7.0, 4)
    assert raster.type[1] == CrustType["OCEAN"]
    assert raster.flags[1] == 0
    assert raster.farDx[1] == 0
    raster.restore(2, snapshot)
    assert raster.snapshot(2) == snapshot
    assert raster.count == 2
