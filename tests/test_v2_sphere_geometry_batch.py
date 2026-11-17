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


def test_batch_spherical_geometry_rejects_invalid_shapes():
    points = _sample_points()
    with pytest.raises(ValueError, match=r"points must have shape"):
        geometry.destination_many(np.zeros((3, 2)), 0.5, 0.1)
    with pytest.raises(ValueError, match=r"bearing must be a scalar"):
        geometry.destination_many(points, np.zeros(2), 0.1)
    with pytest.raises(ValueError, match=r"distance must be a scalar"):
        geometry.destination_many(points, 0.5, np.zeros((len(points), 1)))
    with pytest.raises(ValueError, match="same shape as points"):
        geometry.bearing_to_many(points, np.zeros((len(points) - 1, 3)))


def test_crust_python_fallback_keeps_create_and_snapshot(monkeypatch):
    import orogen.v2.tectonics.crust as crust_module

    monkeypatch.setattr(crust_module, "NUMBA_AVAILABLE", False)
    raster = crust_module.CrustRaster(2)
    raster.create(1, crust_module.CrustType["CONTINENT"], -8.5, 12)
    raster.flags[1] = 3
    snapshot = raster.snapshot(1)
    assert len(snapshot) == len(raster.fields)
    assert snapshot[raster.fields.index(raster.type)] == crust_module.CrustType["CONTINENT"]
    raster.create(1, crust_module.CrustType["OCEAN"], 4.0, 2)
    assert raster.flags[1] == 0
    assert raster.count == 1


def test_crust_batch_create_and_snapshot_match_scalar_order():
    from orogen.v2.tectonics.crust import CrustRaster, CrustType

    cells = np.asarray([1, 2, 1, 3, -1], dtype=np.int32)
    types = np.asarray([1, 2, 3, 1, 2], dtype=np.uint8)
    times = np.asarray([-10.5, -20.0, 4.25, 7.0, -2.0], dtype=np.float64)
    terranes = np.asarray([11, 12, 13, 14, 15], dtype=np.uint16)
    batch = CrustRaster(5)
    scalar = CrustRaster(5)
    batch.create_many(cells, types, times, terranes)
    for cell, crust_type, formed_at, terrane in zip(cells, types, times, terranes):
        scalar.create(int(cell), int(crust_type), float(formed_at), int(terrane))

    assert batch.count == scalar.count == 4
    for cell in range(5):
        assert batch.snapshot(cell) == scalar.snapshot(cell)
    assert batch.snapshot_many(cells) == [batch.snapshot(int(cell)) for cell in cells]


def test_crust_batch_methods_broadcast_and_handle_empty_inputs():
    from orogen.v2.tectonics.crust import CrustRaster, CrustType

    raster = CrustRaster(4)
    raster.create_many([0, 2], CrustType["CONTINENT"], -12.0, 9)
    assert raster.count == 2
    assert raster.snapshot_many([2, 0]) == [raster.snapshot(2), raster.snapshot(0)]
    raster.create_many([], CrustType["OCEAN"], 0.0)
    assert raster.snapshot_many([]) == []
    with pytest.raises(ValueError, match="one-dimensional integer"):
        raster.create_many([[1]], 1, 0.0)
    with pytest.raises(ValueError, match="crust_type must be a scalar"):
        raster.create_many([1, 2], [1, 2, 3], 0.0)
    with pytest.raises(IndexError, match="cell index out of range"):
        raster.snapshot_many([4])
    with pytest.raises(ValueError, match="one-dimensional integer"):
        raster.create_many([1.5], 1, 0.0)


def test_crust_batch_methods_use_python_fallback(monkeypatch):
    import orogen.v2.tectonics.crust as crust_module

    monkeypatch.setattr(crust_module, "NUMBA_AVAILABLE", False)
    raster = crust_module.CrustRaster(3)
    raster.create_many([0, 2], crust_module.CrustType["CONTINENT"], -5.0, [7, 8])
    expected = [raster.snapshot(0), raster.snapshot(2)]
    assert raster.snapshot_many([0, 2]) == expected
    assert raster.count == 2
