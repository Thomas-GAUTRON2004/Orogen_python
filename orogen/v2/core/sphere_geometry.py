"""Small spherical-geometry helpers ported from V2 sphere-geom.js."""

# differential-module: js_module=sphere-geom.mjs params=a:vec3,b:vec3,at:vec3,point:vec3,source:vec3,target:vec3,xyz:number_list,cells:integer_list

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ._numba_kernels import (
    NUMBA_AVAILABLE,
    bearing_to_many_numba,
    destination_many_numba,
)

PLANET_RADIUS_KM = 6_371.0
Vector3 = Sequence[float]


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector3, b: Vector3) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def normalize(a: Vector3) -> list[float]:
    length = math.hypot(a[0], a[1], a[2]) or 1.0
    return [a[0] / length, a[1] / length, a[2] / length]


def angle_between(a: Vector3, b: Vector3) -> float:
    return math.acos(max(-1.0, min(1.0, dot(a, b))))


def tangent_toward(at: Vector3, source: Vector3, target: Vector3) -> list[float]:
    delta = [target[i] - source[i] for i in range(3)]
    projection = dot(delta, at)
    return normalize([delta[i] - projection * at[i] for i in range(3)])


def random_point(rng: Callable[[], float]) -> list[float]:
    z = rng() * 2 - 1
    theta = rng() * 2 * math.pi
    radius = math.sqrt(1 - z * z)
    return [radius * math.cos(theta), radius * math.sin(theta), z]


def tangent_basis(point: Vector3) -> tuple[list[float], list[float]]:
    up = [1.0, 0.0, 0.0] if abs(point[2]) > 0.9 else [0.0, 0.0, 1.0]
    east = normalize(cross(up, point))
    north = cross(point, east)
    return east, north


def destination(point: Vector3, bearing: float, distance: float) -> list[float]:
    east, north = tangent_basis(point)
    dx, dy = math.cos(bearing), math.sin(bearing)
    sine, cosine = math.sin(distance), math.cos(distance)
    return normalize(
        [
            point[0] * cosine + (east[0] * dx + north[0] * dy) * sine,
            point[1] * cosine + (east[1] * dx + north[1] * dy) * sine,
            point[2] * cosine + (east[2] * dx + north[2] * dy) * sine,
        ]
    )


def bearing_to(point: Vector3, target: Vector3) -> float:
    east, north = tangent_basis(point)
    return math.atan2(dot(target, north), dot(target, east))


def _points_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return np.ascontiguousarray(array)


def _parameter_array(values, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return np.full(count, float(array), dtype=np.float64)
    if array.ndim != 1 or array.size != count:
        raise ValueError(f"{name} must be a scalar or have one value per point")
    return np.ascontiguousarray(array)


def destination_many(
    points: Sequence[Vector3],
    bearing: float | Sequence[float],
    distance: float | Sequence[float],
) -> np.ndarray:
    """Vectorised destination() for an (n, 3) array of unit-sphere points.

    Bearing and distance may each be scalars or one-dimensional arrays. The
    Numba path fuses the scalar geometry operations into one compiled loop.
    """
    values = _points_array(points, "points")
    bearings = _parameter_array(bearing, len(values), "bearing")
    distances = _parameter_array(distance, len(values), "distance")
    if NUMBA_AVAILABLE:
        return destination_many_numba(values, bearings, distances)
    out = np.empty((len(values), 3), dtype=np.float64)
    for index in range(len(values)):
        out[index] = destination(values[index], bearings[index], distances[index])
    return out


def bearing_to_many(
    points: Sequence[Vector3],
    targets: Sequence[Vector3],
) -> np.ndarray:
    """Vectorised bearing_to() for matching arrays of points and targets."""
    values = _points_array(points, "points")
    destinations = _points_array(targets, "targets")
    if destinations.shape != values.shape:
        raise ValueError("targets must have the same shape as points")
    if NUMBA_AVAILABLE:
        return bearing_to_many_numba(values, destinations)
    return np.asarray(
        [bearing_to(values[i], destinations[i]) for i in range(len(values))],
        dtype=np.float64,
    )


def cap_fraction(radius_rad: float) -> float:
    return (1 - math.cos(radius_rad)) / 2


# Returns a closure, which the JSON replay protocol cannot serialize.
# differential: skip
def polar_frame(
    pole: Vector3, reference: Vector3
) -> Callable[[float, float, float], dict[str, float]]:
    """Return the upstream Euler-pole frame: colatitude and azimuth in degrees."""
    projection = dot(pole, reference)
    u = normalize(
        [
            reference[0] - projection * pole[0],
            reference[1] - projection * pole[1],
            reference[2] - projection * pole[2],
        ]
    )
    v = cross(pole, u)

    def coordinates(x: float, y: float, z: float) -> dict[str, float]:
        colatitude = (
            math.acos(max(-1.0, min(1.0, x * pole[0] + y * pole[1] + z * pole[2])))
            * 180.0
            / math.pi
        )
        azimuth = (
            math.atan2(
                x * v[0] + y * v[1] + z * v[2],
                x * u[0] + y * u[1] + z * u[2],
            )
            * 180.0
            / math.pi
        )
        return {"colat": colatitude, "azimuth": azimuth}

    return coordinates


def cell_distance_km(
    xyz: Sequence[float] | np.ndarray, cell_a: int, cell_b: int
) -> float:
    a, b = 3 * cell_a, 3 * cell_b
    cosine = (
        float(xyz[a]) * float(xyz[b])
        + float(xyz[a + 1]) * float(xyz[b + 1])
        + float(xyz[a + 2]) * float(xyz[b + 2])
    )
    return math.acos(max(-1.0, min(1.0, cosine))) * PLANET_RADIUS_KM


def centroid_of_cells(xyz: Sequence[float], cells: Iterable[int]) -> list[float]:
    x = y = z = 0.0
    for cell in cells:
        offset = 3 * cell
        x += float(xyz[offset])
        y += float(xyz[offset + 1])
        z += float(xyz[offset + 2])
    length = math.hypot(x, y, z) or 1.0
    return [x / length, y / length, z / length]
