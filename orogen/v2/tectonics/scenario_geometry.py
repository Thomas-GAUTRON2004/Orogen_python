"""Shared spherical motion helpers for interactive tectonic scenarios."""

from __future__ import annotations

import math

import numpy as np

from ..core.sphere_geometry import cross, normalize
from .kinematics import omega_from_euler_pole


def heading(point: list[float] | np.ndarray, bearing_deg: float) -> list[float]:
    """Return the unit tangent at a point toward a compass bearing."""
    position = np.asarray(point, dtype=np.float64)
    north = normalize(
        [
            -position[2] * position[0],
            -position[2] * position[1],
            1.0 - position[2] * position[2],
        ]
    )
    east = cross(north, position.tolist())
    bearing = math.radians(bearing_deg)
    return normalize(
        [
            north[index] * math.cos(bearing) + east[index] * math.sin(bearing)
            for index in range(3)
        ]
    )


def omega_for(
    point: list[float] | np.ndarray,
    direction: list[float] | np.ndarray,
    speed_cm_yr: float,
) -> list[float]:
    """Angular velocity for moving a point along a tangent at a given speed."""
    pole = normalize(
        cross(
            np.asarray(point, dtype=np.float64).tolist(),
            np.asarray(direction, dtype=np.float64).tolist(),
        )
    )
    return omega_from_euler_pole(pole, speed_cm_yr)
