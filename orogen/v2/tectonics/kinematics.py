"""Plate kinematics and time integration ported from V2 step/kinematics.js."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..core.rotation import (
    quat_from_axis_angle,
    quat_integrate,
    quat_mul,
    quat_normalize,
)

CM_PER_YR_TO_RAD_PER_MYR = 10.0 / 6371.0


def lat_lon_to_xyz(lat_deg: float, lon_deg: float) -> list[float]:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]


def xyz_to_lat_lon(x: float, y: float, z: float) -> dict[str, float]:
    return {
        "lat": math.degrees(math.asin(max(-1.0, min(1.0, z)))),
        "lon": math.degrees(math.atan2(y, x)),
    }


def omega_from_euler_pole(pole_xyz: Sequence[float], speed_cm_yr: float) -> list[float]:
    rate = speed_cm_yr * CM_PER_YR_TO_RAD_PER_MYR
    return [pole_xyz[i] * rate for i in range(3)]


def surface_velocity(omega: Sequence[float], px: float, py: float, pz: float, out=None):
    if out is None:
        out = [0.0, 0.0, 0.0]
    out[0] = omega[1] * pz - omega[2] * py
    out[1] = omega[2] * px - omega[0] * pz
    out[2] = omega[0] * py - omega[1] * px
    return out


def speed_cm_yr_at(omega: Sequence[float], px: float, py: float, pz: float) -> float:
    return math.hypot(*surface_velocity(omega, px, py, pz)) / CM_PER_YR_TO_RAD_PER_MYR


def advance_plates(world, dt: float) -> None:
    for plate in world.plates:
        if plate.alive:
            plate.set_quat(quat_integrate(plate.quat, plate.omega, dt))
    world.time += dt
    world.step += 1


def pose_at_constant_omega(
    q0: Sequence[float], omega: Sequence[float], t: float
) -> list[float]:
    rate = math.hypot(*omega)
    if rate < 1e-15:
        return list(q0)
    dq = quat_from_axis_angle(
        omega[0] / rate, omega[1] / rate, omega[2] / rate, rate * t
    )
    return quat_normalize(quat_mul(dq, q0))
