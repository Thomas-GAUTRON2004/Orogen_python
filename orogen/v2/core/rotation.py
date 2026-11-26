"""Quaternion rotations on the unit sphere, ported from V2 rotation.js."""

# differential-module: mode=approx rel_tol=1e-9 abs_tol=1e-9 params=q:quaternion,a:quaternion,b:quaternion,omega:vec3,dt:time_step,angle:angle

from __future__ import annotations

import math
from collections.abc import MutableSequence, Sequence


def quat_identity() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0]


def quat_from_axis_angle(ax: float, ay: float, az: float, angle: float) -> list[float]:
    half = angle * 0.5
    sine = math.sin(half)
    return [math.cos(half), ax * sine, ay * sine, az * sine]


def quat_mul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Hamilton product a·b (apply b first, then a)."""
    return [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]


def quat_normalize(q: Sequence[float]) -> list[float]:
    length = math.hypot(q[0], q[1], q[2], q[3]) or 1.0
    return [q[i] / length for i in range(4)]


# Known divergence: upstream quatToMat3(q, out) requires `out` and throws when it
# is omitted, while this port returns a new list. Remove the skip once fixed.
# differential: skip
def quat_to_mat3(q: Sequence[float], out: MutableSequence[float] | None = None):
    """Write the row-major 3×3 rotation matrix into out or return a new list."""
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    values = [
        1 - 2 * (yy + zz),
        2 * (xy - wz),
        2 * (xz + wy),
        2 * (xy + wz),
        1 - 2 * (xx + zz),
        2 * (yz - wx),
        2 * (xz - wy),
        2 * (yz + wx),
        1 - 2 * (xx + yy),
    ]
    if out is None:
        return values
    for index, value in enumerate(values):
        out[index] = value
    return out


def quat_integrate(
    q: Sequence[float], omega: Sequence[float], dt: float
) -> list[float]:
    """Advance orientation by a world-frame angular velocity over dt."""
    rate = math.hypot(omega[0], omega[1], omega[2])
    if rate < 1e-12:
        return list(q)
    delta = quat_from_axis_angle(
        omega[0] / rate, omega[1] / rate, omega[2] / rate, rate * dt
    )
    return quat_normalize(quat_mul(delta, q))


def quat_to_euler_pole(q: Sequence[float]) -> dict[str, float]:
    """Return a GPlates-style Euler pole in degrees."""
    sine = math.hypot(q[1], q[2], q[3])
    if sine < 1e-12:
        return {"lat": 90.0, "lon": 0.0, "angle": 0.0}
    angle = 2 * math.atan2(sine, q[0])
    x, y, z = q[1] / sine, q[2] / sine, q[3] / sine
    return {
        "lat": math.asin(max(-1.0, min(1.0, z))) * 180 / math.pi,
        "lon": math.atan2(y, x) * 180 / math.pi,
        "angle": angle * 180 / math.pi,
    }
