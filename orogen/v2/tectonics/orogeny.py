"""Orogenic belt geometry helpers ported from Orogen V2 step/orogeny.js."""
from __future__ import annotations

import math


def belt_width_km(kind: str, work_km: float, rate_cm_yr: float, config: dict) -> float:
    """Width of an orogenic belt from accumulated shortening and current rate."""
    belt = config["belts"][kind]
    speed = min(
        float(config["beltSpeedFactorMax"]),
        max(
            float(config["beltSpeedFactorMin"]),
            (max(0.0, float(rate_cm_yr)) / float(config["beltSpeedRefCmYr"]))
            ** float(config["beltSpeedExponent"]),
        ),
    )
    return min(
        float(belt["maxKm"]),
        float(belt["baseKm"]) + float(belt["perKm"]) * float(work_km),
    ) * speed


def belt_shape(head_on: float, slab_age_myr: float | None, config: dict) -> float:
    """Scale belt reach by obliquity and, for subduction, slab youth."""
    h = min(1.0, max(0.0, float(head_on)))
    shape = float(config["beltObliqueMin"]) + (1.0 - float(config["beltObliqueMin"])) * h * h
    if slab_age_myr is not None:
        youth = 1.0 + float(config["beltYoungSlabBoost"]) * (
            1.0 - float(slab_age_myr) / float(config["beltYoungSlabRefMyr"])
        )
        shape *= min(
            1.0 + float(config["beltYoungSlabBoost"]),
            max(float(config["beltOldSlabMin"]), youth),
        )
    return shape


def obliquity_deg(velocity, normal, position, closing: float) -> float:
    """Signed obliquity in degrees; positive is dextral, negative sinistral."""
    px, py, pz = map(float, position)
    nx, ny, nz = map(float, normal)
    vx, vy, vz = map(float, velocity)
    along = (
        vx * (py * nz - pz * ny)
        + vy * (pz * nx - px * nz)
        + vz * (px * ny - py * nx)
    )
    return math.atan2(along, max(1e-12, float(closing))) * 180.0 / math.pi
