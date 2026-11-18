"""Deterministic force-balance primitives ported from V2 torque-balance.js."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .config import CM_PER_YR_TO_RAD_PER_MYR
from .crust import CrustType
from .forces import gather_boundary_loads
from .simulation import fastest_speed


def solve3(matrix: Sequence[float], vector: Sequence[float]) -> list[float]:
    """Solve a row-major 3x3 system; return zero for a numerically singular matrix."""
    if len(matrix) != 9 or len(vector) != 3:
        raise ValueError("solve3 expects a row-major 3x3 matrix and a 3-vector")
    d = [float(x) for x in matrix]
    b = [float(x) for x in vector]
    c00 = d[4] * d[8] - d[5] * d[7]
    c01 = d[3] * d[8] - d[5] * d[6]
    c02 = d[3] * d[7] - d[4] * d[6]
    determinant = d[0] * c00 - d[1] * c01 + d[2] * c02
    if abs(determinant) < 1e-30:
        return [0.0, 0.0, 0.0]
    return [
        (
            b[0] * c00
            - d[1] * (b[1] * d[8] - d[5] * b[2])
            + d[2] * (b[1] * d[7] - d[4] * b[2])
        )
        / determinant,
        (
            d[0] * (b[1] * d[8] - d[5] * b[2])
            - b[0] * c01
            + d[2] * (d[3] * b[2] - b[1] * d[6])
        )
        / determinant,
        (
            d[0] * (d[4] * b[2] - b[1] * d[7])
            - d[1] * (d[3] * b[2] - b[1] * d[6])
            + b[0] * c02
        )
        / determinant,
    ]


def drag_tensor(plate, mesh, config: Mapping) -> np.ndarray:
    """Basal-drag tensor in world coordinates for the plate's current crust."""
    count = len(plate.crust.mask)
    points = np.asarray(mesh.xyz, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] != count:
        raise ValueError("mesh and plate crust must have the same cell count")
    mobility = float(config["mobility"])
    if mobility <= 0:
        raise ValueError("mobility must be positive")
    scale = float(mesh.cell_area_km2) / 1e4 / mobility
    coefficients = {
        CrustType["OCEAN"]: float(config["dragOcean"]) * scale,
        CrustType["CONTINENT"]: float(config["dragContinent"]) * scale,
        CrustType["CRATON"]: float(config["dragCraton"]) * scale,
    }
    local = np.zeros((3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for cell in np.flatnonzero(plate.crust.mask):
        kind = int(plate.crust.type[cell])
        coefficient = coefficients.get(kind, float(config["dragContinent"]) * scale)
        r = points[cell]
        local += coefficient * (identity - np.outer(r, r))
    rotation = np.asarray(plate.mat, dtype=np.float64).reshape(3, 3)
    return rotation @ local @ rotation.T


def _cross(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    return np.cross(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def balanced_rotation(
    plate,
    mesh,
    load: Mapping,
    omega_of: Mapping,
    config: Mapping,
) -> list[float]:
    """Solve instantaneous torque balance with slab and collision coupling.

    load contains torque plus optional slabs and contacts lists, matching the
    records returned by the upstream force-gathering pass. omega_of maps plate
    IDs to the angular velocities at the start of the step.
    """
    area_unit = float(mesh.cell_area_km2) / 1e4
    drag = drag_tensor(plate, mesh, config)
    torque = np.asarray(load["torque"], dtype=np.float64).copy()

    for slab in load.get("slabs", ()):
        r = np.asarray(slab["r"], dtype=np.float64)
        normal = np.asarray(slab["n"], dtype=np.float64)
        coefficient = float(config["slabResistance"]) * float(slab["length"])
        axis = _cross(r, normal)
        drag += coefficient * np.outer(axis, axis)
        other_omega = np.asarray(omega_of[slab["other"]], dtype=np.float64)
        torque += coefficient * float(np.dot(other_omega, axis)) * axis

    self_omega = np.asarray(omega_of[plate.id], dtype=np.float64)
    for contact in load.get("contacts", ()):
        r = np.asarray(contact["r"], dtype=np.float64)
        normal = np.asarray(contact["n"], dtype=np.float64)
        other_omega = np.asarray(omega_of[contact["other"]], dtype=np.float64)
        coefficient = (
            float(config["collisionResistance"]) * area_unit * float(contact["weight"])
        )
        normal_axis = _cross(r, normal)
        closing = float(np.dot(self_omega - other_omega, normal_axis)) > 0.0
        tangent_axis = _cross(r, normal_axis)
        terms = (
            (normal_axis, 1.0 if closing else float(config["collisionSeparateShare"])),
            (tangent_axis, float(config["collisionShearShare"])),
        )
        for axis, share in terms:
            weighted = coefficient * share
            drag += weighted * np.outer(axis, axis)
            torque += weighted * float(np.dot(other_omega, axis)) * axis

    return solve3(drag.ravel().tolist(), torque.tolist())


def soften_speed(omega: Sequence[float], plate, centre, config: Mapping) -> list[float]:
    """Apply smooth continental/oceanic speed saturation."""
    result = [float(value) for value in omega]
    continental = float(plate.stats["continentalCells"]) > float(
        config["continentalPlateShare"]
    ) * float(plate.stats["cells"])
    limit_cm_yr = float(
        config["continentalSoftCmYr"] if continental else config["oceanicSoftCmYr"]
    )
    scale_limit = limit_cm_yr * CM_PER_YR_TO_RAD_PER_MYR
    speed = fastest_speed(result, plate, centre)
    if not scale_limit > 0 or speed < 1e-12:
        return result
    factor = 1.0 / math.sqrt(1.0 + (speed / scale_limit) ** 2)
    return [value * factor for value in result]


def clamp_speed(
    omega: Sequence[float], plate, centre, config: Mapping
) -> tuple[list[float], bool]:
    """Apply the configured hard speed ceiling; return the capped vector and status."""
    result = [float(value) for value in omega]
    speed = fastest_speed(result, plate, centre)
    if speed < 1e-12:
        return result, False
    continental = float(plate.stats["continentalCells"]) > float(
        config["continentalPlateShare"]
    ) * float(plate.stats["cells"])
    limit = (
        float(
            config["maxSpeedContinental"] if continental else config["maxSpeedOceanic"]
        )
        * CM_PER_YR_TO_RAD_PER_MYR
    )
    if speed <= limit:
        return result, False
    factor = limit / speed
    return [value * factor for value in result], True


def _describe_dynamics(load: Mapping, target, centre, capped: bool, mesh) -> dict:
    from .forces import FORCE_KINDS
    from .kinematics import speed_cm_yr_at

    net = math.hypot(
        *(float(load["torque"][i]) - float(load["flowTorque"][i]) for i in range(3))
    )
    forces = []
    for index, name in enumerate(FORCE_KINDS):
        cells = int(load["cells"][index])
        forces.append(
            {
                "name": name,
                "cells": cells,
                "km": cells * float(mesh.edge_km),
                "vector": list(load["forceByKind"][index]),
                "torque": float(np.linalg.norm(load["byKind"][index])),
            }
        )
    total = float(load["torqueMagnitudeSum"])
    contacts = load["contacts"]
    return {
        "forces": forces,
        "collisionCells": len(contacts),
        "collisionKm2": len(contacts) * float(mesh.cell_area_km2),
        "tension": 1.0 - net / total if total > 1e-12 else 0.0,
        "targetCmYr": speed_cm_yr_at(target, *map(float, centre)),
        "capped": bool(capped),
    }


def update_plate_motion(world) -> None:
    """Update every living plate from the currently implemented boundary forces.

    Uses the partial boundary-force gatherer; collision-overlap and mantle-flow
    forces are not yet supplied by that gatherer.
    """
    config, plates = world.config, world.plates
    relax_myr = float(config["relaxMyr"])
    if relax_myr <= 0:
        raise ValueError("relaxMyr must be positive")
    loads = gather_boundary_loads(world)
    before = {int(plate.id): list(plate.omega) for plate in plates}
    blend = 1.0 - math.exp(-float(world.lastDt) / relax_myr)

    for plate in plates:
        if not plate.alive:
            continue
        if plate.isBackground or plate.crust.count == 0:
            plate.omega = [0.0, 0.0, 0.0]
            plate.dynamics = None
            continue
        load = loads[int(plate.id)]
        target = balanced_rotation(plate, world.mesh, load, before, config)
        centre = plate.to_world(*map(float, plate.capCenter))
        target = soften_speed(target, plate, centre, config)
        target, capped = clamp_speed(target, plate, centre, config)
        initial = before[int(plate.id)]
        plate.omega = [initial[i] + (target[i] - initial[i]) * blend for i in range(3)]
        plate.dynamics = _describe_dynamics(load, target, centre, capped, world.mesh)
