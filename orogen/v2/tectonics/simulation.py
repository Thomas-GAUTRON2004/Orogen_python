"""Deterministic adaptive tectonic time-step helpers from V2 simulate.js."""
from __future__ import annotations

import math

from .config import CM_PER_YR_TO_RAD_PER_MYR


def fastest_speed(omega, plate, centre) -> float:
    """Maximum angular surface speed over a plate's spherical cap (rad/Myr)."""
    rate = math.hypot(float(omega[0]), float(omega[1]), float(omega[2]))
    if rate < 1e-12:
        return 0.0
    axis_angle = math.acos(max(-1.0, min(1.0, (
        float(omega[0]) * float(centre[0]) +
        float(omega[1]) * float(centre[1]) +
        float(omega[2]) * float(centre[2])
    ) / rate)))
    cap_radius = math.acos(max(-1.0, min(1.0, float(plate.capCos))))
    reaches_equator = axis_angle - cap_radius <= math.pi / 2 and axis_angle + cap_radius >= math.pi / 2
    return rate * (1.0 if reaches_equator else max(
        math.sin(axis_angle - cap_radius), math.sin(axis_angle + cap_radius)
    ))


def closing_rate(world) -> float:
    """Sum the two fastest living plates' maximum cap speeds."""
    first = second = 0.0
    for plate in world.plates:
        if not plate.alive:
            continue
        if plate.crust.count:
            centre = plate.to_world(*map(float, plate.capCenter))
            rate = fastest_speed(plate.omega, plate, centre)
        else:
            rate = 0.0
        if rate > first:
            second, first = first, rate
        elif rate > second:
            second = rate
    return first + second


def choose_step(world) -> float:
    """Choose an adaptive step that limits two-plate closure in mesh-cell units."""
    config, mesh = world.config, world.mesh
    max_step = float(config["maxStepMyr"])
    max_cells = float(config["maxCellsPerStep"])
    max_speed = float(config["maxSpeedOceanic"])
    rate = closing_rate(world)
    floor = max_cells * mesh.edge_rad / (2 * max_speed * CM_PER_YR_TO_RAD_PER_MYR)
    if rate < 1e-12:
        return max_step
    return max(floor, min(max_step, max_cells * mesh.edge_rad / rate))
