"""Mantle thermal-memory field ported from V2 step/hotspots.js.

This module ports the heat-memory part of the hotspot stage. Plume birth,
LIP construction, plume tails and volcano-track stamping remain separate.
"""

from __future__ import annotations

import math

import numpy as np

from .supercontinent import is_continental


def insulate_at_start(world) -> None:
    """Seed the initial mantle-memory field beneath the starting continents."""
    if len(world.mantleHeat) != int(world.mesh.num_cells):
        raise ValueError("mantleHeat must match the simulation mesh cell count")
    for cell in range(int(world.mesh.num_cells)):
        world.mantleHeat[cell] = 1.0 if is_continental(world, cell) else 0.0


def update_mantle_heat(world, dt: float) -> np.ndarray:
    """Warm mantle beneath continental lids and cool it elsewhere exponentially.

    The first call initializes the pre-history field, then applies the same
    warming/cooling step as later calls. Heat values are stored per world cell
    in the interval [0, 1].
    """
    step = float(dt)
    if not math.isfinite(step) or step < 0:
        raise ValueError("dt must be finite and non-negative")
    heat = world.mantleHeat
    if len(heat) != int(world.mesh.num_cells):
        raise ValueError("mantleHeat must match the simulation mesh cell count")
    if not getattr(world, "hotspotsReady", False):
        insulate_at_start(world)
        world.hotspotsReady = True

    warm_myr = float(world.config["mantleInsulateMyr"])
    cool_myr = float(world.config["mantleCoolMyr"])
    if warm_myr <= 0 or cool_myr <= 0:
        raise ValueError("mantleInsulateMyr and mantleCoolMyr must be positive")
    warm = 1.0 - math.exp(-step / warm_myr)
    cool = 1.0 - math.exp(-step / cool_myr)

    for cell in range(int(world.mesh.num_cells)):
        if is_continental(world, cell):
            heat[cell] += (1.0 - float(heat[cell])) * warm
        else:
            heat[cell] -= float(heat[cell]) * cool
    return heat


def plume_heat_factor(world, plate) -> float:
    """Scale plume-triggered rifting by mean mantle heat under a continental plate."""
    config = world.config
    gain = float(config.get("riftHeatGain", 0.0))
    if not gain > 0:
        return 1.0
    total = 0.0
    count = 0
    for cell in range(len(world.owner)):
        if int(world.owner[cell]) == int(plate.id) and is_continental(world, cell):
            total += float(world.mantleHeat[cell])
            count += 1
    mean_heat = total / count if count else 0.0
    floor = float(config["riftHeatFloor"])
    power = float(config["riftHeatPower"])
    denominator = floor + gain
    if denominator <= 0:
        return 1.0
    return (floor + gain * mean_heat**power) / denominator


def pick_plume_site(world, rng) -> int:
    """Choose a plume origin weighted by ocean allowance plus retained mantle heat.

    A live plume excludes cells within plumeSpacingKm. The caller owns the
    random stream so plume selection remains deterministic and isolated from
    unrelated tectonic mechanisms.
    """
    mesh, config, mantle_heat = world.mesh, world.config, world.mantleHeat
    count = int(mesh.num_cells)
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] != count or len(mantle_heat) != count:
        raise ValueError("mesh coordinates and mantleHeat must match num_cells")
    ocean_weight = float(config["plumeOceanWeight"])
    if not 0.0 <= ocean_weight <= 1.0:
        raise ValueError("plumeOceanWeight must be between zero and one")
    near = math.cos(float(config["plumeSpacingKm"]) / 6371.0)
    live = [
        np.asarray(h["xyz"] if isinstance(h, dict) else h.xyz, dtype=np.float64)
        for h in getattr(world, "hotspots", ())
        if (h.get("alive", True) if isinstance(h, dict) else getattr(h, "alive", True))
    ]
    weights = np.zeros(count, dtype=np.float32)
    total = 0.0
    for cell in range(count):
        position = xyz[cell]
        if any(float(np.dot(position, plume)) > near for plume in live):
            continue
        weight = ocean_weight + (1.0 - ocean_weight) * float(mantle_heat[cell])
        weights[cell] = weight
        total += float(weight)
    if total <= 0.0:
        return -1
    draw = float(rng()) * total
    for cell in range(count):
        draw -= float(weights[cell])
        if draw <= 0.0 and weights[cell] > 0:
            return cell
    return -1
