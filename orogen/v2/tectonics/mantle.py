"""Mantle-flow traction and supercontinent dispersal field from V2.

The cycle tracker (landmass detection, heating and phase transitions) is a
separate stage. This module evaluates an already-computed cycle state.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..core.sim_mesh import PLANET_RADIUS_KM
from .crust import CrustType


_PAIR = 65536


def flow_of_cycle(cycle, config: Mapping, x: float, y: float, z: float, out=None):
    """Evaluate mantle traction at a unit-sphere position for a supplied cycle.

    The vector is tangent to the sphere and expressed per 10^4 km² of
    continental crust. Outflow leaves the last upwelling; inflow points toward
    the current downwelling or, during the assembled hold, toward the upwelling.
    """
    if out is None:
        out = [0.0, 0.0, 0.0]
    out[0] = out[1] = out[2] = 0.0
    if cycle is None or cycle.get("up") is None:
        return out

    position = (float(x), float(y), float(z))

    def add(target, strength):
        dot = (
            float(target[0]) * position[0]
            + float(target[1]) * position[1]
            + float(target[2]) * position[2]
        )
        tx = float(target[0]) - dot * position[0]
        ty = float(target[1]) - dot * position[1]
        tz = float(target[2]) - dot * position[2]
        length = math.hypot(tx, ty, tz)
        if length < 1e-9:
            return dot
        out[0] += float(strength) * tx / length
        out[1] += float(strength) * ty / length
        out[2] += float(strength) * tz / length
        return dot

    up = cycle["up"]
    cosine_up = float(up[0]) * position[0] + float(up[1]) * position[1] + float(up[2]) * position[2]
    sine_up = math.sqrt(max(0.0, 1.0 - cosine_up * cosine_up))
    outflow = float(config.get("mantleOutflow", 0.0))
    if cosine_up > 0 and outflow > 0:
        add(up, -outflow * float(cycle.get("upHeat", 0.0)) * 2.0 * sine_up * cosine_up)

    inflow = float(config.get("mantleInflow", 0.0))
    pack_km = float(config.get("mantleInflowPackKm", 0.0))

    def pull(target, strength):
        target_dot = (
            float(target[0]) * position[0]
            + float(target[1]) * position[1]
            + float(target[2]) * position[2]
        )
        distance = math.acos(max(-1.0, min(1.0, target_dot))) * PLANET_RADIUS_KM
        attenuation = min(1.0, distance / pack_km) if pack_km > 0 else 1.0
        add(target, strength * attenuation)

    if inflow > 0 and float(cycle.get("gather", 0.0)) > 0 and cycle.get("down") is not None:
        pull(cycle["down"], inflow * float(cycle["gather"]))
    if (
        inflow > 0
        and float(cycle.get("hold", 0.0)) > 0
        and math.acos(max(-1.0, min(1.0, cosine_up))) * PLANET_RADIUS_KM
        <= float(cycle.get("holdKm", 0.0))
    ):
        pull(up, inflow * float(cycle["hold"]))
    return out


def mantle_flow_at(world, x: float, y: float, z: float, out=None):
    """Evaluate the mantle traction using the world's cycle and config."""
    if out is None:
        out = [0.0, 0.0, 0.0]
    return flow_of_cycle(getattr(world, "cycle", None), world.config, x, y, z, out)


def _is_continental(world, cell: int) -> bool:
    owner = int(world.owner[cell])
    if owner < 0:
        return False
    plate = world.plates[owner]
    kind = int(plate.crust.type[int(world.ownerCell[cell])])
    return kind in (CrustType["CONTINENT"], CrustType["CRATON"])


def dispersal_field(world):
    """Build the per-continent repulsion field only while the cycle disperses."""
    config = world.config
    if (
        float(config.get("dispersalRepulsion", 0.0)) <= 0
        or getattr(world, "cycle", None) is None
        or world.cycle.get("phase") != "dispersing"
    ):
        return None

    mesh = world.mesh
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape(-1, 3)
    sums = {}
    for cell in range(int(mesh.num_cells)):
        if not _is_continental(world, cell):
            continue
        owner = int(world.owner[cell])
        acc = sums.setdefault(owner, [0.0, 0.0, 0.0, 0])
        acc[0] += float(xyz[cell, 0])
        acc[1] += float(xyz[cell, 1])
        acc[2] += float(xyz[cell, 2])
        acc[3] += 1

    total = sum(record[3] for record in sums.values())
    centres = []
    cell_area = float(mesh.cell_area_km2)
    reach = float(config["dispersalRepulsionKm"]) / PLANET_RADIUS_KM
    for plate_id, record in sums.items():
        count = record[3]
        plate = world.plates[plate_id]
        if count * cell_area < 1e6 or plate.isBackground:
            continue
        length = math.hypot(record[0], record[1], record[2]) or 1.0
        radius = math.sqrt(count * cell_area / math.pi) / PLANET_RADIUS_KM
        far = radius + reach
        cutoff = math.cos(far) - 1e-9 if far < math.pi else -3.0
        centres.append({
            "id": plate_id,
            "c": [record[0] / length, record[1] / length, record[2] / length],
            "share": count / total if total else 0.0,
            "radius": radius,
            "cutoff": cutoff,
        })

    touching = set()
    for cell in range(int(mesh.num_cells)):
        if int(world.landClaims[cell]) > 1 and int(world.collider[cell]) >= 0:
            owner = int(world.owner[cell])
            other = int(world.collider[cell])
            touching.add(owner * _PAIR + other)
            touching.add(other * _PAIR + owner)

    return {
        "centres": centres,
        "touching": touching,
        "k": float(config["dispersalRepulsion"]),
        "reach": reach,
    }


def repel_at(field, plate_id: int, x: float, y: float, z: float, out=None):
    """Accumulate the push away from nearby large continental plates."""
    if out is None:
        out = [0.0, 0.0, 0.0]
    if field is None or float(field["reach"]) <= 0:
        return out
    position = (float(x), float(y), float(z))
    for centre in field["centres"]:
        other_id = int(centre["id"])
        if other_id == int(plate_id):
            continue
        c = centre["c"]
        dot = c[0] * position[0] + c[1] * position[1] + c[2] * position[2]
        if dot < float(centre["cutoff"]):
            continue
        if int(plate_id) * _PAIR + other_id in field["touching"]:
            continue
        gap = max(
            0.0,
            math.acos(max(-1.0, min(1.0, dot))) - float(centre["radius"]),
        )
        reach = float(field["reach"])
        if gap >= reach:
            continue
        tx = dot * position[0] - c[0]
        ty = dot * position[1] - c[1]
        tz = dot * position[2] - c[2]
        length = math.hypot(tx, ty, tz)
        if length < 1e-9:
            continue
        strength = (
            float(field["k"])
            * min(1.0, 2.0 * float(centre["share"]))
            * (1.0 - gap / reach) ** 2
            / length
        )
        out[0] += strength * tx
        out[1] += strength * ty
        out[2] += strength * tz
    return out
