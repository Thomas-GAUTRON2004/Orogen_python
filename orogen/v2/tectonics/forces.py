"""Partial boundary-force gathering ported from V2 step/forces.js.

Gathers slab pull, slab suction, young-ridge push and young-rift push from a
resolved owner map. Collision-overlap contacts and mantle flow are excluded.
"""
# ruff: noqa: I001 - preserve the module's established import grouping.
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..core.mesh_rings import rings_of
from .contact import contact_at
from .crust import CrustFlag, CrustType
from .mantle import dispersal_field, mantle_flow_at, repel_at


FORCE_KINDS = ("slab pull", "slab suction", "ridge push", "rift push", "mantle flow")
_PULL, _SUCTION, _RIDGE, _RIFT = range(4)


def _new_load() -> dict:
    def zero() -> list[float]:
        return [0.0, 0.0, 0.0]
    return {
        "torque": zero(),
        "flowTorque": zero(),
        "byKind": [zero() for _ in FORCE_KINDS],
        "forceByKind": [zero() for _ in FORCE_KINDS],
        "torqueMagnitudeSum": 0.0,
        "cells": [0 for _ in FORCE_KINDS],
        "slabs": [],
        "contacts": [],
    }


def _rift_age(plate_a, crust_a, cell_a: int, plate_b, crust_b, cell_b: int, time: float) -> float:
    born = max(float(plate_a.born), float(plate_b.born))
    if plate_a.riftPartner != plate_b.id or plate_b.riftPartner != plate_a.id:
        return math.inf
    split = plate_a.oceanSplitWith == plate_b.id and plate_b.oceanSplitWith == plate_a.id
    margin_a = split or bool(int(crust_a.flags[cell_a]) & CrustFlag["RIFT_MARGIN"])
    margin_b = split or bool(int(crust_b.flags[cell_b]) & CrustFlag["RIFT_MARGIN"])
    young_a = int(crust_a.type[cell_a]) == CrustType["OCEAN"] and float(crust_a.formedAt[cell_a]) >= born
    young_b = int(crust_b.type[cell_b]) == CrustType["OCEAN"] and float(crust_b.formedAt[cell_b]) >= born
    return float(time) - born if (margin_a or young_a) and (margin_b or young_b) else math.inf


def _are_young_rift_partners(plate_a, plate_b, time: float, config: Mapping) -> bool:
    return (
        plate_a.riftPartner == plate_b.id
        and plate_b.riftPartner == plate_a.id
        and float(time) - max(float(plate_a.born), float(plate_b.born))
        < float(config["riftPushMyr"])
    )


def _front_normal(world, cell: int, plate_a_id: int, plate_b_id: int, position, plates_by_id) -> list[float]:
    """Estimate the collision-front normal from nearby uncontested land."""
    mesh = world.mesh
    ring_data = rings_of(mesh)
    start, cells, stride = ring_data["start"], ring_data["cells"], ring_data["stride"]
    vx = vy = vz = 0.0
    first = int(start[cell * stride + 1])
    stop = int(start[cell * stride + stride - 1])
    for offset in range(first, stop):
        neighbour = int(cells[offset])
        owner_id = int(world.owner[neighbour])
        if owner_id not in (plate_a_id, plate_b_id):
            continue
        if int(world.landClaims[neighbour]) > 1:
            continue
        plate = plates_by_id[owner_id]
        local_cell = int(world.ownerCell[neighbour])
        if int(plate.crust.type[local_cell]) == CrustType["OCEAN"]:
            continue
        sign = 1.0 if owner_id == plate_b_id else -1.0
        vx += sign * float(mesh.xyz[3 * neighbour])
        vy += sign * float(mesh.xyz[3 * neighbour + 1])
        vz += sign * float(mesh.xyz[3 * neighbour + 2])

    r = np.asarray(position, dtype=np.float64)
    vector = np.array([vx, vy, vz], dtype=np.float64)
    vector -= float(np.dot(vector, r)) * r
    length = float(np.linalg.norm(vector))
    if length > 1e-9:
        return (vector / length).tolist()

    plate_a, plate_b = plates_by_id[plate_a_id], plates_by_id[plate_b_id]
    centre_a = plate_a.to_world(*map(float, plate_a.capCenter))
    centre_b = plate_b.to_world(*map(float, plate_b.capCenter))
    vector = np.asarray(centre_b) - np.asarray(centre_a)
    vector -= float(np.dot(vector, r)) * r
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    return (vector / length).tolist()



def _slab_force(
    crust_a,
    cell_a: int,
    crust_b,
    cell_b: int,
    ocean_a: bool,
    ocean_b: bool,
    trench_a: bool,
    trench_b: bool,
    time: float,
    config: Mapping,
) -> tuple[int, float] | None:
    """Return the slab pull/suction category and magnitude when applicable."""
    if trench_b and ocean_a and not trench_a:
        age = math.sqrt(
            min(
                1.0,
                max(0.0, time - float(crust_a.formedAt[cell_a]))
                / float(config["slabPullAgeRefMyr"]),
            )
        )
        maturity = min(
            1.0, float(crust_b.slabKm[cell_b]) / float(config["slabMatureKm"])
        )
        return _PULL, float(config["slabPull"]) * age * maturity

    if trench_a and ocean_b and not trench_b:
        age = math.sqrt(
            min(
                1.0,
                max(0.0, time - float(crust_b.formedAt[cell_b]))
                / float(config["slabPullAgeRefMyr"]),
            )
        )
        maturity = min(
            1.0, float(crust_a.slabKm[cell_a]) / float(config["slabMatureKm"])
        )
        return _SUCTION, float(config["slabSuction"]) * age * maturity
    return None


def _rift_or_ridge_force(
    plate,
    other,
    crust_a,
    cell_a: int,
    crust_b,
    cell_b: int,
    ocean_a: bool,
    ocean_b: bool,
    direction: np.ndarray,
    time: float,
    config: Mapping,
) -> tuple[int, float, np.ndarray]:
    """Classify a non-slab contact as young-rift or young-ridge push."""
    rift_age = _rift_age(
        plate, crust_a, cell_a, other, crust_b, cell_b, time
    )
    if rift_age < float(config["riftPushMyr"]):
        size = -float(config["riftPush"]) * (
            1.0 - rift_age / float(config["riftPushMyr"])
        )
        if plate.riftPushDir is not None:
            direction = np.asarray(
                plate.to_world(*map(float, plate.riftPushDir)),
                dtype=np.float64,
            )
            size = abs(size)
        return _RIFT, size, direction

    if (
        ocean_a
        and ocean_b
        and float(crust_a.formedAt[cell_a]) >= 0
        and float(crust_b.formedAt[cell_b]) >= 0
        and time - float(crust_a.formedAt[cell_a]) < float(config["ridgeMaxAgeMyr"])
        and time - float(crust_b.formedAt[cell_b]) < float(config["ridgeMaxAgeMyr"])
    ):
        return _RIDGE, -float(config["ridgePush"]), direction
    return -1, 0.0, direction


def _accumulate_boundary_force(
    load: dict,
    kind: int,
    size: float,
    length: float,
    direction: np.ndarray,
    position: np.ndarray,
) -> None:
    """Accumulate one classified force into its torque and force channels."""
    force = size * length * direction
    torque = np.cross(position, force)
    load["torque"] = (np.asarray(load["torque"]) + torque).tolist()
    load["byKind"][kind] = (np.asarray(load["byKind"][kind]) + torque).tolist()
    load["forceByKind"][kind] = (
        np.asarray(load["forceByKind"][kind]) + force
    ).tolist()
    load["torqueMagnitudeSum"] += float(np.linalg.norm(torque))
    load["cells"][kind] += 1

def gather_boundary_loads(world) -> dict[int, dict]:
    """Gather four local boundary-force families for living non-background plates.

    Output records can be passed to balanced_rotation. Requires a resolved
    owner/ownerCell map and includes slab, collision-lock and boundary-force records.
    """
    mesh = world.mesh
    config = world.config
    n_cells = int(mesh.num_cells)
    if len(world.owner) != n_cells or len(world.ownerCell) != n_cells:
        raise ValueError("world owner arrays must match the simulation mesh")
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] != n_cells:
        raise ValueError("mesh xyz must contain one 3-vector per simulation cell")
    edge_length = float(mesh.edge_km) / math.sqrt(3.0) / 100.0
    loads = {
        int(plate.id): _new_load()
        for plate in world.plates
        if plate.alive and not plate.isBackground and plate.crust.count > 0
    }
    plates_by_id = {int(plate.id): plate for plate in world.plates}
    normal_out = np.zeros(4, dtype=np.float64)
    cycle = getattr(world, "cycle", None)
    flowing = bool(
        cycle is not None and cycle.get("up") is not None
        and (float(config.get("mantleOutflow", 0.0)) > 0
             or float(config.get("mantleInflow", 0.0)) > 0
             or float(config.get("dispersalRepulsion", 0.0)) > 0)
    )
    apart = dispersal_field(world) if flowing else None
    flow = np.zeros(3, dtype=np.float64)
    flow_area = float(mesh.cell_area_km2) / 1e4

    for world_cell in range(n_cells):
        owner_id = int(world.owner[world_cell])
        if owner_id < 0:
            continue
        plate = plates_by_id.get(owner_id)
        if plate is None:
            continue
        load = loads.get(owner_id)
        if flowing and load is not None:
            cell = int(world.ownerCell[world_cell])
            if int(plate.crust.type[cell]) in (CrustType["CONTINENT"], CrustType["CRATON"]):
                position = xyz[world_cell]
                mantle_flow_at(world, *map(float, position), flow)
                if apart is not None:
                    repel_at(apart, owner_id, *map(float, position), flow)
                force = flow * flow_area
                torque = np.cross(position, force)
                load["torque"] = (np.asarray(load["torque"]) + torque).tolist()
                load["flowTorque"] = (np.asarray(load["flowTorque"]) + torque).tolist()
                load["byKind"][4] = (np.asarray(load["byKind"][4]) + torque).tolist()
                load["forceByKind"][4] = (np.asarray(load["forceByKind"][4]) + force).tolist()
                load["cells"][4] += 1
        if int(world.landClaims[world_cell]) > 1 and int(world.collider[world_cell]) >= 0:
            other_id = int(world.collider[world_cell])
            other = plates_by_id.get(other_id)
            if other is None:
                continue
            r0, r1, r2 = map(float, xyz[world_cell])
            other_cell = int(other.cell_at(mesh.index, r0, r1, r2))
            crust_a, crust_b = plate.crust, other.crust
            flags_a = int(crust_a.flags[int(world.ownerCell[world_cell])])
            flags_b = int(other.crust.flags[other_cell])
            if (
                _are_young_rift_partners(plate, other, float(world.time), config)
                and flags_a & CrustFlag["RIFT_MARGIN"]
                and flags_b & CrustFlag["RIFT_MARGIN"]
            ):
                continue
            cell_a = int(world.ownerCell[world_cell])
            if (
                int(crust_a.type[cell_a]) == CrustType["ARC"]
                or int(crust_b.type[other_cell]) == CrustType["ARC"]
            ):
                continue
            craton = (
                int(crust_a.type[cell_a]) == CrustType["CRATON"]
                or int(crust_b.type[other_cell]) == CrustType["CRATON"]
            )
            weight = float(config["cratonLockFactor"]) if craton else 1.0
            position = xyz[world_cell]
            normal = _front_normal(
                world, world_cell, owner_id, other_id, position, plates_by_id
            )
            load = loads.get(owner_id)
            if load is not None:
                load["contacts"].append({
                    "r": position.tolist(), "n": normal, "other": other_id,
                    "weight": weight,
                })
            other_load = loads.get(other_id)
            if other_load is not None:
                other_load["contacts"].append({
                    "r": position.tolist(),
                    "n": [-normal[0], -normal[1], -normal[2]],
                    "other": owner_id,
                    "weight": weight,
                })
            continue
        load = loads.get(owner_id)
        if load is None:
            continue
        across = contact_at(world, world_cell, normal_out)
        if across < 0:
            continue
        other_id = int(world.owner[across])
        other = plates_by_id.get(other_id)
        if other is None or other_id == owner_id:
            continue
        cell_a = int(world.ownerCell[world_cell])
        cell_b = int(world.ownerCell[across])
        crust_a, crust_b = plate.crust, other.crust
        ocean_a = int(crust_a.type[cell_a]) == CrustType["OCEAN"]
        ocean_b = int(crust_b.type[cell_b]) == CrustType["OCEAN"]
        trench_a = bool(int(crust_a.flags[cell_a]) & CrustFlag["TRENCH"])
        trench_b = bool(int(crust_b.flags[cell_b]) & CrustFlag["TRENCH"])
        length = edge_length * float(normal_out[3])
        if length <= 0:
            continue

        direction = normal_out[:3].copy()
        slab_force = _slab_force(
            crust_a,
            cell_a,
            crust_b,
            cell_b,
            ocean_a,
            ocean_b,
            trench_a,
            trench_b,
            float(world.time),
            config,
        )
        if slab_force is not None:
            kind, size = slab_force
            if kind == _PULL:
                load["slabs"].append({
                    "r": xyz[world_cell].tolist(),
                    "n": direction.tolist(),
                    "other": other_id,
                    "length": length,
                })
        else:
            kind, size, direction = _rift_or_ridge_force(
                plate,
                other,
                crust_a,
                cell_a,
                crust_b,
                cell_b,
                ocean_a,
                ocean_b,
                direction,
                float(world.time),
                config,
            )
        if kind < 0 or size == 0:
            continue

        _accumulate_boundary_force(
            load, kind, size, length, direction, xyz[world_cell]
        )

    return loads
