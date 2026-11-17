"""Partial boundary-force gathering ported from V2 step/forces.js.

Gathers slab pull, slab suction, young-ridge push and young-rift push from a
resolved owner map. Collision-overlap contacts and mantle flow are excluded.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .contact import contact_at
from .crust import CrustFlag, CrustType


FORCE_KINDS = ("slab pull", "slab suction", "ridge push", "rift push")
_PULL, _SUCTION, _RIDGE, _RIFT = range(4)


def _new_load() -> dict:
    zero = lambda: [0.0, 0.0, 0.0]
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


def gather_boundary_loads(world) -> dict[int, dict]:
    """Gather four local boundary-force families for living non-background plates.

    Output records can be passed to balanced_rotation. Requires a resolved
    owner/ownerCell map and returns contacts empty by design.
    """
    mesh = world.mesh
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

    for world_cell in range(n_cells):
        owner_id = int(world.owner[world_cell])
        if owner_id < 0:
            continue
        load = loads.get(owner_id)
        plate = plates_by_id.get(owner_id)
        if load is None or plate is None:
            continue
        if int(world.landClaims[world_cell]) > 1 and int(world.collider[world_cell]) >= 0:
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

        kind = -1
        size = 0.0
        direction = normal_out[:3].copy()
        if trench_b and ocean_a and not trench_a:
            kind = _PULL
            age = math.sqrt(min(1.0, max(0.0, float(world.time) - float(crust_a.formedAt[cell_a])) / float(world.config["slabPullAgeRefMyr"])))
            maturity = min(1.0, float(crust_b.slabKm[cell_b]) / float(world.config["slabMatureKm"]))
            size = float(world.config["slabPull"]) * age * maturity
            load["slabs"].append({
                "r": xyz[world_cell].tolist(),
                "n": direction.tolist(),
                "other": other_id,
                "length": length,
            })
        elif trench_a and ocean_b and not trench_b:
            kind = _SUCTION
            age = math.sqrt(min(1.0, max(0.0, float(world.time) - float(crust_b.formedAt[cell_b])) / float(world.config["slabPullAgeRefMyr"])))
            maturity = min(1.0, float(crust_a.slabKm[cell_a]) / float(world.config["slabMatureKm"]))
            size = float(world.config["slabSuction"]) * age * maturity
        else:
            rift_age = _rift_age(plate, crust_a, cell_a, other, crust_b, cell_b, float(world.time))
            if rift_age < float(world.config["riftPushMyr"]):
                kind = _RIFT
                size = -float(world.config["riftPush"]) * (1.0 - rift_age / float(world.config["riftPushMyr"]))
                if plate.riftPushDir is not None:
                    direction = plate.to_world(*map(float, plate.riftPushDir))
                    size = abs(size)
            elif (
                ocean_a and ocean_b
                and float(crust_a.formedAt[cell_a]) >= 0
                and float(crust_b.formedAt[cell_b]) >= 0
                and float(world.time) - float(crust_a.formedAt[cell_a]) < float(world.config["ridgeMaxAgeMyr"])
                and float(world.time) - float(crust_b.formedAt[cell_b]) < float(world.config["ridgeMaxAgeMyr"])
            ):
                kind = _RIDGE
                size = -float(world.config["ridgePush"])
        if kind < 0 or size == 0:
            continue

        force = size * length * direction
        torque = np.cross(xyz[world_cell], force)
        load["torque"] = (np.asarray(load["torque"]) + torque).tolist()
        load["byKind"][kind] = (np.asarray(load["byKind"][kind]) + torque).tolist()
        load["forceByKind"][kind] = (np.asarray(load["forceByKind"][kind]) + force).tolist()
        load["torqueMagnitudeSum"] += float(np.linalg.norm(torque))
        load["cells"][kind] += 1

    return loads
