"""Boundary-motion primitives ported from Orogen V2 step/boundaries.js.

This module classifies contacts and maintains trench records. When the matching
upstream configuration group is supplied, it also records strongly coupled
subduction margins as sources for the far-field deformation stage. Local
Andean/Laramide belt spreading remains a separate, unported stage.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh_bfs import reaches_within
from .config import CM_PER_YR_TO_KM_PER_MYR, CM_PER_YR_TO_RAD_PER_MYR
from .contact import contact_at
from .crust import CrustFlag, CrustType
from .far_field import feed_far_field
from .orogeny import belt_shape
from .polarity import overrides
from .world import BoundaryKind, EventKind


class Motion:
    LOCKED = 0
    OPENING = 1
    CLOSING = 2
    SLIDING = 3


def classify_motion(
    closing: float, shear: float, active_speed: float, oblique_limit: float
) -> int:
    """Classify relative contact motion from normal closing rate and shear."""
    normal_dominates = abs(closing) > oblique_limit * shear
    if closing < -active_speed and normal_dominates:
        return Motion.OPENING
    if closing > active_speed and normal_dominates:
        return Motion.CLOSING
    return Motion.SLIDING if shear > active_speed else Motion.LOCKED


def vote_segments(world, contacts: list[dict]) -> None:
    """Vote one subduction polarity per connected ocean-ocean contact segment."""
    mesh, overrider = world.mesh, world.oceanOverrider
    overrider.fill(-1)
    if not contacts:
        return
    by_pair: dict[str, list[dict]] = {}
    for contact in contacts:
        by_pair.setdefault(contact["key"], []).append(contact)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    for group in by_pair.values():
        at = {contact["w"]: contact for contact in group}
        seen: set[int] = set()
        for start in group:
            if start["w"] in seen:
                continue
            run = [start]
            seen.add(start["w"])
            index = 0
            while index < len(run):
                current = run[index]
                index += 1
                for k in range(
                    int(offsets[current["w"]]), int(offsets[current["w"] + 1])
                ):
                    neighbour = at.get(int(adjacency[k]))
                    if neighbour is not None and neighbour["w"] not in seen:
                        seen.add(neighbour["w"])
                        run.append(neighbour)
            score = sum(c["weight"] if c["prefersLower"] else -c["weight"] for c in run)
            if score == 0:
                winner = (
                    run[0]["lowerId"] if run[0]["prefersLower"] else run[0]["upperId"]
                )
            else:
                winner = run[0]["lowerId"] if score > 0 else run[0]["upperId"]
            for contact in run:
                overrider[contact["w"]] = winner


def near_land(plate, mesh, cell: int, config: dict) -> bool:
    """Whether non-ocean crust is within the subduction apron reach."""
    hops = round((config["apronKm"] + 2 * mesh.edge_km) / mesh.edge_km)
    crust = plate.crust
    return reaches_within(
        mesh,
        cell,
        hops,
        lambda c: bool(crust.mask[c]),
        lambda c: int(crust.type[c]) != CrustType["OCEAN"],
    )


def feed_trench(world, world_cell: int, crust, cell: int, subducted_km: float) -> bool:
    """Establish/extend a trench only when this cell or a neighbour consumed crust."""
    mesh = world.mesh
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    if not (int(crust.flags[cell]) & CrustFlag["TRENCH"]):
        fed = bool(world.consumedAt[world_cell])
        for k in range(int(offsets[world_cell]), int(offsets[world_cell + 1])):
            if fed:
                break
            fed = bool(world.consumedAt[int(adjacency[k])])
        if not fed:
            return False
        crust.flags[cell] |= CrustFlag["TRENCH"]
    crust.record_subduction(cell, subducted_km)
    return subducted_km > 0


_SUBDUCTION_FAR_FIELD_CONFIG = frozenset(
    {
        "apronKm",
        "orogenyQuietMyr",
        "flatSlabMaxAgeMyr",
        "flatSlabAdvanceCmYr",
        "flatSlabOnShare",
        "flatSlabOffShare",
        "andeanWorkShare",
        "beltObliqueMin",
        "farFieldEveryMyr",
        "farReachKm",
        "farFrontRefKm",
        "farMaxKm",
        "farCratonCost",
        "farWeakCost",
        "farCratonTake",
        "farWeakTake",
        "farFoldMyr",
        "farAndeanShare",
        "farAndeanRateCmYr",
        "farFlatShare",
    }
)


def _feed_subduction_far_field(world, fronts: dict, dt: float) -> None:
    """Feed active overriding margins into the far-field deformation clock.

    The flat-slab share and quiet-gap reset follow V2 step/boundaries.js.
    This records far-field sources only; local Andean/Laramide belt spreading
    is deliberately not performed by this partial boundary-classification pass.
    """
    if not fronts:
        return

    config, mesh = world.config, world.mesh
    rate_reference = float(config["farAndeanRateCmYr"])
    if rate_reference <= 0.0:
        raise ValueError("farAndeanRateCmYr must be positive")

    for plate, front in sorted(fronts.items(), key=lambda item: int(item[0].id)):
        if float(world.time) - float(plate.lastFedAt) > float(
            config["orogenyQuietMyr"]
        ):
            plate.flatSlab = False
            plate.flatSlabKm = 0.0
        plate.lastFedAt = float(world.time)

        source_indices = [
            index
            for index, cell in enumerate(front["cells"])
            if int(plate.crust.type[cell]) != CrustType["OCEAN"]
            or near_land(plate, mesh, cell, config)
        ]
        andean_cells = [front["cells"][index] for index in source_indices]
        if not andean_cells or dt <= 0.0:
            if not andean_cells:
                plate.flatSlabKm = 0.0
            continue

        flat_share = float(front["flat"]) / len(front["cells"])
        flat = (
            flat_share > float(config["flatSlabOffShare"])
            if plate.flatSlab
            else flat_share > float(config["flatSlabOnShare"])
        )
        if flat != plate.flatSlab:
            plate.flatSlab = flat
            if flat:
                world.log(
                    EventKind["SLAB_FLAT"],
                    f"the slab going down beneath {plate.name} is young and it "
                    "drives at it: the slab flattens, deformation reaches far "
                    "inland (a Laramide belt) and the arc volcanoes fall silent",
                    {"plate": plate.id},
                )
            else:
                world.log(
                    EventKind["SLAB_STEEP"],
                    f"the slab beneath {plate.name} steepens again: the belt "
                    "narrows to an Andean one",
                    {"plate": plate.id},
                )

        plate.flatSlabKm = (
            float(plate.flatSlabKm) + float(front["km"]) / len(front["cells"])
            if flat
            else 0.0
        )

        pushes = []
        for index in source_indices:
            km = float(front["kms"][index])
            rate_cm_yr = km / float(dt) / CM_PER_YR_TO_KM_PER_MYR
            work = (
                km
                * float(config["andeanWorkShare"])
                * belt_shape(float(front["headOn"][index]), None, config)
            )
            share = (
                float(config["farFlatShare"])
                if flat
                else float(config["farAndeanShare"])
                * min(1.0, (rate_cm_yr / rate_reference) ** 2)
            )
            pushes.append(work * share)

        feed_far_field(
            world,
            plate,
            f"s{plate.id}",
            andean_cells,
            pushes,
            len(andean_cells) * float(mesh.edge_km) * 0.87,
            1.0,
        )


def classify_contact_boundaries(world, dt: float) -> int:
    """Classify contacts, update trench state, and collect far-field sources.

    With the complete far-field/flat-slab configuration group, overriding
    contacts also feed subduction work into the far-field stage. Local Andean
    and Laramide belt spreading remains outside this partial pass.
    """
    mesh = world.mesh
    owner, owner_cell, plates = world.owner, world.ownerCell, world.plates
    boundary, convergence = world.boundaryKind, world.convergence
    boundary.fill(BoundaryKind["NONE"])
    convergence.fill(0)
    normal = [0.0, 0.0, 0.0, 0.0]
    contacts = []
    fed: dict[object, dict] = {}
    subduction_far_field = _SUBDUCTION_FAR_FIELD_CONFIG.issubset(world.config)
    classified = 0
    xyz = mesh.xyz.reshape(-1, 3)
    active = float(world.config["boundaryActiveCmYr"]) * CM_PER_YR_TO_RAD_PER_MYR
    oblique = float(world.config["obliqueLimit"])
    for w in range(mesh.num_cells):
        a_id = int(owner[w])
        if a_id < 0:
            continue
        if int(world.landClaims[w]) > 1:
            boundary[w] = BoundaryKind["COLLISION"]
            classified += 1
            continue
        plate_a = plates[a_id]
        cell_a = int(owner_cell[w])
        real = not bool(world.provisional[w])
        other_w = contact_at(world, w, normal)
        if other_w < 0:
            if real and int(plate_a.crust.type[cell_a]) == CrustType["OCEAN"]:
                plate_a.crust.flags[cell_a] &= np.uint8(0xFF ^ CrustFlag["TRENCH"])
            continue
        b_id = int(owner[other_w])
        if b_id < 0 or b_id == a_id:
            continue
        plate_b = plates[b_id]
        cell_b = int(owner_cell[other_w])
        px, py, pz = map(float, xyz[w])
        ox = float(plate_a.omega[0]) - float(plate_b.omega[0])
        oy = float(plate_a.omega[1]) - float(plate_b.omega[1])
        oz = float(plate_a.omega[2]) - float(plate_b.omega[2])
        vx, vy, vz = oy * pz - oz * py, oz * px - ox * pz, ox * py - oy * px
        closing = vx * normal[0] + vy * normal[1] + vz * normal[2]
        shear = (
            float(
                (vx - closing * normal[0]) ** 2
                + (vy - closing * normal[1]) ** 2
                + (vz - closing * normal[2]) ** 2
            )
            ** 0.5
        )
        convergence[w] = closing / CM_PER_YR_TO_RAD_PER_MYR
        motion = classify_motion(closing, shear, active, oblique)
        if motion == Motion.OPENING:
            boundary[w] = BoundaryKind["RIDGE"]
        elif motion == Motion.SLIDING:
            boundary[w] = BoundaryKind["TRANSFORM"]
        elif motion == Motion.CLOSING:
            a_overrides = overrides(
                plate_a, cell_a, plate_b, cell_b, int(world.oceanOverrider[w])
            )
            if (
                int(plate_a.crust.type[cell_a]) == CrustType["OCEAN"]
                and int(plate_b.crust.type[cell_b]) == CrustType["OCEAN"]
            ):
                lower, upper = (plate_a, plate_b) if a_id < b_id else (plate_b, plate_a)
                lower_cell, upper_cell = (
                    (cell_a, cell_b) if a_id < b_id else (cell_b, cell_a)
                )
                trench = bool(
                    (
                        int(plate_a.crust.flags[cell_a])
                        | int(plate_b.crust.flags[cell_b])
                    )
                    & CrustFlag["TRENCH"]
                )
                contacts.append(
                    {
                        "w": w,
                        "key": f"{lower.id}:{upper.id}",
                        "lowerId": lower.id,
                        "upperId": upper.id,
                        "prefersLower": overrides(lower, lower_cell, upper, upper_cell)
                        is True,
                        "weight": float(world.config.get("segmentTrenchWeight", 1.0))
                        if trench
                        else 1.0,
                    }
                )
            if a_overrides is None:
                boundary[w] = BoundaryKind["COLLISION"]
            elif a_overrides:
                boundary[w] = BoundaryKind["OVERRIDING"]
                if real:
                    subducted_km = closing * dt * 6371.0
                    if (
                        feed_trench(world, w, plate_a.crust, cell_a, subducted_km)
                        and subduction_far_field
                    ):
                        front = fed.setdefault(
                            plate_a,
                            {
                                "cells": [],
                                "kms": [],
                                "headOn": [],
                                "flat": 0,
                                "km": 0.0,
                            },
                        )
                        front["cells"].append(cell_a)
                        front["kms"].append(subducted_km)
                        front["headOn"].append(
                            closing / float(np.hypot(closing, shear))
                        )
                        front["km"] += subducted_km

                        slab_age = float(world.time) - float(
                            plate_b.crust.formedAt[cell_b]
                        )
                        wx, wy, wz = map(float, plate_a.omega)
                        advance = (
                            (wy * pz - wz * py) * normal[0]
                            + (wz * px - wx * pz) * normal[1]
                            + (wx * py - wy * px) * normal[2]
                        ) / CM_PER_YR_TO_RAD_PER_MYR
                        if slab_age < float(
                            world.config["flatSlabMaxAgeMyr"]
                        ) and advance > float(world.config["flatSlabAdvanceCmYr"]):
                            front["flat"] += 1
            else:
                boundary[w] = BoundaryKind["SUBDUCTING"]
                if (
                    real
                    and int(plate_a.crust.type[cell_a]) == CrustType["OCEAN"]
                    and int(plate_b.crust.type[cell_b]) != CrustType["OCEAN"]
                ):
                    plate_a.crust.flags[cell_a] &= np.uint8(0xFF ^ CrustFlag["TRENCH"])
        if boundary[w] != BoundaryKind["NONE"]:
            classified += 1
    if world.config.get("segmentPolarity", False):
        vote_segments(world, contacts)
    else:
        world.oceanOverrider.fill(-1)
    if subduction_far_field:
        _feed_subduction_far_field(world, fed, float(dt))
    return classified
