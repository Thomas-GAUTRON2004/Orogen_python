"""Subduction rollback and back-arc basin formation (V2 step/backarc.js)."""

from __future__ import annotations

import math
import re
from collections import deque

import numpy as np

from ..core.mesh_bfs import hops_from, largest_piece_of, pieces_of
from ..core.sphere_geometry import PLANET_RADIUS_KM
from .config import CM_PER_YR_TO_RAD_PER_MYR
from .contact import contact_at
from .crust import CrustFlag, CrustType
from .ops.plate_transfer import transfer_cells
from .plate import Plate
from .world import BoundaryKind, EventKind

_REJOIN_CONFIG = frozenset({"backArcActiveMyr", "backArcSeamKm"})
_BACKARC_CONFIG = frozenset(
    {
        "rollbackMinAgeMyr",
        "rollbackFullAgeMyr",
        "rollbackMaxCmYr",
        "backArcRatePerMyr",
        "backArcMinLengthKm",
        "backArcWidthKm",
        "backArcContinentalWidthKm",
        "backArcContinentalShare",
        "backArcMinLandKm2",
        "backArcActiveMyr",
        "backArcSeamKm",
        "backArcTenureMyr",
        "riftPushMyr",
    }
)


def rollback_cm_yr(slab_age_myr: float, config: dict) -> float:
    """Return slab-hinge rollback rate in cm/yr, capped at the configured maximum."""
    minimum = float(config["rollbackMinAgeMyr"])
    span = float(config["rollbackFullAgeMyr"]) - minimum
    if span <= 0.0:
        raise ValueError("rollbackFullAgeMyr must exceed rollbackMinAgeMyr")
    fraction = (float(slab_age_myr) - minimum) / span
    if fraction <= 0.0:
        return 0.0
    return float(config["rollbackMaxCmYr"]) * min(1.0, fraction)


def open_back_arcs(world, dt: float) -> int:
    """Open eligible rollback basins and rejoin strips whose spreading window ended.

    Topology-changing results are returned as a count so the stepper can rebuild
    owner/ownerCell before a later stage reads the world view.
    """
    config = world.config
    rejoined = _rejoin_finished_strips(world) if _REJOIN_CONFIG.issubset(config) else 0
    if not _BACKARC_CONFIG.issubset(config) or dt <= 0.0:
        return rejoined
    if float(config["backArcRatePerMyr"]) <= 0.0:
        return rejoined

    fronts = _rollback_fronts(world)
    opened = 0
    for front in fronts.values():
        if _try_open_back_arc(world, front, float(dt)):
            opened += 1
    return opened + rejoined


def _rollback_fronts(world) -> dict:
    """Group old rolling-back overriding contacts by the two plates they join."""
    mesh, config = world.mesh, world.config
    owner, owner_cell, plates = world.owner, world.ownerCell, world.plates
    normal = [0.0, 0.0, 0.0, 0.0]
    xyz = mesh.xyz.reshape(-1, 3)
    fronts: dict[tuple[int, int], dict] = {}

    for w in range(mesh.num_cells):
        if int(world.boundaryKind[w]) != BoundaryKind["OVERRIDING"]:
            continue
        a_id = int(owner[w])
        if a_id < 0:
            continue
        plate_a = plates[a_id]
        if not plate_a.alive or plate_a.isBackground or plate_a.reshaped:
            continue

        other_w = contact_at(world, w, normal)
        if other_w < 0:
            continue
        b_id = int(owner[other_w])
        if b_id < 0:
            continue
        plate_b = plates[b_id]
        cell_b = int(owner_cell[other_w])
        if (
            not plate_b.alive
            or plate_b.reshaped
            or plate_b.isBackground
            or not plate_b.crust.mask[cell_b]
            or int(plate_b.crust.type[cell_b]) != CrustType["OCEAN"]
        ):
            continue

        slab_age = float(world.time) - float(plate_b.crust.formedAt[cell_b])
        if slab_age < float(config["rollbackMinAgeMyr"]):
            continue
        position = xyz[w]
        wx, wy, wz = map(float, plate_a.omega)
        px, py, pz = map(float, position)
        advance = (
            (wy * pz - wz * py) * normal[0]
            + (wz * px - wx * pz) * normal[1]
            + (wx * py - wy * px) * normal[2]
        ) / CM_PER_YR_TO_RAD_PER_MYR
        if advance >= rollback_cm_yr(slab_age, config):
            continue

        key = (a_id, b_id)
        front = fronts.setdefault(
            key,
            {"plate": plate_a, "cells": [], "ages": 0.0, "direction": [0.0, 0.0, 0.0]},
        )
        front["cells"].append(w)
        front["ages"] += slab_age
        for axis in range(3):
            front["direction"][axis] += float(normal[axis])

    return fronts


def _try_open_back_arc(world, front: dict, dt: float) -> bool:
    """Validate one grouped margin and split its coherent strip when it qualifies."""
    plate, cells = front["plate"], front["cells"]
    if not plate.alive or plate.reshaped:
        return False
    config, mesh = world.config, world.mesh
    if float(world.time) - float(plate.lastBackArcAt) < float(
        config["backArcTenureMyr"]
    ):
        return False

    segment = largest_piece_of(mesh, cells)
    length_km = len(segment) * float(mesh.edge_km)
    minimum_km = float(config["backArcMinLengthKm"])
    if not segment or length_km < minimum_km:
        return False
    chance = float(config["backArcRatePerMyr"]) * dt * length_km / minimum_km
    if float(world.rngBackArc()) >= chance:
        return False

    strip = _strip_behind(world, plate, segment)
    if strip is None:
        return False
    if strip["continental"] and float(world.rngBackArc()) >= float(
        config["backArcContinentalShare"]
    ):
        return False

    mean_age = float(front["ages"]) / len(cells)
    fresh = _split_strip(
        world, plate, strip, front["direction"], rollback_cm_yr(mean_age, config)
    )
    kind = "a strip of its coast" if strip["continental"] else "its island arc"
    world.log(
        EventKind["BACKARC_OPEN"],
        (
            f"{plate.name}'s trench rolls back from under it ({mean_age:.0f} Myr old): "
            f"{kind} — {length_km:.0f} km of it — rifts away as {fresh.name}"
        ),
        {
            "parent": int(plate.id),
            "strip": int(fresh.id),
            "continental": bool(strip["continental"]),
            "slabAgeMyr": mean_age,
            "lengthKm": length_km,
        },
    )
    return True


def _strip_behind(world, plate: Plate, segment: list[int]) -> dict | None:
    """Find a viable island-arc or continental strip behind a rollback segment."""
    config = world.config
    in_strip = _band_behind(
        world, plate, segment, float(config["backArcContinentalWidthKm"])
    )
    has_land = any(
        in_strip[cell] and int(plate.crust.type[cell]) == CrustType["CONTINENT"]
        for cell in range(world.mesh.num_cells)
    )
    if not has_land:
        in_strip = _band_behind(world, plate, segment, float(config["backArcWidthKm"]))
    return _judge_strip(world, plate, in_strip)


def _band_behind(world, plate: Plate, segment: list[int], width_km: float):
    """Map the nearest trench-segment source over the plate's geodesic width band."""
    mesh, crust = world.mesh, plate.crust
    xyz = mesh.xyz.reshape(-1, 3)
    near = math.cos(float(width_km) / PLANET_RADIUS_KM)
    source = np.full(mesh.num_cells, -1, dtype=np.int32)
    best = np.full(mesh.num_cells, -2.0, dtype=np.float64)
    in_strip = np.zeros(mesh.num_cells, dtype=np.uint8)
    ring: deque[int] = deque()

    for world_cell in segment:
        cell = int(world.ownerCell[world_cell])
        if crust.mask[cell] and source[cell] < 0:
            source[cell] = cell
            best[cell] = 1.0
            in_strip[cell] = 1
            ring.append(cell)

    while ring:
        cell = ring.popleft()
        seed = int(source[cell])
        seed_xyz = xyz[seed]
        for edge in range(int(mesh.adj_offset[cell]), int(mesh.adj_offset[cell + 1])):
            neighbour = int(mesh.adj_list[edge])
            if not crust.mask[neighbour]:
                continue
            distance_cos = float(np.dot(xyz[neighbour], seed_xyz))
            if distance_cos < near or distance_cos <= best[neighbour]:
                continue
            best[neighbour] = distance_cos
            source[neighbour] = seed
            in_strip[neighbour] = 1
            ring.append(neighbour)

    return in_strip


def _judge_strip(world, plate: Plate, in_strip) -> dict | None:
    """Reject cratons, slivers, most-of-plate cuts, and detached remainders."""
    mesh, config, crust = world.mesh, world.config, plate.crust
    cells: list[int] = []
    land_cells: list[int] = []
    land = arc = 0
    for cell in range(mesh.num_cells):
        if not in_strip[cell]:
            continue
        crust_type = int(crust.type[cell])
        if crust_type == CrustType["CRATON"]:
            return None
        cells.append(cell)
        if crust_type == CrustType["CONTINENT"]:
            land += 1
            land_cells.append(cell)
        elif crust_type == CrustType["ARC"]:
            arc += 1
            land_cells.append(cell)

    if not cells or len(cells) * 2 > int(crust.count):
        return None
    continental = land > arc
    if continental:
        if land * float(mesh.cell_area_km2) < float(config["backArcMinLandKm2"]):
            return None
        if not land_cells or len(largest_piece_of(mesh, land_cells)) < len(land_cells):
            return None

    remaining = [
        cell
        for cell in range(mesh.num_cells)
        if crust.mask[cell] and not in_strip[cell]
    ]
    if len(largest_piece_of(mesh, remaining)) < 0.9 * len(remaining):
        return None
    return {"cells": cells, "continental": continental}


def _split_strip(
    world, parent: Plate, strip: dict, direction_world, rollback_rate_cm_yr: float
) -> Plate:
    """Move a selected strip to a new plate and mark the two rifted margins."""
    mesh, crust = world.mesh, parent.crust
    fresh = world.add_plate(_strip_name(world, parent, bool(strip["continental"])))
    fresh.set_quat(parent.quat[:])
    fresh.omega = list(parent.omega)

    cells = [int(cell) for cell in strip["cells"]]
    for cell in cells:
        fresh.crust.copy_from(cell, crust, cell)
        crust.remove(cell)
    for cell in cells:
        fresh.crust.flags[cell] |= CrustFlag["RIFT_MARGIN"]
        for edge in range(int(mesh.adj_offset[cell]), int(mesh.adj_offset[cell + 1])):
            neighbour = int(mesh.adj_list[edge])
            if crust.mask[neighbour]:
                crust.flags[neighbour] |= CrustFlag["RIFT_MARGIN"]

    direction = np.asarray(direction_world, dtype=np.float64)
    length = float(np.linalg.norm(direction)) or 1.0
    push_direction = fresh.to_plate_frame(*(direction / length))
    fresh.backArcPushDir = [float(value) for value in push_direction]

    partner_id = int(parent.riftPartner)
    partner = world.plates[partner_id] if 0 <= partner_id < len(world.plates) else None
    still_rifting = bool(
        partner is not None
        and partner.alive
        and int(partner.riftPartner) == int(parent.id)
        and float(world.time) - max(float(parent.born), float(partner.born))
        < float(world.config["riftPushMyr"])
    )
    if not still_rifting:
        if partner is not None and int(partner.riftPartner) == int(parent.id):
            partner.riftPartner = -1
            partner.riftPushDir = None
        fresh.riftPushDir = fresh.backArcPushDir[:]
        parent.riftPushDir = [-value for value in fresh.backArcPushDir]
        fresh.riftPartner = int(parent.id)
        parent.riftPartner = int(fresh.id)

    fresh.backArcOf = int(parent.id)
    fresh.rollbackCmYr = float(rollback_rate_cm_yr)
    fresh.born = float(world.time)
    fresh.lastBackArcAt = float(world.time)
    parent.lastBackArcAt = float(world.time)
    fresh.mark_reshaped()
    parent.mark_reshaped()
    fresh.refresh_extent(mesh.xyz)
    parent.refresh_extent(mesh.xyz)
    return fresh


def _strip_name(world, parent: Plate, continental: bool) -> str:
    """Name the fragment uniquely while retaining its originating landmass."""
    root = re.sub(r" (Arc|Isles)( \d+)?$", "", str(parent.name))
    suffix = "Isles" if continental else "Arc"
    base = f"{root} {suffix}"
    taken = {str(plate.name) for plate in world.plates if plate.alive}
    if base not in taken:
        return base
    number = 2
    while f"{base} {number}" in taken:
        number += 1
    return f"{base} {number}"


def _rejoin_finished_strips(world) -> int:
    """Rejoin fragments after their back-arc spreading interval has elapsed."""
    mesh, config = world.mesh, world.config
    closed = 0
    for strip in world.plates:
        if (
            not strip.alive
            or int(strip.backArcOf) < 0
            or float(world.time) - float(strip.born) < float(config["backArcActiveMyr"])
        ):
            continue

        parent_id = int(strip.backArcOf)
        strip.backArcOf = -1
        parent = world.plates[parent_id] if parent_id < len(world.plates) else None
        name = str(strip.name)
        if parent is None or not parent.alive:
            world.log(
                EventKind["BACKARC_CLOSE"],
                f"the back-arc sea behind {name} stops spreading; its parent is gone",
                {"strip": int(strip.id), "parent": -1, "rejoined": False},
            )
            continue

        cells = np.flatnonzero(strip.crust.mask).tolist()
        transfer_cells(world, strip, parent, cells)
        filled = _bridge_pieces(parent, mesh, float(config["backArcSeamKm"]))
        strip.alive = False
        strip.omega = [0.0, 0.0, 0.0]
        strip.dynamics = None
        if int(parent.riftPartner) == int(strip.id):
            parent.riftPartner = -1
        world.log(
            EventKind["BACKARC_CLOSE"],
            (
                f"the back-arc sea behind {name} stops spreading; its arc and young "
                f"sea rejoin {parent.name}"
            ),
            {
                "strip": int(strip.id),
                "parent": int(parent.id),
                "rejoined": True,
                "seamCells": int(filled),
            },
        )
        closed += 1
    return closed


def _bridge_pieces(plate: Plate, mesh, max_km: float) -> int:
    """Fill short empty seams between disconnected pieces with ocean crust."""
    pieces = pieces_of(mesh, np.flatnonzero(plate.crust.mask))
    if len(pieces) < 2:
        return 0
    pieces.sort(key=len, reverse=True)
    crust = plate.crust
    max_hops = max(2, math.floor(float(max_km) / float(mesh.edge_km) + 0.5))

    def over_empty(cells):
        return hops_from(mesh, cells, lambda cell: not crust.mask[cell], max_hops)[0]

    main_distance = over_empty(pieces[0])
    filled = 0
    for piece in pieces[1:]:
        distance = over_empty(piece)
        between = [
            cell
            for cell in range(mesh.num_cells)
            if not crust.mask[cell]
            and distance[cell] > 0
            and main_distance[cell] > 0
            and distance[cell] + main_distance[cell] <= max_hops + 1
        ]
        between.sort(key=lambda cell: min(distance[cell], main_distance[cell]))
        for cell in between:
            source = -1
            for edge in range(
                int(mesh.adj_offset[cell]), int(mesh.adj_offset[cell + 1])
            ):
                neighbour = int(mesh.adj_list[edge])
                if crust.mask[neighbour] and (
                    source < 0 or int(crust.type[neighbour]) == CrustType["OCEAN"]
                ):
                    source = neighbour
            if source < 0:
                continue
            if int(crust.type[source]) == CrustType["OCEAN"]:
                crust.copy_from(cell, crust, source)
            else:
                crust.create(
                    cell,
                    CrustType["OCEAN"],
                    float(crust.formedAt[source]),
                )
            filled += 1
    if filled:
        plate.mark_reshaped()
    return filled


def steer_back_arc_strips(world) -> int:
    """Override active strip velocities so they retreat with their slab hinge."""
    if "backArcActiveMyr" not in world.config:
        return 0
    active_myr = float(world.config["backArcActiveMyr"])
    steered = 0
    for strip in world.plates:
        parent_id = int(strip.backArcOf)
        if (
            not strip.alive
            or parent_id < 0
            or strip.backArcPushDir is None
            or float(world.time) - float(strip.born) >= active_myr
            or parent_id >= len(world.plates)
        ):
            continue
        parent = world.plates[parent_id]
        if not parent.alive:
            continue
        centre = np.asarray(strip.to_world(*strip.capCenter), dtype=np.float64)
        direction = np.asarray(strip.to_world(*strip.backArcPushDir), dtype=np.float64)
        tangent = direction - float(np.dot(direction, centre)) * centre
        pole = np.cross(centre, tangent)
        length = float(np.linalg.norm(pole)) or 1.0
        rate = float(strip.rollbackCmYr) * CM_PER_YR_TO_RAD_PER_MYR / length
        strip.omega = [
            float(parent.omega[axis]) + float(pole[axis]) * rate for axis in range(3)
        ]
        steered += 1
    return steered


__all__ = ["open_back_arcs", "rollback_cm_yr", "steer_back_arc_strips"]
