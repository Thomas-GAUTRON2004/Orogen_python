"""Reusable continental-collision primitives ported from Orogen V2."""

from __future__ import annotations

import math

from .crust import CrustFlag, CrustType
from ..core.mesh_bfs import hops_from


def collect_contacts(world) -> dict:
    """Collect land/land overlap cells and touching-land contacts by plate pair."""
    mesh = world.mesh
    owner, owner_cell = world.owner, world.ownerCell
    land_claims, collider = world.landClaims, world.collider
    plates = world.plates
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    overlaps: dict[str, list[int]] = {}
    touching: dict[str, list[int]] = {}

    def pair_key(a: int, b: int) -> str:
        return f"{min(a, b)}:{max(a, b)}"

    for cell in range(int(mesh.num_cells)):
        if int(land_claims[cell]) > 1 and int(collider[cell]) >= 0:
            a, b = int(owner[cell]), int(collider[cell])
            if a >= 0 and b >= 0 and a != b:
                overlaps.setdefault(pair_key(a, b), []).append(cell)
            continue

        a = int(owner[cell])
        if a < 0:
            continue
        plate_a = plates[a]
        raster_a = int(owner_cell[cell])
        if (
            plate_a.isBackground
            or not plate_a.crust.mask[raster_a]
            or plate_a.crust.type[raster_a] == CrustType["OCEAN"]
            or int(plate_a.crust.flags[raster_a]) & CrustFlag["RIFT_MARGIN"]
        ):
            continue

        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            b = int(owner[neighbour])
            if b < 0 or b == a or int(land_claims[neighbour]) > 1:
                continue
            plate_b = plates[b]
            raster_b = int(owner_cell[neighbour])
            if (
                plate_b.isBackground
                or not plate_b.crust.mask[raster_b]
                or plate_b.crust.type[raster_b] == CrustType["OCEAN"]
                or int(plate_b.crust.flags[raster_b]) & CrustFlag["RIFT_MARGIN"]
            ):
                continue
            touching.setdefault(pair_key(a, b), []).append(cell)

    return {"overlap": overlaps, "touching": touching}


def forget_stale_collisions(world) -> int:
    """Remove unsutured pair records that have been inactive too long."""
    time = float(world.time)
    forget_after = float(world.config["collisionForgetMyr"])
    stale = [
        key
        for key, record in world.collisions.items()
        if not record.get("sutured", False)
        and time - float(record["lastActive"]) > forget_after
    ]
    for key in stale:
        del world.collisions[key]
    return len(stale)


def are_young_rift_partners(plate_a, plate_b, time: float, config) -> bool:
    """Whether two conjugate plates are still in their post-rift push window."""
    return bool(
        plate_a.riftPartner == plate_b.id
        and plate_b.riftPartner == plate_a.id
        and float(time) - max(float(plate_a.born), float(plate_b.born))
        < float(config["riftPushMyr"])
    )


def choose_sides(world, plate_a, plate_b, cells) -> dict:
    """Choose the upper plate from trench flags at the contact front."""
    mesh = world.mesh
    xyz, index = mesh.xyz, mesh.index
    offsets, adjacency = mesh.adj_offset, mesh.adj_list

    def flagged(plate) -> int:
        count = 0
        seen = set()
        for world_cell_value in cells:
            world_cell = int(world_cell_value)
            for edge in range(
                int(offsets[world_cell]),
                int(offsets[world_cell + 1]),
            ):
                neighbour = int(adjacency[edge])
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                offset = 3 * neighbour
                local = plate.cell_at(
                    index,
                    float(xyz[offset]),
                    float(xyz[offset + 1]),
                    float(xyz[offset + 2]),
                )
                crust = plate.crust
                if (
                    crust.mask[local]
                    and crust.type[local] != CrustType["OCEAN"]
                    and int(crust.flags[local]) & CrustFlag["TRENCH"]
                ):
                    count += 1
        return count

    flag_a, flag_b = flagged(plate_a), flagged(plate_b)
    if flag_a != flag_b:
        upper, lower = (
            (plate_a, plate_b) if flag_a > flag_b else (plate_b, plate_a)
        )
    else:
        upper, lower = (
            (plate_a, plate_b)
            if plate_a.stats["continentalCells"]
            >= plate_b.stats["continentalCells"]
            else (plate_b, plate_a)
        )
    return {"upper": upper, "lower": lower}


def underthrust(world, upper, lower, cells, record: dict, shortening_km: float) -> int:
    """Consume advancing lower-plate land under the upper plate.

    The lower plate's most advanced non-cratonic land cells are removed in
    proportion to convergence and front length. Matching upper-plate land is
    thickened and rooted above the consumed cells.
    """
    mesh = world.mesh
    config = world.config
    share = float(config["underthrustShare"])
    km = float(shortening_km)
    if not share > 0.0 or km <= 0.0:
        return 0

    xyz, index = mesh.xyz, mesh.index
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    owner, owner_cell, land_claims = (
        world.owner, world.ownerCell, world.landClaims
    )
    cell_area = float(mesh.cell_area_km2)
    front_km = float(record["frontKm"])
    record["underthrustDue"] = (
        float(record.get("underthrustDue", 0.0))
        + share * km * front_km / cell_area
    )

    in_overlap = {int(cell) for cell in cells}
    lower_side = []
    for cell in in_overlap:
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if (
                neighbour not in in_overlap
                and int(land_claims[neighbour]) <= 1
                and int(owner[neighbour]) == lower.id
                and lower.crust.type[int(owner_cell[neighbour])]
                != CrustType["OCEAN"]
            ):
                lower_side.append(cell)
                break

    if not lower_side:
        return 0

    depths, _sources = hops_from(
        mesh, lower_side, lambda cell: cell in in_overlap
    )
    candidates = [
        int(cell)
        for cell in cells
        if int(depths[int(cell)]) > 0
    ]
    candidates.sort(key=lambda cell: (-int(depths[cell]), cell))

    overlap_cap = max(
        1,
        math.floor(
            front_km * float(config["overlapCapKm"]) / cell_area
        ),
    )
    excess = max(0, len(cells) - overlap_cap)
    target = max(
        math.floor(float(record["underthrustDue"])),
        excess,
    )

    consumed = 0
    for world_cell in candidates:
        if consumed >= target:
            break
        offset = 3 * world_cell
        x, y, z = (
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )
        lower_cell = lower.cell_at(index, x, y, z)
        if (
            not lower.crust.mask[lower_cell]
            or lower.crust.type[lower_cell] in (
                CrustType["OCEAN"], CrustType["CRATON"]
            )
        ):
            continue
        lower.crust.remove(lower_cell)
        consumed += 1

        upper_cell = upper.cell_at(index, x, y, z)
        if (
            upper.crust.mask[upper_cell]
            and upper.crust.type[upper_cell] != CrustType["OCEAN"]
        ):
            thickening = float(config["underthrustThickenKm"])
            maximum = float(config["underthrustMaxThickKm"])
            upper.crust.thickKm[upper_cell] = min(
                maximum,
                float(upper.crust.thickKm[upper_cell]) + thickening,
            )
            upper.crust.rootKm[upper_cell] = min(
                maximum,
                float(upper.crust.rootKm[upper_cell]) + thickening,
            )

    record["underthrustDue"] = min(
        max(0.0, float(record["underthrustDue"]) - consumed),
        1.0,
    )
    record["underthrustCells"] = (
        int(record.get("underthrustCells", 0)) + consumed
    )
    world.budget["underthrustCells"] += consumed
    if consumed:
        lower.mark_reshaped()
        upper.dirty = True
    return consumed


__all__ = [
    "are_young_rift_partners",
    "choose_sides",
    "collect_contacts",
    "forget_stale_collisions",
    "underthrust",
]
