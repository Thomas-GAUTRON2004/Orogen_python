"""Reusable continental-collision primitives ported from Orogen V2."""

from __future__ import annotations

import math

from ..core.mesh_bfs import hops_from
from ..core.sphere_geometry import (
    PLANET_RADIUS_KM,
    centroid_of_cells,
    cross,
    dot,
    tangent_toward,
)
from .config import CM_PER_YR_TO_RAD_PER_MYR
from .crust import BeltSide, CrustFlag, CrustType, OrogenyType
from .far_field import feed_far_field
from .ops.merge_plates import merge_plates
from .orogeny import belt_shape, belt_width_km, obliquity_deg
from .spread_belt import spread_belt
from .world import EventKind


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
        upper, lower = (plate_a, plate_b) if flag_a > flag_b else (plate_b, plate_a)
    else:
        upper, lower = (
            (plate_a, plate_b)
            if plate_a.stats["continentalCells"] >= plate_b.stats["continentalCells"]
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
    share = float(config.get("underthrustShare", 0.0))
    km = float(shortening_km)
    if not share > 0.0 or km <= 0.0:
        return 0

    xyz, index = mesh.xyz, mesh.index
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    owner, owner_cell, land_claims = (world.owner, world.ownerCell, world.landClaims)
    cell_area = float(mesh.cell_area_km2)
    front_km = float(record["frontKm"])
    record["underthrustDue"] = (
        float(record.get("underthrustDue", 0.0)) + share * km * front_km / cell_area
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
                and lower.crust.type[int(owner_cell[neighbour])] != CrustType["OCEAN"]
            ):
                lower_side.append(cell)
                break

    if not lower_side:
        return 0

    depths, _sources = hops_from(mesh, lower_side, lambda cell: cell in in_overlap)
    candidates = [int(cell) for cell in cells if int(depths[int(cell)]) > 0]
    candidates.sort(key=lambda cell: (-int(depths[cell]), cell))

    overlap_cap = max(
        1,
        math.floor(front_km * float(config["overlapCapKm"]) / cell_area),
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
        if not lower.crust.mask[lower_cell] or lower.crust.type[lower_cell] in (
            CrustType["OCEAN"],
            CrustType["CRATON"],
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
    record["underthrustCells"] = int(record.get("underthrustCells", 0)) + consumed
    world.budget["underthrustCells"] += consumed
    if consumed:
        lower.mark_reshaped()
        upper.dirty = True
    return consumed


__all__ = [
    "track_collisions",
    "are_young_rift_partners",
    "choose_sides",
    "collect_contacts",
    "forget_stale_collisions",
    "underthrust",
    "accrete_arcs",
    "dock_if_terrane",
    "enclosed_sea_between_km2",
    "is_rift_jitter",
    "open_sea_between",
    "recycle",
    "recycle_small_arc",
]


def enclosed_sea_between_km2(world, upper, lower) -> float:
    """Find the largest enclosed sea component bordered by both plates, in km²."""
    mesh = world.mesh
    owner = world.owner
    owner_cell = world.ownerCell
    land_claims = world.landClaims
    plates = world.plates
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    n = int(mesh.num_cells)

    def is_sea(cell: int) -> bool:
        plate_id = int(owner[cell])
        return bool(
            plate_id < 0
            or (
                int(land_claims[cell]) <= 1
                and plates[plate_id].crust.type[int(owner_cell[cell])]
                == CrustType["OCEAN"]
            )
        )

    component = [-1] * n
    sizes: list[int] = []
    for start in range(n):
        if not is_sea(start) or component[start] >= 0:
            continue
        identifier = len(sizes)
        queue = [start]
        component[start] = identifier
        cursor = 0
        while cursor < len(queue):
            cell = queue[cursor]
            cursor += 1
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if component[neighbour] < 0 and is_sea(neighbour):
                    component[neighbour] = identifier
                    queue.append(neighbour)
        sizes.append(len(queue))

    if not sizes:
        return 0.0
    world_ocean = max(range(len(sizes)), key=lambda identifier: sizes[identifier])
    borders_upper = [False] * len(sizes)
    borders_lower = [False] * len(sizes)
    for cell in range(n):
        plate_id = int(owner[cell])
        if plate_id not in (upper.id, lower.id) or is_sea(cell):
            continue
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            identifier = component[int(adjacency[edge])]
            if identifier >= 0:
                if plate_id == upper.id:
                    borders_upper[identifier] = True
                else:
                    borders_lower[identifier] = True

    largest = 0
    for identifier, size in enumerate(sizes):
        if (
            identifier != world_ocean
            and borders_upper[identifier]
            and borders_lower[identifier]
        ):
            largest = max(largest, size)
    return float(largest) * float(mesh.cell_area_km2)


def open_sea_between(world, upper, lower) -> dict[str, float]:
    """Measure the still-open sea directly between two opposed continental fronts."""
    mesh = world.mesh
    owner, owner_cell = world.owner, world.ownerCell
    land_claims, plates = world.landClaims, world.plates
    xyz = mesh.xyz
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    edge_km = float(mesh.edge_km)
    cell_area = float(mesh.cell_area_km2)
    config = world.config
    n = int(mesh.num_cells)

    def is_sea(cell: int) -> bool:
        plate_id = int(owner[cell])
        return bool(
            plate_id < 0
            or (
                int(land_claims[cell]) <= 1
                and plates[plate_id].crust.type[int(owner_cell[cell])]
                == CrustType["OCEAN"]
            )
        )

    coast_upper: list[int] = []
    coast_lower: list[int] = []
    for cell in range(n):
        plate_id = int(owner[cell])
        if plate_id not in (upper.id, lower.id) or is_sea(cell):
            continue
        touches_sea = any(
            is_sea(int(adjacency[edge]))
            for edge in range(int(offsets[cell]), int(offsets[cell + 1]))
        )
        if touches_sea:
            (coast_upper if plate_id == upper.id else coast_lower).append(cell)

    hops = max(
        1,
        math.floor(float(config["sutureGapKm"]) / edge_km + 0.5),
    )
    dist_upper, source_upper = hops_from(mesh, coast_upper, is_sea, hops)
    dist_lower, source_lower = hops_from(mesh, coast_lower, is_sea, hops)

    def direction_to(cell: int, source: int) -> list[float]:
        return tangent_toward(
            _position(xyz, cell),
            _position(xyz, cell),
            _position(xyz, int(source)),
        )

    count = 0
    widest = 0
    for cell in range(n):
        du = int(dist_upper[cell])
        dl = int(dist_lower[cell])
        if du <= 0 or dl <= 0 or du + dl > hops + 1:
            continue
        to_upper = direction_to(cell, int(source_upper[cell]))
        to_lower = direction_to(cell, int(source_lower[cell]))
        facing = dot(to_upper, to_lower)
        if facing < -0.5:
            count += 1
        if facing < -0.87:
            widest = max(widest, du + dl - 1)

    return {
        "km2": count * cell_area,
        "widestKm": widest * edge_km,
    }


def is_rift_jitter(world, plate_a, plate_b, cells) -> bool:
    """Detect a one-cell sliver where conjugate, young rift margins jitter."""
    mesh = world.mesh
    if len(cells) * float(mesh.cell_area_km2) / float(mesh.edge_km) > float(
        world.config["riftJitterFrontKm"]
    ):
        return False

    xyz, index = mesh.xyz, mesh.index
    for world_cell in cells:
        offset = 3 * int(world_cell)
        point = (
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )
        cell_a = plate_a.cell_at(index, *point)
        cell_b = plate_b.cell_at(index, *point)
        if not (
            int(plate_a.crust.flags[cell_a]) & CrustFlag["RIFT_MARGIN"]
            and int(plate_b.crust.flags[cell_b]) & CrustFlag["RIFT_MARGIN"]
        ):
            return False
    return True


def recycle(world, donor, cells) -> int:
    """Remove an arc fragment into the mantle and retire an empty donor."""
    area_km2 = len(cells) * float(world.mesh.cell_area_km2)
    for cell in cells:
        donor.crust.remove(int(cell))
    donor.mark_reshaped()
    world.budget["recycledCells"] += len(cells)
    if area_km2 >= float(world.config["terraneMinAreaKm2"]):
        world.log(
            EventKind["ARC_RECYCLED"],
            (
                f"a small island arc of {donor.name} "
                f"({area_km2 / 1e3:.0f}k km²) goes down with its slab "
                "instead of accreting"
            ),
            {"donor": donor.id, "areaKm2": area_km2},
        )
    if donor.crust.count == 0:
        donor.alive = False
        donor.omega = [0.0, 0.0, 0.0]
        donor.dynamics = None
    return len(cells)


def recycle_small_arc(world, donor, cell: int) -> bool:
    """Recycle a connected arc fragment below the configured accretion area."""
    mesh = world.mesh
    crust = donor.crust
    if not crust.mask[cell] or crust.type[cell] != CrustType["ARC"]:
        return False

    limit = math.ceil(
        float(world.config["arcAccreteMinKm2"]) / float(mesh.cell_area_km2)
    )
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    component = [int(cell)]
    seen = {int(cell)}
    cursor = 0
    while cursor < len(component) and len(component) < limit:
        current = component[cursor]
        cursor += 1
        for edge in range(int(offsets[current]), int(offsets[current + 1])):
            neighbour = int(adjacency[edge])
            if (
                neighbour not in seen
                and crust.mask[neighbour]
                and crust.type[neighbour] == CrustType["ARC"]
            ):
                seen.add(neighbour)
                component.append(neighbour)
    if len(component) >= limit:
        return False
    recycle(world, donor, component)
    return True


def accrete_arcs(world, plate_a, plate_b, overlap) -> dict:
    """Scrape overlapping island arcs onto the opposite plate before core collision."""
    from .ops.scrape_arc import scrape_arc

    mesh = world.mesh
    xyz, index = mesh.xyz, mesh.index

    def cell_of(plate, world_cell: int) -> int:
        offset = 3 * int(world_cell)
        return plate.cell_at(
            index,
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )

    def is_arc(plate, cell: int) -> bool:
        return bool(
            plate.crust.mask[cell] and plate.crust.type[cell] == CrustType["ARC"]
        )

    core = []
    from_a: list[int] = []
    from_b: list[int] = []
    arc_a_loses = plate_a.stats["continentalCells"] < plate_b.stats[
        "continentalCells"
    ] or (
        plate_a.stats["continentalCells"] == plate_b.stats["continentalCells"]
        and plate_a.id > plate_b.id
    )
    for world_cell in overlap:
        cell_a = cell_of(plate_a, world_cell)
        cell_b = cell_of(plate_b, world_cell)
        arc_a = is_arc(plate_a, cell_a)
        arc_b = is_arc(plate_b, cell_b)
        if not arc_a and not arc_b:
            core.append(int(world_cell))
        elif arc_a and arc_b:
            (from_a if arc_a_loses else from_b).append(
                cell_a if arc_a_loses else cell_b
            )
        elif arc_a:
            from_a.append(cell_a)
        else:
            from_b.append(cell_b)

    accreted = 0
    for donor, host, arriving in (
        (plate_a, plate_b, from_a),
        (plate_b, plate_a, from_b),
    ):
        if not arriving or not donor.alive or not host.alive:
            continue
        seeds = []
        for cell in dict.fromkeys(arriving):
            if not recycle_small_arc(world, donor, int(cell)):
                seeds.append(int(cell))
        if not seeds:
            accreted += 1
            continue
        transfer = scrape_arc(world, host, donor, seeds)
        moved = int(transfer["moved"])
        accreted += moved + int(transfer["absorbed"])
        key = f"{donor.id}:{host.id}"
        record = world.arcScrapes.get(key)
        if record is None:
            record = {
                "cells": 0,
                "since": float(world.time),
                "donorName": donor.name,
            }
            world.arcScrapes[key] = record
        record["cells"] += moved
        record["last"] = float(world.time)
        if not record.get("logged", False) and record["cells"] * float(
            mesh.cell_area_km2
        ) >= float(world.config["arcScrapeLogKm2"]):
            record["logged"] = True
            area_mkm2 = record["cells"] * float(mesh.cell_area_km2) / 1e6
            world.log(
                EventKind["ARC_SCRAPE"],
                (
                    f"island arc of {record['donorName']} is scraped onto "
                    f"{host.name} ({area_mkm2:.2f} Mkm² so far): the ocean "
                    f"behind it keeps going down"
                    + (f"; {record['donorName']} is gone" if not donor.alive else "")
                ),
                {
                    "host": host.id,
                    "donor": donor.id,
                    "areaMkm2": area_mkm2,
                    "donorGone": not donor.alive,
                },
            )
    return {"core": core, "accreted": bool(accreted)}


def dock_if_terrane(world, plate_a, plate_b, cells) -> bool:
    """Dock a small landmass or mostly-arc piece as a terrane rather than equal plates."""
    from .ops.plate_accretion import accrete_terrane, landmass_of

    mesh, config = world.mesh, world.config
    xyz, index = mesh.xyz, mesh.index
    cell_area = float(mesh.cell_area_km2)

    def contact_cells(plate):
        result = []
        for world_cell in cells:
            offset = 3 * int(world_cell)
            result.append(
                plate.cell_at(
                    index,
                    float(xyz[offset]),
                    float(xyz[offset + 1]),
                    float(xyz[offset + 2]),
                )
            )
        return result

    mass_a = landmass_of(plate_a, mesh, contact_cells(plate_a))
    mass_b = landmass_of(plate_b, mesh, contact_cells(plate_b))
    area_a = len(mass_a) * cell_area
    area_b = len(mass_b) * cell_area

    def mostly_arc(plate, mass) -> bool:
        return bool(
            mass
            and sum(plate.crust.type[cell] == CrustType["ARC"] for cell in mass) * 2
            > len(mass)
        )

    terrane_a = bool(
        area_a > 0
        and (area_a < float(config["terraneMaxAreaKm2"]) or mostly_arc(plate_a, mass_a))
    )
    terrane_b = bool(
        area_b > 0
        and (area_b < float(config["terraneMaxAreaKm2"]) or mostly_arc(plate_b, mass_b))
    )
    if not terrane_a and not terrane_b:
        return False
    if area_a == 0 or area_b == 0:
        return False

    a_hosts = not terrane_a or (
        terrane_b
        and plate_a.stats["continentalCells"] >= plate_b.stats["continentalCells"]
    )
    host, donor, mass = (
        (plate_a, plate_b, mass_b) if a_hosts else (plate_b, plate_a, mass_a)
    )
    relative_omega = [float(donor.omega[i]) - float(host.omega[i]) for i in range(3)]
    centre = centroid_of_cells(xyz, cells)
    velocity = cross(relative_omega, centre)
    rate = math.hypot(*velocity) / CM_PER_YR_TO_RAD_PER_MYR
    arc_piece = mostly_arc(donor, mass)

    if arc_piece and len(mass) * cell_area < float(config["arcAccreteMinKm2"]):
        recycle(world, donor, mass)
        return True

    donor_name = donor.name
    result = accrete_terrane(world, host, donor, mass, rate)
    if result.get("scrap", False):
        return True

    area_mkm2 = len(mass) * cell_area / 1e6
    kind = "island arc" if arc_piece else "terrane"
    world.log(
        EventKind["DOCK"],
        (
            f"{kind} of {donor_name} ({area_mkm2:.2f} Mkm²) accretes to "
            f"{host.name}: {result['cells']} cells transferred"
            + (
                f", {result['sealed']} cells of strait closed"
                if result["sealed"]
                else ""
            )
            + "; subduction jumps to its outboard side"
            + (f"; {donor_name} is gone" if not donor.alive else "")
        ),
        {
            "host": host.id,
            "donor": donor.id,
            "arc": arc_piece,
            "areaMkm2": area_mkm2,
            "cells": result["cells"],
            "sealed": result["sealed"],
            "donorGone": not donor.alive,
        },
    )
    return True


def _position(xyz, cell: int):
    offset = 3 * int(cell)
    return (
        float(xyz[offset]),
        float(xyz[offset + 1]),
        float(xyz[offset + 2]),
    )


def _suture_is_ready(
    record: dict,
    cells: list[int],
    cell_area: float,
    config,
) -> tuple[bool, bool]:
    """Return the raw readiness and minimum-front candidate flags."""
    ready = (
        float(record["shorteningKm"])
        >= float(config.get("sutureShorteningKm", math.inf))
        or (
            float(record["stalledMyr"]) >= float(config.get("sutureStallMyr", math.inf))
            and float(record["shorteningKm"])
            >= float(config.get("sutureStallMinKm", math.inf))
        )
        or float(record["contactMyr"])
        >= float(config.get("sutureContactMyr", math.inf))
        or len(cells) * cell_area >= float(config.get("sutureOverlapKm2", math.inf))
    )
    candidate = bool(
        ready
        and float(record["frontMaxKm"])
        >= float(config.get("sutureMinContactKm", math.inf))
    )
    return ready, candidate


def _refresh_suture_gap(
    world,
    upper,
    lower,
    record: dict,
    config,
    time: float,
) -> tuple[bool, bool]:
    """Update periodic open-sea measurements and report whether the gap is closed."""
    due = not (
        time - float(record.get("checkedAt", -math.inf))
        < float(config["sutureCheckMyr"])
    )
    if due:
        gap = open_sea_between(world, upper, lower)
        record["gapKm2"] = float(gap["km2"])
        record["gapWidestKm"] = float(gap["widestKm"])
        best_gap = float(record.get("gapBestKm2", math.inf))
        if not record["gapKm2"] >= 0.9 * best_gap:
            record["gapBestKm2"] = record["gapKm2"]
            record["gapShrankAt"] = time

    record.setdefault("gapKm2", math.inf)
    closed = (
        float(record.get("gapWidestKm", math.inf)) <= float(config["sutureGapWidthKm"])
        or float(record["stalledMyr"]) >= float(config["sutureStallGiveUpMyr"])
        or time - float(record.get("gapShrankAt", time))
        >= float(config["sutureGapStallMyr"])
    )
    return due, closed


def _update_suture_weld_flags(
    world,
    upper,
    lower,
    cells: list[int],
    under_now: int,
    edge_km: float,
    cell_area: float,
    record: dict,
    config,
    time: float,
    due: bool,
    ready: bool,
    closed: bool,
) -> tuple[bool, bool]:
    """Apply stacking and ring-sea gates and record weld eligibility."""
    overdue = float(record["contactMyr"]) >= float(config["sutureLockMaxMyr"])
    stacked = (len(cells) - under_now) * cell_area / max(
        float(record["frontKm"]), edge_km
    ) >= float(config["sutureStackKm"])
    would_weld = bool(
        (ready and (closed or stacked) or overdue)
        and float(record["frontMaxKm"]) >= float(config["sutureMinContactKm"])
    )
    if would_weld and (due or "ringKm2" not in record):
        record["ringKm2"] = enclosed_sea_between_km2(world, upper, lower)
    if due:
        record["checkedAt"] = time

    ring_area = float(record.get("ringKm2", math.inf))
    ring_too_large = bool(would_weld and ring_area > float(config["ringSeaKm2"]))
    ring_blocked = bool(ring_too_large and not stacked and not overdue)
    record["weldCandidate"] = would_weld
    record["weldBlockedByRing"] = ring_blocked
    record["mergeEligible"] = bool(would_weld and not ring_blocked)
    if ring_too_large and not record.get("ringLogged", False):
        record["ringLogged"] = True
        world.log(
            EventKind["COLLISION"],
            (
                f"{lower.name} and {upper.name} stay apart: their "
                f"land rings a {ring_area / 1e6:.1f} Mkm² sea"
            ),
            {
                "upper": upper.id,
                "lower": lower.id,
                "ringKm2": ring_area,
            },
        )
    return overdue, stacked


def _merge_eligible_suture(
    world,
    upper,
    lower,
    cells: list[int],
    record: dict,
    config,
    time: float,
    closed: bool,
    overdue: bool,
    stacked: bool,
) -> bool:
    """Perform a fully configured plate weld and emit its suture event."""
    if not record["mergeEligible"] or not _SUTURE_MERGE_CONFIG_KEYS.issubset(config):
        return False

    why = (
        "stacked"
        if stacked and not closed
        else "overdue"
        if not closed
        else "stalled"
        if float(record["stalledMyr"]) >= float(config.get("sutureStallMyr", math.inf))
        else "contact"
        if float(record["shorteningKm"])
        < float(config.get("sutureShorteningKm", math.inf))
        else "shortening"
    )
    result = merge_plates(world, upper, lower, cells, record)
    record["sutured"] = True
    record["sutureReason"] = why
    after_myr = time - float(record["startedAt"])
    world.log(
        EventKind["SUTURE"],
        (
            f"{lower.name} sutured to {upper.name} after "
            f"{after_myr:.0f} Myr: "
            f"{record['frontKm']:.0f} km front, "
            f"{record['shorteningKm']:.0f} km of shortening, "
            f"outlines redrawn (indented up to "
            f"{result['maxIndentKm']:.0f} km; "
            f"{result['landLostCells']} cells of land absorbed "
            f"into the belt"
            + (
                f", {result['sealedCells']} cells of sea between them closed"
                if result["sealedCells"]
                else ""
            )
            + ")"
        ),
        {
            "upper": upper.id,
            "lower": lower.id,
            "afterMyr": after_myr,
            "frontKm": float(record["frontKm"]),
            "shorteningKm": float(record["shorteningKm"]),
            "maxIndentKm": float(result["maxIndentKm"]),
            "landLostCells": int(result["landLostCells"]),
            "sealedCells": int(result["sealedCells"]),
            "gapKm2": float(record["gapKm2"]),
            "gapWidestKm": float(record["gapWidestKm"]),
            "why": why,
        },
    )
    return True


def _update_suture_state(
    world,
    upper,
    lower,
    cells: list[int],
    record: dict,
    under_now: int,
    edge_km: float,
    cell_area: float,
    config,
) -> bool:
    """Advance suture readiness, gap/ring checks and configured plate welding."""
    time = float(world.time)
    ready, candidate = _suture_is_ready(record, cells, cell_area, config)
    record["sutureCandidate"] = candidate
    record["weldCandidate"] = False
    record["weldBlockedByRing"] = False
    record["mergeEligible"] = False
    if not candidate or not _SUTURE_GAP_CONFIG_KEYS.issubset(config):
        return False

    due, closed = _refresh_suture_gap(world, upper, lower, record, config, time)
    overdue, stacked = _update_suture_weld_flags(
        world,
        upper,
        lower,
        cells,
        under_now,
        edge_km,
        cell_area,
        record,
        config,
        time,
        due,
        ready,
        closed,
    )
    return _merge_eligible_suture(
        world,
        upper,
        lower,
        cells,
        record,
        config,
        time,
        closed,
        overdue,
        stacked,
    )


_BELT_CONFIG_KEYS = {
    "belts",
    "beltSpeedFactorMin",
    "beltSpeedFactorMax",
    "beltSpeedRefCmYr",
    "beltSpeedExponent",
    "beltObliqueMin",
    "apronKm",
    "orogenyQuietMyr",
    "beltWorkShare",
    "thickeningPerWorkKm",
    "weakCellCost",
    "cratonRimKm",
    "cratonRimCost",
    "cratonRimShare",
    "sutureMinContactKm",
}

_SUTURE_GAP_CONFIG_KEYS = frozenset(
    {
        "sutureCheckMyr",
        "sutureGapKm",
        "sutureGapWidthKm",
        "sutureStallGiveUpMyr",
        "sutureGapStallMyr",
        "sutureLockMaxMyr",
        "sutureStackKm",
        "ringSeaKm2",
    }
)
_SUTURE_MERGE_CONFIG_KEYS = _BELT_CONFIG_KEYS | {
    "indentFalloffKm",
    "indentGain",
    "escapeShare",
    "sutureHalfWidthKm",
    "sealStraitKm",
    "inlandSeaMaxKm2",
}


def _collision_front_cells(
    world, upper, lower, cells: list[int]
) -> tuple[list[int], list[int]]:
    """Return unique local-raster cells bordering each side of a collision.

    Ambiguous world claims and oceanic neighbours do not contribute to either
    continental front. The returned indices are expressed in each plate's
    local crust raster, not in world-mesh coordinates.
    """
    mesh = world.mesh
    xyz, index = mesh.xyz, mesh.index
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    owner, owner_cell = world.owner, world.ownerCell
    land_claims = world.landClaims
    upper_front: list[int] = []
    lower_front: list[int] = []

    for world_cell in cells:
        touches_upper = False
        touches_lower = False
        for edge in range(int(offsets[world_cell]), int(offsets[world_cell + 1])):
            neighbour = int(adjacency[edge])
            if int(land_claims[neighbour]) > 1 or int(owner[neighbour]) < 0:
                continue
            neighbour_plate = int(owner[neighbour])
            local_cell = int(owner_cell[neighbour])
            if (
                neighbour_plate == upper.id
                and upper.crust.type[local_cell] != CrustType["OCEAN"]
            ):
                touches_upper = True
            if (
                neighbour_plate == lower.id
                and lower.crust.type[local_cell] != CrustType["OCEAN"]
            ):
                touches_lower = True

        offset = 3 * world_cell
        point = (
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )
        if touches_upper:
            upper_front.append(upper.cell_at(index, *point))
        if touches_lower:
            lower_front.append(lower.cell_at(index, *point))

    return list(dict.fromkeys(upper_front)), list(dict.fromkeys(lower_front))


def _plates_alive(plates, id_a: int, id_b: int) -> bool:
    """Whether both collision participants still exist and are active."""
    return (
        0 <= id_a < len(plates)
        and 0 <= id_b < len(plates)
        and plates[id_a].alive
        and plates[id_b].alive
    )


CollisionProgress = tuple[
    list[float],
    list[float],
    list[float],
    float,
    float,
    list[int],
    list[int],
]


def _update_collision_progress(
    world,
    key: str,
    mesh,
    upper,
    lower,
    cells: list[int],
    record: dict,
    step_dt: float,
    edge_km: float,
    cell_area: float,
    config,
) -> CollisionProgress | None:
    """Update collision kinematics, shortening and active continental fronts."""
    xyz = mesh.xyz
    record["cells"] = len(cells)
    record["lastActive"] = float(world.time)
    record["contactMyr"] = float(record.get("contactMyr", 0.0)) + step_dt

    centre = centroid_of_cells(xyz, cells)
    upper_centre = upper.to_world(*map(float, upper.capCenter))
    lower_centre = lower.to_world(*map(float, lower.capCenter))
    normal = tangent_toward(centre, lower_centre, upper_centre)
    relative_omega = [
        float(lower.omega[axis]) - float(upper.omega[axis]) for axis in range(3)
    ]
    closing = dot(cross(relative_omega, centre), normal)
    closing_cm_yr = closing / CM_PER_YR_TO_RAD_PER_MYR
    record["closingCmYr"] = closing_cm_yr

    # A newly detected one-cell sliver between plates that are parting is
    # a raster timing artefact, not a persistent collision.
    front_limit = float(config.get("collisionJitterFrontKm", 0.0))
    if (
        record.get("announce", False)
        and closing_cm_yr < 0.0
        and len(cells) * cell_area / max(edge_km, 1e-12) <= front_limit
    ):
        world.collisions.pop(key, None)
        return None

    if record.get("announce", False):
        record["announce"] = False
        world.log(
            EventKind["COLLISION"],
            f"{lower.name} meets {upper.name}",
            {"upper": upper.id, "lower": lower.id},
        )

    shortening_km = max(0.0, closing) * step_dt * PLANET_RADIUS_KM
    record["shorteningKm"] = float(record["shorteningKm"]) + shortening_km
    stall_rate = float(config.get("sutureStallCmYr", 0.4))
    record["stalledMyr"] = (
        float(record.get("stalledMyr", 0.0)) + step_dt
        if closing_cm_yr < stall_rate
        else 0.0
    )
    elapsed = max(
        step_dt,
        float(world.time) - float(record["startedAt"]),
    )
    record["meanRateCmYr"] = float(record["shorteningKm"]) / elapsed / 10.0

    upper_front, lower_front = _collision_front_cells(world, upper, lower, cells)
    record["frontKm"] = (
        max(len(upper_front), len(lower_front)) * edge_km * math.sqrt(3.0) / 2.0
    )
    record["frontMaxKm"] = max(
        float(record.get("frontMaxKm", 0.0)),
        float(record["frontKm"]),
    )
    return (
        centre,
        normal,
        relative_omega,
        closing,
        shortening_km,
        upper_front,
        lower_front,
    )


def _feed_collision_far_field(
    world,
    key: str,
    upper,
    lower,
    upper_front: list[int],
    lower_front: list[int],
    shortening_km: float,
    record: dict,
    config,
) -> None:
    """Feed the two collision margins into their far-field deformation fields."""
    far_share = float(config.get("farCollisionShare", 0.0))
    if shortening_km <= 0.0 or far_share <= 0.0:
        return
    feed_far_field(
        world,
        upper,
        f"c{key}:u",
        list(dict.fromkeys(upper_front)),
        shortening_km,
        float(record["frontKm"]),
        far_share,
    )
    feed_far_field(
        world,
        lower,
        f"c{key}:l",
        list(dict.fromkeys(lower_front)),
        shortening_km,
        float(record["frontKm"]),
        far_share,
    )


def _spread_collision_belts(
    world,
    mesh,
    upper,
    lower,
    record: dict,
    centre: list[float],
    normal: list[float],
    relative_omega: list[float],
    closing: float,
    shortening_km: float,
    upper_front: list[int],
    lower_front: list[int],
    config,
) -> None:
    """Spread the plateau and foreland belts resulting from collision shortening."""
    if shortening_km <= 0.0 or not _BELT_CONFIG_KEYS.issubset(config):
        return

    major = float(record["frontKm"]) >= float(config["sutureMinContactKm"])
    kind = OrogenyType["HIMALAYAN"] if major else OrogenyType["URAL"]
    upper_kind = "himalayan" if major else "ural"
    lower_kind = "foreland" if major else "ural"
    velocity = cross(relative_omega, centre)
    relative_speed = math.hypot(*velocity)
    shape = belt_shape(closing / max(1e-12, relative_speed), None, config)
    accumulated = float(record["shorteningKm"])
    mean_rate = float(record["meanRateCmYr"])
    upper_width = belt_width_km(upper_kind, accumulated, mean_rate, config) * shape
    lower_width = belt_width_km(lower_kind, accumulated, mean_rate, config) * shape
    shear = obliquity_deg(velocity, normal, centre, closing)
    if upper_front:
        spread_belt(
            upper,
            mesh,
            list(dict.fromkeys(upper_front)),
            {
                "type": kind,
                "widthKm": upper_width,
                "workKm": shortening_km,
                "time": float(world.time),
                "side": BeltSide["UPPER"],
                "shear": shear,
            },
            config,
        )
    if lower_front:
        spread_belt(
            lower,
            mesh,
            list(dict.fromkeys(lower_front)),
            {
                "type": kind,
                "widthKm": lower_width,
                "workKm": shortening_km,
                "time": float(world.time),
                "side": BeltSide["LOWER"],
                "shear": shear,
            },
            config,
        )


def _process_collision_overlap(
    world,
    key: str,
    overlap_values: list[int],
    step_dt: float,
    config,
    mesh,
    edge_km: float,
    cell_area: float,
) -> tuple[int, int, int, bool]:
    """Advance one active collision pair by one tectonic time step."""
    plates = world.plates
    started = 0
    tracked = 0
    underthrust_total = 0
    changed = False

    id_a, id_b = map(int, key.split(":"))
    if not _plates_alive(plates, id_a, id_b):
        return (started, tracked, underthrust_total, changed)

    # Arcs are transferred before continent-on-continent overlap is
    # counted. Keep this operation gated until its full config is present.
    if {"arcAccreteMinKm2", "terraneMinAreaKm2", "arcScrapeLogKm2"}.issubset(config):
        arc_result = accrete_arcs(world, plates[id_a], plates[id_b], overlap_values)
        core = arc_result["core"]
        changed = changed or bool(arc_result["accreted"])
    else:
        core = [int(cell) for cell in overlap_values]

    if not core or not _plates_alive(plates, id_a, id_b):
        return (started, tracked, underthrust_total, changed)

    record = world.collisions.get(key)
    if record is None:
        if "riftJitterFrontKm" in config and is_rift_jitter(
            world, plates[id_a], plates[id_b], core
        ):
            return (started, tracked, underthrust_total, changed)
        if (
            "riftPushMyr" in config
            and "riftJitterFrontKm" in config
            and are_young_rift_partners(
                plates[id_a], plates[id_b], float(world.time), config
            )
            and len(core) * cell_area / max(edge_km, 1e-12)
            <= float(config["riftJitterFrontKm"])
        ):
            return (started, tracked, underthrust_total, changed)

        sides = choose_sides(world, plates[id_a], plates[id_b], core)
        record = {
            "upper": sides["upper"].id,
            "lower": sides["lower"].id,
            "startedAt": float(world.time),
            "lastActive": float(world.time),
            "shorteningKm": 0.0,
            "frontKm": 0.0,
            "closingCmYr": 0.0,
            "meanRateCmYr": 0.0,
            "stalledMyr": 0.0,
            "contactMyr": 0.0,
            "cells": 0,
            "sutured": False,
            "ringLogged": False,
            "announce": True,
            "underthrustDue": 0.0,
            "underthrustCells": 0,
            "frontMaxKm": 0.0,
            "sutureCandidate": False,
        }
        world.collisions[key] = record
        started += 1

        if "terraneMaxAreaKm2" in config and dock_if_terrane(
            world, sides["upper"], sides["lower"], core
        ):
            del world.collisions[key]
            changed = True
            return (started, tracked, underthrust_total, changed)

    if record.get("sutured", False):
        return (started, tracked, underthrust_total, changed)

    upper = plates[int(record["upper"])]
    lower = plates[int(record["lower"])]
    if not upper.alive or not lower.alive:
        return (started, tracked, underthrust_total, changed)

    cells = [int(cell) for cell in core]
    progress = _update_collision_progress(
        world,
        key,
        mesh,
        upper,
        lower,
        cells,
        record,
        step_dt,
        edge_km,
        cell_area,
        config,
    )
    if progress is None:
        return (started, tracked, underthrust_total, changed)

    centre, normal, relative_omega, closing, shortening_km, upper_front, lower_front = (
        progress
    )
    _feed_collision_far_field(
        world,
        key,
        upper,
        lower,
        upper_front,
        lower_front,
        shortening_km,
        record,
        config,
    )
    _spread_collision_belts(
        world,
        mesh,
        upper,
        lower,
        record,
        centre,
        normal,
        relative_omega,
        closing,
        shortening_km,
        upper_front,
        lower_front,
        config,
    )

    under_now = underthrust(world, upper, lower, cells, record, shortening_km)
    underthrust_total += under_now
    changed = changed or under_now > 0

    changed = (
        _update_suture_state(
            world,
            upper,
            lower,
            cells,
            record,
            under_now,
            edge_km,
            cell_area,
            config,
        )
        or changed
    )

    tracked += 1

    return started, tracked, underthrust_total, changed


def track_collisions(world, dt: float) -> dict:
    """Track continent-on-continent contacts, shortening and underthrusting.

    This is the first orchestration slice of upstream step/collision.js.
    Terrane/arc docking and overlap underthrusting are handled when their
    prerequisites are configured. Belt propagation and plate welding remain
    separate follow-up stages because their Python implementations are not
    yet complete.
    """
    step_dt = float(dt)
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("dt must be a finite positive number")

    config = world.config
    contacts = collect_contacts(world)
    overlaps = contacts["overlap"]
    touching = contacts["touching"]
    forget_stale_collisions(world)

    plates = world.plates
    changed = False
    started = 0
    tracked = 0
    underthrust_total = 0

    # Small landmasses dock on contact, even before they overlap.
    if "terraneMaxAreaKm2" in config:
        for key, cells in touching.items():
            if key in overlaps or key in world.collisions:
                continue
            id_a, id_b = map(int, key.split(":"))
            if _plates_alive(plates, id_a, id_b) and dock_if_terrane(
                world, plates[id_a], plates[id_b], cells
            ):
                changed = True

    mesh = world.mesh
    edge_km = float(mesh.edge_km)
    cell_area = float(mesh.cell_area_km2)

    for key, overlap_values in overlaps.items():
        pair_started, pair_tracked, pair_underthrust, pair_changed = (
            _process_collision_overlap(
                world,
                key,
                overlap_values,
                step_dt,
                config,
                mesh,
                edge_km,
                cell_area,
            )
        )
        started += pair_started
        tracked += pair_tracked
        underthrust_total += pair_underthrust
        changed = changed or pair_changed

    return {
        "tracked": tracked,
        "started": started,
        "underthrustCells": underthrust_total,
        "changed": changed,
    }
