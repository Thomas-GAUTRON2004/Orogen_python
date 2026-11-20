"""Terrane accretion and narrow-sea closure, ported from Orogen V2."""

from __future__ import annotations

import math

import numpy as np

from ...core.mesh_bfs import hops_from
from ..crust import BeltSide, CrustFlag, CrustType, OrogenyType
from ..orogeny import belt_width_km
from ..plate import frame_rotation
from ..spread_belt import spread_belt
from .plate_transfer import free_cell_beside


def close_seas_between(out, origin, mesh, upper, config, time: float) -> int:
    """Close narrow straits between two landmasses and small enclosed seas."""
    num_cells = int(mesh.num_cells)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list

    def is_land(cell: int) -> bool:
        return bool(out.mask[cell] and out.type[cell] != CrustType["OCEAN"])

    def is_sea(cell: int) -> bool:
        return not is_land(cell)

    land_a = [
        cell for cell in range(num_cells) if is_land(cell) and int(origin[cell]) == 1
    ]
    land_b = [
        cell for cell in range(num_cells) if is_land(cell) and int(origin[cell]) != 1
    ]
    reach_hops = max(
        1,
        math.floor(float(config["sealStraitKm"]) / float(mesh.edge_km) + 0.5),
    )
    from_a_dist, _from_a_source = hops_from(mesh, land_a, is_sea, reach_hops)
    from_b_dist, _from_b_source = hops_from(mesh, land_b, is_sea, reach_hops)
    fill = np.zeros(num_cells, dtype=np.uint8)
    for cell in range(num_cells):
        if is_sea(cell) and from_a_dist[cell] > 0 and from_b_dist[cell] > 0:
            fill[cell] = 1

    # Sea connected to the far side of the planet stays open.
    centre = upper.capCenter
    far = int(mesh.index.nearest(-centre[0], -centre[1], -centre[2]))
    opened = np.zeros(num_cells, dtype=np.uint8)
    stack = [far] if is_sea(far) and not fill[far] else []
    if stack:
        opened[far] = 1
    while stack:
        cell = stack.pop()
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if not opened[neighbour] and is_sea(neighbour) and not fill[neighbour]:
                opened[neighbour] = 1
                stack.append(neighbour)

    seen = np.zeros(num_cells, dtype=np.uint8)
    max_cells = float(config["inlandSeaMaxKm2"]) / float(mesh.cell_area_km2)
    for start in range(num_cells):
        if seen[start] or not is_sea(start) or fill[start] or opened[start]:
            continue
        component = [start]
        seen[start] = 1
        cursor = 0
        while cursor < len(component):
            cell = component[cursor]
            cursor += 1
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if (
                    not seen[neighbour]
                    and is_sea(neighbour)
                    and not fill[neighbour]
                    and not opened[neighbour]
                ):
                    seen[neighbour] = 1
                    component.append(neighbour)
        if len(component) <= max_cells:
            for cell in component:
                fill[cell] = 1

    to_fill = [cell for cell in range(num_cells) if fill[cell]]
    if not to_fill:
        return 0

    _nearest_dist, nearest_source = hops_from(
        mesh, land_a + land_b, lambda cell: bool(fill[cell])
    )
    fallback = land_a[0]
    for cell in to_fill:
        source = int(nearest_source[cell]) if nearest_source[cell] >= 0 else fallback
        out.copy_from(cell, out, source)
        if out.type[cell] == CrustType["CRATON"]:
            out.type[cell] = CrustType["CONTINENT"]
        out.formedAt[cell] = time
        out.flags[cell] = CrustFlag["SUTURE"]
        out.slabKm[cell] = 0.0
        origin[cell] = 3
    return len(to_fill)


def accrete_terrane(world, host, donor, terrane, rate_cm_yr: float) -> dict:
    """Accrete land cells and fore-arc from one plate onto a host plate."""
    mesh, config, time = world.mesh, world.config, float(world.time)
    xyz, num_cells = mesh.xyz, int(mesh.num_cells)
    offsets, adjacency, index = mesh.adj_offset, mesh.adj_list, mesh.index
    host_crust, donor_crust = host.crust, donor.crust
    origin = np.zeros(num_cells, dtype=np.int8)
    for cell in range(num_cells):
        if host_crust.mask[cell] and host_crust.type[cell] != CrustType["OCEAN"]:
            origin[cell] = 1

    is_terrane = np.zeros(num_cells, dtype=np.uint8)
    for cell_value in terrane:
        is_terrane[int(cell_value)] = 1

    fore_arc = []
    seen = set(int(cell) for cell in terrane)
    ring = [int(cell) for cell in terrane]
    fore_arc_hops = max(
        1,
        math.floor(float(config["terraneForeArcKm"]) / float(mesh.edge_km) + 0.5),
    )
    for hop in range(fore_arc_hops):
        following = []
        for cell in ring:
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if (
                    neighbour in seen
                    or not donor_crust.mask[neighbour]
                    or donor_crust.type[neighbour] != CrustType["OCEAN"]
                    or (
                        hop > 0
                        and not (
                            int(donor_crust.flags[neighbour]) & CrustFlag["TRENCH"]
                        )
                    )
                ):
                    continue
                seen.add(neighbour)
                following.append(neighbour)
                fore_arc.append(neighbour)
        ring = following

    moved = 0
    absorbed = 0
    rotation = None
    for cell in [int(value) for value in terrane] + fore_arc:
        offset = 3 * cell
        world_point = donor.to_world(
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )
        target = host.cell_at(
            index,
            float(world_point[0]),
            float(world_point[1]),
            float(world_point[2]),
        )
        land = bool(is_terrane[cell])
        if land and origin[target]:
            target = free_cell_beside(
                host,
                mesh,
                target,
                world_point,
                lambda candidate: bool(origin[candidate]),
            )
            if target < 0:
                absorbed += 1
                donor_crust.remove(cell)
                continue

        if origin[target] != 1 and (land or not host_crust.mask[target]):
            if rotation is None:
                rotation = frame_rotation(donor, host)
            host_crust.copy_from(target, donor_crust, cell, rotation)
            if land:
                host_crust.flags[target] = int(host_crust.flags[target]) & (
                    0xFF ^ CrustFlag["TRENCH"]
                )
                origin[target] = 2
                moved += 1
        donor_crust.remove(cell)

    donor.mark_reshaped()
    host.mark_reshaped()
    if donor_crust.count == 0:
        donor.alive = False
        donor.omega = [0.0, 0.0, 0.0]
        donor.dynamics = None

    if len(terrane) * float(mesh.cell_area_km2) < float(config["terraneMinAreaKm2"]):
        host.refresh_extent(xyz)
        donor.refresh_extent(xyz)
        return {
            "cells": moved,
            "sealed": 0,
            "absorbed": absorbed,
            "scrap": True,
        }

    sealed = close_seas_between(host_crust, origin, mesh, host, config, time)
    seam = []
    for cell in range(num_cells):
        if not host_crust.mask[cell] or host_crust.type[cell] == CrustType["OCEAN"]:
            continue
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if (
                host_crust.mask[neighbour]
                and host_crust.type[neighbour] != CrustType["OCEAN"]
                and origin[neighbour]
                and origin[neighbour] != origin[cell]
            ):
                seam.append(cell)
                break

    for cell in seam:
        if host_crust.type[cell] != CrustType["CRATON"]:
            host_crust.flags[cell] |= CrustFlag["SUTURE"]

    width = belt_width_km("ural", 0.0, rate_cm_yr, config)
    work = float(config["belts"]["ural"]["baseKm"]) * 0.5
    upper_cells = [cell for cell in seam if origin[cell] == 1]
    lower_cells = [cell for cell in seam if origin[cell] != 1]
    if upper_cells:
        spread_belt(
            host,
            mesh,
            upper_cells,
            {
                "type": OrogenyType["URAL"],
                "widthKm": width,
                "workKm": work,
                "time": time,
                "side": BeltSide["UPPER"],
            },
            config,
            lambda cell: origin[cell] == 1,
        )
    if lower_cells:
        spread_belt(
            host,
            mesh,
            lower_cells,
            {
                "type": OrogenyType["URAL"],
                "widthKm": width,
                "workKm": work,
                "time": time,
                "side": BeltSide["LOWER"],
            },
            config,
            lambda cell: origin[cell] != 1,
        )

    host.refresh_extent(xyz)
    donor.refresh_extent(xyz)
    return {"cells": moved, "sealed": sealed, "absorbed": absorbed}


def landmass_of(plate, mesh, seeds) -> list[int]:
    """Return connected land cells of a plate containing any supplied seed."""
    crust = plate.crust
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    seen = np.zeros(len(crust.mask), dtype=np.uint8)
    result = []
    for seed_value in seeds:
        seed = int(seed_value)
        if (
            crust.mask[seed]
            and crust.type[seed] != CrustType["OCEAN"]
            and not seen[seed]
        ):
            seen[seed] = 1
            result.append(seed)
    cursor = 0
    while cursor < len(result):
        cell = result[cursor]
        cursor += 1
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if (
                not seen[neighbour]
                and crust.mask[neighbour]
                and crust.type[neighbour] != CrustType["OCEAN"]
            ):
                seen[neighbour] = 1
                result.append(neighbour)
    return result


__all__ = ["accrete_terrane", "close_seas_between", "landmass_of"]
