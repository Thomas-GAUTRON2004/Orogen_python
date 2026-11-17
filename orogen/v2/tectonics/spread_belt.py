"""Spatial propagation of orogenic belts over a plate crust raster.

This ports the record/geometry part of V2 step/orogeny.js. Height synthesis
(buildHeights) and sediment-cover synthesis (thickenCover) remain separate
systems and are intentionally not approximated here.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from ..core.sphere_geometry import cell_distance_km
from .crust import BeltSide, CrustFlag, CrustType, OrogenyType


def _value(values, index: int, fallback: float) -> float:
    if values is None or index >= len(values):
        return float(fallback)
    return float(values[index])


def spread_belt(
    plate,
    mesh,
    sources: Sequence[int],
    belt: Mapping,
    config: Mapping,
    allow: Callable[[int], bool] | None = None,
) -> list[int]:
    """Propagate one belt from source cells and update per-cell geological records.

    belt requires type, widthKm, workKm and time. Optional per-source
    widths/works/shears override the scalar values. Returns shaped land cells.
    """
    if not sources:
        return []
    crust = plate.crust
    count = len(crust.mask)
    if len(mesh.xyz) != 3 * count:
        raise ValueError("mesh and plate crust must have the same cell count")
    offsets, adjacency, xyz = mesh.adj_offset, mesh.adj_list, mesh.xyz
    widths = belt.get("widths")
    works = belt.get("works")
    width_for = lambda i: _value(widths, i, belt["widthKm"])
    work_for = lambda i: _value(works, i, belt["workKm"])
    rim_max = float(config.get("cratonRimKm", 0.0))
    sea_limit = float(config["apronKm"]) * 1.5 + float(mesh.edge_km)

    # Multi-source shortest path; normalized distance is specific to each front source.
    best_u = np.full(count, np.inf, dtype=np.float64)
    best_src = np.full(count, -1, dtype=np.int32)
    heap: list[tuple[float, int, int, float, float]] = []
    for source_index, cell in enumerate(sources):
        cell = int(cell)
        if cell < 0 or cell >= count:
            raise IndexError("source cell out of range")
        reach = width_for(source_index)
        if reach <= 0 or work_for(source_index) <= 0:
            continue
        if best_src[cell] < 0 or reach > width_for(int(best_src[cell])):
            best_u[cell] = 0.0
            best_src[cell] = source_index
            heapq.heappush(heap, (0.0, cell, source_index, 0.0, 0.0))

    while heap:
        u, cell, source_index, sea_km, rim_km = heapq.heappop(heap)
        if u > best_u[cell] + 1e-12 or source_index != best_src[cell]:
            continue
        cell_type = int(crust.type[cell])
        at_sea = cell_type == CrustType["OCEAN"]
        reach = width_for(source_index)
        for pos in range(int(offsets[cell]), int(offsets[cell + 1])):
            nb = int(adjacency[pos])
            if not crust.mask[nb] or (allow is not None and not allow(nb)):
                continue
            nb_type = int(crust.type[nb])
            distance = cell_distance_km(xyz, cell, nb)
            if nb_type == CrustType["OCEAN"]:
                if not at_sea:
                    continue
                next_sea = sea_km + distance
                if next_sea > sea_limit or best_u[nb] < math.inf:
                    continue
                best_u[nb] = 0.0
                best_src[nb] = source_index
                heapq.heappush(heap, (0.0, nb, source_index, next_sea, 0.0))
                continue

            hop = distance
            if int(crust.flags[nb]) & (
                CrustFlag["SUTURE"] | CrustFlag["FAILED_RIFT"] | CrustFlag["RIFT_MARGIN"]
            ):
                hop *= float(config["weakCellCost"])
            rim = 0.0
            if nb_type == CrustType["CRATON"]:
                rim = distance * 0.5 if cell_type != CrustType["CRATON"] else 0.0
                if rim > rim_max:
                    continue
                hop *= float(config["cratonRimCost"])
            next_u = 0.0 if at_sea else u + hop / reach
            if next_u >= best_u[nb] or next_u > 1.0:
                continue
            best_u[nb] = next_u
            best_src[nb] = source_index
            heapq.heappush(heap, (next_u, nb, source_index, 0.0, rim))

    reached: list[int] = []
    oro_type = int(belt["type"])
    time = float(belt["time"])
    quiet = float(config["orogenyQuietMyr"])
    side = int(belt.get("side", BeltSide["NONE"]))
    for cell_value in np.flatnonzero(np.isfinite(best_u)):
        cell = int(cell_value)
        if int(crust.type[cell]) == CrustType["OCEAN"]:
            continue
        rim_cell = int(crust.type[cell]) == CrustType["CRATON"]
        if rim_cell and rim_max <= 0:
            continue
        source_index = int(best_src[cell])
        work = work_for(source_index) * (
            float(config["cratonRimShare"]) if rim_cell else 1.0
        )
        u = float(best_u[cell])
        record_work = work if u == 0 else work * float(config["beltWorkShare"]) * (1.0 - u)
        was_type = int(crust.oroType[cell])
        was_start = float(crust.oroStart[cell])
        was_last = float(crust.oroLast[cell])
        crust.record_orogeny(cell, oro_type, record_work, time, quiet)
        if (
            was_type != OrogenyType["NONE"]
            and float(crust.beltReach[cell]) > 0
            and (int(crust.oroType[cell]) != was_type or float(crust.oroStart[cell]) != was_start)
        ):
            crust.prevType[cell] = was_type
            crust.prevLast[cell] = was_last
            crust.prevU[cell] = crust.beltU[cell]
            crust.prevReach[cell] = crust.beltReach[cell]
            crust.prevSide[cell] = crust.beltSide[cell]
            crust.prevShear[cell] = crust.beltShear[cell]
            crust.prevRugged[cell] = crust.rugged[cell]
            crust.rugged[cell] = 0
            crust.beltReach[cell] = 0

        reach = width_for(source_index)
        if int(crust.oroType[cell]) == oro_type and (
            crust.beltReach[cell] == 0
            or crust.beltAt[cell] < time
            or u < float(crust.beltU[cell])
        ):
            crust.beltU[cell] = u
            crust.beltReach[cell] = reach
            crust.beltAt[cell] = time
            crust.beltSide[cell] = side
            if not belt.get("keepShear", False):
                shear = _value(belt.get("shears"), source_index, belt.get("shear", 0))
                crust.beltShear[cell] = max(-90, min(90, int(math.floor(shear + 0.5))))

        thickened = 0.0
        if not rim_cell and not belt.get("noRoot", False):
            thickened = (
                work * float(config["beltWorkShare"]) * (1.0 - u * u)
                * float(config["thickeningPerWorkKm"])
            )
        crust.thickKm[cell] += thickened
        reached.append(cell)

    return reached
