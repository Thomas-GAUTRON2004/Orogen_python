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
from .geology import thicken_cover
from .heights import build_heights


def _value(values, index: int, fallback: float) -> float:
    if values is None or index >= len(values):
        return float(fallback)
    return float(values[index])


def _source_width(belt: Mapping, source_index: int) -> float:
    return _value(belt.get("widths"), source_index, belt["widthKm"])


def _source_work(belt: Mapping, source_index: int) -> float:
    return _value(belt.get("works"), source_index, belt["workKm"])


def _seed_front(
    count: int,
    sources: Sequence[int],
    belt: Mapping,
    best_u: np.ndarray,
    best_src: np.ndarray,
    heap: list[tuple[float, int, int, float, float]],
) -> None:
    """Initialize the multi-source priority queue in source order."""
    for source_index, cell_value in enumerate(sources):
        cell = int(cell_value)
        if cell < 0 or cell >= count:
            raise IndexError("source cell out of range")
        reach = _source_width(belt, source_index)
        if reach <= 0 or _source_work(belt, source_index) <= 0:
            continue
        if best_src[cell] < 0 or reach > _source_width(belt, int(best_src[cell])):
            best_u[cell] = 0.0
            best_src[cell] = source_index
            heapq.heappush(heap, (0.0, cell, source_index, 0.0, 0.0))


def _visit_front_neighbours(
    crust,
    mesh,
    config: Mapping,
    allow: Callable[[int], bool] | None,
    best_u: np.ndarray,
    best_src: np.ndarray,
    heap: list[tuple[float, int, int, float, float]],
    *,
    cell: int,
    source_index: int,
    u: float,
    sea_km: float,
    rim_km: float,
    cell_type: int,
    reach: float,
    rim_max: float,
    sea_limit: float,
) -> None:
    """Propagate a popped front cell to eligible adjacent cells."""
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    for pos in range(int(offsets[cell]), int(offsets[cell + 1])):
        neighbour = int(adjacency[pos])
        if not crust.mask[neighbour] or (allow is not None and not allow(neighbour)):
            continue
        neighbour_type = int(crust.type[neighbour])
        distance = cell_distance_km(mesh.xyz, cell, neighbour)
        if neighbour_type == CrustType["OCEAN"]:
            if cell_type != CrustType["OCEAN"]:
                continue
            next_sea = sea_km + distance
            if next_sea > sea_limit or best_u[neighbour] < math.inf:
                continue
            best_u[neighbour] = 0.0
            best_src[neighbour] = source_index
            heapq.heappush(
                heap,
                (0.0, neighbour, source_index, next_sea, 0.0),
            )
            continue

        hop = distance
        weak_flags = (
            CrustFlag["SUTURE"]
            | CrustFlag["FAILED_RIFT"]
            | CrustFlag["RIFT_MARGIN"]
        )
        if int(crust.flags[neighbour]) & weak_flags:
            hop *= float(config["weakCellCost"])
        next_rim = 0.0
        if neighbour_type == CrustType["CRATON"]:
            next_rim = (
                rim_km + distance
                if cell_type == CrustType["CRATON"]
                else 0.5 * distance
            )
            if next_rim > rim_max:
                continue
            hop *= float(config["cratonRimCost"])
        next_u = 0.0 if cell_type == CrustType["OCEAN"] else u + hop / reach
        if next_u >= best_u[neighbour] or next_u > 1.0:
            continue
        best_u[neighbour] = next_u
        best_src[neighbour] = source_index
        heapq.heappush(
            heap,
            (next_u, neighbour, source_index, 0.0, next_rim),
        )


def _propagate_front(
    plate,
    mesh,
    sources: Sequence[int],
    belt: Mapping,
    config: Mapping,
    allow: Callable[[int], bool] | None,
    rim_max: float,
    sea_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run multi-source shortest-path propagation for a single belt."""
    crust = plate.crust
    count = len(crust.mask)
    best_u = np.full(count, np.inf, dtype=np.float64)
    best_src = np.full(count, -1, dtype=np.int32)
    heap: list[tuple[float, int, int, float, float]] = []
    _seed_front(count, sources, belt, best_u, best_src, heap)

    while heap:
        u, cell, source_index, sea_km, rim_km = heapq.heappop(heap)
        if u > best_u[cell] + 1e-12 or source_index != best_src[cell]:
            continue
        cell_type = int(crust.type[cell])
        _visit_front_neighbours(
            crust,
            mesh,
            config,
            allow,
            best_u,
            best_src,
            heap,
            cell=cell,
            source_index=source_index,
            u=u,
            sea_km=sea_km,
            rim_km=rim_km,
            cell_type=cell_type,
            reach=_source_width(belt, source_index),
            rim_max=rim_max,
            sea_limit=sea_limit,
        )
    return best_u, best_src


def _preserve_previous_belt(
    crust,
    cell: int,
    was_type: int,
    was_start: float,
    was_last: float,
) -> None:
    """Keep the previous orogenic record before a new belt takes over."""
    if (
        was_type == OrogenyType["NONE"]
        or float(crust.beltReach[cell]) <= 0
        or (
            int(crust.oroType[cell]) == was_type
            and float(crust.oroStart[cell]) == was_start
        )
    ):
        return
    crust.prevType[cell] = was_type
    crust.prevLast[cell] = was_last
    crust.prevU[cell] = crust.beltU[cell]
    crust.prevReach[cell] = crust.beltReach[cell]
    crust.prevSide[cell] = crust.beltSide[cell]
    crust.prevShear[cell] = crust.beltShear[cell]
    crust.prevRugged[cell] = crust.rugged[cell]
    crust.rugged[cell] = 0
    crust.beltReach[cell] = 0


def _update_belt_front(
    crust,
    cell: int,
    belt: Mapping,
    source_index: int,
    oro_type: int,
    side: int,
    u: float,
    reach: float,
    time: float,
) -> None:
    """Store the source and normalized distance that define a belt front."""
    if int(crust.oroType[cell]) != oro_type:
        return
    if not (
        crust.beltReach[cell] == 0
        or crust.beltAt[cell] < time
        or u < float(crust.beltU[cell])
    ):
        return
    crust.beltU[cell] = u
    crust.beltReach[cell] = reach
    crust.beltAt[cell] = time
    crust.beltSide[cell] = side
    if belt.get("keepShear", False):
        return
    shear = _value(belt.get("shears"), source_index, belt.get("shear", 0))
    crust.beltShear[cell] = max(-90, min(90, int(math.floor(shear + 0.5))))


def _record_belt_cell(
    crust,
    cell: int,
    belt: Mapping,
    config: Mapping,
    source_index: int,
    u: float,
    rim_cell: bool,
    time: float,
    quiet: float,
    oro_type: int,
    side: int,
) -> None:
    """Apply geological records, thickening, relief, and sediment cover."""
    work = _source_work(belt, source_index)
    if rim_cell:
        work *= float(config["cratonRimShare"])
    record_work = (
        work
        if u == 0
        else work * float(config["beltWorkShare"]) * (1.0 - u)
    )
    was_type = int(crust.oroType[cell])
    was_start = float(crust.oroStart[cell])
    was_last = float(crust.oroLast[cell])
    crust.record_orogeny(cell, oro_type, record_work, time, quiet)
    _preserve_previous_belt(crust, cell, was_type, was_start, was_last)

    reach = _source_width(belt, source_index)
    _update_belt_front(
        crust, cell, belt, source_index, oro_type, side, u, reach, time
    )
    thickened = 0.0
    if not rim_cell and not belt.get("noRoot", False):
        thickened = (
            work
            * float(config["beltWorkShare"])
            * (1.0 - u * u)
            * float(config["thickeningPerWorkKm"])
        )
    crust.thickKm[cell] += thickened
    amp = _value(belt.get("amps"), source_index, 1.0)
    relief_share = float(config["cratonRimShare"]) if rim_cell else 1.0
    build_heights(
        crust,
        cell,
        oro_type,
        side,
        u,
        reach,
        work,
        thickened,
        config,
        relief_share * amp,
    )
    geology = config.get("geology")
    if thickened > 0 and geology and geology.get("records"):
        thicken_cover(crust, cell, thickened, time, geology)


def _record_reached_cells(
    plate,
    belt: Mapping,
    config: Mapping,
    best_u: np.ndarray,
    best_src: np.ndarray,
    rim_max: float,
) -> list[int]:
    """Apply the belt to reached non-ocean cells in ascending cell order."""
    crust = plate.crust
    oro_type = int(belt["type"])
    time = float(belt["time"])
    quiet = float(config["orogenyQuietMyr"])
    side = int(belt.get("side", BeltSide["NONE"]))
    reached: list[int] = []
    for cell_value in np.flatnonzero(np.isfinite(best_u)):
        cell = int(cell_value)
        if int(crust.type[cell]) == CrustType["OCEAN"]:
            continue
        rim_cell = int(crust.type[cell]) == CrustType["CRATON"]
        if rim_cell and rim_max <= 0:
            continue
        _record_belt_cell(
            crust,
            cell,
            belt,
            config,
            int(best_src[cell]),
            float(best_u[cell]),
            rim_cell,
            time,
            quiet,
            oro_type,
            side,
        )
        reached.append(cell)
    return reached


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
    count = len(plate.crust.mask)
    if len(mesh.xyz) != 3 * count:
        raise ValueError("mesh and plate crust must have the same cell count")
    rim_max = float(config.get("cratonRimKm", 0.0))
    sea_limit = float(config["apronKm"]) * 1.5 + float(mesh.edge_km)
    best_u, best_src = _propagate_front(
        plate,
        mesh,
        sources,
        belt,
        config,
        allow,
        rim_max,
        sea_limit,
    )
    return _record_reached_cells(
        plate, belt, config, best_u, best_src, rim_max
    )
