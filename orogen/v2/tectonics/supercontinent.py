"""Cohesive continental landmass analysis for the V2 supercontinent cycle."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..core.min_heap import spread_km
from ..core.sim_mesh import PLANET_RADIUS_KM
from .crust import CrustType


def is_continental(world, cell: int) -> bool:
    """True for continental or cratonic crust; island arcs do not count as land."""
    owner = int(world.owner[cell])
    if owner < 0:
        return False
    kind = int(world.plates[owner].crust.type[int(world.ownerCell[cell])])
    return kind in (CrustType["CONTINENT"], CrustType["CRATON"])


def landmasses(world) -> dict:
    """Return the largest cohesive continental mass, its share, and large-piece count.

    A narrow land neck does not unite two continental cores. Core cells lie
    farther than cycleNeckKm/2 inland; a multi-source geodesic expansion then
    assigns remaining continental cells to their nearest core. The separate
    count measures plain connected continental components of at least 1 Mkm².
    """
    mesh, config = world.mesh, world.config
    n = int(mesh.num_cells)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    cell_area = float(mesh.cell_area_km2)

    continental = np.zeros(n, dtype=np.uint8)
    total = 0
    for cell in range(n):
        if is_continental(world, cell):
            continental[cell] = 1
            total += 1

    # Count components joined by any land (the cycle's headline count).
    seen = np.zeros(n, dtype=np.uint8)
    count = 0
    for start in range(n):
        if not continental[start] or seen[start]:
            continue
        queue = [start]
        seen[start] = 1
        for cell in queue:
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if continental[neighbour] and not seen[neighbour]:
                    seen[neighbour] = 1
                    queue.append(neighbour)
        if len(queue) * cell_area >= 1e6:
            count += 1

    # Distance inland from the ocean: ocean cells are Dijkstra sources.
    inland = np.full(n, np.inf, dtype=np.float32)
    inland[continental == 0] = 0.0
    spread_km(mesh, inland)
    core_depth = float(config["cycleNeckKm"]) / 2.0

    # Split continental interiors at narrow necks.
    core = np.full(n, -1, dtype=np.int32)
    core_count = 0
    for start in range(n):
        if not continental[start] or float(inland[start]) <= core_depth or core[start] >= 0:
            continue
        queue = [start]
        core[start] = core_count
        for cell in queue:
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if (
                    continental[neighbour]
                    and float(inland[neighbour]) > core_depth
                    and core[neighbour] < 0
                ):
                    core[neighbour] = core_count
                    queue.append(neighbour)
        core_count += 1

    # Grow every core back through land; each border cell follows its nearest core.
    distance = np.full(n, np.inf, dtype=np.float32)
    distance[core >= 0] = 0.0

    def on_reach(source: int, target: int) -> None:
        core[target] = core[source]

    spread_km(
        mesh,
        distance,
        can_enter=lambda source, target: bool(continental[target]),
        on_reach=on_reach,
    )

    sizes = np.zeros(core_count, dtype=np.int32)
    for cell in range(n):
        label = int(core[cell])
        if continental[cell] and label >= 0:
            sizes[label] += 1

    largest_label = -1
    for label in range(core_count):
        if largest_label < 0 or sizes[label] > sizes[largest_label]:
            largest_label = label

    largest = (
        [cell for cell in range(n) if continental[cell] and core[cell] == largest_label]
        if largest_label >= 0 else []
    )
    return {
        "largest": largest,
        "share": len(largest) / total if total else 0.0,
        "count": count,
    }
