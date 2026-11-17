"""Far-field shortening records spread inland from tectonic fronts.

Port of V2 step/far-field.js record and diffusion stages. This module
records how much shortening reaches each land cell, its weighted path distance,
its decaying recent component, and the direction away from the source front.
Old-line reactivation is a separate, not-yet-scheduled substage.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..core.min_heap import MinHeap
from ..core.sphere_geometry import cell_distance_km
from .crust import CrustFlag, CrustType

_WEAK = (
    CrustFlag["SUTURE"]
    | CrustFlag["FAILED_RIFT"]
    | CrustFlag["RIFT_MARGIN"]
)


def feed_far_field(
    world,
    plate,
    key: str,
    cells: Sequence[int],
    works: float | Sequence[float],
    front_km: float,
    share: float,
) -> None:
    """Accumulate one front's work until the next far-field spread."""
    if not float(share) > 0.0 or not cells:
        return
    pending = getattr(world, "farPending", None)
    if pending is None:
        pending = {}
        world.farPending = pending

    group = pending.get(key)
    if group is None or group["plate"] is not plate:
        group = {"plate": plate, "cells": {}, "frontKm": 0.0}
        pending[key] = group
    group["frontKm"] = max(float(group["frontKm"]), float(front_km))

    scalar = np.isscalar(works)
    for index, cell_value in enumerate(cells):
        cell = int(cell_value)
        work = float(works) if scalar else float(works[index])
        pushed = work * float(share)
        if pushed > 0.0:
            group["cells"][cell] = group["cells"].get(cell, 0.0) + pushed


def evolve_far_field(world, dt: float) -> int:
    """Spread accumulated fronts whenever the far-field clock is due.

    The V2 far-field stage records geological history only and returns zero;
    it does not change plate motion or resolve crust ownership.
    """
    step_dt = float(dt)
    if not math.isfinite(step_dt) or step_dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    world.farClock = float(getattr(world, "farClock", 0.0)) + step_dt
    interval = float(world.config["farFieldEveryMyr"])
    if world.farClock < interval - 1e-9:
        return 0

    window = world.farClock
    world.farClock = 0.0
    pending = getattr(world, "farPending", None)
    if not pending:
        return 0

    groups = sorted(
        pending.values(),
        key=lambda item: int(item["plate"].id),
    )
    for group in groups:
        plate = group["plate"]
        if not plate.alive:
            continue
        spread_push(
            world,
            plate,
            group["cells"],
            float(group["frontKm"]),
            float(world.time),
        )
    pending.clear()
    # Retain the elapsed spread window for the future reactivation substage.
    world.farLastWindow = window
    return 0


def spread_push(
    world,
    plate,
    sources: Mapping[int, float],
    front_km: float,
    time: float,
) -> dict:
    """Diffuse one front's strongest attenuated push through occupied crust.

    Path cost is reduced through cratons (which pass the push on) and raised
    through sutures/failed-rift/rift-margin cells (which absorb it). Ocean
    cells are not traversed and receive no records.
    """
    config, mesh, crust = world.config, world.mesh, plate.crust
    xyz = mesh.xyz
    count = len(crust.mask)
    if len(xyz) != 3 * count or int(mesh.num_cells) != count:
        raise ValueError("mesh and plate crust must have the same cell count")

    reach_km = float(config["farReachKm"])
    reference_front = float(config["farFrontRefKm"])
    if reach_km <= 0.0 or reference_front <= 0.0:
        raise ValueError("farReachKm and farFrontRefKm must be positive")
    reach_length = reach_km * min(
        1.6, max(0.5, math.sqrt(max(0.0, float(front_km)) / reference_front))
    )

    active_sources = [
        (int(cell), float(work))
        for cell, work in sources.items()
        if 0 <= int(cell) < count and crust.mask[int(cell)] and float(work) > 0.0
    ]
    if not active_sources:
        return {
            "cells": [], "inc": np.zeros(0, dtype=np.float32),
            "from": np.zeros(0, dtype=np.int32),
            "dist": np.zeros(0, dtype=np.float64),
        }

    strongest = max(work for _, work in active_sources)
    distance = np.full(count, np.inf, dtype=np.float64)
    source_of = np.full(count, -1, dtype=np.int32)
    heap = MinHeap(max(64, 2 * len(active_sources)))
    for cell, work in active_sources:
        initial = -reach_length * math.log(work / strongest)
        if initial < distance[cell]:
            distance[cell] = initial
            source_of[cell] = cell
            heap.push(initial, cell)

    offsets = mesh.adj_offset
    adjacency = mesh.adj_list
    max_cost = float(config["farMaxKm"])
    reached: list[int] = []
    while heap.size:
        cell = heap.pop()
        current = heap.last_key
        if current > distance[cell]:
            continue
        reached.append(cell)
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if not crust.mask[neighbour]:
                continue
            neighbour_type = int(crust.type[neighbour])
            if neighbour_type == CrustType["CRATON"]:
                factor = float(config["farCratonCost"])
            elif int(crust.flags[neighbour]) & _WEAK:
                factor = float(config["farWeakCost"])
            else:
                factor = 1.0
            candidate = current + cell_distance_km(xyz, cell, neighbour) * factor
            if candidate >= distance[neighbour] or candidate > max_cost:
                continue
            distance[neighbour] = candidate
            source_of[neighbour] = int(source_of[cell])
            heap.push(candidate, neighbour)

    output_cells: list[int] = []
    increments: list[float] = []
    output_sources: list[int] = []
    output_distances: list[float] = []
    fold_myr = float(config["farFoldMyr"])
    for cell in reached:
        if int(crust.type[cell]) == CrustType["OCEAN"]:
            continue
        arrived = strongest * math.exp(-float(distance[cell]) / reach_length)
        if not arrived > 1e-4:
            continue
        inc = arrived * far_take(config, int(crust.type[cell]), int(crust.flags[cell]))
        record_push(
            crust,
            cell,
            arrived,
            float(distance[cell]),
            xyz,
            int(source_of[cell]),
            float(time),
            fold_myr,
        )
        output_cells.append(cell)
        increments.append(inc)
        output_sources.append(int(source_of[cell]))
        output_distances.append(float(distance[cell]))

    return {
        "cells": output_cells,
        "inc": np.asarray(increments, dtype=np.float32),
        "from": np.asarray(output_sources, dtype=np.int32),
        "dist": np.asarray(output_distances, dtype=np.float64),
    }


def record_push(
    crust,
    cell: int,
    felt: float,
    distance_km: float,
    xyz,
    source: int,
    time: float,
    fold_myr: float,
) -> None:
    """Accumulate whole-history and exponentially fading far-field records."""
    old_work = float(crust.farWork[cell])
    total_work = old_work + float(felt)
    crust.farWork[cell] = total_work
    crust.farDist[cell] = (
        float(crust.farDist[cell]) * old_work
        + float(distance_km) * float(felt)
    ) / total_work
    elapsed = max(0.0, float(time) - float(crust.farAt[cell]))
    previous_fold = float(crust.farFold[cell])
    crust.farFold[cell] = (
        previous_fold * math.exp(-elapsed / float(fold_myr)) + float(felt)
    )
    crust.farAt[cell] = float(time)

    if source < 0 or source == cell:
        return
    offset = 3 * cell
    source_offset = 3 * source
    px, py, pz = (float(v) for v in xyz[offset:offset + 3])
    qx, qy, qz = (float(v) for v in xyz[source_offset:source_offset + 3])
    cosine = px * qx + py * qy + pz * qz
    ax, ay, az = cosine * px - qx, cosine * py - qy, cosine * pz - qz
    length = math.hypot(ax, ay, az)
    if length < 1e-9:
        return
    ax, ay, az = ax / length, ay / length, az / length

    vx = float(crust.farDx[cell]) / 127.0
    vy = float(crust.farDy[cell]) / 127.0
    vz = float(crust.farDz[cell]) / 127.0
    if vx * ax + vy * ay + vz * az < 0.0:
        vx, vy, vz = -vx, -vy, -vz
    x = vx * old_work + ax * float(felt)
    y = vy * old_work + ay * float(felt)
    z = vz * old_work + az * float(felt)
    norm = math.hypot(x, y, z) or 1.0
    crust.farDx[cell] = max(-127, min(127, math.floor(127.0 * x / norm + 0.5)))
    crust.farDy[cell] = max(-127, min(127, math.floor(127.0 * y / norm + 0.5)))
    crust.farDz[cell] = max(-127, min(127, math.floor(127.0 * z / norm + 0.5)))


def far_take(config: Mapping, crust_type: int, flags: int) -> float:
    """Return the share of arrived push recorded as deforming work."""
    if crust_type == CrustType["CRATON"]:
        return float(config["farCratonTake"])
    if flags & _WEAK:
        return float(config["farWeakTake"])
    return 1.0


def recent_push(far_fold: float, far_at: float, time: float, fold_myr: float) -> float:
    """Return the recent far-field push remaining at time."""
    if float(far_fold) <= 0.0:
        return 0.0
    return float(far_fold) * math.exp(
        -max(0.0, float(time) - float(far_at)) / float(fold_myr)
    )
