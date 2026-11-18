"""Cohesive continental landmass analysis for the V2 supercontinent cycle."""

from __future__ import annotations

import math

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
        if (
            not continental[start]
            or float(inland[start]) <= core_depth
            or core[start] >= 0
        ):
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
        if largest_label >= 0
        else []
    )
    return {
        "largest": largest,
        "share": len(largest) / total if total else 0.0,
        "count": count,
    }


def _centroid(mesh, cells):
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape(-1, 3)
    total = xyz[np.asarray(cells, dtype=np.int32)].sum(axis=0)
    length = float(np.linalg.norm(total)) or 1.0
    return (total / length).tolist()


def _basis_around(position):
    p = np.asarray(position, dtype=np.float64)
    axis = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(p[0])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    first = np.cross(p, axis)
    first_length = float(np.linalg.norm(first)) or 1.0
    first /= first_length
    second = np.cross(p, first)
    return first, second


def _heat_near(world, cycle) -> float:
    if cycle.get("up") is None:
        return 0.0
    if cycle.get("near") is None:
        xyz = np.asarray(world.mesh.xyz, dtype=np.float64).reshape(-1, 3)
        up = np.asarray(cycle["up"], dtype=np.float64)
        cutoff = math.cos(float(world.config["cycleUpwellingKm"]) / PLANET_RADIUS_KM)
        cells = np.flatnonzero(xyz @ up >= cutoff)
        cycle["near"] = cells.astype(np.int32)
    near = cycle["near"]
    if len(near) == 0:
        return 0.0
    return float(np.mean(np.asarray(world.mantleHeat)[near]))


def track_cycle(world):
    """Advance the supercontinent cycle using the resolved continental owner map.

    Cycle creation, breakup and gathering timing are deterministic for a fixed
    world seed because downwelling placement and cycle duration use rngCycle.
    This function does not update mantleHeat itself; the hotspot/thermal stage
    is responsible for that field.
    """
    from .world import EventKind

    config = world.config
    if not (
        float(config.get("mantleOutflow", 0.0)) > 0
        or float(config.get("mantleInflow", 0.0)) > 0
        or float(config.get("riftHeatGain", 0.0)) > 0
    ):
        return getattr(world, "cycle", None)

    time = float(world.time)
    events = world.events
    cycle = getattr(world, "cycle", None)
    if cycle is None:
        cycle = {
            "up": None,
            "down": None,
            "upHeat": 0.0,
            "gather": 0.0,
            "share": 1.0,
            "landmasses": 1,
            "state": "dispersed",
            "since": 0.0,
            "phase": "none",
            "formedAt": 0.0,
            "period": 0.0,
            "breakupAt": None,
            "gatherAt": math.inf,
            "hold": 0.0,
            "holdKm": 0.0,
            "checkedAt": -math.inf,
            "near": None,
            "highSince": None,
            "lowSince": None,
            "seenEvents": 0,
        }
        world.cycle = cycle

    while int(cycle["seenEvents"]) < len(events):
        event = events[int(cycle["seenEvents"])]
        cycle["seenEvents"] += 1
        if (
            event["kind"] != EventKind["RIFT"]
            or cycle["up"] is None
            or cycle["breakupAt"] is not None
            or float(event["time"])
            < float(cycle["formedAt"]) - float(config["cycleHoldMyr"]) - 1e-9
        ):
            continue
        cycle["breakupAt"] = float(event["time"])
        cycle["gatherAt"] = max(
            float(cycle["formedAt"])
            + float(cycle["period"])
            - float(config["cycleGatherMyr"]),
            float(event["time"]) + float(config["cycleDisperseMinMyr"]),
        )

    if (
        time - float(cycle["checkedAt"]) >= float(config["cycleCheckMyr"]) - 1e-9
        or cycle["up"] is None
    ):
        cycle["checkedAt"] = time
        result = landmasses(world)
        largest, share, count = (
            result["largest"],
            float(result["share"]),
            int(result["count"]),
        )
        cycle["share"] = share
        cycle["landmasses"] = count

        by_plate = {}
        for cell in largest:
            plate_id = int(world.owner[cell])
            by_plate[plate_id] = by_plate.get(plate_id, 0) + 1
        cycle["largestPlates"] = [
            plate_id
            for plate_id, size in by_plate.items()
            if size >= 0.15 * len(largest)
        ]

        cycle["highSince"] = (
            cycle["highSince"]
            if share >= float(config["superShare"]) and cycle["highSince"] is not None
            else time
            if share >= float(config["superShare"])
            else None
        )
        cycle["lowSince"] = (
            cycle["lowSince"]
            if share <= float(config["dispersedShare"])
            and cycle["lowSince"] is not None
            else time
            if share <= float(config["dispersedShare"])
            else None
        )

        def held(since):
            return (
                since is not None
                and time - float(since) >= float(config["cycleHoldMyr"]) - 1e-9
            )

        may_form = (cycle["up"] is None and cycle["highSince"] is not None) or (
            cycle["up"] is not None
            and cycle["state"] != "assembled"
            and time >= float(cycle["gatherAt"]) - 1e-9
            and held(cycle["highSince"])
        )
        if may_form and held(cycle["highSince"]):
            up = _centroid(world.mesh, largest)
            first, second = _basis_around(up)
            angle = 2.0 * math.pi * float(world.rngCycle())
            down = (math.cos(angle) * first + math.sin(angle) * second).tolist()
            period = float(config["cyclePeriodMinMyr"]) + (
                float(config["cyclePeriodMaxMyr"]) - float(config["cyclePeriodMinMyr"])
            ) * float(world.rngCycle())
            cycle["up"] = up
            cycle["down"] = down
            cycle["period"] = period
            cycle["formedAt"] = time
            cycle["breakupAt"] = None
            cycle["gatherAt"] = math.inf
            cycle["holdKm"] = math.sqrt(
                len(largest) * float(world.mesh.cell_area_km2) / math.pi
            ) + float(config["superHoldMarginKm"])
            cycle["near"] = None
            cycle["state"] = "assembled"
            cycle["since"] = time
            world.log(
                EventKind["SUPERCONTINENT"],
                (
                    f"a supercontinent: {100 * share:.0f} % of the continental crust "
                    f"in one landmass ({len(largest) * float(world.mesh.cell_area_km2) / 1e6:.0f} Mkm²) "
                    f"— the mantle under it will warm; the next one will gather ~90° away, "
                    f"~{period:.0f} Myr from now"
                ),
                {
                    "share": share,
                    "areaMkm2": len(largest) * float(world.mesh.cell_area_km2) / 1e6,
                    "up": cycle["up"],
                    "down": cycle["down"],
                    "periodMyr": period,
                },
            )
        elif cycle["state"] == "assembled" and held(cycle["lowSince"]):
            cycle["state"] = "dispersed"
            cycle["since"] = time
            world.log(
                EventKind["DISPERSED"],
                f"the supercontinent has broken up: the largest landmass holds {100 * share:.0f} % of the continental crust",
                {"share": share, "landmasses": count},
            )

    cycle["upHeat"] = _heat_near(world, cycle)
    if cycle["up"] is None:
        cycle["phase"] = "none"
    elif cycle["breakupAt"] is None:
        cycle["phase"] = "assembled"
    elif time < float(cycle["gatherAt"]):
        cycle["phase"] = "dispersing"
    else:
        cycle["phase"] = "gathering"

    if cycle["phase"] == "gathering":
        ramp = float(config["cycleRampMyr"])
        cycle["gather"] = (
            min(1.0, (time - float(cycle["gatherAt"])) / ramp) if ramp > 0 else 1.0
        )
    else:
        cycle["gather"] = 0.0
    cycle["clump"] = (
        cycle["largestPlates"]
        if cycle["phase"] == "dispersing" and share > float(config["dispersalMaxShare"])
        else None
    )
    cycle["hold"] = (
        1.0 if cycle["phase"] == "assembled" and float(cycle["formedAt"]) > 0 else 0.0
    )
    return cycle
