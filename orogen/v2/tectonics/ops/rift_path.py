"""Search for a continent-crossing rift path, ported from Orogen V2."""

from __future__ import annotations

import math

import numpy as np

from ...core.fast_noise import FastNoise
from ...core.min_heap import MinHeap
from ...core.sphere_geometry import (
    PLANET_RADIUS_KM,
    cell_distance_km,
    cross,
    dot,
    normalize,
    tangent_toward,
)
from ..crust import CrustFlag, CrustType
from .rift_partition import exterior_sea, masked_cells, sides_of

_RIFT_ARMS_APART_DEG = 135.0
_ARM_SEPARATION_DEG = 60.0
_PAIRS_TRIED = 12
_DISTINCT_CUT_KM = 500.0


def grain_of(world):
    """Return the cached seeded multiplier describing the plate's rift fabric."""
    config = world.config
    if not (config["riftCurveAmp"] > 0.0 or config["riftJagAmp"] > 0.0):
        return lambda _cell: 1.0

    noise = getattr(world, "riftGrain", None)
    if noise is None:
        noise = FastNoise((int(world.seed) ^ 0x6A41) & 0xFFFFFFFF)
        world.riftGrain = noise
    f1 = PLANET_RADIUS_KM / float(config["riftCurveKm"])
    f2 = PLANET_RADIUS_KM / float(config["riftJagKm"])
    xyz = world.mesh.xyz

    known = getattr(world, "riftGrainOf", None)
    if known is None:
        known = np.full(int(world.mesh.num_cells), np.nan, dtype=np.float64)
        world.riftGrainOf = known

    def grain(cell: int) -> float:
        value = float(known[cell])
        if not math.isnan(value):
            return value
        offset = 3 * cell
        x, y, z = (
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )
        value = max(
            0.2,
            1.0
            + float(config["riftCurveAmp"]) * noise.noise3d(x * f1, y * f1, z * f1)
            + float(config["riftJagAmp"])
            * noise.noise3d(
                x * f2 + 31.7,
                y * f2 + 17.3,
                z * f2 + 5.1,
            ),
        )
        known[cell] = value
        return value

    return grain


def rift_context(world, plate) -> dict:
    """Precompute values shared by all junction searches on one plate."""
    mesh = world.mesh
    crust = plate.crust
    exterior = exterior_sea(mesh, crust)
    return {
        "cells": masked_cells(crust),
        "exterior": exterior,
        "from_coast": coast_distance(mesh, crust, exterior),
        "grain": grain_of(world),
    }


def _coast_candidates(mesh, crust, junction, pull_axis, costs, ctx, config):
    """Rank oceanic coast cells reachable from a proposed rift junction."""
    xyz, offsets, adjacency = mesh.xyz, mesh.adj_offset, mesh.adj_list
    exterior = ctx["exterior"]
    junction_position = _position(xyz, junction)

    def land(cell: int) -> bool:
        return bool(crust.mask[cell] and crust.type[cell] != CrustType["OCEAN"])

    coast = []
    for cell_value in ctx["cells"]:
        cell = int(cell_value)
        if not land(cell) or not math.isfinite(float(costs["cost"][cell])):
            continue
        is_coast = False
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            if exterior[int(adjacency[edge])]:
                is_coast = True
                break
        if not is_coast or cell_distance_km(xyz, junction, cell) < float(
            config["riftMinArmKm"]
        ):
            continue
        direction = tangent_toward(
            junction_position,
            junction_position,
            _position(xyz, cell),
        )
        along = abs(dot(direction, pull_axis)) if pull_axis is not None else 0.0
        coast.append(
            {
                "cell": cell,
                "direction": direction,
                "score": float(costs["cost"][cell])
                * (1.0 + float(config["riftAcrossPullBias"]) * along),
            }
        )
    return coast


def _trace_parent_path(end: int, parent: np.ndarray) -> list[int]:
    """Reconstruct a root-to-end path from a predecessor array."""
    result = []
    cell = int(end)
    while cell >= 0:
        result.append(cell)
        cell = int(parent[cell])
    result.reverse()
    return result


def _angular_separation(first: dict, second: dict) -> float:
    """Return the angular separation of two candidate coast-arm directions."""
    cosine = max(-1.0, min(1.0, dot(first["direction"], second["direction"])))
    return math.degrees(math.acos(cosine))


def _nearby_rift_cuts(xyz, first: dict, second: dict) -> bool:
    """Whether two candidate cuts are within the duplicate-cut threshold."""
    return cell_distance_km(xyz, first["cell"], second["cell"]) < _DISTINCT_CUT_KM


def _select_rift_coast_pair(
    mesh,
    crust,
    junction: int,
    pull_axis,
    costs: dict,
    ctx: dict,
    config,
    need_cratons_both_sides: bool,
):
    """Rank coast endpoints and select the lowest-cost valid crossing."""
    xyz = mesh.xyz
    edge_km = float(mesh.edge_km)
    from_coast = ctx["from_coast"]
    coast = _coast_candidates(mesh, crust, junction, pull_axis, costs, ctx, config)

    if len(coast) < 2:
        return coast, None, None, math.inf
    coast.sort(key=lambda item: item["score"])
    top_count = max(
        8,
        math.floor(float(config["riftArmEndCoastKm"]) / edge_km + 0.5),
    )
    top = coast[:top_count]
    for item in top:
        item["length_km"] = (
            len(_trace_parent_path(item["cell"], costs["from"])) * edge_km
        )

    pairs = []
    for first in top:
        for second in top:
            if (
                first is second
                or _angular_separation(first, second) < _RIFT_ARMS_APART_DEG
            ):
                continue
            chord = cell_distance_km(xyz, first["cell"], second["cell"])
            if (
                chord < float(config["riftMinChordKm"])
                or first["length_km"] + second["length_km"]
                > float(config["riftTortuosityMax"]) * chord
            ):
                continue
            pairs.append(
                {
                    "a": first,
                    "b": second,
                    "per_km": (first["score"] + second["score"]) / chord,
                }
            )
    pairs.sort(key=lambda item: item["per_km"])

    chosen = None
    chosen_path = None
    best_value = math.inf
    tried: list[dict] = []
    for pair in pairs:
        if len(tried) >= _PAIRS_TRIED or (
            tried
            and pair["per_km"] > float(config["riftCostSlack"]) * pairs[0]["per_km"]
        ):
            break
        if any(
            (
                _nearby_rift_cuts(xyz, item["a"], pair["a"])
                and _nearby_rift_cuts(xyz, item["b"], pair["b"])
            )
            or (
                _nearby_rift_cuts(xyz, item["a"], pair["b"])
                and _nearby_rift_cuts(xyz, item["b"], pair["a"])
            )
            for item in tried
        ):
            continue
        tried.append(pair)

        path = (
            list(reversed(_trace_parent_path(pair["a"]["cell"], costs["from"])[1:]))
            + [junction]
            + _trace_parent_path(pair["b"]["cell"], costs["from"])[1:]
        )
        if not _is_monotone(mesh, path, config):
            continue
        if not _crosses_rather_than_hugs(mesh, path, from_coast, config):
            continue

        sides = sides_of(mesh, crust, path, ctx["cells"])
        if sides["share"] < float(config["riftMinPieceShare"]) or (
            need_cratons_both_sides and not sides["cratonsBothSides"]
        ):
            continue
        value = pair["per_km"] * (
            1.0 + float(config["riftElongationBias"]) * (sides["elongation"] - 1.0)
        )
        if value < best_value:
            best_value = value
            chosen = pair
            chosen_path = path

    return coast, chosen, chosen_path, best_value


def _trace_failed_rift_arms(
    mesh,
    xyz,
    crust,
    junction: int,
    coast: list[dict],
    arm1: dict,
    arm2: dict,
    grain,
    config,
) -> tuple[list[int], list[int]]:
    """Trace the abandoned coast arm and optionally retain its full path."""
    plain = _cheapest_paths(mesh, crust, junction, grain)
    failed_arm: list[int] = []
    taken = [arm1, arm2]
    plain_coast = [
        item for item in coast if math.isfinite(float(plain["cost"][item["cell"]]))
    ]
    plain_coast.sort(key=lambda item: float(plain["cost"][item["cell"]]))
    for item in plain_coast:
        if len(taken) >= 2 + int(config["riftFailedArms"]):
            break
        if any(
            _angular_separation(existing, item) < _ARM_SEPARATION_DEG
            for existing in taken
        ):
            continue
        taken.append(item)
        path = _trace_parent_path(item["cell"], plain["from"])
        distance = 0.0
        for index in range(1, len(path)):
            if distance >= float(config["riftFailedArmKm"]):
                break
            failed_arm.append(path[index])
            distance += cell_distance_km(xyz, path[index - 1], path[index])

    arm3_trace = (
        _trace_parent_path(taken[2]["cell"], plain["from"]) if len(taken) > 2 else []
    )
    return failed_arm, arm3_trace
def find_rift(
    world,
    plate,
    junction: int,
    pull_axis=None,
    need_cratons_both_sides: bool = True,
    ctx: dict | None = None,
):
    """Find the best valid coast-to-coast rift crossing the given junction.

    The result contains the chosen path, failed arm, three arm traces,
    great-circle normal and a cost/shape score. None means no valid cut.
    """
    mesh, config = world.mesh, world.config
    xyz = mesh.xyz
    edge_km = float(mesh.edge_km)
    crust = plate.crust

    def land(cell: int) -> bool:
        return bool(crust.mask[cell] and crust.type[cell] != CrustType["OCEAN"])

    if not land(junction) or crust.type[junction] == CrustType["CRATON"]:
        return None
    if ctx is None:
        ctx = rift_context(world, plate)

    weak = CrustFlag["SUTURE"] | CrustFlag["FAILED_RIFT"] | CrustFlag["RIFT_MARGIN"]
    grain = ctx["grain"]
    costs = _cheapest_paths(
        mesh,
        crust,
        junction,
        lambda cell: (
            (
                float(config["riftWeaknessCost"])
                if int(crust.flags[cell]) & weak
                else 1.0
            )
            * grain(cell)
        ),
    )

    coast, chosen, chosen_path, best_value = _select_rift_coast_pair(
        mesh,
        crust,
        junction,
        pull_axis,
        costs,
        ctx,
        config,
        need_cratons_both_sides,
    )
    if chosen is None or chosen_path is None:
        return None

    arm1 = chosen["a"]
    arm2 = chosen["b"]
    arm1_trace = _trace_parent_path(arm1["cell"], costs["from"])
    arm2_trace = _trace_parent_path(arm2["cell"], costs["from"])

    failed_arm, arm3_trace = _trace_failed_rift_arms(
        mesh,
        xyz,
        crust,
        junction,
        coast,
        arm1,
        arm2,
        grain,
        config,
    )
    ends = [chosen_path[0], chosen_path[-1]]
    across = normalize(
        cross(
            _position(xyz, ends[0]),
            _position(xyz, ends[1]),
        )
    )
    return {
        "path": chosen_path,
        "failed_arm": failed_arm,
        "arms": [arm1_trace, arm2_trace, arm3_trace],
        "across": across,
        "value": best_value,
    }


def coast_distance(mesh, crust, exterior: np.ndarray) -> np.ndarray:
    """Return land-cell hop distance from the plate's exterior coast."""
    num_cells = int(mesh.num_cells)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    distance = np.full(num_cells, -1, dtype=np.int16)

    def land(cell: int) -> bool:
        return bool(crust.mask[cell] and crust.type[cell] != CrustType["OCEAN"])

    ring = []
    for cell in range(num_cells):
        if not land(cell):
            continue
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            if exterior[int(adjacency[edge])]:
                distance[cell] = 0
                ring.append(cell)
                break

    depth = 1
    while ring:
        following = []
        for cell in ring:
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if land(neighbour) and distance[neighbour] < 0:
                    distance[neighbour] = depth
                    following.append(neighbour)
        ring = following
        depth += 1
    return distance


def _crosses_rather_than_hugs(mesh, path, from_coast, config) -> bool:
    edge_km = float(mesh.edge_km)
    clearance = math.floor(float(config["riftCoastClearanceKm"]) / edge_km + 0.5)
    max_hug = float(config["riftCoastHugKm"]) / edge_km
    end_free = max(1, clearance)
    hug = 0
    for index in range(end_free, len(path) - end_free):
        distance = int(from_coast[path[index]])
        if distance >= 0 and distance < clearance:
            hug += 1
            if hug > max_hug:
                return False
        else:
            hug = 0
    return True


def _cheapest_paths(mesh, crust, junction: int, cost_of) -> dict:
    """Dijkstra costs and predecessor tree; cratons cannot be entered."""
    xyz = mesh.xyz
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    num_cells = int(mesh.num_cells)
    edge_km = float(mesh.edge_km)
    cost = np.full(num_cells, math.inf, dtype=np.float64)
    parent = np.full(num_cells, -1, dtype=np.int32)

    def land(cell: int) -> bool:
        return bool(crust.mask[cell] and crust.type[cell] != CrustType["OCEAN"])

    heap = MinHeap()
    cost[junction] = 0.0
    heap.push(0.0, int(junction))
    while heap.size:
        cell = heap.pop()
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if not land(neighbour) or crust.type[neighbour] == CrustType["CRATON"]:
                continue
            next_cost = (
                float(cost[cell])
                + float(cost_of(neighbour))
                * cell_distance_km(xyz, cell, neighbour)
                / edge_km
            )
            if next_cost < cost[neighbour]:
                cost[neighbour] = next_cost
                parent[neighbour] = cell
                heap.push(next_cost, neighbour)
    return {"cost": cost, "from": parent}


def _is_monotone(mesh, path, config) -> bool:
    """Reject a path that backtracks too far along its endpoint great circle."""
    xyz = mesh.xyz
    first = _position(xyz, path[0])
    last = _position(xyz, path[-1])
    normal = normalize(cross(first, last))
    tolerance = max(
        float(config["riftOverhangKm"]) / PLANET_RADIUS_KM,
        float(mesh.edge_rad),
    )
    previous_angle = -math.inf
    for cell in path:
        point = _position(xyz, cell)
        projection = dot(point, normal)
        projected = normalize(
            [
                point[0] - projection * normal[0],
                point[1] - projection * normal[1],
                point[2] - projection * normal[2],
            ]
        )
        angle = math.atan2(
            dot(cross(first, projected), normal),
            dot(first, projected),
        )
        if angle < previous_angle - tolerance:
            return False
        previous_angle = max(previous_angle, angle)
    return True


def _position(xyz, cell: int) -> tuple[float, float, float]:
    offset = 3 * cell
    return (
        float(xyz[offset]),
        float(xyz[offset + 1]),
        float(xyz[offset + 2]),
    )
