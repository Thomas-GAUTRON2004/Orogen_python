"""Rift initiation and plate splitting operations, ported from Orogen V2."""

from __future__ import annotations

import math

import numpy as np

from ...core.sphere_geometry import dot, normalize, tangent_toward
from ..crust import CrustFlag, CrustType, OrogenyType
from ..spread_belt import spread_belt
from .rift_partition import partition
from .rift_path import find_rift, rift_context


def split_plate(world, plate, path, failed_arm, name_for, across=None):
    """Split a plate along a coast-to-coast rift path and return the new plate."""
    mesh = world.mesh
    xyz = mesh.xyz
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    num_cells = int(mesh.num_cells)
    crust = plate.crust
    path = [int(cell) for cell in path]
    failed_arm = [int(cell) for cell in failed_arm]

    on_path = np.zeros(num_cells, dtype=np.uint8)
    for cell in path:
        on_path[cell] = 1

    side, _land, _cratons = partition(mesh, crust, path)
    fresh = world.add_plate("")
    fresh.set_quat(plate.quat.copy())
    fresh.omega = plate.omega.copy()

    for cell in range(num_cells):
        if not crust.mask[cell] or side[cell] != 2:
            continue
        fresh.crust.copy_from(cell, crust, cell)
        crust.remove(cell)

    # Iterate over a snapshot because Python dictionaries cannot be mutated
    # during iteration (the upstream JavaScript Map can).
    for tip, direction in list(plate.wings.items()):
        if side[int(tip)] == 2:
            fresh.wings[tip] = direction
            del plate.wings[tip]

    shoulders = [[], []]
    for cell in path:
        if crust.mask[cell]:
            crust.flags[cell] |= CrustFlag["RIFT_MARGIN"]
            shoulders[0].append(cell)

    for cell in range(num_cells):
        if not fresh.crust.mask[cell]:
            continue
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            if on_path[int(adjacency[edge])]:
                fresh.crust.flags[cell] |= CrustFlag["RIFT_MARGIN"]
                shoulders[1].append(cell)
                break

    for cell in failed_arm:
        owner = (
            crust
            if crust.mask[cell]
            else fresh.crust
            if fresh.crust.mask[cell]
            else None
        )
        if owner is not None:
            owner.flags[cell] |= CrustFlag["FAILED_RIFT"]

    for cell in path:
        if crust.mask[cell]:
            crust.riftAt[cell] = world.time
    for cell in shoulders[1]:
        fresh.crust.riftAt[cell] = world.time
    for cell in failed_arm:
        if crust.mask[cell]:
            crust.riftAt[cell] = world.time
        if fresh.crust.mask[cell]:
            fresh.crust.riftAt[cell] = world.time

    shoulder = {
        "type": OrogenyType["RIFT"],
        "widthKm": world.config["riftShoulderKm"],
        "workKm": world.config["riftShoulderWorkKm"],
        "time": world.time,
    }

    def not_failed(raster):
        return lambda cell: not (
            int(raster.flags[cell]) & CrustFlag["FAILED_RIFT"]
        )

    lifted = [
        [
            cell for cell in cells
            if not (int(raster.flags[cell]) & CrustFlag["FAILED_RIFT"])
        ]
        for cells, raster in (
            (shoulders[0], crust),
            (shoulders[1], fresh.crust),
        )
    ]
    if lifted[0]:
        spread_belt(
            plate, mesh, lifted[0], shoulder, world.config,
            not_failed(crust),
        )
    if lifted[1]:
        spread_belt(
            fresh, mesh, lifted[1], shoulder, world.config,
            not_failed(fresh.crust),
        )

    plate.born = float(world.time)
    fresh.born = float(world.time)
    if fresh.crust.count > crust.count:
        fresh.hue = plate.hue
        plate.hue = fresh.id
    plate.bornOfRift = True
    fresh.bornOfRift = True
    plate.riftPartner = fresh.id
    fresh.riftPartner = plate.id

    centroid_a = _continental_centroid(crust, xyz, num_cells)
    centroid_b = _continental_centroid(fresh.crust, xyz, num_cells)
    if across is not None:
        midpoint = normalize([
            centroid_a[0] + centroid_b[0],
            centroid_a[1] + centroid_b[1],
            centroid_a[2] + centroid_b[2],
        ])
        to_a = tangent_toward(midpoint, centroid_b, centroid_a)
        sign = 1.0 if dot(across, to_a) >= 0.0 else -1.0
        plate.riftPushDir = [sign * float(value) for value in across]
        fresh.riftPushDir = [-sign * float(value) for value in across]
    else:
        plate.riftPushDir = tangent_toward(
            centroid_a, centroid_b, centroid_a
        )
        fresh.riftPushDir = tangent_toward(
            centroid_b, centroid_a, centroid_b
        )

    plate.mark_reshaped()
    fresh.mark_reshaped()
    plate.refresh_extent(xyz)
    fresh.refresh_extent(xyz)
    name_a, name_b = name_for(plate, fresh)
    plate.name = name_a
    fresh.name = name_b
    return fresh


def choose_junction(world, plate, rng) -> int:
    """Choose a weighted interior rift junction; return -1 if none exists."""
    mesh = world.mesh
    config = world.config
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    num_cells = int(mesh.num_cells)
    edge_km = float(mesh.edge_km)
    crust = plate.crust

    def land(cell: int) -> bool:
        return bool(
            crust.mask[cell]
            and crust.type[cell] != CrustType["OCEAN"]
        )

    distance = np.full(num_cells, -1, dtype=np.int16)
    frontier = []
    for cell in range(num_cells):
        if not land(cell):
            continue
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            if not land(int(adjacency[edge])):
                distance[cell] = 0
                frontier.append(cell)
                break

    depth = 1
    while frontier:
        following = []
        for cell in frontier:
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if land(neighbour) and distance[neighbour] < 0:
                    distance[neighbour] = depth
                    following.append(neighbour)
        frontier = following
        depth += 1

    min_hops = math.ceil(float(config["riftMinArmKm"]) / edge_km)
    weak = CrustFlag["SUTURE"] | CrustFlag["FAILED_RIFT"]
    candidates = []
    weights = []
    total = 0.0
    for cell in range(num_cells):
        if (
            not land(cell)
            or distance[cell] < min_hops
            or crust.type[cell] == CrustType["CRATON"]
        ):
            continue
        interior_factor = min(
            1.0, float(distance[cell]) / (2.0 * min_hops)
        )
        weight = (
            1.0
            if int(crust.flags[cell]) & weak
            else float(config["riftWeaknessCost"])
        ) * interior_factor
        candidates.append(cell)
        weights.append(weight)
        total += weight

    if not candidates:
        return -1

    pick = float(rng()) * total
    for cell, weight in zip(candidates, weights):
        pick -= weight
        if pick <= 0.0:
            return cell
    return candidates[-1]


def _continental_centroid(crust, xyz, num_cells: int) -> list[float]:
    x = y = z = 0.0
    for cell in range(num_cells):
        if (
            crust.mask[cell]
            and crust.type[cell] != CrustType["OCEAN"]
        ):
            offset = 3 * cell
            x += float(xyz[offset])
            y += float(xyz[offset + 1])
            z += float(xyz[offset + 2])
    length = math.hypot(x, y, z) or 1.0
    return [x / length, y / length, z / length]


__all__ = [
    "choose_junction",
    "find_rift",
    "rift_context",
    "split_plate",
]
