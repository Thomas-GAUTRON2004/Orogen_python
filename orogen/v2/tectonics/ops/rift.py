"""Rift initiation and plate splitting operations, ported from Orogen V2."""

from __future__ import annotations

import math
import re

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
    "pull_axis_of",
    "names_after_split",
    "try_rift",
    "plume_junctions",
]


def pull_axis_of(plate, mesh):
    """Return the principal axis of the plate's opposing trench pulls, if any."""
    xyz = mesh.xyz
    mask, flags = plate.crust.mask, plate.crust.flags
    centre = plate.capCenter
    matrix = np.zeros(9, dtype=np.float64)
    trench_cells = 0
    for cell in range(len(mask)):
        if not mask[cell] or not (int(flags[cell]) & CrustFlag["TRENCH"]):
            continue
        offset = 3 * cell
        direction = tangent_toward(
            centre,
            centre,
            [
                float(xyz[offset]),
                float(xyz[offset + 1]),
                float(xyz[offset + 2]),
            ],
        )
        for row in range(3):
            for col in range(3):
                matrix[3 * row + col] += direction[row] * direction[col]
        trench_cells += 1

    if trench_cells * float(mesh.edge_km) < 485.0:
        return None

    vector = [float(centre[1]), -float(centre[0]), 0.0]
    for _ in range(30):
        following = [
            matrix[3 * row] * vector[0]
            + matrix[3 * row + 1] * vector[1]
            + matrix[3 * row + 2] * vector[2]
            for row in range(3)
        ]
        length = math.hypot(*following)
        if length < 1e-12:
            return None
        vector = [value / length for value in following]
    return vector


def names_after_split(world, plate_a, plate_b, before: str) -> tuple[str, str]:
    """Name new halves by craton letters, falling back to unused numbered names."""
    name_a = world.name_by_cratons(plate_a, None)
    name_b = world.name_by_cratons(plate_b, None)
    if name_a and name_b and name_a != name_b:
        return name_a, name_b

    base = re.sub(r"(·\\d+)+$", "", before)
    taken = {
        str(plate.name)
        for plate in world.plates
        if plate.alive
    }
    names = []
    number = 1
    while len(names) < 2:
        candidate = f"{base}·{number}"
        if candidate not in taken:
            names.append(candidate)
        number += 1
    return names[0], names[1]


def try_rift(
    world,
    plate,
    junctions,
    cause: dict,
    need_cratons_both_sides: bool = True,
) -> bool:
    """Search all supplied junctions and execute the best valid rift, if any."""
    mesh = world.mesh
    pull = pull_axis_of(plate, mesh)
    context = rift_context(world, plate)
    best = None
    for junction in junctions:
        junction = int(junction)
        if junction < 0:
            continue
        found = find_rift(
            world,
            plate,
            junction,
            pull,
            need_cratons_both_sides,
            context,
        )
        if found is not None and (
            best is None or float(found["value"]) < float(best["value"])
        ):
            best = found
    if best is None:
        return False

    weak = (
        CrustFlag["SUTURE"]
        | CrustFlag["FAILED_RIFT"]
        | CrustFlag["RIFT_MARGIN"]
    )
    weak_km = sum(
        1 for cell in best["path"]
        if int(plate.crust.flags[int(cell)]) & weak
    ) * float(mesh.edge_km)

    before = str(plate.name)
    age_myr = float(world.time) - float(plate.born)
    fresh = split_plate(
        world,
        plate,
        best["path"],
        best["failed_arm"],
        lambda left, right: names_after_split(world, left, right, before),
        best["across"],
    )
    length_km = len(best["path"]) * float(mesh.edge_km)
    failed_km = len(best["failed_arm"]) * float(mesh.edge_km)

    def fixed0(value: float) -> str:
        return str(math.floor(float(value) + 0.5))

    message = (
        f"{before} rifts ({cause['text']}): {plate.name} and {fresh.name} "
        f"part along {fixed0(length_km)} km of rift"
    )
    if best["failed_arm"]:
        message += (
            f", a failed arm of {fixed0(failed_km)} km left behind"
        )
    world.log(
        EventKind["RIFT"],
        message,
        {
            "plate": int(plate.id),
            "fresh": int(fresh.id),
            "cause": cause["kind"],
            "hotspot": int(cause["hotspot"]),
            "lengthKm": length_km,
            "weakKm": weak_km,
            "failedArmKm": failed_km,
            "ageMyr": age_myr,
        },
    )
    world.lastRiftAt = world.time
    return True


def plume_junctions(world, plate, xyz) -> list[int]:
    """Find land cells near a plume head that can serve as rift junctions."""
    mesh = world.mesh
    config = world.config
    offsets, adjacency, index = mesh.adj_offset, mesh.adj_list, mesh.index
    crust = plate.crust
    centre_cell = plate.cell_at(
        index, float(xyz[0]), float(xyz[1]), float(xyz[2])
    )
    max_hops = max(
        1,
        math.floor(
            float(config["plumeJunctionKm"]) / float(mesh.edge_km) + 0.5
        ),
    )
    try_hops = max(
        1,
        math.floor(220.0 / float(mesh.edge_km) + 0.5),
    )

    def eligible(cell: int) -> bool:
        return bool(
            crust.mask[cell]
            and crust.type[cell] != CrustType["OCEAN"]
            and crust.type[cell] != CrustType["CRATON"]
        )

    candidates = [centre_cell] if eligible(centre_cell) else []
    seen = {centre_cell}
    ring = [centre_cell]
    for hop in range(max_hops):
        if not ring or not (hop < try_hops or not candidates):
            break
        following = []
        for cell in ring:
            for edge in range(
                int(offsets[cell]), int(offsets[cell + 1])
            ):
                neighbour = int(adjacency[edge])
                if neighbour in seen or not crust.mask[neighbour]:
                    continue
                seen.add(neighbour)
                following.append(neighbour)
                if eligible(neighbour):
                    candidates.append(neighbour)
        ring = following

    if len(candidates) <= 12:
        return candidates
    return [
        candidates[
            math.floor(
                index * (len(candidates) - 1) / 11.0 + 0.5
            )
        ]
        for index in range(12)
    ]
