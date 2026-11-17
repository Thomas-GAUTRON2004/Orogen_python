"""Plate partitioning along a rift path, ported from Orogen V2.

The path removes a strip of crust. Connected land components vote for the
left/right side of the path; the plate's ocean follows its nearest land.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ...core.sphere_geometry import tangent_toward
from ..crust import CrustType


_RIFT_OCEAN_REACH_KM = 7300.0


def masked_cells(crust) -> np.ndarray:
    """Return occupied plate cells in ascending cell order."""
    out = np.empty(int(crust.count), dtype=np.int32)
    count = 0
    for cell in range(len(crust.mask)):
        if crust.mask[cell]:
            out[count] = cell
            count += 1
    return out[:count]


def sides_of(mesh, crust, path: Iterable[int], cells=None) -> dict:
    """Describe the land blocks left by a rift path.

    `share` is the smaller block's share of continental land.
    `elongation` is the largest block aspect ratio.
    `cratonsBothSides` indicates whether each block retains a craton.
    """
    if cells is None:
        cells = masked_cells(crust)
    side, land, cratons = partition(mesh, crust, path, cells)
    a, b = int(land[1]), int(land[2])
    if a + b == 0:
        return {"share": 0.0, "elongation": 1.0, "cratonsBothSides": False}
    return {
        "share": min(a, b) / (a + b),
        "elongation": max(
            elongation_of(mesh, side, 1, cells),
            elongation_of(mesh, side, 2, cells),
        ),
        "cratonsBothSides": bool(cratons[1] > 0 and cratons[2] > 0),
    }


def partition(mesh, crust, path: Iterable[int], cells=None):
    """Partition all occupied crust cells into the two sides of `path`.

    Returns an int8 side array (1 = left, 2 = the largest right-hand land
    component, 0 = outside the plate) and land/craton counts by side.
    """
    if cells is None:
        cells = masked_cells(crust)
    cells = [int(cell) for cell in cells]
    path = _through_ocean(mesh, crust, [int(cell) for cell in path])

    xyz = mesh.xyz
    num_cells = int(mesh.num_cells)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    on_path = np.zeros(num_cells, dtype=np.uint8)
    side = np.zeros(num_cells, dtype=np.int8)
    for cell in path:
        on_path[cell] = 1

    def is_land(cell: int) -> bool:
        return bool(
            crust.mask[cell]
            and crust.type[cell] != CrustType["OCEAN"]
        )

    # Accumulate the signed offset of each path-adjacent cell. Positive is
    # left of the directed path, negative is right.
    lean = np.zeros(num_cells, dtype=np.float32)
    for index, cell in enumerate(path):
        previous = path[max(0, index - 1)]
        following = path[min(index + 1, len(path) - 1)]
        point = _position(xyz, cell)
        along = tangent_toward(
            point, _position(xyz, previous), _position(xyz, following)
        )
        left = (
            along[1] * point[2] - along[2] * point[1],
            along[2] * point[0] - along[0] * point[2],
            along[0] * point[1] - along[1] * point[0],
        )
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if on_path[neighbour]:
                continue
            offset = 3 * neighbour
            value = (
                (float(xyz[offset]) - point[0]) * left[0]
                + (float(xyz[offset + 1]) - point[1]) * left[1]
                + (float(xyz[offset + 2]) - point[2]) * left[2]
            )
            lean[neighbour] = np.float32(float(lean[neighbour]) + value)

    # Land components left after removing the path each vote by their
    # path-adjacent cells. The largest component voting right becomes side 2.
    piece_of = np.full(num_cells, -1, dtype=np.int32)
    pieces: list[dict] = []
    for start in cells:
        if not is_land(start) or on_path[start] or piece_of[start] >= 0:
            continue
        identifier = len(pieces)
        component = [start]
        piece_of[start] = identifier
        vote = 0.0
        cursor = 0
        while cursor < len(component):
            cell = component[cursor]
            cursor += 1
            vote += float(lean[cell])
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if (
                    is_land(neighbour)
                    and not on_path[neighbour]
                    and piece_of[neighbour] < 0
                ):
                    piece_of[neighbour] = identifier
                    component.append(neighbour)
        pieces.append({"cells": component, "vote": vote})

    right = None
    for piece in pieces:
        if piece["vote"] < 0.0 and (
            right is None or len(piece["cells"]) > len(right["cells"])
        ):
            right = piece

    for cell in cells:
        if is_land(cell):
            side[cell] = 1
    if right is not None:
        for cell in right["cells"]:
            side[cell] = 2

    # Oceanic crust follows its nearest land block by a multi-source BFS.
    frontier = [cell for cell in cells if side[cell]]
    while frontier:
        following_frontier = []
        for cell in frontier:
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if crust.mask[neighbour] and not side[neighbour]:
                    side[neighbour] = side[cell]
                    following_frontier.append(neighbour)
        frontier = following_frontier

    land = [0, 0, 0]
    cratons = [0, 0, 0]
    seen_cratons = set()
    for cell in cells:
        if not side[cell]:
            side[cell] = 1
        if not is_land(cell):
            continue
        side_id = int(side[cell])
        land[side_id] += 1
        terrane = int(crust.terrane[cell])
        if (
            crust.type[cell] == CrustType["CRATON"]
            and terrane not in seen_cratons
        ):
            seen_cratons.add(terrane)
            cratons[side_id] += 1
    return side, land, cratons


def exterior_sea(mesh, crust) -> np.ndarray:
    """Mark sea connected to the outside of the plate.

    If the plate covers the entire mesh, the largest connected ocean basin
    acts as its exterior sea.
    """
    num_cells = int(mesh.num_cells)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    out = np.zeros(num_cells, dtype=np.uint8)

    def is_sea(cell: int) -> bool:
        return bool(
            not crust.mask[cell]
            or crust.type[cell] == CrustType["OCEAN"]
        )

    frontier = []
    for cell in range(num_cells):
        if not crust.mask[cell]:
            out[cell] = 1
            frontier.append(cell)

    if not frontier:
        seen = np.zeros(num_cells, dtype=np.uint8)
        largest = []
        for start in range(num_cells):
            if seen[start] or crust.type[start] != CrustType["OCEAN"]:
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
                        and crust.type[neighbour] == CrustType["OCEAN"]
                    ):
                        seen[neighbour] = 1
                        component.append(neighbour)
            if len(component) > len(largest):
                largest = component
        for cell in largest:
            out[cell] = 1
        frontier = largest

    while frontier:
        following_frontier = []
        for cell in frontier:
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if not out[neighbour] and is_sea(neighbour):
                    out[neighbour] = 1
                    following_frontier.append(neighbour)
        frontier = following_frontier
    return out


def elongation_of(mesh, side, which: int, cells=None) -> float:
    """Return the aspect ratio of one land block from its position covariance."""
    xyz = mesh.xyz
    num_cells = int(mesh.num_cells)
    if cells is None:
        cells = [cell for cell in range(num_cells) if side[cell]]

    count = 0
    mean = [0.0, 0.0, 0.0]
    for cell in cells:
        if side[cell] == which:
            count += 1
            offset = 3 * cell
            mean[0] += float(xyz[offset])
            mean[1] += float(xyz[offset + 1])
            mean[2] += float(xyz[offset + 2])
    if count < 6:
        return 1.0

    mean = [value / count for value in mean]
    covariance = np.zeros(9, dtype=np.float64)
    for cell in cells:
        if side[cell] != which:
            continue
        offset = 3 * cell
        delta = [
            float(xyz[offset]) - mean[0],
            float(xyz[offset + 1]) - mean[1],
            float(xyz[offset + 2]) - mean[2],
        ]
        for i in range(3):
            for j in range(3):
                covariance[3 * i + j] += delta[i] * delta[j]

    p1 = (
        covariance[1] ** 2
        + covariance[2] ** 2
        + covariance[5] ** 2
    )
    q = (covariance[0] + covariance[4] + covariance[8]) / 3.0
    p2 = (
        (covariance[0] - q) ** 2
        + (covariance[4] - q) ** 2
        + (covariance[8] - q) ** 2
        + 2.0 * p1
    )
    p = math.sqrt(p2 / 6.0)
    if p < 1e-18:
        return 1.0

    matrix = [
        [
            (covariance[3 * i + j] - (q if i == j else 0.0)) / p
            for j in range(3)
        ]
        for i in range(3)
    ]
    determinant = (
        matrix[0][0] * (
            matrix[1][1] * matrix[2][2]
            - matrix[1][2] * matrix[2][1]
        )
        - matrix[0][1] * (
            matrix[1][0] * matrix[2][2]
            - matrix[1][2] * matrix[2][0]
        )
        + matrix[0][2] * (
            matrix[1][0] * matrix[2][1]
            - matrix[1][1] * matrix[2][0]
        )
    )
    phi = math.acos(max(-1.0, min(1.0, determinant / 2.0))) / 3.0
    largest_eigenvalue = q + 2.0 * p * math.cos(phi)
    smallest_eigenvalue = q + 2.0 * p * math.cos(
        phi + 2.0 * math.pi / 3.0
    )
    middle_eigenvalue = (
        3.0 * q - largest_eigenvalue - smallest_eigenvalue
    )
    eigenvalues = sorted(
        (
            largest_eigenvalue,
            middle_eigenvalue,
            smallest_eigenvalue,
        ),
        reverse=True,
    )
    if eigenvalues[1] > 1e-12:
        return math.sqrt(eigenvalues[0] / eigenvalues[1])
    return 10.0


def _through_ocean(mesh, crust, path: list[int]) -> list[int]:
    """Extend both ends of a rift path through plate oceanic crust."""
    on_path = set(path)
    if len(path) < 2:
        return path

    def extend(end: int, previous: int) -> list[int]:
        result = []
        current = end
        coming_from = previous
        steps = math.floor(_RIFT_OCEAN_REACH_KM / float(mesh.edge_km) + 0.5)
        for _ in range(steps):
            current_position = _position(mesh.xyz, current)
            direction = tangent_toward(
                current_position,
                _position(mesh.xyz, coming_from),
                current_position,
            )
            best = -1
            best_dot = 0.3
            for edge in range(
                int(mesh.adj_offset[current]),
                int(mesh.adj_offset[current + 1]),
            ):
                neighbour = int(mesh.adj_list[edge])
                if (
                    not crust.mask[neighbour]
                    or neighbour in on_path
                    or crust.type[neighbour] != CrustType["OCEAN"]
                ):
                    continue
                neighbour_direction = tangent_toward(
                    current_position,
                    current_position,
                    _position(mesh.xyz, neighbour),
                )
                along = sum(
                    neighbour_direction[i] * direction[i]
                    for i in range(3)
                )
                if along > best_dot:
                    best_dot = along
                    best = neighbour
            if best < 0:
                break
            result.append(best)
            on_path.add(best)
            coming_from, current = current, best
        return result

    head = list(reversed(extend(path[0], path[1])))
    tail = extend(path[-1], path[-2])
    return head + path + tail


def _position(xyz, cell: int) -> tuple[float, float, float]:
    offset = 3 * cell
    return (
        float(xyz[offset]),
        float(xyz[offset + 1]),
        float(xyz[offset + 2]),
    )
