"""Breadth-first utilities for the V2 simulation mesh.

Port of v2/js/core/mesh-bfs.js. Mesh adjacency is stored in CSR form:
adj_offset has num_cells + 1 entries and adj_list stores neighbour indices.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ._numba_kernels import (
    NUMBA_AVAILABLE,
    hop_distance_numba,
    largest_piece_numba,
    pieces_of_numba,
)
from .sphere_geometry import PLANET_RADIUS_KM

Predicate = Callable[[int], bool]


def within_km(sim_mesh: Any, is_source: Predicate, radius_km: float) -> np.ndarray:
    """Mark cells within great-circle radius of any source cell."""
    n = int(sim_mesh.num_cells)
    offsets, adjacency, xyz = sim_mesh.adj_offset, sim_mesh.adj_list, sim_mesh.xyz
    min_dot = math.cos(radius_km / PLANET_RADIUS_KM)
    reached = np.zeros(n, dtype=np.uint8)
    seen = np.full(n, -1, dtype=np.int32)
    queue: list[int] = []

    for source in range(n):
        if not is_source(source):
            continue
        reached[source] = 1
        edge = any(
            not is_source(int(adjacency[k]))
            for k in range(int(offsets[source]), int(offsets[source + 1]))
        )
        if not edge:
            continue

        sx, sy, sz = (float(value) for value in xyz[3 * source : 3 * source + 3])
        queue.clear()
        queue.append(source)
        seen[source] = source
        head = 0
        while head < len(queue):
            cell = queue[head]
            head += 1
            for k in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[k])
                if seen[neighbour] == source:
                    continue
                offset = 3 * neighbour
                cosine = (
                    float(xyz[offset]) * sx
                    + float(xyz[offset + 1]) * sy
                    + float(xyz[offset + 2]) * sz
                )
                if cosine < min_dot:
                    continue
                seen[neighbour] = source
                reached[neighbour] = 1
                queue.append(neighbour)
    return reached


def hop_distance(
    sim_mesh: Any,
    is_source: Predicate,
    max_hops: int | float = math.inf,
    can_enter: Predicate | None = None,
) -> np.ndarray:
    """Return hop distance from all source cells; -1 means unreachable."""
    n = int(sim_mesh.num_cells)
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    if NUMBA_AVAILABLE and can_enter is None:
        sources = np.fromiter(
            (bool(is_source(cell)) for cell in range(n)), dtype=np.bool_, count=n
        )
        return hop_distance_numba(
            np.ascontiguousarray(offsets, dtype=np.int32),
            np.ascontiguousarray(adjacency, dtype=np.int32),
            sources,
            int(max_hops) if math.isfinite(max_hops) else n,
        )
    distance = np.full(n, -1, dtype=np.int16)
    frontier = []
    for cell in range(n):
        if is_source(cell):
            distance[cell] = 0
            frontier.append(cell)

    depth = 1
    while frontier and depth <= max_hops:
        next_frontier = []
        for cell in frontier:
            for k in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[k])
                if distance[neighbour] >= 0:
                    continue
                if can_enter is not None and not can_enter(neighbour):
                    continue
                distance[neighbour] = depth
                next_frontier.append(neighbour)
        frontier = next_frontier
        depth += 1
    return distance


def flood_from(sim_mesh: Any, start: int, can_enter: Predicate) -> np.ndarray:
    """Mark cells reachable from start through cells accepted by can_enter."""
    n = int(sim_mesh.num_cells)
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    seen = np.zeros(n, dtype=np.uint8)
    if not can_enter(start):
        return seen
    stack = [start]
    seen[start] = 1
    while stack:
        cell = stack.pop()
        for k in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[k])
            if not seen[neighbour] and can_enter(neighbour):
                seen[neighbour] = 1
                stack.append(neighbour)
    return seen


def touches(sim_mesh: Any, cell: int, test: Predicate) -> bool:
    """Return whether any neighbour of cell satisfies test."""
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    return any(
        test(int(adjacency[k]))
        for k in range(int(offsets[cell]), int(offsets[cell + 1]))
    )


def hops_from(
    sim_mesh: Any,
    sources: Iterable[int],
    can_enter: Predicate | None = None,
    max_hops: int | float = math.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hop distances and the originating source for every reached cell."""
    n = int(sim_mesh.num_cells)
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    distance = np.full(n, -1, dtype=np.int16)
    source_of = np.full(n, -1, dtype=np.int32)
    frontier = []
    for cell in sources:
        cell = int(cell)
        if distance[cell] < 0:
            distance[cell] = 0
            source_of[cell] = cell
            frontier.append(cell)

    depth = 1
    while frontier and depth <= max_hops:
        next_frontier = []
        for cell in frontier:
            for k in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[k])
                if distance[neighbour] >= 0:
                    continue
                if can_enter is not None and not can_enter(neighbour):
                    continue
                distance[neighbour] = depth
                source_of[neighbour] = source_of[cell]
                next_frontier.append(neighbour)
        frontier = next_frontier
        depth += 1
    return distance, source_of


def reaches_within(
    sim_mesh: Any,
    cell: int,
    max_hops: int,
    can_enter: Predicate,
    is_target: Predicate,
) -> bool:
    """Test whether a target can be reached within max_hops."""
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    seen = {cell}
    ring = [cell]
    for _ in range(max_hops):
        if not ring:
            break
        next_ring = []
        for current in ring:
            for k in range(int(offsets[current]), int(offsets[current + 1])):
                neighbour = int(adjacency[k])
                if neighbour in seen or not can_enter(neighbour):
                    continue
                if is_target(neighbour):
                    return True
                seen.add(neighbour)
                next_ring.append(neighbour)
        ring = next_ring
    return False


def pieces_of(sim_mesh: Any, cells: Iterable[int]) -> list[list[int]]:
    """Return connected components of a selected cell set.

    When Numba is installed, the CSR traversal runs in one compiled call.
    The Python implementation remains the reference and fallback path.
    """
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    ordered_cells = [int(cell) for cell in cells]
    n = int(sim_mesh.num_cells)
    if NUMBA_AVAILABLE and all(0 <= cell < n for cell in ordered_cells):
        flat, starts = pieces_of_numba(
            np.ascontiguousarray(offsets, dtype=np.int32),
            np.ascontiguousarray(adjacency, dtype=np.int32),
            np.asarray(ordered_cells, dtype=np.int32),
        )
        return [
            flat[int(starts[i]) : int(starts[i + 1])].tolist()
            for i in range(len(starts) - 1)
        ]

    in_set = set(ordered_cells)
    seen: set[int] = set()
    pieces: list[list[int]] = []
    for start in ordered_cells:
        if start in seen:
            continue
        piece = [start]
        seen.add(start)
        index = 0
        while index < len(piece):
            cell = piece[index]
            index += 1
            for k in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[k])
                if neighbour in in_set and neighbour not in seen:
                    seen.add(neighbour)
                    piece.append(neighbour)
        pieces.append(piece)
    return pieces


def largest_piece_of(sim_mesh: Any, cells: Iterable[int]) -> list[int]:
    """Return the largest connected component; the first wins ties."""
    ordered_cells = [int(cell) for cell in cells]
    n = int(sim_mesh.num_cells)
    if NUMBA_AVAILABLE and all(0 <= cell < n for cell in ordered_cells):
        return largest_piece_numba(
            np.ascontiguousarray(sim_mesh.adj_offset, dtype=np.int32),
            np.ascontiguousarray(sim_mesh.adj_list, dtype=np.int32),
            np.asarray(ordered_cells, dtype=np.int32),
        ).tolist()

    best: list[int] = []
    for piece in pieces_of(sim_mesh, ordered_cells):
        if len(piece) > len(best):
            best = piece
    return best
