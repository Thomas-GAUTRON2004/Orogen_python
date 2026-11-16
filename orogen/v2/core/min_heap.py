"""Binary min-heap and multi-source Dijkstra helpers from V2 min-heap.js."""
from __future__ import annotations

import math
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np

from .sphere_geometry import PLANET_RADIUS_KM


class MinHeap:
    """Binary min-heap of (key, id) pairs; duplicate ids are permitted."""

    def __init__(self, capacity: int = 1024):
        if capacity < 1:
            raise ValueError("capacity must be at least one")
        self.keys = np.empty(capacity, dtype=np.float64)
        self.ids = np.empty(capacity, dtype=np.int32)
        self.size = 0
        self.last_key = math.nan

    def push(self, key: float, identifier: int) -> None:
        if self.size == len(self.keys):
            new_keys = np.empty(self.size * 2, dtype=np.float64)
            new_ids = np.empty(self.size * 2, dtype=np.int32)
            new_keys[:self.size] = self.keys
            new_ids[:self.size] = self.ids
            self.keys, self.ids = new_keys, new_ids
        index = self.size
        self.size += 1
        while index > 0:
            parent = (index - 1) >> 1
            if self.keys[parent] <= key:
                break
            self.keys[index] = self.keys[parent]
            self.ids[index] = self.ids[parent]
            index = parent
        self.keys[index] = key
        self.ids[index] = identifier

    def peek_key(self) -> float:
        return float(self.keys[0])

    def pop(self) -> int:
        if self.size == 0:
            raise IndexError("pop from empty MinHeap")
        top_id = int(self.ids[0])
        self.last_key = float(self.keys[0])
        count = self.size - 1
        self.size = count
        if count > 0:
            key, identifier = self.keys[count], self.ids[count]
            index = 0
            while True:
                left = 2 * index + 1
                if left >= count:
                    break
                right = left + 1
                child = right if right < count and self.keys[right] < self.keys[left] else left
                if self.keys[child] >= key:
                    break
                self.keys[index] = self.keys[child]
                self.ids[index] = self.ids[child]
                index = child
            self.keys[index] = key
            self.ids[index] = identifier
        return top_id


_EDGE_KM: weakref.WeakKeyDictionary[Any, np.ndarray] = weakref.WeakKeyDictionary()


def forget_edge_km(mesh: Any) -> None:
    """Release cached edge lengths for a mesh."""
    _EDGE_KM.pop(mesh, None)


def _edge_km_of(mesh: Any) -> np.ndarray:
    cached = _EDGE_KM.get(mesh)
    if cached is not None:
        return cached
    offsets, adjacency, xyz = mesh.adj_offset, mesh.adj_list, mesh.xyz
    edge_km = np.empty(len(adjacency), dtype=np.float64)
    for cell in range(len(offsets) - 1):
        cx, cy, cz = (float(value) for value in xyz[3 * cell:3 * cell + 3])
        for k in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[k])
            offset = 3 * neighbour
            cosine = (
                cx * xyz[offset] + cy * xyz[offset + 1] + cz * xyz[offset + 2]
            )
            edge_km[k] = math.acos(min(1.0, float(cosine))) * PLANET_RADIUS_KM
    _EDGE_KM[mesh] = edge_km
    return edge_km


def spread_km(
    mesh: Any,
    dist: np.ndarray,
    *,
    can_enter: Callable[[int, int], bool] | None = None,
    on_reach: Callable[[int, int], None] | None = None,
    max_km: float = math.inf,
) -> np.ndarray:
    """Multi-source Dijkstra over mesh edges measured in great-circle km."""
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    edge = _edge_km_of(mesh)
    heap = MinHeap(4096)
    for cell in range(len(dist)):
        if dist[cell] < math.inf:
            heap.push(float(dist[cell]), cell)

    while heap.size:
        cell = heap.pop()
        current_distance = heap.last_key
        if current_distance > dist[cell]:
            continue
        for k in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[k])
            if can_enter is not None and not can_enter(cell, neighbour):
                continue
            candidate = current_distance + edge[k]
            if candidate >= dist[neighbour] or candidate > max_km:
                continue
            dist[neighbour] = candidate
            if on_reach is not None:
                on_reach(cell, neighbour)
            # Queue the value after assignment: float32 arrays round it.
            heap.push(float(dist[neighbour]), neighbour)
    return dist
