"""Precomputed concentric mesh neighbourhoods from V2 mesh-rings.js."""

# differential-module:

from __future__ import annotations

from typing import Any, TypedDict, cast

import numpy as np

from .mesh_types import MeshGraph


class MeshRings(TypedDict):
    start: np.ndarray
    cells: np.ndarray
    stride: int


RING_COUNT = 3


def rings_of(mesh: MeshGraph) -> MeshRings:
    """Build and cache breadth-first rings 0..RING_COUNT for every cell."""
    cached = getattr(mesh, "_rings", None)
    if cached is not None:
        return cast(MeshRings, cached)

    num_cells = int(mesh.num_cells)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    stride = RING_COUNT + 2
    start = np.empty(num_cells * stride, dtype=np.int32)
    seen = np.zeros(num_cells, dtype=np.uint32)
    cells = np.empty(max(1, num_cells * 40), dtype=np.int32)
    count = 0
    ring: list[int] = []
    next_ring: list[int] = []

    for cell in range(num_cells):
        stamp = cell + 1
        ring.clear()
        ring.append(cell)
        seen[cell] = stamp
        for radius in range(RING_COUNT + 1):
            start[cell * stride + radius] = count
            required = count + len(ring)
            if required > len(cells):
                grown = np.empty(max(required, len(cells) * 2), dtype=np.int32)
                grown[:count] = cells[:count]
                cells = grown
            for member in ring:
                cells[count] = member
                count += 1

            next_ring.clear()
            if radius < RING_COUNT:
                for member in ring:
                    for k in range(int(offsets[member]), int(offsets[member + 1])):
                        neighbour = int(adjacency[k])
                        if seen[neighbour] != stamp:
                            seen[neighbour] = stamp
                            next_ring.append(neighbour)
            ring, next_ring = next_ring, ring
        start[cell * stride + RING_COUNT + 1] = count

    result: MeshRings = {
        "start": start,
        "cells": cells[:count],
        "stride": stride,
    }
    cast(Any, mesh)._rings = result
    return result
