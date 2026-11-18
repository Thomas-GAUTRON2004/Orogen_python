"""Cube-map starting table plus greedy exact nearest-cell search.

Port of v2/js/core/sphere-index.js. The index operates on unit-vector cell
centres and CSR neighbour adjacency.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class SphereIndex:
    """Nearest-cell lookup on a spherical Delaunay mesh."""

    def __init__(self, xyz: np.ndarray, mesh: Any, texels_per_cell: int = 2):
        self.xyz = np.asarray(xyz)
        self.adj_offset = np.asarray(mesh.adj_offset)
        self.adj_list = np.asarray(mesh.adj_list)
        self.num_cells = int(mesh.num_cells)
        self.res = max(8, math.ceil(math.sqrt(self.num_cells / 6) * texels_per_cell))
        self.table = np.empty(6 * self.res * self.res, dtype=np.int32)
        self._build()

    def _build(self) -> None:
        res, table, xyz = self.res, self.table, self.xyz
        offsets, adjacency = self.adj_offset, self.adj_list
        current = 0
        for face in range(6):
            for j in range(res):
                for scan_index in range(res):
                    i = res - 1 - scan_index if j & 1 else scan_index
                    u = (2 * (i + 0.5)) / res - 1
                    v = (2 * (j + 0.5)) / res - 1
                    if face == 0:
                        x, y, z = 1.0, u, v
                    elif face == 1:
                        x, y, z = -1.0, u, v
                    elif face == 2:
                        x, y, z = u, 1.0, v
                    elif face == 3:
                        x, y, z = u, -1.0, v
                    elif face == 4:
                        x, y, z = u, v, 1.0
                    else:
                        x, y, z = u, v, -1.0
                    inverse = 1.0 / math.hypot(x, y, z)
                    x, y, z = x * inverse, y * inverse, z * inverse
                    offset = 3 * current
                    best = x * xyz[offset] + y * xyz[offset + 1] + z * xyz[offset + 2]
                    improved = True
                    while improved:
                        improved = False
                        for k in range(
                            int(offsets[current]), int(offsets[current + 1])
                        ):
                            neighbour = int(adjacency[k])
                            offset = 3 * neighbour
                            candidate = (
                                x * xyz[offset]
                                + y * xyz[offset + 1]
                                + z * xyz[offset + 2]
                            )
                            if candidate > best:
                                best = candidate
                                current = neighbour
                                improved = True
                    table[(face * res + j) * res + i] = current

    def start(self, x: float, y: float, z: float) -> int:
        """Return the cube-table starting guess for a direction."""
        ax, ay, az = abs(x), abs(y), abs(z)
        res = self.res
        if ax >= ay and ax >= az:
            face = 0 if x > 0 else 1
            u, v = y / ax, z / ax
        elif ay >= az:
            face = 2 if y > 0 else 3
            u, v = x / ay, z / ay
        else:
            face = 4 if z > 0 else 5
            u, v = x / az, y / az
        i = int((u + 1) * 0.5 * res)
        j = int((v + 1) * 0.5 * res)
        i = min(res - 1, i)
        j = min(res - 1, j)
        return int(self.table[(face * res + j) * res + i])

    def walk(self, current: int, x: float, y: float, z: float) -> int:
        """Greedily walk to the nearest cell from current."""
        xyz, offsets, adjacency = self.xyz, self.adj_offset, self.adj_list
        offset = 3 * current
        best = x * xyz[offset] + y * xyz[offset + 1] + z * xyz[offset + 2]
        improved = True
        while improved:
            improved = False
            for k in range(int(offsets[current]), int(offsets[current + 1])):
                neighbour = int(adjacency[k])
                offset = 3 * neighbour
                candidate = x * xyz[offset] + y * xyz[offset + 1] + z * xyz[offset + 2]
                if candidate > best:
                    best = candidate
                    current = neighbour
                    improved = True
        return current

    def nearest(self, x: float, y: float, z: float) -> int:
        """Return the nearest cell to a direction (normalization is unnecessary)."""
        return self.walk(self.start(x, y, z), x, y, z)
