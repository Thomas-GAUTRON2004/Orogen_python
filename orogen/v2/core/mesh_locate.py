"""Piecewise-linear point location on a spherical mesh, ported from V2."""
from __future__ import annotations

from typing import Any

import numpy as np


def locate(
    mesh: Any,
    x: float,
    y: float,
    z: float,
    cells: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate a direction in a Delaunay triangle and return 3 cells and weights.

    Output arrays may be supplied for reuse. Weights are non-negative and sum
    to one; if no usable triangle is found, the nearest cell gets weight one.
    """
    if cells is None:
        cells = np.empty(3, dtype=np.int32)
    if weights is None:
        weights = np.empty(3, dtype=np.float64)

    xyz, offsets, adjacency = mesh.xyz, mesh.adj_offset, mesh.adj_list
    centre = int(mesh.index.nearest(x, y, z))
    offset = 3 * centre
    cx, cy, cz = xyz[offset:offset + 3]
    start, end = int(offsets[centre]), int(offsets[centre + 1])

    best_min = -float("inf")
    best_a = best_b = -1
    weight_c, weight_a, weight_b = 1.0, 0.0, 0.0

    for k in range(start, end):
        a = int(adjacency[k])
        b = int(adjacency[k + 1 if k + 1 < end else start])
        oa, ob = 3 * a, 3 * b
        ax, ay, az = xyz[oa:oa + 3]
        bx, by, bz = xyz[ob:ob + 3]

        # Signed volumes are the spherical triangle's unnormalized barycentrics.
        sc = (ay * bz - az * by) * x + (az * bx - ax * bz) * y + (ax * by - ay * bx) * z
        sa = (by * cz - bz * cy) * x + (bz * cx - bx * cz) * y + (bx * cy - by * cx) * z
        sb = (cy * az - cz * ay) * x + (cz * ax - cx * az) * y + (cx * ay - cy * ax) * z
        total = sc + sa + sb
        if not abs(total) > 1e-18:
            continue

        nc, na, nb = sc / total, sa / total, sb / total
        lowest = min(nc, na, nb)
        if lowest > best_min:
            best_min = lowest
            best_a, best_b = a, b
            weight_c, weight_a, weight_b = nc, na, nb
            if lowest >= 0:
                break

    if best_a < 0:
        cells[:] = (centre, centre, centre)
        weights[:] = (1.0, 0.0, 0.0)
        return cells, weights

    wc, wa, wb = max(0.0, weight_c), max(0.0, weight_a), max(0.0, weight_b)
    total = wc + wa + wb or 1.0
    cells[:] = (centre, best_a, best_b)
    weights[:] = (wc / total, wa / total, wb / total)
    return cells, weights
