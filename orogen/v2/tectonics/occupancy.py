"""Coarse spherical occupancy bins used to accelerate tectonic gap resolution.

Port of V2 step/occupancy.js. The index is conservative: it may retain extra
candidate plates, but must never exclude a plate within the configured reach.
"""

# differential-module:

from __future__ import annotations

import math

import numpy as np

from ..core.sphere_geometry import PLANET_RADIUS_KM

BIN_DEG = 6
REACH_KM = 700.0
REACH_CELLS = 6.5


class BinIndex:
    def __init__(
        self, bin_of: np.ndarray, neighbours: tuple[np.ndarray, ...], count: int
    ):
        self.bin_of = bin_of
        self.neighbours = neighbours
        self.count = count

    def at(self, x: float, y: float, z: float) -> int:
        rows, cols = 180 // BIN_DEG, 360 // BIN_DEG
        row = min(
            rows - 1,
            math.floor((math.asin(max(-1.0, min(1.0, z))) / math.pi + 0.5) * rows),
        )
        col = min(cols - 1, math.floor((math.atan2(y, x) / (2 * math.pi) + 0.5) * cols))
        return row * cols + col


def bin_index(mesh) -> BinIndex:
    cached = getattr(mesh, "_bins", None)
    if cached is not None:
        return cached
    xyz = np.asarray(mesh.xyz).reshape(-1, 3)
    rows, cols = 180 // BIN_DEG, 360 // BIN_DEG
    index = BinIndex(np.empty(len(xyz), dtype=np.uint16), (), rows * cols)
    for w, (x, y, z) in enumerate(xyz):
        index.bin_of[w] = index.at(float(x), float(y), float(z))
    rad = math.pi / 180
    centres = []
    radii = []
    for row in range(rows):
        for col in range(cols):
            lat = (-90 + (row + 0.5) * BIN_DEG) * rad
            lon = (-180 + (col + 0.5) * BIN_DEG) * rad
            c = (
                math.cos(lat) * math.cos(lon),
                math.cos(lat) * math.sin(lon),
                math.sin(lat),
            )
            radius = 0.0
            for dlat in (-0.5, 0.5):
                for dlon in (-0.5, 0.5):
                    a = lat + dlat * BIN_DEG * rad
                    b = lon + dlon * BIN_DEG * rad
                    d = math.acos(
                        max(
                            -1.0,
                            min(
                                1.0,
                                c[0] * math.cos(a) * math.cos(b)
                                + c[1] * math.cos(a) * math.sin(b)
                                + c[2] * math.sin(a),
                            ),
                        )
                    )
                    radius = max(radius, d)
            centres.append(c)
            radii.append(radius)
    reach = max(REACH_KM, REACH_CELLS * float(mesh.edge_km)) / PLANET_RADIUS_KM
    neighbours = []
    for i, c in enumerate(centres):
        near = []
        for j, neighbour_centre in enumerate(centres):
            angle = math.acos(
                max(
                    -1.0,
                    min(
                        1.0,
                        c[0] * neighbour_centre[0]
                        + c[1] * neighbour_centre[1]
                        + c[2] * neighbour_centre[2],
                    ),
                )
            )
            if angle <= radii[i] + radii[j] + reach:
                near.append(j)
        neighbours.append(np.asarray(near, dtype=np.int32))
    index.neighbours = tuple(neighbours)
    from contextlib import suppress

    with suppress(AttributeError, TypeError):
        mesh._bins = index
    return index


def mark_bin(plate, bin_id: int, stamp: int, count: int) -> None:
    marks = getattr(plate, "bins", None)
    if marks is None or len(marks) != count:
        marks = np.zeros(count, dtype=np.uint32)
        plate.bins = marks
    marks[bin_id] = stamp


def may_have_crust_near(plate, bin_id: int, stamp: int, bins: BinIndex) -> bool:
    marks = getattr(plate, "bins", None)
    if marks is None:
        return False
    return bool(np.any(marks[bins.neighbours[bin_id]] == stamp))
