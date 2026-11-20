"""Cached world-frame plate-contact geometry, ported from Orogen V2 contact.js."""

from __future__ import annotations

from weakref import WeakKeyDictionary

import numpy as np

_UNIT_DIRECTIONS: WeakKeyDictionary = WeakKeyDictionary()


def _unit_directions(mesh) -> np.ndarray:
    """Return one normalized direction per CSR adjacency entry, cached per mesh."""
    cached = _UNIT_DIRECTIONS.get(mesh)
    if cached is not None:
        return cached
    xyz, offsets, neighbours = mesh.xyz, mesh.adj_offset, mesh.adj_list
    directions = np.zeros(3 * len(neighbours), dtype=np.float64)
    for cell in range(mesh.num_cells):
        x, y, z = xyz[3 * cell : 3 * cell + 3]
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            other = int(neighbours[edge])
            dx, dy, dz = (
                float(xyz[3 * other] - x),
                float(xyz[3 * other + 1] - y),
                float(xyz[3 * other + 2] - z),
            )
            length = float(np.hypot(np.hypot(dx, dy), dz)) or 1.0
            directions[3 * edge : 3 * edge + 3] = (
                dx / length,
                dy / length,
                dz / length,
            )
    _UNIT_DIRECTIONS[mesh] = directions
    return directions


def _contact_uncached(world, cell: int, out) -> int:
    """Compute the foreign contact cell and averaged outward normal."""
    owner, mesh = world.owner, world.mesh
    xyz, offsets, neighbours = mesh.xyz, mesh.adj_offset, mesh.adj_list
    unit = _unit_directions(mesh)
    plate_id = int(owner[cell])
    x, y, z = (float(v) for v in xyz[3 * cell : 3 * cell + 3])
    nx = ny = nz = 0.0
    shared = 0
    for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
        other = int(neighbours[edge])
        other_owner = int(owner[other])
        if other_owner < 0 or other_owner == plate_id:
            continue
        shared += 1
        nx += unit[3 * edge]
        ny += unit[3 * edge + 1]
        nz += unit[3 * edge + 2]
    if shared == 0:
        return -1
    normal_length = float(np.hypot(np.hypot(nx, ny), nz))
    if normal_length < 1e-9:
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            other = int(neighbours[edge])
            if int(owner[other]) >= 0 and int(owner[other]) != plate_id:
                nx = float(xyz[3 * other] - x)
                ny = float(xyz[3 * other + 1] - y)
                nz = float(xyz[3 * other + 2] - z)
                break
        normal_length = float(np.hypot(np.hypot(nx, ny), nz)) or 1.0
    nx, ny, nz = nx / normal_length, ny / normal_length, nz / normal_length
    across, best_alignment = -1, -2.0
    for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
        other = int(neighbours[edge])
        other_owner = int(owner[other])
        if other_owner < 0 or other_owner == plate_id:
            continue
        alignment = (
            (float(xyz[3 * other]) - x) * nx
            + (float(xyz[3 * other + 1]) - y) * ny
            + (float(xyz[3 * other + 2]) - z) * nz
        )
        if alignment > best_alignment:
            best_alignment, across = alignment, other
    out[0], out[1], out[2], out[3] = nx, ny, nz, shared
    return across


def contact_at(world, cell: int, out) -> int:
    """Return the neighbouring world cell across a plate contact.

    The writable output must have four entries: unit normal (nx, ny, nz)
    and the count of foreign neighbouring edges. Returns -1 for an interior
    cell. Cached results are invalidated by incrementing occupancyStamp.
    """
    if len(out) < 4:
        raise ValueError("out must provide at least four writable entries")
    n = len(world.owner)
    if cell < 0 or cell >= n:
        raise IndexError("cell index out of range")
    cache = getattr(world, "_contact", None)
    if cache is None or len(cache["across"]) != n:
        cache = {
            "done": np.zeros(n, dtype=np.uint32),
            "across": np.full(n, -1, dtype=np.int32),
            "normal": np.zeros(4 * n, dtype=np.float64),
        }
        world._contact = cache
    version = (int(world.occupancyStamp) + 1) & 0xFFFFFFFF
    if version == 0:
        version = 1
    if int(cache["done"][cell]) == version:
        across = int(cache["across"][cell])
        if across >= 0:
            out[0:4] = cache["normal"][4 * cell : 4 * cell + 4]
        return across
    cache["done"][cell] = version
    across = _contact_uncached(world, cell, out)
    cache["across"][cell] = across
    if across >= 0:
        cache["normal"][4 * cell : 4 * cell + 4] = out[0:4]
    return across
