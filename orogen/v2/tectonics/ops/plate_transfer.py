"""Crust transfer between rigid plates, ported from Orogen V2 tidy-plates."""

from __future__ import annotations

import math

from ..crust import CrustFlag, CrustType
from ..plate import frame_rotation


def transfer_cells(world, source, target, cells) -> int:
    """Move raster cells from one plate to another, resampling into target frame.

    Existing target land is preserved. Source land that would overlap target
    land is placed in the nearest free neighbouring target cell; ocean is only
    copied into unoccupied target cells. Source cells are removed regardless
    of whether the target can accept them.
    """
    mesh = world.mesh
    xyz, index = mesh.xyz, mesh.index
    source_crust, target_crust = source.crust, target.crust
    moved = 0
    rotation = None

    for cell_value in cells:
        cell = int(cell_value)
        offset = 3 * cell
        world_point = source.to_world(
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )
        target_cell = target.cell_at(
            index,
            float(world_point[0]),
            float(world_point[1]),
            float(world_point[2]),
        )

        land = int(source_crust.type[cell]) != CrustType["OCEAN"]
        if land and _is_host_land(target_crust, target_cell):
            target_cell = free_cell_beside(
                target,
                mesh,
                target_cell,
                world_point,
                lambda candidate: _is_host_land(target_crust, candidate),
            )

        if target_cell >= 0 and (land or not target_crust.mask[target_cell]):
            if rotation is None:
                rotation = frame_rotation(source, target)
            target_crust.copy_from(target_cell, source_crust, cell, rotation)
            if land:
                target_crust.flags[target_cell] = int(
                    target_crust.flags[target_cell]
                ) & (0xFF ^ CrustFlag["TRENCH"])
            moved += 1

        source_crust.remove(cell)

    source.mark_reshaped()
    target.mark_reshaped()
    source.refresh_extent(xyz)
    target.refresh_extent(xyz)
    return moved


def free_cell_beside(target, mesh, cell: int, world_point, taken) -> int:
    """Find the nearest free neighbour of a taken target-frame cell."""
    xyz = mesh.xyz
    local = target.to_plate_frame(
        float(world_point[0]),
        float(world_point[1]),
        float(world_point[2]),
    )
    best = -1
    best_distance = math.inf
    for edge in range(
        int(mesh.adj_offset[cell]),
        int(mesh.adj_offset[cell + 1]),
    ):
        neighbour = int(mesh.adj_list[edge])
        if taken(neighbour):
            continue
        offset = 3 * neighbour
        dx = float(xyz[offset]) - float(local[0])
        dy = float(xyz[offset + 1]) - float(local[1])
        dz = float(xyz[offset + 2]) - float(local[2])
        distance = dx * dx + dy * dy + dz * dz
        if distance < best_distance:
            best_distance = distance
            best = neighbour
    return best


def _is_host_land(crust, cell: int) -> bool:
    return bool(crust.mask[cell] and crust.type[cell] != CrustType["OCEAN"])


__all__ = ["free_cell_beside", "transfer_cells"]
