"""World-frame crust claims gathered from plate-frame rasters (V2 gather.js)."""

from __future__ import annotations

import numpy as np


def _is_identity(matrix: np.ndarray) -> bool:
    return bool(np.array_equal(matrix, np.eye(3, dtype=np.float64)))


def clear_world_view(world) -> None:
    """Reset per-cell ownership and claim metadata before a gather pass."""
    world.owner.fill(-1)
    world.claimCount.fill(0)
    world.landClaims.fill(0)
    world.collider.fill(-1)
    world.consumedAt.fill(0)
    world.provisional.fill(0)


def for_each_claim(world, on_claim) -> None:
    """Call on_claim(world_cell, plate, raster_cell) for every crust claim.

    This is the direct, correctness-oriented port of the upstream gather loop.
    It deliberately omits the JS claim-map and near-edge caches for now; those
    are performance shortcuts and do not alter the claim set.
    """
    mesh = world.mesh
    xyz = mesh.xyz.reshape(-1, 3)
    for plate in world.plates:
        if not plate.alive or plate.crust.count == 0:
            continue
        cx, cy, cz, cap_cos = plate.cap_in_world(mesh)
        mask = plate.crust.mask
        if _is_identity(plate.mat):
            for w in range(mesh.num_cells):
                x, y, z = xyz[w]
                if x * cx + y * cy + z * cz < cap_cos:
                    continue
                if mask[w]:
                    on_claim(w, plate, w)
            continue

        rotation = plate.mat
        for w, (x, y, z) in enumerate(xyz):
            if x * cx + y * cy + z * cz < cap_cos:
                continue
            # JS applies the transpose of the plate pose matrix in this lookup.
            px = rotation[0, 0] * x + rotation[1, 0] * y + rotation[2, 0] * z
            py = rotation[0, 1] * x + rotation[1, 1] * y + rotation[2, 1] * z
            pz = rotation[0, 2] * x + rotation[1, 2] * y + rotation[2, 2] * z
            cell = int(mesh.index.nearest(float(px), float(py), float(pz)))
            if mask[cell]:
                on_claim(w, plate, cell)
