"""Read-only reconstruction of world-frame crust ownership (V2 reconstruct.js)."""
from __future__ import annotations

from .crust import CrustType
from .gather import clear_world_view, for_each_claim


def reconstruct(world) -> None:
    """Rebuild owner, ownerCell and claim counters without changing plate crust."""
    clear_world_view(world)

    def claim(world_cell, plate, raster_cell):
        land = int(plate.crust.type[raster_cell]) != CrustType["OCEAN"]
        count = int(world.claimCount[world_cell])
        lands = int(world.landClaims[world_cell])
        if count == 0 or (land and lands == 0):
            world.owner[world_cell] = plate.id
            world.ownerCell[world_cell] = raster_cell
        if count < 255:
            world.claimCount[world_cell] = count + 1
        if land and lands < 255:
            world.landClaims[world_cell] = lands + 1

    for_each_claim(world, claim)
