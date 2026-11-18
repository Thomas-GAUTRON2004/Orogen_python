"""Birth a mobile oceanic plate out of the passive background ocean."""

from __future__ import annotations

import math

from .plate_transfer import transfer_cells
from ..world import EventKind


def birth_ocean_plate(world, background, seed_cells, reach_km: float, name: str):
    """Carve background crust within graph reach of a world-frame margin.

    The selected background-owned cells are transferred into a new plate in
    the background frame. The new plate starts stationary and is subsequently
    driven by the ordinary force-balance stage.
    """
    mesh = world.mesh
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    owner, owner_cell = world.owner, world.ownerCell
    edge_km = float(mesh.edge_km)
    cell_area = float(mesh.cell_area_km2)
    config = world.config
    max_hops = max(
        1,
        math.floor(float(reach_km) / edge_km + 0.5),
    )

    seen = set()
    take = []
    frontier = []
    for world_cell_value in seed_cells:
        world_cell = int(world_cell_value)
        for edge in range(
            int(offsets[world_cell]),
            int(offsets[world_cell + 1]),
        ):
            neighbour = int(adjacency[edge])
            if int(owner[neighbour]) == int(background.id) and neighbour not in seen:
                seen.add(neighbour)
                frontier.append(neighbour)
                take.append(neighbour)

    for _ in range(max_hops):
        if not frontier:
            break
        following = []
        for world_cell in frontier:
            for edge in range(
                int(offsets[world_cell]),
                int(offsets[world_cell + 1]),
            ):
                neighbour = int(adjacency[edge])
                if int(owner[neighbour]) != int(background.id) or neighbour in seen:
                    continue
                seen.add(neighbour)
                following.append(neighbour)
                take.append(neighbour)
        frontier = following

    area_km2 = len(take) * cell_area
    if area_km2 < float(config["oceanPlateMinAreaKm2"]):
        return None

    fresh = world.add_plate(name)
    fresh.set_quat(background.quat.copy())
    fresh.omega = [0.0, 0.0, 0.0]

    cells = []
    cells_seen = set()
    for world_cell in take:
        raster_cell = int(owner_cell[world_cell])
        if background.crust.mask[raster_cell] and raster_cell not in cells_seen:
            cells_seen.add(raster_cell)
            cells.append(raster_cell)

    moved = transfer_cells(world, background, fresh, cells)
    if moved == 0:
        fresh.alive = False
        return None

    fresh.born = float(world.time)
    fresh.isOceanBorn = True
    area_mkm2 = area_km2 / 1e6
    world.log(
        EventKind["PLATE_BORN"],
        (
            f"{name} is born: {area_mkm2:.1f} Mkm² of the old ocean "
            "becomes a plate of its own"
        ),
        {"plate": fresh.id, "areaMkm2": area_mkm2},
    )
    return fresh


def ocean_plate_name(world) -> str:
    """Return the next deterministic ocean-born plate name."""
    count = sum(1 for plate in world.plates if plate.isOceanBorn)
    return "Thalassa" if count == 0 else f"Thalassa {count + 1}"


__all__ = ["birth_ocean_plate", "ocean_plate_name"]
