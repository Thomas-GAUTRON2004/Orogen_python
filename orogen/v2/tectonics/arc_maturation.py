"""Island-arc maturation into continental crust, ported from Orogen V2."""

from __future__ import annotations

import math

from .crust import CrustFlag, CrustType


def mature_arcs(world, dt: float) -> int:
    """Convert old island-arc cells touching continental crust, one ring per call.

    The arc front advances by arcContinentKmPerMyr * dt. A ring is processed
    only after at least one characteristic mesh-cell spacing has accumulated.
    Existing terrane, age, volcanic and geological fields are retained; only
    the crust type changes.
    """
    mesh, plates, config = world.mesh, world.plates, world.config
    rate = float(config["arcContinentKmPerMyr"])
    step_dt = float(dt)
    if not rate > 0.0 or step_dt <= 0.0:
        return 0

    spacing = float(mesh.edge_km) * math.sqrt(3.0) / 2.0
    changed = 0
    time = float(world.time)
    minimum_age = float(config["arcContinentAgeMyr"])
    offsets, adjacency = mesh.adj_offset, mesh.adj_list

    for plate in plates:
        crust = plate.crust
        if not plate.alive or plate.isBackground or crust.count == 0:
            continue

        plate.arcMatureKm = (
            float(getattr(plate, "arcMatureKm", 0.0)) + rate * step_dt
        )
        if plate.arcMatureKm < spacing:
            continue
        plate.arcMatureKm -= spacing

        mask, kinds = crust.mask, crust.type
        formed_at, flags = crust.formedAt, crust.flags
        ring = []
        for cell in range(int(mesh.num_cells)):
            if (
                not mask[cell]
                or kinds[cell] != CrustType["ARC"]
                or int(flags[cell]) & CrustFlag["MARGIN_ISLE"]
                or time - float(formed_at[cell]) < minimum_age
            ):
                continue
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if (
                    mask[neighbour]
                    and int(kinds[neighbour]) in (CrustType["CONTINENT"], CrustType["CRATON"])
                ):
                    ring.append(cell)
                    break

        if not ring:
            continue

        for cell in ring:
            kinds[cell] = CrustType["CONTINENT"]
        world.budget["arcMaturedCells"] = (
            int(world.budget.get("arcMaturedCells", 0)) + len(ring)
        )
        plate.dirty = True
        changed += 1

    return changed


__all__ = ["mature_arcs"]
