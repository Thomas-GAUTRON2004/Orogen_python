"""Transfer an overriding island-arc strip onto its host plate."""

from __future__ import annotations

from ..crust import CrustFlag, CrustType
from ..plate import frame_rotation


def scrape_arc(world, host, donor, cells) -> dict[str, int]:
    """Move donor arc cells into the host plate, absorbing cells over existing land.

    A cell that falls on host continental/cratonic/arc crust is already land and
    is counted as absorbed. Otherwise all crust-history fields are copied into
    the corresponding host-frame cell; the new seam loses its trench flag and
    gains a suture flag. The donor cell is removed in either case.
    """
    mesh = world.mesh
    xyz, index = mesh.xyz, mesh.index
    host_crust, donor_crust = host.crust, donor.crust
    moved = 0
    absorbed = 0
    rotation = None

    for cell_value in cells:
        cell = int(cell_value)
        if not donor_crust.mask[cell]:
            continue

        offset = 3 * cell
        world_position = donor.to_world(
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )
        target = host.cell_at(
            index,
            float(world_position[0]),
            float(world_position[1]),
            float(world_position[2]),
        )

        if (
            host_crust.mask[target]
            and host_crust.type[target] != CrustType["OCEAN"]
        ):
            absorbed += 1
        else:
            if rotation is None:
                rotation = frame_rotation(donor, host)
            host_crust.copy_from(target, donor_crust, cell, rotation)
            host_crust.flags[target] = (
                int(host_crust.flags[target]) & ~CrustFlag["TRENCH"]
            ) | CrustFlag["SUTURE"]
            moved += 1

        donor_crust.remove(cell)

    donor.mark_reshaped()
    host.mark_reshaped()
    if donor_crust.count == 0:
        donor.alive = False
        donor.omega = [0.0, 0.0, 0.0]
        donor.dynamics = None

    host.refresh_extent(xyz)
    donor.refresh_extent(xyz)
    return {"moved": moved, "absorbed": absorbed}


__all__ = ["scrape_arc"]
