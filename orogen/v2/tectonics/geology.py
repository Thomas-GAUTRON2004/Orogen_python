"""Geological cover records ported from V2 step/geology.js."""
from __future__ import annotations

from collections.abc import Mapping


def thicken_cover(crust, cell: int, thickened_km: float, time_myr: float, geology: Mapping) -> float:
    """Bury existing sediment in proportion to crustal thickening; return added cover."""
    cover = float(crust.coverKm[cell])
    if not cover > 0:
        return 0.0
    crustal_scale = float(geology["crustKm"])
    if crustal_scale <= 0:
        raise ValueError("geology.crustKm must be positive")
    added = cover * float(thickened_km) / crustal_scale
    new_cover = cover + added
    crust.coverKm[cell] = new_cover
    if new_cover > float(crust.coverMaxKm[cell]):
        crust.coverMaxKm[cell] = new_cover
        crust.coverMaxAt[cell] = time_myr
    crust.unroofKm[cell] -= added
    return added
