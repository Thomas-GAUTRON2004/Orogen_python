"""Convergence polarity rules from Orogen V2 step/polarity.js."""

from __future__ import annotations

from .crust import CrustFlag, CrustType


def overrides(plate_a, cell_a: int, plate_b, cell_b: int, decided: int = -1):
    """Return True if A overrides B, False if B overrides A, or None for collision."""
    a, b = plate_a.crust, plate_b.crust
    land_a = int(a.type[cell_a]) != CrustType["OCEAN"]
    land_b = int(b.type[cell_b]) != CrustType["OCEAN"]
    if land_a and land_b:
        return None
    if land_a != land_b:
        return land_a

    if decided >= 0 and (decided == plate_a.id or decided == plate_b.id):
        return decided == plate_a.id

    trench_a = bool(int(a.flags[cell_a]) & CrustFlag["TRENCH"])
    trench_b = bool(int(b.flags[cell_b]) & CrustFlag["TRENCH"])
    if trench_a != trench_b:
        return trench_a

    if plate_a.isBackground != plate_b.isBackground:
        return bool(plate_b.isBackground)

    age_a, age_b = float(a.formedAt[cell_a]), float(b.formedAt[cell_b])
    if age_a != age_b:
        return age_a > age_b

    return (
        ((cell_a ^ cell_b) & 1) == 0
        and plate_a.id < plate_b.id
        or (((cell_a ^ cell_b) & 1) != 0 and plate_a.id > plate_b.id)
    )
