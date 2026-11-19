"""Shared geological record model and compact frame encodings (V2 model/geology.js).

These fields are a serialization-facing view of records stored on CrustRaster.
The Python helpers intentionally preserve the upstream quantization rules.
"""

from __future__ import annotations

from collections.abc import Mapping

GeoFlag = {
    "WAS_ARC": 1,
    "FORELAND": 2,
    "RIFT_BASIN": 4,
    "OPHIOLITE": 8,
    "SLAB_CHANGE": 16,
    "SHELF": 32,
    "EPEIRIC": 64,
}

GEO_FRAME_FIELDS = [
    "geoFlags", "coverKm", "coverMaxKm", "coverMaxAge", "facies",
    "exhumKm", "unroofKm", "floodMyr", "floodAge", "arcMyr",
]
GEOLOGY_READS = ["oroStart"]

_COVER_MAX_KM = 16.0
_EXHUM_STEP_KM = 0.25
_FLOOD_STEP_MYR = 2.0
_ARC_STEP_MYR = 2.0
_AGE_STEP_MYR = 8.0
_FLOOD_AGE_STEP_MYR = 5.0


def _byte(value: float) -> int:
    return min(255, max(0, int(round(value))))


def encode_cover_km(km: float) -> int:
    return _byte(255.0 * (_positive(km) / _COVER_MAX_KM) ** 0.5)


def decode_cover_km(value: int) -> float:
    return _COVER_MAX_KM * (int(value) / 255.0) ** 2


def encode_exhum_km(km: float) -> int:
    return _byte(_positive(km) / _EXHUM_STEP_KM)


def decode_exhum_km(value: int) -> float:
    return int(value) * _EXHUM_STEP_KM


def encode_unroof_km(km: float) -> int:
    return max(-127, min(127, int(round(float(km) / _EXHUM_STEP_KM))))


def decode_unroof_km(value: int) -> float:
    return int(value) * _EXHUM_STEP_KM


def encode_flood_myr(myr: float) -> int:
    return _byte(_positive(myr) / _FLOOD_STEP_MYR)


def decode_flood_myr(value: int) -> float:
    return int(value) * _FLOOD_STEP_MYR


def encode_arc_myr(myr: float) -> int:
    return _byte(_positive(myr) / _ARC_STEP_MYR)


def decode_arc_myr(value: int) -> float:
    return int(value) * _ARC_STEP_MYR


def encode_age(myr: float) -> int:
    return _byte(_positive(myr) / _AGE_STEP_MYR)


def decode_age(value: int) -> float:
    return int(value) * _AGE_STEP_MYR


def encode_flood_age(last: float, time: float, ever: bool) -> int:
    if not ever:
        return 0
    return 1 + min(254, max(0, int(round((float(time) - float(last)) / _FLOOD_AGE_STEP_MYR))))


def decode_flood_age(value: int) -> float:
    return (int(value) - 1) * _FLOOD_AGE_STEP_MYR if value else float("inf")


def encode_facies(carb: float, salt: float, coal: float, organic: float) -> int:
    def nibble(value: float) -> int:
        return min(15, max(0, int(round(15.0 * float(value)))))

    return nibble(carb) | (nibble(salt) << 4) | (nibble(coal) << 8) | (nibble(organic) << 12)


def decode_facies(value: int, out: dict[str, float] | None = None) -> dict[str, float]:
    if out is None:
        out = {"carb": 0.0, "salt": 0.0, "coal": 0.0, "organic": 0.0}
    packed = int(value)
    out["carb"] = (packed & 15) / 15.0
    out["salt"] = ((packed >> 4) & 15) / 15.0
    out["coal"] = ((packed >> 8) & 15) / 15.0
    out["organic"] = ((packed >> 12) & 15) / 15.0
    return out


def _positive(value: float) -> float:
    return max(0.0, float(value))


def thicken_cover(
    crust, cell: int, thickened_km: float, time_myr: float, geology: Mapping
) -> float:
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
