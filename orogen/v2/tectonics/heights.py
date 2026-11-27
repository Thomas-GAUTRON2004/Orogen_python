"""Orogenic relief records ported from V2 step/heights.js.

This module implements the belt-building path only. Time-dependent erosion,
isostatic rebound and terrain synthesis are separate simulation stages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypedDict, cast

from .crust import BeltSide, CrustType, OrogenyType


class HeightConfig(TypedDict):
    """Typed relief parameters used when building and rejuvenating mountain belts."""

    reliefKm: Mapping[int, float]
    fullReachKm: Mapping[int, float]
    youngShare: float
    upliftBuildKm: Mapping[int, float]
    rootGain: Mapping[int, float]
    arcUpliftShare: float


DEFAULT_HEIGHTS: HeightConfig = {
    "reliefKm": {
        OrogenyType["ANDEAN"]: 3.4,
        OrogenyType["LARAMIDE"]: 1.5,
        OrogenyType["URAL"]: 1.7,
        OrogenyType["HIMALAYAN"]: 3.2,
        OrogenyType["ARC"]: 0.8,
        OrogenyType["RIFT"]: 1.4,
        OrogenyType["TRANSPRESSION"]: 1.9,
        OrogenyType["INTRAPLATE"]: 3.0,
    },
    "fullReachKm": {
        OrogenyType["ANDEAN"]: 450,
        OrogenyType["LARAMIDE"]: 900,
        OrogenyType["URAL"]: 250,
        OrogenyType["HIMALAYAN"]: 900,
        OrogenyType["ARC"]: 200,
        OrogenyType["RIFT"]: 220,
        OrogenyType["TRANSPRESSION"]: 80,
        OrogenyType["INTRAPLATE"]: 300,
    },
    "youngShare": 0.55,
    "upliftBuildKm": {
        OrogenyType["ANDEAN"]: 20,
        OrogenyType["LARAMIDE"]: 20,
        OrogenyType["URAL"]: 30,
        OrogenyType["HIMALAYAN"]: 50,
        OrogenyType["ARC"]: 20,
        OrogenyType["RIFT"]: 1,
        OrogenyType["TRANSPRESSION"]: 30,
        OrogenyType["INTRAPLATE"]: 15,
    },
    "rootGain": {OrogenyType["ANDEAN"]: 1.25},
    "arcUpliftShare": 0.25,
}


def _smooth(a: float, b: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def _bump(x: float, centre: float, sigma: float, g: float = 0.0) -> float:
    if not g:
        t = (x - centre) / sigma
        return math.exp(-t * t)
    width = math.sqrt(sigma * sigma + g)
    t = (x - centre) / width
    return (sigma / width) * math.exp(-t * t)


def _plateau(u: float, a: float, b: float, edge: float) -> float:
    return _smooth(a - edge, a + edge, u) * (1.0 - _smooth(b - edge, b + edge, u))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rift_wear(crest_width: float, g: float) -> float:
    if not g:
        return 1.0
    sigma = 0.6 * crest_width
    return sigma / math.sqrt(sigma * sigma + g)


def broad_profile(
    oro_type: int,
    side: int,
    distance_km: float,
    reach_km: float,
    jitter: float = 0.0,
    wear_g: float = 0.0,
) -> float:
    """Smooth, evolving-uplift component of a belt cross-section."""
    width = max(40.0, float(reach_km))
    u = float(distance_km) / width
    fade = 1.0 - _smooth(0.7 + jitter, 1.05 + jitter, u)
    d = float(distance_km)
    if oro_type == OrogenyType["ANDEAN"]:
        mature = _smooth(180, 450, width)
        return fade * (
            0.575
            * _bump(d, _clamp(0.3 * width, 70, 260), max(110, 0.2 * width), wear_g)
            + 0.55 * mature * _plateau(u, 0.33, 0.72, 0.1)
            + 0.3
            * _smooth(200, 350, width)
            * _bump(d, 0.76 * width, max(100, 0.12 * width), wear_g)
        )
    if oro_type == OrogenyType["LARAMIDE"]:
        return fade * (
            0.3 * _bump(d, _clamp(0.1 * width, 40, 150), max(100, 0.1 * width), wear_g)
            + 0.55 * _plateau(u, 0.15, 0.86, 0.1)
            + 0.45 * _bump(d, 0.88 * width, max(100, 0.08 * width), wear_g)
        )
    if oro_type == OrogenyType["URAL"]:
        if side == BeltSide["LOWER"]:
            return fade * (
                0.5 * _bump(d, 0, max(100, 0.3 * width), wear_g)
                - 0.25 * _bump(d, 0.85 * width, max(80, 0.15 * width), wear_g)
            )
        return fade * 0.6 * _bump(d, 0, max(110, 0.35 * width), wear_g)
    if oro_type == OrogenyType["HIMALAYAN"]:
        if side == BeltSide["LOWER"]:
            return fade * (
                0.675
                * _bump(
                    d, _clamp(0.22 * width, 40, 130), max(110, 0.15 * width), wear_g
                )
                - 0.3 * _bump(d, 0.86 * width, max(90, 0.12 * width), wear_g)
            )
        return fade * (
            0.525
            * _bump(d, _clamp(0.05 * width, 20, 80), max(110, 0.08 * width), wear_g)
            + 0.8 * _smooth(300, 800, width) * _plateau(u, 0.1, 0.84, 0.1)
            + 0.525 * _bump(d, 0.88 * width, max(110, 0.08 * width), wear_g)
        )
    if oro_type == OrogenyType["ARC"]:
        return fade * 0.675 * _bump(d, 0, max(90, 0.35 * width), wear_g)
    if oro_type == OrogenyType["TRANSPRESSION"]:
        return fade * 0.525 * _bump(d, 0, max(70, 0.45 * width), wear_g)
    if oro_type == OrogenyType["RIFT"]:
        crest = _clamp(0.12 * width, 25, 90)
        scale = _rift_wear(crest, wear_g)
        expanded = crest + (d - crest) * scale if wear_g else d
        shape = (
            _bump(expanded, crest, max(70, crest))
            if expanded < crest
            else math.exp(-(expanded - crest) / (0.45 * width))
        )
        return fade * 0.9 * scale * shape
    if oro_type == OrogenyType["INTRAPLATE"]:
        return fade * (
            0.5 * _bump(d, 0, max(100, 0.3 * width), wear_g)
            - 0.2 * _bump(d, 0.8 * width, max(80, 0.15 * width), wear_g)
        )
    return 0.0


def belt_relief_km(oro_type: int, reach_km: float, heights: Mapping) -> float:
    """Target crest relief, scaled down for a young/narrow belt."""
    relief = float(heights["reliefKm"][oro_type])
    full_reach = float(heights["fullReachKm"][oro_type])
    maturity = min(1.0, float(reach_km) / full_reach)
    young = float(heights["youngShare"])
    return relief * (young + (1.0 - young) * maturity)


def build_heights(
    crust,
    cell: int,
    oro_type: int,
    side: int,
    u: float,
    reach_km: float,
    work_km: float,
    thickened_km: float,
    config: Mapping,
    amp: float = 1.0,
) -> None:
    """Accumulate crustal root, broad uplift and ruggedness for one belt cell."""
    heights = cast(HeightConfig, config.get("heights") or DEFAULT_HEIGHTS)
    crust.rootKm[cell] += float(thickened_km) * float(
        heights.get("rootGain", {}).get(oro_type, 1.0)
    )
    amplitude = float(amp) * belt_relief_km(oro_type, reach_km, heights)
    share = 1.0 - math.exp(-float(work_km) / float(heights["upliftBuildKm"][oro_type]))
    own = (
        float(heights["arcUpliftShare"])
        if int(crust.type[cell]) == CrustType["ARC"]
        and oro_type in (OrogenyType["ANDEAN"], OrogenyType["ARC"])
        else 1.0
    )
    target = (
        own
        * amplitude
        * broad_profile(
            oro_type, int(side), float(u) * float(reach_km), float(reach_km)
        )
    )
    current = float(crust.upliftKm[cell])
    if (target >= 0 and target > current) or (target < 0 and target < current):
        crust.upliftKm[cell] += (target - current) * share
    if amplitude > float(crust.rugged[cell]):
        crust.rugged[cell] += (amplitude - float(crust.rugged[cell])) * share
