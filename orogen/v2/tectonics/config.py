"""Shared V2 tectonic configuration defaults and validation.

Reference: upstream v3.0.1, v2/js/tectonics/config.js.
This is the first Python configuration slice; unported upstream keys may be
provided by callers and are preserved rather than silently discarded.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from numbers import Real
from typing import Any, Mapping

CM_PER_YR_TO_KM_PER_MYR = 10.0
CM_PER_YR_TO_RAD_PER_MYR = CM_PER_YR_TO_KM_PER_MYR / 6371.0

TECTONICS: dict[str, Any] = {
    "simCells": 40000, "maxStepMyr": 4.0, "stepGridMyr": 2.0,
    "maxCellsPerStep": 1.5, "slowRulesEveryMyr": 0.0, "snapshotEveryMyr": 10.0,
    "boundaryActiveCmYr": 0.3, "obliqueLimit": 0.35, "closingCmYr": 0.2,
    "orogenyQuietMyr": 40.0, "slabPull": 1.0, "slabResistance": 30.0,
    "slabPullAgeRefMyr": 80.0, "slabMatureKm": 600.0, "slabSuction": 0.25,
    "ridgePush": 0.05, "ridgeMaxAgeMyr": 40.0, "collisionResistance": 100.0,
    "collisionShearShare": 0.5, "collisionSeparateShare": 0.05,
    "dragOcean": 1.0, "dragContinent": 5.0, "dragCraton": 8.0,
    "mobility": 3.0, "relaxMyr": 6.0, "maxSpeedOceanic": 25.0,
    "maxSpeedContinental": 10.0, "continentalPlateShare": 0.15,
    "continentalSoftCmYr": 7.0, "oceanicSoftCmYr": 12.0,
    "belts": {
        "andean": {"baseKm": 120.0, "perKm": 1.0, "maxKm": 400.0},
        "himalayan": {"baseKm": 220.0, "perKm": 0.6, "maxKm": 1000.0},
        "foreland": {"baseKm": 120.0, "perKm": 0.2, "maxKm": 350.0},
        "ural": {"baseKm": 150.0, "perKm": 0.3, "maxKm": 400.0},
        "laramide": {"baseKm": 200.0, "perKm": 0.6, "maxKm": 1800.0},
        "transpression": {"baseKm": 40.0, "perKm": 0.3, "maxKm": 150.0},
    },
    "beltSpeedRefCmYr": 5.0, "beltSpeedExponent": 0.5,
    "beltSpeedFactorMin": 0.6, "beltSpeedFactorMax": 1.6,
    "sutureShorteningKm": 600.0, "sutureStallCmYr": 0.4,
    "sutureStallMyr": 12.0, "sutureStallMinKm": 200.0,
    "sutureContactMyr": 40.0, "sutureOverlapKm2": 400000.0,
}

_POSITIVE = frozenset({
    "simCells", "maxStepMyr", "stepGridMyr", "maxCellsPerStep",
    "snapshotEveryMyr", "slabPullAgeRefMyr", "slabMatureKm",
    "ridgeMaxAgeMyr", "mobility", "relaxMyr", "maxSpeedOceanic",
    "maxSpeedContinental", "beltSpeedRefCmYr", "sutureStallMyr",
    "sutureContactMyr",
})
_NON_NEGATIVE = frozenset({
    "slowRulesEveryMyr", "boundaryActiveCmYr", "closingCmYr",
    "orogenyQuietMyr", "slabPull", "slabResistance", "slabSuction",
    "ridgePush", "collisionResistance", "dragOcean", "dragContinent",
    "dragCraton", "continentalSoftCmYr", "oceanicSoftCmYr",
    "sutureShorteningKm", "sutureStallCmYr", "sutureStallMinKm",
    "sutureOverlapKm2",
})
_UNIT_INTERVAL = frozenset({
    "obliqueLimit", "collisionShearShare", "collisionSeparateShare",
    "continentalPlateShare",
})


def _merge(base: dict[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in values.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_tectonics_config(config: Mapping[str, Any]) -> None:
    """Validate known constraints while allowing forward-compatible extra keys."""
    if not isinstance(config, Mapping):
        raise TypeError("tectonics config must be a mapping")
    for key in _POSITIVE | _NON_NEGATIVE | _UNIT_INTERVAL | {
        "beltSpeedExponent", "beltSpeedFactorMin", "beltSpeedFactorMax"
    }:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{key} must be a real number")
        if not isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
        if key in _POSITIVE and value <= 0:
            raise ValueError(f"{key} must be positive")
        if key in _NON_NEGATIVE and value < 0:
            raise ValueError(f"{key} must be non-negative")
        if key in _UNIT_INTERVAL and not 0 <= value <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
        if key in {"beltSpeedFactorMin", "beltSpeedFactorMax"} and value < 0:
            raise ValueError(f"{key} must be non-negative")
    if config.get("beltSpeedFactorMin", 0.6) > config.get("beltSpeedFactorMax", 1.6):
        raise ValueError("beltSpeedFactorMin cannot exceed beltSpeedFactorMax")

    belts = config.get("belts")
    if belts is not None:
        if not isinstance(belts, Mapping):
            raise TypeError("belts must be a mapping")
        for name, belt in belts.items():
            if not isinstance(belt, Mapping):
                raise TypeError(f"belts.{name} must be a mapping")
            for field in ("baseKm", "perKm", "maxKm"):
                if field in belt:
                    value = belt[field]
                    if isinstance(value, bool) or not isinstance(value, Real):
                        raise TypeError(f"belts.{name}.{field} must be a real number")
                    if not isfinite(float(value)) or value < 0:
                        raise ValueError(f"belts.{name}.{field} must be finite and non-negative")
            if all(field in belt for field in ("baseKm", "maxKm")) and belt["baseKm"] > belt["maxKm"]:
                raise ValueError(f"belts.{name}.baseKm cannot exceed maxKm")


def make_tectonics_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return an isolated config with recursive overrides and validated known fields."""
    if overrides is not None and not isinstance(overrides, Mapping):
        raise TypeError("tectonics overrides must be a mapping")
    config = _merge(TECTONICS, overrides or {})
    validate_tectonics_config(config)
    return config
