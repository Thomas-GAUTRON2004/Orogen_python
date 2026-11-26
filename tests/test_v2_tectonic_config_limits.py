"""Boundary values of the Python configuration validator.

These tests pin the Python contract. Upstream JavaScript has no validator; where
the Python rule differs from what the JavaScript accepts, the difference is listed
in docs/ADAPTATIONS.md with its evidence.
"""

from __future__ import annotations

import math

import pytest

from orogen.v2.tectonics.config import (
    _NON_NEGATIVE,
    _POSITIVE,
    _UNIT_INTERVAL,
    validate_tectonics_config,
)


@pytest.mark.parametrize("key", sorted(_POSITIVE))
def test_strictly_positive_bound(key: str):
    validate_tectonics_config({key: 1e-6})
    validate_tectonics_config({key: 2})
    for rejected in (0, -1, -1e-12):
        with pytest.raises(ValueError, match=f"{key} must be positive"):
            validate_tectonics_config({key: rejected})


@pytest.mark.parametrize("key", sorted(_NON_NEGATIVE))
def test_non_negative_bound(key: str):
    validate_tectonics_config({key: 0})
    validate_tectonics_config({key: 1e-6})
    with pytest.raises(ValueError, match=f"{key} must be non-negative"):
        validate_tectonics_config({key: -1e-12})


@pytest.mark.parametrize("key", sorted(_UNIT_INTERVAL))
def test_unit_interval_bounds(key: str):
    validate_tectonics_config({key: 0})
    validate_tectonics_config({key: 1})
    validate_tectonics_config({key: 0.5})
    for rejected in (-0.001, 1.001):
        with pytest.raises(ValueError, match=f"{key} must be in"):
            validate_tectonics_config({key: rejected})


@pytest.mark.parametrize("key", sorted(_POSITIVE | _NON_NEGATIVE | _UNIT_INTERVAL))
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(key: str, value: float):
    with pytest.raises(ValueError, match="must be finite"):
        validate_tectonics_config({key: value})


@pytest.mark.parametrize("key", sorted(_POSITIVE | _NON_NEGATIVE | _UNIT_INTERVAL))
@pytest.mark.parametrize("value", [True, "1", None])
def test_non_numeric_values_are_rejected(key: str, value):
    with pytest.raises(TypeError, match="must be a real number"):
        validate_tectonics_config({key: value})


@pytest.mark.parametrize("key", ["startArcKm", "startStepKm"])
def test_width_pairs_need_positive_ordered_bounds(key: str):
    validate_tectonics_config({key: [100, 200]})
    validate_tectonics_config({key: [100, 100]})
    with pytest.raises(ValueError, match="must be positive"):
        validate_tectonics_config({key: [0, 100]})
    with pytest.raises(ValueError, match="cannot exceed upper bound"):
        validate_tectonics_config({key: [200, 100]})


def test_latitude_limit_is_bounded_by_the_pole():
    validate_tectonics_config({"startMaxLatitudeDeg": 90})
    with pytest.raises(ValueError, match="cannot exceed 90 degrees"):
        validate_tectonics_config({"startMaxLatitudeDeg": 90.5})
