"""Tests for the pure orogenic belt geometry rules."""

import math

import pytest

from orogen.v2.tectonics.orogeny import belt_width_km, belt_shape, obliquity_deg


def _config():
    return {
        "belts": {"andean": {"baseKm": 120, "perKm": 1.0, "maxKm": 400}},
        "beltSpeedFactorMax": 1.6,
        "beltSpeedFactorMin": 0.6,
        "beltSpeedRefCmYr": 5.0,
        "beltSpeedExponent": 0.5,
        "beltObliqueMin": 0.35,
        "beltYoungSlabBoost": 0.5,
        "beltYoungSlabRefMyr": 60.0,
        "beltOldSlabMin": 0.75,
    }


def test_belt_width_clamps_rate_and_work():
    cfg = _config()
    assert belt_width_km("andean", 100, 5, cfg) == 220
    assert belt_width_km("andean", 1000, 5, cfg) == 400
    assert belt_width_km("andean", 100, 0, cfg) == 132
    assert belt_width_km("andean", 1000, 100, cfg) == 640


def test_belt_shape_obliquity_and_slab_age():
    cfg = _config()
    assert belt_shape(1, None, cfg) == 1.0
    assert belt_shape(0, None, cfg) == 0.35
    assert belt_shape(1, 60, cfg) == 1.0
    assert belt_shape(1, 0, cfg) == 1.5
    assert belt_shape(1, 1000, cfg) == 0.75


def test_obliquity_signed_angle_and_zero_closing_guard():
    assert obliquity_deg([0, 1, 0], [1, 0, 0], [0, 0, 1], 1) == pytest.approx(45)
    assert obliquity_deg([0, -1, 0], [1, 0, 0], [0, 0, 1], 1) == pytest.approx(-45)
    assert obliquity_deg([0, 1, 0], [1, 0, 0], [0, 0, 1], 0) == pytest.approx(
        90, abs=1e-8
    )
