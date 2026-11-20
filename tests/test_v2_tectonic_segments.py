"""Tests for ridge-segment fracture coordinates and physical conversions."""

import math

import pytest

from orogen.v2.tectonics.config import CM_PER_YR_TO_KM_PER_MYR, CM_PER_YR_TO_RAD_PER_MYR
from orogen.v2.tectonics.segments import SEGMENT_KM, fracture_coordinate


def test_physical_unit_conversions_match_upstream_constants():
    assert CM_PER_YR_TO_KM_PER_MYR == 10.0
    assert pytest.approx(10.0 / 6371.0) == CM_PER_YR_TO_RAD_PER_MYR


def test_fracture_coordinate_zero_and_tiny_relative_rotation():
    assert fracture_coordinate(1.0, 0.0, 0.0, (0.0, 0.0, 0.0)) == 0.5
    assert fracture_coordinate(1.0, 0.0, 0.0, (1e-13, 0.0, 0.0)) == 0.5


def test_fracture_coordinate_is_distance_from_normalized_pole():
    # The pole is deliberately not unit length: the implementation normalizes it.
    value = fracture_coordinate(0.0, 1.0, 0.0, (0.0, 0.0, 4.0))
    assert value == pytest.approx(math.pi * 6371.0 / (2.0 * SEGMENT_KM))


def test_fracture_coordinate_avoids_reserved_zero_and_clamps_dot():
    assert fracture_coordinate(1.0, 0.0, 0.0, (1.0, 0.0, 0.0)) == 1e-6
    # A slightly non-unit input is clamped before acos, avoiding a domain error.
    assert math.isfinite(fracture_coordinate(1.0000000001, 0.0, 0.0, (1.0, 0.0, 0.0)))
