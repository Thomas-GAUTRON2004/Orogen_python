"""Numerical compatibility tests for the far-field vector norm."""

import math

from orogen.v2.tectonics.far_field import _norm3


def test_norm3_matches_hypot_and_avoids_intermediate_square_overflow():
    assert _norm3(3.0, 4.0, 12.0) == 13.0
    # Node 22: Math.hypot(1e308, 1e308, 1e308)
    assert _norm3(1e308, 1e308, 1e308) == 1.7320508075688772e308
