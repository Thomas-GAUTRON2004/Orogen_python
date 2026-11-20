"""Focused characterization tests for the upstream initial-craton geometry."""

from __future__ import annotations

import math

import pytest

from orogen.v2.tectonics.initial_cratons import (
    _frame_at,
    _piece_shape,
    _trace_outline,
    craton_edge,
)


def _circle_piece():
    return {
        "body": {
            "u0": 0.0,
            "v0": 0.0,
            "a": 1.0,
            "b": 1.0,
            "psi": 0.0,
            "lobes": [],
        },
        "cuts": [],
        "tu": 0.0,
        "tv": 0.0,
    }


def test_craton_body_signed_distance_and_outline_area():
    piece = _circle_piece()

    assert _piece_shape(piece, 0.0, 0.0)["d"] == pytest.approx(-1.0)
    assert _piece_shape(piece, 2.0, 0.0)["d"] == pytest.approx(1.0)
    outline = _trace_outline(piece)

    assert outline["area"] == pytest.approx(math.pi, rel=2e-3)
    assert outline["compact"] == pytest.approx(1.0, abs=2e-2)
    assert len(outline["points"]) == 72


def test_craton_edge_projects_world_point_into_block_frame():
    piece = _circle_piece()
    block = {
        "frame": _frame_at([1.0, 0.0, 0.0], 0.0),
        "scale": 1.0,
        "cx": 0.0,
        "cy": 0.0,
    }
    craton = {"block": block, "piece": piece}

    centre = craton_edge(craton, 1.0, 0.0, 0.0)
    outside = craton_edge(craton, math.cos(2.0), math.sin(2.0), 0.0)

    assert centre["d"] == pytest.approx(-1.0)
    assert outside["d"] == pytest.approx(1.0)
    assert centre["rift"] == 0.0
