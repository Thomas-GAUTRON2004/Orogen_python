"""Focused characterization tests for the upstream initial-craton geometry."""

from __future__ import annotations

import math
import random

import pytest

from orogen.v2.tectonics import initial_cratons
from orogen.v2.tectonics.config import make_tectonics_config
from orogen.v2.tectonics.initial_cratons import (
    _craton_letter,
    _frame_at,
    _piece_shape,
    _random_centre_within_latitude,
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


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, "A"), (25, "Z"), (26, "AA"), (27, "AB"), (701, "ZZ"), (702, "AAA")],
)
def test_craton_letters_remain_alphanumeric_beyond_26(index, expected):
    assert _craton_letter(index) == expected


def test_random_centre_within_tiny_latitude_band_has_bounded_fallback(monkeypatch):
    monkeypatch.setattr(initial_cratons, "random_point", lambda _rng: [1.0, 0.0, 0.0])
    centre = _random_centre_within_latitude(lambda: 0.5, 0.001)
    assert len(centre) == 3
    assert sum(value * value for value in centre) == pytest.approx(1.0)
    assert abs(centre[2]) <= math.sin(math.radians(0.001))


def test_craton_letter_rejects_negative_index():
    with pytest.raises(ValueError, match="non-negative"):
        _craton_letter(-1)


def test_random_centre_accepts_first_sample_inside_latitude_band(monkeypatch):
    monkeypatch.setattr(initial_cratons, "random_point", lambda _rng: [1.0, 0.0, 0.0])
    centre = _random_centre_within_latitude(lambda: 0.5, 90.0)
    assert centre == [1.0, 0.0, 0.0]


def test_single_piece_craton_block_builds_without_rift_gap(monkeypatch):
    outline = {
        "points": [
            [math.cos(i * math.tau / 8), math.sin(i * math.tau / 8)]
            for i in range(8)
        ],
        "area": math.pi,
        "cx": 0.0,
        "cy": 0.0,
        "compact": 1.0,
    }
    monkeypatch.setattr(initial_cratons, "_trace_outline", lambda _piece: outline)
    block = initial_cratons._make_block(
        random.Random(12).random,
        make_tectonics_config(),
        [100.0],
        0.0,
    )
    assert len(block["pieces"]) == 1
    assert block["reach"] > 0.0
    assert block["inner"] > 0.0
