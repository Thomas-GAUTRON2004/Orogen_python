"""Tests for shared tectonic scenario construction helpers."""

import math

from orogen.v2.tectonics import scene_support


def test_add_ocean_corridor_covers_skirt_and_corridor_predicate_branches(monkeypatch):
    points = [
        (1.0, 0.0, 0.0),
        (math.cos(0.5), math.sin(0.5), 0.0),
        (
            math.cos(0.5) * math.cos(0.2),
            math.sin(0.5) * math.cos(0.2),
            math.sin(0.2),
        ),
        (math.cos(-0.5), math.sin(-0.5), 0.0),
        (-1.0, 0.0, 0.0),
    ]
    observed = []

    def capture_ocean(_canvas, _plate, wanted, age):
        observed.extend(wanted(*point) for point in points)
        assert age == -90.0

    monkeypatch.setattr(scene_support, "add_ocean", capture_ocean)
    scene_support.add_ocean_corridor(
        {},
        object(),
        [1.0, 0.0, 0.0],
        [math.cos(1.0), math.sin(1.0), 0.0],
        100.0,
        500.0,
    )

    assert observed == [True, True, False, False, False]
