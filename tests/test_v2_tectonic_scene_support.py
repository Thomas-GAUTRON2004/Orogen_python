"""Tests for shared tectonic scenario construction helpers."""

import math

import pytest

from orogen.v2.tectonics import scene_support
from orogen.v2.tectonics.force_scenes import EAST, WEST
from orogen.v2.tectonics.margin_scenes import LARAMIDE, QUIET, ROLLBACK_ONLY
from orogen.v2.tectonics.terrain_scenes import FAR_TERRANES, WHOLE_PLANET_PACE


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


def test_shared_scenario_configuration_templates_are_immutable():
    for template in (
        LARAMIDE,
        QUIET,
        ROLLBACK_ONLY,
        FAR_TERRANES,
        WHOLE_PLANET_PACE,
    ):
        with pytest.raises(TypeError):
            template["mutation_probe"] = True

    with pytest.raises(TypeError):
        WEST[0][0] = 0.0
    with pytest.raises(TypeError):
        EAST[0][0] = 0.0
