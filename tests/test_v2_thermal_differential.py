"""Differential checks for mantle heat memory and plume-site selection."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.supercontinent import is_continental
from orogen.v2.tectonics.thermal import (
    insulate_at_start,
    pick_plume_site,
    update_mantle_heat,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/thermal-differential-runner.mjs"


def _xyz(latitude: float, longitude: float) -> list[float]:
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    return [
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ]


def _base_arrays() -> dict[str, Any]:
    xyz = [
        _xyz(0.0, 0.0),
        _xyz(0.0, 1.0),
        _xyz(0.0, 4.0),
        _xyz(0.0, 10.0),
        _xyz(-20.0, 0.0),
        _xyz(90.0, 0.0),
    ]
    return {
        "xyz": [value for point in xyz for value in point],
        "owners": [0, 0, 1, 1, 2, -1],
        "ownerCells": [0, 1, 0, 1, 0, 0],
        "plates": [
            {
                "types": [
                    CrustType["CONTINENT"],
                    CrustType["CRATON"],
                ]
            },
            {
                "types": [
                    CrustType["OCEAN"],
                    CrustType["ARC"],
                ]
            },
            {"types": [CrustType["CRATON"]]},
        ],
    }


def _payload() -> dict[str, Any]:
    arrays = _base_arrays()
    insulation = {
        **arrays,
        "config": {},
        "heat": [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
        "hotspots": [],
    }
    warm = {
        **arrays,
        "config": {
            "mantleInsulateMyr": 40.0,
            "mantleCoolMyr": 200.0,
            "plumeRatePerMyr": 0.0,
        },
        "heat": [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
        "hotspots": [],
        "dt": 15.0,
    }

    live_hotspots = [
        {"xyz": _xyz(0.0, 0.0), "alive": True},
        {"xyz": _xyz(0.0, 10.0), "alive": False},
    ]
    pick_live = {
        **arrays,
        "config": {
            "plumeOceanWeight": 0.2,
            "plumeSpacingKm": 200.0,
        },
        "heat": [0.9, 0.8, 0.0, 1.0, 0.2, 0.8],
        "hotspots": live_hotspots,
        "draws": [0.0, 0.4, 0.8, 0.999],
    }
    pick_zero = {
        **arrays,
        "config": {
            "plumeOceanWeight": 0.0,
            "plumeSpacingKm": 200.0,
        },
        "heat": [0.0] * 6,
        "hotspots": [],
        "draws": [0.0, 0.5, 0.999],
    }
    return {
        "insulation": insulation,
        "warm": warm,
        "pickCases": [pick_live, pick_zero],
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> dict[str, Any]:
    return run_js_reference(RUNNER, payload)


def _world(spec: dict[str, Any]):
    return SimpleNamespace(
        config=spec["config"],
        mesh=SimpleNamespace(
            num_cells=len(spec["owners"]),
            xyz=np.asarray(spec["xyz"], dtype=np.float64),
        ),
        owner=np.asarray(spec["owners"], dtype=np.int16),
        ownerCell=np.asarray(spec["ownerCells"], dtype=np.int32),
        plates=[
            SimpleNamespace(
                crust=SimpleNamespace(
                    type=np.asarray(plate["types"], dtype=np.uint8)
                )
            )
            for plate in spec["plates"]
        ],
        mantleHeat=np.asarray(spec["heat"], dtype=np.float32),
        hotspots=spec.get("hotspots", []),
        hotspotsReady=True,
    )


def test_initial_mantle_insulation_matches_upstream(javascript, payload):
    world = _world(payload["insulation"])
    flags = [
        is_continental(world, cell) for cell in range(len(world.owner))
    ]
    insulate_at_start(world)
    assert_js_python_equal(
        {"continental": flags, "heat": world.mantleHeat},
        javascript["insulation"],
        atol=1e-7,
    )


def test_mantle_warming_and_cooling_match_upstream(javascript, payload):
    scenario = payload["warm"]
    world = _world(scenario)
    update_mantle_heat(world, scenario["dt"])
    assert_js_python_equal(world.mantleHeat, javascript["warmed"], atol=2e-7)


def test_weighted_plume_site_selection_matches_upstream(javascript, payload):
    expected = []
    for scenario in payload["pickCases"]:
        world = _world(scenario)
        selected = [
            pick_plume_site(world, lambda value=draw: value)
            for draw in scenario["draws"]
        ]
        expected.append(selected)
    assert_js_python_equal(expected, javascript["picks"])
