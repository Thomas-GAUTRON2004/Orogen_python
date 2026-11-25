"""Differential checks for ported tectonic model and numerical primitives.

The same deterministic inputs are evaluated by the pinned upstream JavaScript
modules and their Python ports. These checks cover compact geology records,
belt relief, fracture-zone coordinates, force-balance math and adaptive steps.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import run_js_reference

from orogen.v2.tectonics.crust import BeltSide, CrustType, OrogenyType
from orogen.v2.tectonics.heights import (
    DEFAULT_HEIGHTS,
    build_heights,
)
from orogen.v2.tectonics.simulation import choose_step

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/tectonics-primitives-differential-runner.mjs"


def _payload() -> dict[str, Any]:
    profiles = [
        {
            "type": oro_type,
            "side": side,
            "distance": distance,
            "reach": reach,
            "jitter": 0.08 if (oro_type + side) % 2 else 0.0,
            "wear": 180.0 if side == BeltSide["LOWER"] else 0.0,
        }
        for oro_type in range(1, 9)
        for side, distance, reach in (
            (BeltSide["UPPER"], 0.18 * 500.0, 500.0),
            (BeltSide["LOWER"], 0.76 * 850.0, 850.0),
        )
    ]
    reliefs = [
        {"type": oro_type, "reach": reach}
        for oro_type in range(1, 9)
        for reach in (
            0.0,
            0.5 * DEFAULT_HEIGHTS["fullReachKm"][oro_type],
            1.5 * DEFAULT_HEIGHTS["fullReachKm"][oro_type],
        )
    ]
    height_cases = [
        {
            "type": OrogenyType["ANDEAN"],
            "side": BeltSide["LOWER"],
            "u": 0.28,
            "reach": 620.0,
            "work": 65.0,
            "thickened": 2.4,
            "amp": 1.0,
            "crustType": CrustType["CONTINENT"],
            "rootKm": 1.2,
            "upliftKm": 0.1,
            "rugged": 0.2,
        },
        {
            "type": OrogenyType["URAL"],
            "side": BeltSide["LOWER"],
            "u": 0.82,
            "reach": 320.0,
            "work": 90.0,
            "thickened": 1.1,
            "amp": 0.85,
            "crustType": CrustType["CONTINENT"],
            "rootKm": 0.5,
            "upliftKm": -0.02,
            "rugged": 0.05,
        },
        {
            "type": OrogenyType["ARC"],
            "side": BeltSide["UPPER"],
            "u": 0.2,
            "reach": 240.0,
            "work": 30.0,
            "thickened": 0.7,
            "amp": 0.75,
            "crustType": CrustType["ARC"],
            "rootKm": 0.0,
            "upliftKm": 0.0,
            "rugged": 0.0,
        },
        {
            "type": OrogenyType["RIFT"],
            "side": BeltSide["UPPER"],
            "u": 0.48,
            "reach": 260.0,
            "work": 18.0,
            "thickened": 0.9,
            "amp": 1.1,
            "crustType": CrustType["CONTINENT"],
            "rootKm": 0.2,
            "upliftKm": 0.0,
            "rugged": 0.1,
        },
        {
            "type": OrogenyType["INTRAPLATE"],
            "side": BeltSide["LOWER"],
            "u": 0.64,
            "reach": 410.0,
            "work": 52.0,
            "thickened": 1.6,
            "amp": 0.9,
            "crustType": CrustType["CRATON"],
            "rootKm": 2.0,
            "upliftKm": 0.4,
            "rugged": 0.6,
        },
    ]
    return {
        "codecs": {
            "cover": [-1.0, 0.0, 0.0001, 0.25, 1.0, 4.0, 16.0, 40.0],
            "exhum": [-1.0, 0.0, 0.1, 0.25, 1.0, 20.0, 100.0],
            "unroof": [-100.0, -31.125, -0.125, 0.0, 0.125, 31.125, 100.0],
            "flood": [0.0, 0.5, 2.0, 15.0, 120.0, 800.0],
            "arc": [0.0, 0.5, 2.0, 22.0, 130.0, 800.0],
            "age": [0.0, 1.0, 4.0, 160.0, 2048.0, 4000.0],
            "floodAge": [
                [-10.0, 10.0, False],
                [10.0, 10.0, True],
                [2.5, 12.5, True],
                [100.0, 10.0, True],
                [-100.0, 10000.0, True],
            ],
            "facies": [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.25, 0.5],
                [0.33, 0.66, 0.12, 0.91],
                [-1.0, 2.0, 0.5, 0.5],
            ],
            "decodeFacies": [0, 0x1234, 0xFFFF, 0x8001],
        },
        "covers": [
            {
                "coverKm": [2.0],
                "coverMaxKm": [2.5],
                "coverMaxAt": [4.0],
                "unroofKm": [-1.0],
                "thickenedKm": 3.0,
                "time": 30.0,
                "geology": {"crustKm": 30.0},
            },
            {
                "coverKm": [8.0],
                "coverMaxKm": [4.0],
                "coverMaxAt": [10.0],
                "unroofKm": [0.5],
                "thickenedKm": 2.0,
                "time": 42.0,
                "geology": {"crustKm": 25.0},
            },
            {
                "coverKm": [0.0],
                "coverMaxKm": [0.0],
                "coverMaxAt": [0.0],
                "unroofKm": [0.0],
                "thickenedKm": 4.0,
                "time": 50.0,
                "geology": {"crustKm": 30.0},
            },
        ],
        "segments": [
            {"point": [1.0, 0.0, 0.0], "omega": [0.0, 0.0, 0.0]},
            {"point": [1.0, 0.0, 0.0], "omega": [1.0, 0.0, 0.0]},
            {"point": [0.0, 1.0, 0.0], "omega": [0.0, 0.0, 2.0]},
            {"point": [-1.0, 0.0, 0.0], "omega": [1.0, 0.0, 0.0]},
            {"point": [0.0, 0.0, 1.0], "omega": [0.2, 0.3, 0.4]},
        ],
        "profiles": profiles,
        "reliefs": reliefs,
        "heights": DEFAULT_HEIGHTS,
        "heightCases": height_cases,
        "linearSystems": [
            {"matrix": [3, 1, 0, 1, 4, 1, 0, 1, 2], "vector": [7, 8, 3]},
            {"matrix": [2, 0, 0, 0, 5, 0, 0, 0, 4], "vector": [6, 10, -8]},
            {"matrix": [1, 2, 3, 2, 4, 6, 0, 0, 0], "vector": [1, 2, 0]},
        ],
        "speedCases": [
            {"omega": [0.0, 0.0, 0.0], "capCos": 0.8, "centre": [0, 0, 1]},
            {"omega": [0.2, -0.3, 0.4], "capCos": 0.98, "centre": [0, 0, 1]},
            {"omega": [0.7, 0.0, 0.1], "capCos": -0.2, "centre": [0, 1, 0]},
        ],
        "simulation": {
            "edgeRad": 0.012,
            "config": {
                "maxStepMyr": 4.0,
                "maxCellsPerStep": 1.5,
                "maxSpeedOceanic": 15.0,
            },
            "plates": [
                {
                    "alive": True,
                    "count": 4,
                    "omega": [0.002, 0.001, -0.003],
                    "capCos": 0.85,
                    "capCenter": [0.0, 0.0, 1.0],
                },
                {
                    "alive": True,
                    "count": 2,
                    "omega": [-0.001, 0.003, 0.002],
                    "capCos": 0.72,
                    "capCenter": [0.0, 1.0, 0.0],
                },
                {
                    "alive": True,
                    "count": 0,
                    "omega": [0.5, 0.5, 0.5],
                    "capCos": 0.9,
                    "capCenter": [1.0, 0.0, 0.0],
                },
                {
                    "alive": False,
                    "count": 5,
                    "omega": [1.0, 1.0, 1.0],
                    "capCos": 0.9,
                    "capCenter": [0.0, 0.0, 1.0],
                },
            ],
        },
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> dict[str, Any]:
    return run_js_reference(RUNNER, payload)


def _assert_close(actual, expected, *, atol: float = 2e-7) -> None:
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=atol)

\n
def test_adaptive_step_matches_upstream(javascript, payload):

    scenario = payload["simulation"]
    plates = [
        SimpleNamespace(
            alive=case["alive"],
            omega=case["omega"],
            capCos=case["capCos"],
            capCenter=case["capCenter"],
            crust=SimpleNamespace(count=case["count"]),
        )
        for case in scenario["plates"]
    ]
    for plate in plates:
        plate.to_world = lambda x, y, z: (x, y, z)
    world = SimpleNamespace(
        config=scenario["config"],
        mesh=SimpleNamespace(edge_rad=scenario["edgeRad"]),
        plates=plates,
    )
    _assert_close(javascript["step"], choose_step(world), atol=1e-12)
