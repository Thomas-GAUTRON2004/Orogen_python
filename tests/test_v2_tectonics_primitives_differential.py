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
from orogen.v2.tectonics.dynamics import solve3
from orogen.v2.tectonics.geology import (
    GEO_FRAME_FIELDS,
    GEOLOGY_READS,
    decode_age,
    decode_arc_myr,
    decode_cover_km,
    decode_exhum_km,
    decode_facies,
    decode_flood_age,
    decode_flood_myr,
    decode_unroof_km,
    encode_age,
    encode_arc_myr,
    encode_cover_km,
    encode_exhum_km,
    encode_facies,
    encode_flood_age,
    encode_flood_myr,
    encode_unroof_km,
    thicken_cover,
)
from orogen.v2.tectonics.heights import (
    DEFAULT_HEIGHTS,
    belt_relief_km,
    broad_profile,
    build_heights,
)
from orogen.v2.tectonics.segments import fracture_coordinate
from orogen.v2.tectonics.simulation import choose_step, fastest_speed

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


def _codec_pairs(values: list[float], encode: Callable, decode: Callable):
    return [[encode(value), decode(encode(value))] for value in values]


def test_geology_frame_encodings_match_upstream(javascript, payload):
    codecs = payload["codecs"]
    expected = {
        "cover": _codec_pairs(codecs["cover"], encode_cover_km, decode_cover_km),
        "exhum": _codec_pairs(codecs["exhum"], encode_exhum_km, decode_exhum_km),
        "unroof": _codec_pairs(codecs["unroof"], encode_unroof_km, decode_unroof_km),
        "flood": _codec_pairs(codecs["flood"], encode_flood_myr, decode_flood_myr),
        "arc": _codec_pairs(codecs["arc"], encode_arc_myr, decode_arc_myr),
        "age": _codec_pairs(codecs["age"], encode_age, decode_age),
        "floodAge": [
            [
                encode_flood_age(last, time, ever),
                None
                if math.isinf(decode_flood_age(encode_flood_age(last, time, ever)))
                else decode_flood_age(encode_flood_age(last, time, ever)),
            ]
            for last, time, ever in codecs["floodAge"]
        ],
        "facies": [],
        "fields": GEO_FRAME_FIELDS,
        "reads": GEOLOGY_READS,
    }
    for values in codecs["facies"]:
        packed = encode_facies(*values)
        expected["facies"].append([packed, decode_facies(packed)])
    assert javascript["codecs"]["fields"] == expected["fields"]
    assert javascript["codecs"]["reads"] == expected["reads"]
    for key in (
        "cover",
        "exhum",
        "unroof",
        "flood",
        "arc",
        "age",
        "floodAge",
        "facies",
    ):
        actual = javascript["codecs"][key]
        wanted = expected[key]
        assert [x[0] for x in actual] == [x[0] for x in wanted]
        if key == "floodAge":
            assert [x[1] is None for x in actual] == [
                x[1] is None for x in wanted
            ]
            actual_finite = [x[1] for x in actual if x[1] is not None]
            wanted_finite = [x[1] for x in wanted if x[1] is not None]
            _assert_close(actual_finite, wanted_finite, atol=1e-12)
        else:
            _assert_close(
                [x[1] for x in actual],
                [x[1] for x in wanted],
                atol=1e-12,
            )
    _assert_close(
        [
            javascript["codecs"]["decodeFacies"][i][field]
            for i in range(len(codecs["decodeFacies"]))
            for field in ("carb", "salt", "coal", "organic")
        ],
        [
            decode_facies(value)[field]
            for value in codecs["decodeFacies"]
            for field in ("carb", "salt", "coal", "organic")
        ],
        atol=1e-12,
    )


def test_geology_cover_thickening_matches_upstream(javascript, payload):
    for scenario, actual in zip(payload["covers"], javascript["covers"], strict=True):
        crust = SimpleNamespace(
            coverKm=np.asarray(scenario["coverKm"], dtype=np.float32),
            coverMaxKm=np.asarray(scenario["coverMaxKm"], dtype=np.float32),
            coverMaxAt=np.asarray(scenario["coverMaxAt"], dtype=np.float32),
            unroofKm=np.asarray(scenario["unroofKm"], dtype=np.float32),
        )
        thicken_cover(
            crust, 0, scenario["thickenedKm"], scenario["time"], scenario["geology"]
        )
        expected = [
            crust.coverKm[0],
            crust.coverMaxKm[0],
            crust.coverMaxAt[0],
            crust.unroofKm[0],
        ]
        _assert_close(
            [
                actual["coverKm"],
                actual["coverMaxKm"],
                actual["coverMaxAt"],
                actual["unroofKm"],
            ],
            expected,
            atol=2e-6,
        )


def test_fracture_zone_coordinates_match_upstream(javascript, payload):
    expected = [
        fracture_coordinate(*case["point"], case["omega"])
        for case in payload["segments"]
    ]
    _assert_close(javascript["segments"], expected, atol=1e-12)


def test_belt_profiles_and_relief_match_upstream(javascript, payload):
    expected_profiles = [
        broad_profile(
            case["type"],
            case["side"],
            case["distance"],
            case["reach"],
            case["jitter"],
            case["wear"],
        )
        for case in payload["profiles"]
    ]
    expected_reliefs = [
        belt_relief_km(case["type"], case["reach"], payload["heights"])
        for case in payload["reliefs"]
    ]
    _assert_close(javascript["profiles"], expected_profiles, atol=2e-12)
    _assert_close(javascript["reliefs"], expected_reliefs, atol=2e-12)


def test_belt_height_accumulation_matches_upstream(javascript, payload):
    for case, actual in zip(
        payload["heightCases"], javascript["heightStates"], strict=True
    ):
        crust = SimpleNamespace(
            type=np.asarray([case["crustType"]], dtype=np.uint8),
            rootKm=np.asarray([case["rootKm"]], dtype=np.float32),
            upliftKm=np.asarray([case["upliftKm"]], dtype=np.float32),
            rugged=np.asarray([case["rugged"]], dtype=np.float32),
        )
        build_heights(
            crust,
            0,
            case["type"],
            case["side"],
            case["u"],
            case["reach"],
            case["work"],
            case["thickened"],
            {"heights": payload["heights"]},
            case["amp"],
        )
        _assert_close(
            actual,
            [crust.rootKm[0], crust.upliftKm[0], crust.rugged[0]],
            atol=2e-6,
        )


def test_force_balance_linear_solver_matches_upstream(javascript, payload):
    expected = [
        solve3(case["matrix"], case["vector"]) for case in payload["linearSystems"]
    ]
    _assert_close(javascript["systems"], expected, atol=1e-12)


def test_fastest_plate_speed_and_adaptive_step_match_upstream(javascript, payload):
    speed_cases = payload["speedCases"]
    expected_speeds = [
        fastest_speed(
            case["omega"],
            SimpleNamespace(capCos=case["capCos"]),
            case["centre"],
        )
        for case in speed_cases
    ]
    _assert_close(javascript["speeds"], expected_speeds, atol=1e-12)

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
