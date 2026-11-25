"""Differential checks for back-arc rollback rates and strip steering."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.backarc import rollback_cm_yr, steer_back_arc_strips

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/backarc-differential-runner.mjs"


def _payload() -> dict[str, Any]:
    return {
        "rollback": {
            "config": {
                "rollbackMinAgeMyr": 10.0,
                "rollbackFullAgeMyr": 30.0,
                "rollbackMaxCmYr": 6.0,
            },
            "ages": [0.0, 9.999, 10.0, 10.0001, 11.0, 15.0, 19.999, 20.0, 29.999, 30.0, 30.001, 50.0, 1_000_000.0],
        },
        "steering": {
            "time": 10.0,
            "activeMyr": 8.0,
            "plates": [
                {
                    "alive": True,
                    "backArcOf": -1,
                    "backArcPushDir": None,
                    "born": 0.0,
                    "rollbackCmYr": 0.0,
                    "capCenter": [1.0, 0.0, 0.0],
                    "omega": [0.01, 0.02, 0.03],
                },
                {
                    "alive": True,
                    "backArcOf": 0,
                    "backArcPushDir": [0.0, 1.0, 0.0],
                    "born": 5.0,
                    "rollbackCmYr": 3.0,
                    "capCenter": [1.0, 0.0, 0.0],
                    "omega": [0.0, 0.0, 0.0],
                },
                {
                    "alive": True,
                    "backArcOf": 0,
                    "backArcPushDir": [0.0, 1.0, 0.0],
                    "born": 0.0,
                    "rollbackCmYr": 4.0,
                    "capCenter": [0.6, 0.8, 0.0],
                    "omega": [1.0, 2.0, 3.0],
                },
                {
                    "alive": True,
                    "backArcOf": 4,
                    "backArcPushDir": [0.0, 1.0, 0.0],
                    "born": 5.0,
                    "rollbackCmYr": 2.0,
                    "capCenter": [0.0, 0.6, 0.8],
                    "omega": [4.0, 5.0, 6.0],
                },
                {
                    "alive": False,
                    "backArcOf": -1,
                    "backArcPushDir": None,
                    "born": 0.0,
                    "rollbackCmYr": 0.0,
                    "capCenter": [1.0, 0.0, 0.0],
                    "omega": [0.0, 0.0, 0.0],
                },
                {
                    "alive": True,
                    "backArcOf": 0,
                    "backArcPushDir": None,
                    "born": 5.0,
                    "rollbackCmYr": 2.0,
                    "capCenter": [1.0, 0.0, 0.0],
                    "omega": [7.0, 8.0, 9.0],
                },
                {
                    "alive": True,
                    "backArcOf": 999,
                    "backArcPushDir": [0.0, 1.0, 0.0],
                    "born": 5.0,
                    "rollbackCmYr": 2.0,
                    "capCenter": [0.0, 0.0, 1.0],
                    "omega": [10.0, 11.0, 12.0],
                },
            ],
        },
    }


def test_back_arc_rollback_rate_matches_upstream_javascript():
    payload = _payload()
    config = payload["rollback"]["config"]
    python_result = [
        rollback_cm_yr(age, config) for age in payload["rollback"]["ages"]
    ]

    javascript = run_js_reference(RUNNER, payload)

    assert_js_python_equal(python_result, javascript["rollback"], atol=0.0)


def test_back_arc_strip_steering_matches_upstream_javascript():
    payload = _payload()
    plates = []
    for spec in payload["steering"]["plates"]:
        plate = SimpleNamespace(
            alive=spec["alive"],
            backArcOf=spec["backArcOf"],
            backArcPushDir=spec["backArcPushDir"],
            born=spec["born"],
            rollbackCmYr=spec["rollbackCmYr"],
            capCenter=spec["capCenter"],
            omega=list(spec["omega"]),
            to_world=lambda x, y, z: np.asarray([x, y, z], dtype=np.float64),
        )
        plates.append(plate)

    world = SimpleNamespace(
        plates=plates,
        config={"backArcActiveMyr": payload["steering"]["activeMyr"]},
        time=payload["steering"]["time"],
    )
    steer_back_arc_strips(world)
    python_result = [plate.omega for plate in plates]

    javascript = run_js_reference(RUNNER, payload)

    assert_js_python_equal(python_result, javascript["steering"])



def test_back_arc_rollback_rate_matches_upstream_with_zero_maximum():
    payload = _payload()
    payload["rollback"]["config"]["rollbackMaxCmYr"] = 0.0
    config = payload["rollback"]["config"]
    python_result = [
        rollback_cm_yr(age, config) for age in payload["rollback"]["ages"]
    ]

    javascript = run_js_reference(RUNNER, payload)

    assert_js_python_equal(python_result, javascript["rollback"], atol=0.0)


def test_back_arc_strip_steering_stops_at_the_active_window_boundary():
    payload = _payload()
    payload["steering"]["time"] = 13.0
    plates = []
    for spec in payload["steering"]["plates"]:
        plates.append(
            SimpleNamespace(
                alive=spec["alive"],
                backArcOf=spec["backArcOf"],
                backArcPushDir=spec["backArcPushDir"],
                born=spec["born"],
                rollbackCmYr=spec["rollbackCmYr"],
                capCenter=spec["capCenter"],
                omega=list(spec["omega"]),
                to_world=lambda x, y, z: np.asarray(
                    [x, y, z], dtype=np.float64
                ),
            )
        )

    world = SimpleNamespace(
        plates=plates,
        config={"backArcActiveMyr": payload["steering"]["activeMyr"]},
        time=payload["steering"]["time"],
    )
    steer_back_arc_strips(world)
    python_result = [plate.omega for plate in plates]

    javascript = run_js_reference(RUNNER, payload)

    assert_js_python_equal(python_result, javascript["steering"])
