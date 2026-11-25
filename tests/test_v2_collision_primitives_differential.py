"""Differential checks for collision bookkeeping and post-rift partner timing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.collision import (
    are_young_rift_partners,
    forget_stale_collisions,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/collision-primitives-differential-runner.mjs"


def _payload() -> dict[str, Any]:
    young_cases = [
        {
            "a": {"id": 0, "partner": 1, "born": 10.0},
            "b": {"id": 1, "partner": 0, "born": 15.0},
            "time": 20.0,
            "window": 10.0,
        },
        {
            "a": {"id": 0, "partner": 1, "born": 10.0},
            "b": {"id": 1, "partner": 0, "born": 15.0},
            "time": 25.0,
            "window": 10.0,
        },
        {
            "a": {"id": 0, "partner": 2, "born": 10.0},
            "b": {"id": 1, "partner": 0, "born": 15.0},
            "time": 20.0,
            "window": 10.0,
        },
        {
            "a": {"id": 0, "partner": 1, "born": 10.0},
            "b": {"id": 1, "partner": -1, "born": 15.0},
            "time": 20.0,
            "window": 10.0,
        },
        {
            "a": {"id": 0, "partner": 1, "born": 15.0},
            "b": {"id": 1, "partner": 0, "born": 10.0},
            "time": 14.0,
            "window": 10.0,
        },
    ]
    stale_case = {
        "time": 100.0,
        "forgetAfter": 50.0,
        "owners": [-1, -1, -1],
        "records": {
            "0:1": {"lastActive": 10.0, "sutured": False, "shorteningKm": 20.0},
            "0:2": {"lastActive": 80.0, "sutured": False, "shorteningKm": 5.0},
            "1:2": {"lastActive": 0.0, "sutured": True, "shorteningKm": 50.0},
            "2:3": {"lastActive": 50.0, "sutured": False, "shorteningKm": 0.0},
        },
    }
    return {"youngCases": young_cases, "staleCase": stale_case}


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> dict[str, Any]:
    return run_js_reference(RUNNER, payload)


def test_post_rift_partner_window_matches_javascript(javascript, payload):
    expected = []
    for case in payload["youngCases"]:
        plate_a = SimpleNamespace(
            id=case["a"]["id"],
            riftPartner=case["a"]["partner"],
            born=case["a"]["born"],
        )
        plate_b = SimpleNamespace(
            id=case["b"]["id"],
            riftPartner=case["b"]["partner"],
            born=case["b"]["born"],
        )
        expected.append(
            are_young_rift_partners(
                plate_a,
                plate_b,
                case["time"],
                {"riftPushMyr": case["window"]},
            )
        )
    assert expected == javascript["young"]


def test_stale_collision_records_match_javascript(javascript, payload):
    spec = payload["staleCase"]
    collisions = {key: dict(value) for key, value in spec["records"].items()}
    world = SimpleNamespace(
        time=spec["time"],
        config={"collisionForgetMyr": spec["forgetAfter"]},
        collisions=collisions,
    )
    forgotten = forget_stale_collisions(world)
    expected = {
        "forgotten": forgotten,
        "remaining": {
            key: collisions[key]
            for key in sorted(collisions)
        },
    }
    assert_js_python_equal(
        {
            "young": javascript["young"],
            "stale": {
                "merged": False,
                "remaining": expected["remaining"],
            },
        },
        {
            "young": javascript["young"],
            "stale": {
                "merged": javascript["stale"]["merged"],
                "remaining": javascript["stale"]["remaining"],
            },
        },
        atol=0,
    )
    assert forgotten == 1
