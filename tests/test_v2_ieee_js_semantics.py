"""IEEE helpers match JavaScript `/`, `Math.exp`, `Math.log` and `Math.max`.

The expected results come from ``tests/reference/v2/config/ieee_dump.mjs`` (Node).
They keep the coastUnionSoftnessKm = 0 path faithful to upstream without exceptions.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from orogen.v2.tectonics.initial_landmass import (
    _ieee_div,
    _ieee_exp,
    _ieee_log,
    _js_max,
)

REF = Path(__file__).parent / "reference" / "v2" / "config"
CASES = json.loads((REF / "js-ieee-cases.json").read_text(encoding="utf-8"))
_NAMES = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}


def _value(text: str) -> float:
    return _NAMES[text] if text in _NAMES else float(text)


def _same(actual: float, expected: str) -> bool:
    if expected == "NaN":
        return math.isnan(actual)
    if expected in ("Infinity", "-Infinity"):
        return actual == _NAMES[expected]
    if actual == 0.0:
        # The sign of zero matters (1/-0 is -Infinity): compare it explicitly.
        return math.copysign(1.0, actual) == (-1.0 if expected == "-0" else 1.0)
    return actual == float(expected)


@pytest.mark.parametrize("case", CASES["div"], ids=lambda c: f"{c['a']}/{c['b']}")
def test_division_matches_javascript(case):
    assert _same(_ieee_div(_value(case["a"]), _value(case["b"])), case["r"])


@pytest.mark.parametrize("case", CASES["exp"], ids=lambda c: f"exp({c['x']})")
def test_exp_matches_javascript(case):
    assert _same(_ieee_exp(_value(case["x"])), case["r"])


@pytest.mark.parametrize("case", CASES["log"], ids=lambda c: f"log({c['x']})")
def test_log_matches_javascript(case):
    assert _same(_ieee_log(_value(case["x"])), case["r"])


@pytest.mark.parametrize("case", CASES["max"], ids=lambda c: f"max({c['a']},{c['b']})")
def test_max_matches_javascript(case):
    assert _same(_js_max(_value(case["a"]), _value(case["b"])), case["r"])


def test_fixture_is_current_with_node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, str(REF / "ieee_dump.mjs")],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert json.loads(result.stdout) == CASES
