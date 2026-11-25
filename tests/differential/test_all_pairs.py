"""Run every pair registered in the generic differential configuration."""

from __future__ import annotations

import pytest

from _differential import assert_js_python_equal, run_js_reference

from .config import PAIRS


RUNNER = "tests/js/generic-differential-runner.mjs"


@pytest.mark.parametrize(
    ("pair", "case"),
    [
        pytest.param(pair, case, id=f"{pair.javascript_id}::{case.id}")
        for pair in PAIRS
        for case in pair.cases
    ],
)
def test_python_matches_pinned_javascript(pair, case):
    python_result = pair.python_call(case.input)
    javascript_result = run_js_reference(
        RUNNER,
        {
            "function": pair.javascript_id,
            "input": case.input,
        },
    )

    assert_js_python_equal(
        python_result,
        javascript_result,
        rtol=pair.rtol,
        atol=pair.atol,
    )
