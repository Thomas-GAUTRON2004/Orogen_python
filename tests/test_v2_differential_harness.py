"""Canonical example of a behavioral test using the shared differential harness."""

from __future__ import annotations

from pathlib import Path

from orogen.v2.core.seed import is_word_seed, make_rng, seed_of

from _differential import assert_js_python_equal, run_js_reference

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/reference/v2/differential-runner.mjs"


def test_seed_and_rng_behavior_matches_upstream_javascript():
    texts = ["", "0", "42", "-17", "3.14159", "0x10", "hello", "Orogen 3.0.1", "é"]
    rng = make_rng(123456)
    python_result = {
        "seeds": [
            {"text": text, "word": is_word_seed(text), "seed": seed_of(text)}
            for text in texts
        ],
        "random": [rng() for _ in range(12)],
    }

    javascript_result = run_js_reference(RUNNER, {})
    assert_js_python_equal(
        python_result,
        {"seeds": javascript_result["seeds"], "random": javascript_result["random"]},
        rtol=0.0,
        atol=0.0,
    )
