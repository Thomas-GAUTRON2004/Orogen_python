"""Canonical example of a behavioral test using the upstream submodule."""

from __future__ import annotations

from pathlib import Path

from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.core.seed import is_word_seed, make_rng, seed_of

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/seed-differential-runner.mjs"


def test_seed_and_rng_behavior_matches_upstream_javascript():
    payload = {
        "texts": ["", "0", "42", "-17", "3.14159", "0x10", "hello", "Orogen 3.0.1", "é"],
        "rngSeed": 123456,
        "randomCount": 12,
    }
    rng = make_rng(payload["rngSeed"])
    python_result = {
        "seeds": [
            {"text": text, "word": is_word_seed(text), "seed": seed_of(text)}
            for text in payload["texts"]
        ],
        "random": [rng() for _ in range(payload["randomCount"])],
    }

    javascript_result = run_js_reference(RUNNER, payload)
    assert_js_python_equal(
        python_result,
        javascript_result,
        rtol=0.0,
        atol=0.0,
    )
