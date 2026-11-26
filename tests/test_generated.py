"""Configurable property-based comparisons against the upstream JavaScript code."""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from orogen.v2.core.seed import is_word_seed, seed_of

hypothesis = pytest.importorskip(
    "hypothesis",
    reason="install Hypothesis with 'python -m pip install hypothesis'",
)
given = hypothesis.given
seed = hypothesis.seed
settings = hypothesis.settings
st = hypothesis.strategies

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "tests" / "differential" / "config.json"
JS_RUNNER = ROOT / "tests" / "reference" / "v2" / "property-runner.mjs"
JS_BRIDGE_MODULE = ROOT / "tests" / "differential" / "jspybridge-loader.cjs"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

# The registry is deliberately static: config files select known callables and
# strategies; they cannot import arbitrary modules or execute user-supplied code.
TARGETS = {
    "seed_of": seed_of,
    "is_word_seed": is_word_seed,
}

STRATEGIES = {
    "seed_text": lambda: st.one_of(
        st.text(
            alphabet=st.characters(blacklist_categories=("Cc", "Cf", "Cs")),
            max_size=256,
        ),
        st.sampled_from(
            [
                "",
                "0",
                "-0",
                "+0",
                "42",
                "-17",
                "3.14159",
                "1e308",
                "Infinity",
                "-Infinity",
                "NaN",
                "0x10",
                "0b101",
                "0o17",
                "  42  ",
                "hello",
                "Orogen 3.0.1",
                "é",
                "東京",
            ]
        ),
    )
}


def _normalise(value: Any) -> Any:
    """Make Python non-finite numbers match the JS runner's JSON encoding."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        label = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return {"__number__": label}
    return value


class JavaScriptSession:
    """Keep one Node process alive while Hypothesis submits generated examples."""

    def __init__(self) -> None:
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Node.js is required for generated differential tests")
        self.process = subprocess.Popen(
            [node, str(JS_RUNNER)],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def call(self, target: str, value: str) -> Any:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("JavaScript runner streams are unavailable")
        request = json.dumps(
            {"target": target, "value": value},
            ensure_ascii=False,
            allow_nan=False,
        )
        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(f"JavaScript runner exited unexpectedly: {stderr}")
        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def close(self) -> None:
        if self.process.stdin is not None and self.process.poll() is None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class JSPyBridgeSession:
    """Call the upstream ES module through JSPyBridge's persistent Node process."""

    def __init__(self) -> None:
        try:
            import javascript
        except ImportError as exc:
            raise RuntimeError(
                "JSPyBridge is optional; install it with "
                "'python -m pip install javascript==1!1.2.6'"
            ) from exc

        self._bridge = javascript
        module = javascript.require(str(JS_BRIDGE_MODULE))
        self._targets = {
            "seed_of": module.seedOf,
            "is_word_seed": module.isWordSeed,
        }

    def call(self, target: str, value: str) -> Any:
        function = self._targets.get(target)
        if function is None:
            raise RuntimeError(f"Unknown JSPyBridge target: {target}")
        return function(value)

    def close(self) -> None:
        self._bridge.terminate()


_BACKEND_PARAMETERS = [
    pytest.param(
        backend_id,
        marks=(
            pytest.mark.skipif(
                importlib.util.find_spec("javascript") is None,
                reason="install JSPyBridge with 'python -m pip install javascript==1!1.2.6'",
            )
            if backend_id == "jspybridge"
            else ()
        ),
        id=backend_id,
    )
    for backend_id in CONFIG["settings"].get("backends", ["node"])
]


@pytest.fixture(scope="module")
def javascript(request: pytest.FixtureRequest) -> Iterator[Any]:
    """Create one persistent JavaScript runtime for a complete Hypothesis run."""
    backend_id = request.param
    if backend_id == "node":
        session = JavaScriptSession()
    elif backend_id == "jspybridge":
        session = JSPyBridgeSession()
    else:
        raise ValueError(f"Unsupported JavaScript backend: {backend_id}")
    yield session
    session.close()


def _assert_equivalent(actual: Any, expected: Any, comparison: dict[str, Any]) -> None:
    mode = comparison.get("mode", "exact")
    if mode == "exact":
        assert actual == expected
        return
    if mode == "approx":
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool)
        assert isinstance(expected, (int, float)) and not isinstance(expected, bool)
        assert math.isclose(
            float(actual),
            float(expected),
            rel_tol=float(comparison.get("rel_tol", 1e-9)),
            abs_tol=float(comparison.get("abs_tol", 0.0)),
        )
        return
    raise ValueError(f"Unsupported comparison mode: {mode}")


@pytest.mark.parametrize(
    "target_config", CONFIG["targets"], ids=lambda item: item["id"]
)
@pytest.mark.parametrize("javascript", _BACKEND_PARAMETERS, indirect=True)
@settings(
    max_examples=CONFIG["settings"]["max_examples"],
    deadline=timedelta(milliseconds=CONFIG["settings"]["deadline_ms"]),
    derandomize=CONFIG["settings"].get("seed") is None,
)
@given(data=st.data())
def test_generated_differential_cases(
    target_config: dict[str, Any],
    data: Any,
    javascript: Any,
) -> None:
    target_id = target_config["id"]
    strategy_id = target_config["strategy"]
    assert target_id in TARGETS, f"Unknown differential target: {target_id}"
    assert strategy_id in STRATEGIES, f"Unknown Hypothesis strategy: {strategy_id}"

    value = data.draw(STRATEGIES[strategy_id](), label=strategy_id)
    python_result = _normalise(TARGETS[target_id](value))
    javascript_result = _normalise(javascript.call(target_id, value))
    _assert_equivalent(
        python_result,
        javascript_result,
        target_config.get("comparison", {}),
    )


if CONFIG["settings"].get("seed") is not None:
    test_generated_differential_cases = seed(CONFIG["settings"]["seed"])(
        test_generated_differential_cases
    )
