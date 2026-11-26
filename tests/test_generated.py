"""Configurable and auto-generated property-based comparisons against the upstream JavaScript code."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import re
import shutil
import subprocess
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from orogen.v2.core import rotation as core_rotation
from orogen.v2.core import seed as core_seed

hypothesis = pytest.importorskip(
    "hypothesis",
    reason="install Hypothesis with 'python -m pip install hypothesis'",
)
given = hypothesis.given
seed = hypothesis.seed
settings = hypothesis.settings
st = hypothesis.strategies

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "tests" / "config.json"
JS_CORE_DIR = ROOT / "tests" / "reference" / "v2" / "core"
JS_RUNNER = ROOT / "tests" / "reference" / "v2" / "property-runner.mjs"
JS_BRIDGE_MODULE = ROOT / "tests" / "jspybridge-loader.cjs"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SETTINGS = CONFIG["settings"]

# Modules are a static allowlist. Config cannot import arbitrary code, and every
# module listed here must have a same-named re-export under tests/reference/v2/core.
TARGET_MODULES = {
    "seed": core_seed,
    "rotation": core_rotation,
}
DEFAULT_TARGET_MODULE = "seed"

# Strategies are the only hand-written part of the input domains. A parameter
# gets one of these from its name (config "param_strategies") or, failing that,
# from its annotation (_ANNOTATION_STRATEGIES).
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
    ),
    "unit_float": lambda: st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    "angle": lambda: st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    "time_step": lambda: st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    "quaternion": lambda: st.lists(
        st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
    ),
    "vec3": lambda: st.lists(
        st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
}

_ANNOTATION_STRATEGIES = {
    "float": "unit_float",
    "object": "seed_text",
}


def _js_name(snake: str) -> str:
    """Map a snake_case Python name to the camelCase JavaScript export name."""
    return re.sub(r"_([a-z0-9])", lambda match: match.group(1).upper(), snake)


def _js_export_names(module_id: str) -> set[str]:
    """Read the export names of a JavaScript module without importing it in Python."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required to discover JavaScript exports")
    script = (
        f"import({json.dumps('./' + module_id + '.mjs')})"
        ".then(m => console.log(JSON.stringify(Object.keys(m))))"
    )
    completed = subprocess.run(
        [node, "-e", script],
        cwd=JS_CORE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return set(json.loads(completed.stdout))


def _strategies_for(function: Any, overrides: dict[str, str]) -> list[str] | None:
    """Return one strategy id per required parameter, or None if one cannot be inferred."""
    strategies: list[str] = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return None
        if parameter.name in overrides:
            strategies.append(overrides[parameter.name])
            continue
        annotation = parameter.annotation
        if not isinstance(annotation, str):
            annotation = getattr(annotation, "__name__", str(annotation))
        if annotation not in _ANNOTATION_STRATEGIES:
            return None
        strategies.append(_ANNOTATION_STRATEGIES[annotation])
    return strategies


def _generated_targets() -> list[dict[str, Any]]:
    """Generate one comparison per public function that exists on both sides.

    A function is included when its module is listed under config "generate",
    it is not excluded, a same-named camelCase export exists in the pinned
    JavaScript module, and every required parameter has a strategy.
    """
    spec = CONFIG.get("generate", {})
    excluded = set(spec.get("exclude", []))
    overrides = spec.get("param_strategies", {})
    comparison = spec.get("comparison", {"mode": "approx"})
    generated: list[dict[str, Any]] = []
    for module_id in spec.get("modules", []):
        if module_id not in TARGET_MODULES:
            raise AssertionError(f"Unknown differential module: {module_id}")
        module = TARGET_MODULES[module_id]
        js_names = _js_export_names(module_id)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_") or name in excluded:
                continue
            if function.__module__ != module.__name__:
                continue
            if _js_name(name) not in js_names:
                continue
            strategies = _strategies_for(function, overrides)
            if strategies is None:
                continue
            generated.append(
                {
                    "id": name,
                    "module": module_id,
                    "strategies": strategies,
                    "comparison": comparison,
                    "generated": True,
                }
            )
    return generated


def _target_module(target_config: dict[str, Any]) -> str:
    module_id = target_config.get("module", DEFAULT_TARGET_MODULE)
    if module_id not in TARGET_MODULES:
        raise AssertionError(f"Unknown differential module: {module_id}")
    return module_id


def _target_strategies(target_config: dict[str, Any]) -> list[str]:
    if "strategies" in target_config:
        return list(target_config["strategies"])
    return [target_config["strategy"]]


def _target_label(target_config: dict[str, Any]) -> str:
    prefix = "generated" if target_config.get("generated") else "configured"
    return f"{prefix}:{_target_module(target_config)}.{target_config['id']}"


def _resolve_python_target(target_config: dict[str, Any]) -> Any:
    name = target_config["id"]
    function = getattr(TARGET_MODULES[_target_module(target_config)], name, None)
    if name.startswith("_") or not callable(function):
        raise AssertionError(f"Differential target is not a public function: {name}")
    return function


def _normalise(value: Any) -> Any:
    """Make non-finite numbers match the JavaScript runner's JSON encoding."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        label = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return {"__number__": label}
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
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

    def call(self, module: str, target: str, args: list[Any]) -> Any:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("JavaScript runner streams are unavailable")
        request = json.dumps(
            {"module": module, "target": target, "args": args},
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
    """Call the upstream ES modules through JSPyBridge's persistent Node process."""

    def __init__(self) -> None:
        try:
            import javascript
        except ImportError as exc:
            raise RuntimeError(
                "JSPyBridge is optional; install it with "
                "'python -m pip install javascript==1!1.2.6'"
            ) from exc

        self._bridge = javascript
        self._loader = javascript.require("./jspybridge-loader.cjs", __file__)

    def call(self, module: str, target: str, args: list[Any]) -> Any:
        # The loader returns JSON text so results cross the bridge as plain strings,
        # exactly as they do through the Node runner.
        text = self._loader.invoke(
            module, target, json.dumps(args, ensure_ascii=False, allow_nan=False)
        )
        return json.loads(text)

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
    for backend_id in SETTINGS.get("backends", ["node"])
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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _assert_node(
    actual: Any, expected: Any, mode: str, comparison: dict[str, Any], path: str
) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected an object"
        assert set(actual) == set(expected), f"{path}: keys differ"
        for key in expected:
            _assert_node(actual[key], expected[key], mode, comparison, f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list"
        assert len(actual) == len(expected), f"{path}: lengths differ"
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_node(left, right, mode, comparison, f"{path}[{index}]")
        return
    if mode == "approx" and _is_number(actual) and _is_number(expected):
        assert math.isclose(
            float(actual),
            float(expected),
            rel_tol=float(comparison.get("rel_tol", 1e-9)),
            abs_tol=float(comparison.get("abs_tol", 0.0)),
        ), f"{path}: {actual!r} != {expected!r}"
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _assert_equivalent(actual: Any, expected: Any, comparison: dict[str, Any]) -> None:
    mode = comparison.get("mode", "exact")
    if mode not in ("exact", "approx"):
        raise ValueError(f"Unsupported comparison mode: {mode}")
    _assert_node(actual, expected, mode, comparison, "result")


CONFIGURED_TARGETS = list(CONFIG["targets"])
GENERATED_TARGETS = _generated_targets()
ALL_TARGETS = CONFIGURED_TARGETS + GENERATED_TARGETS


@pytest.mark.parametrize("target_config", ALL_TARGETS, ids=_target_label)
@pytest.mark.parametrize("javascript", _BACKEND_PARAMETERS, indirect=True)
@settings(
    max_examples=SETTINGS["max_examples"],
    deadline=timedelta(milliseconds=SETTINGS["deadline_ms"]),
    derandomize=SETTINGS.get("seed") is None,
)
@given(data=st.data())
def test_generated_differential_cases(
    target_config: dict[str, Any],
    data: Any,
    javascript: Any,
) -> None:
    target_id = target_config["id"]
    module_id = _target_module(target_config)
    strategy_ids = _target_strategies(target_config)
    for strategy_id in strategy_ids:
        assert strategy_id in STRATEGIES, f"Unknown Hypothesis strategy: {strategy_id}"

    python_target = _resolve_python_target(target_config)
    args = [
        data.draw(STRATEGIES[strategy_id](), label=strategy_id)
        for strategy_id in strategy_ids
    ]
    python_result = _normalise(python_target(*args))
    javascript_result = _normalise(javascript.call(module_id, target_id, args))
    _assert_equivalent(
        python_result,
        javascript_result,
        target_config.get("comparison", {}),
    )


if SETTINGS.get("seed") is not None:
    test_generated_differential_cases = seed(SETTINGS["seed"])(
        test_generated_differential_cases
    )
