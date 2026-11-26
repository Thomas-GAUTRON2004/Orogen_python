"""Property-based comparisons against the upstream JavaScript, generated from source.

Tests are discovered at collection time from the Python source directory named in
config.json (generate.source). A public function gets a test when a JavaScript
module of the same name exists under tests/reference/v2/core and exports the
camelCase name. Comments above a function can refine the generated test:

    # differential: js=quatMul strategies=quaternion,quaternion mode=approx
    # differential: skip
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import math
import re
import shutil
import subprocess
import warnings
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

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
GENERATE = CONFIG.get("generate", {})

SOURCE_DIR = ROOT / GENERATE.get("source", "orogen/v2/core")
PACKAGE = ".".join(SOURCE_DIR.relative_to(ROOT).parts)
JS_MODULE_OVERRIDES: dict[str, str] = GENERATE.get("js_modules", {})
DEFAULT_TARGET_MODULE = "seed"

_JS_FILE = re.compile(r"^[A-Za-z0-9_-]+\.mjs$")

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


def _js_module_file(stem: str) -> str | None:
    """Return the JavaScript module file for a Python module stem, if one exists."""
    candidates = [
        JS_MODULE_OVERRIDES.get(stem),
        f"{stem}.mjs",
        f"{stem.replace('_', '-')}.mjs",
    ]
    for name in candidates:
        if name and (JS_CORE_DIR / name).is_file():
            return name
    return None


_JS_EXPORTS_CACHE: dict[str, set[str]] = {}


def _js_export_names(js_file: str) -> set[str]:
    """Read the export names of a JavaScript module without importing it in Python."""
    if js_file not in _JS_EXPORTS_CACHE:
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Node.js is required to discover JavaScript exports")
        script = (
            f"import({json.dumps('./' + js_file)})"
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
        _JS_EXPORTS_CACHE[js_file] = set(json.loads(completed.stdout))
    return _JS_EXPORTS_CACHE[js_file]


def _parse_marker(text: str) -> dict[str, Any]:
    """Parse the body of a '# differential:' comment into a dict."""
    marker: dict[str, Any] = {}
    for token in text.split():
        if "=" not in token:
            if token == "skip":
                marker["skip"] = True
                continue
            raise AssertionError(f"Invalid differential marker token: {token!r}")
        key, value = token.split("=", 1)
        if key == "strategies":
            marker[key] = [item for item in value.split(",") if item]
        elif key in ("js", "mode"):
            marker[key] = value
        else:
            raise AssertionError(f"Unknown differential marker key: {key!r}")
    return marker


def _markers(path: Path) -> dict[str, dict[str, Any]]:
    """Read '# differential:' comments placed directly above top-level functions."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    found: dict[str, dict[str, Any]] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        start = min(
            [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
        )
        index = start - 2
        block: list[str] = []
        while index >= 0 and lines[index].strip().startswith("#"):
            block.insert(0, lines[index].strip()[1:].strip())
            index -= 1
        for comment in block:
            if comment.startswith("differential:"):
                found[node.name] = _parse_marker(comment[len("differential:") :])
    return found


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


_PY_MODULES: dict[str, Any] = {}


def _import_python_module(stem: str) -> Any:
    if stem not in _PY_MODULES:
        if not (SOURCE_DIR / f"{stem}.py").is_file():
            raise AssertionError(f"No Python module {stem}.py under {SOURCE_DIR}")
        _PY_MODULES[stem] = importlib.import_module(f"{PACKAGE}.{stem}")
    return _PY_MODULES[stem]


def _discover_stems() -> list[str]:
    """Return the Python module stems selected by generate.modules."""
    available = sorted(
        path.stem for path in SOURCE_DIR.glob("*.py") if not path.stem.startswith("_")
    )
    selected = GENERATE.get("modules", "all")
    if selected == "all":
        return available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise AssertionError(f"generate.modules names unknown modules: {unknown}")
    return list(selected)


def _generated_targets() -> list[dict[str, Any]]:
    """Generate one comparison per eligible public function in the source directory."""
    excluded = set(GENERATE.get("exclude", []))
    overrides = GENERATE.get("param_strategies", {})
    default_comparison = GENERATE.get("comparison", {"mode": "approx"})
    generated: list[dict[str, Any]] = []

    for stem in _discover_stems():
        js_file = _js_module_file(stem)
        if js_file is None:
            continue
        markers = _markers(SOURCE_DIR / f"{stem}.py")
        try:
            module = _import_python_module(stem)
        except ImportError as exc:
            warnings.warn(
                f"Skipping differential tests for {stem}: {exc}", stacklevel=2
            )
            continue
        js_exports = _js_export_names(js_file)

        for name, function in inspect.getmembers(module, inspect.isfunction):
            if (
                name.startswith("_")
                or name in excluded
                or function.__module__ != module.__name__
            ):
                continue
            marker = markers.get(name, {})
            if marker.get("skip"):
                continue
            js_name = marker.get("js", _js_name(name))
            explicit = "js" in marker
            if js_name not in js_exports:
                if explicit:
                    raise AssertionError(
                        f"{stem}.{name} names JavaScript export {js_name!r}, "
                        f"which {js_file} does not export"
                    )
                continue
            strategies = marker.get("strategies") or _strategies_for(
                function, overrides
            )
            if strategies is None:
                continue
            generated.append(
                {
                    "id": name,
                    "module": stem,
                    "js": js_name,
                    "strategies": strategies,
                    "comparison": {
                        "mode": marker.get("mode", default_comparison["mode"]),
                        **{
                            key: value
                            for key, value in default_comparison.items()
                            if key != "mode"
                        },
                    },
                    "generated": True,
                }
            )
    return generated


def _target_module(target_config: dict[str, Any]) -> str:
    return target_config.get("module", DEFAULT_TARGET_MODULE)


def _target_js_name(target_config: dict[str, Any]) -> str:
    return target_config.get("js", _js_name(target_config["id"]))


def _target_strategies(target_config: dict[str, Any]) -> list[str]:
    if "strategies" in target_config:
        return list(target_config["strategies"])
    return [target_config["strategy"]]


def _target_label(target_config: dict[str, Any]) -> str:
    prefix = "generated" if target_config.get("generated") else "configured"
    return f"{prefix}:{_target_module(target_config)}.{target_config['id']}"


def _resolve_python_target(target_config: dict[str, Any]) -> Any:
    name = target_config["id"]
    if name.startswith("_"):
        raise AssertionError(f"Differential target is private: {name}")
    function = getattr(_import_python_module(_target_module(target_config)), name, None)
    if not callable(function):
        raise AssertionError(f"Differential target is not a function: {name}")
    return function


def _js_file_for(target_config: dict[str, Any]) -> str:
    js_file = _js_module_file(_target_module(target_config))
    if js_file is None:
        raise AssertionError(
            f"No JavaScript module for {_target_module(target_config)}"
        )
    return js_file


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

    def call(self, js_file: str, export: str, args: list[Any]) -> Any:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("JavaScript runner streams are unavailable")
        request = json.dumps(
            {"module": js_file, "target": export, "args": args},
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

    def call(self, js_file: str, export: str, args: list[Any]) -> Any:
        # The loader returns JSON text so results cross the bridge as plain
        # strings, exactly as they do through the Node runner.
        text = self._loader.invoke(
            js_file, export, json.dumps(args, ensure_ascii=False, allow_nan=False)
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
GENERATED_TARGETS = _generated_targets() if GENERATE else []
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
    strategy_ids = _target_strategies(target_config)
    for strategy_id in strategy_ids:
        assert strategy_id in STRATEGIES, f"Unknown Hypothesis strategy: {strategy_id}"

    python_target = _resolve_python_target(target_config)
    args = [
        data.draw(STRATEGIES[strategy_id](), label=strategy_id)
        for strategy_id in strategy_ids
    ]
    python_result = _normalise(python_target(*args))
    javascript_result = _normalise(
        javascript.call(
            _js_file_for(target_config), _target_js_name(target_config), args
        )
    )
    _assert_equivalent(
        python_result,
        javascript_result,
        target_config.get("comparison", {}),
    )


if SETTINGS.get("seed") is not None:
    test_generated_differential_cases = seed(SETTINGS["seed"])(
        test_generated_differential_cases
    )
