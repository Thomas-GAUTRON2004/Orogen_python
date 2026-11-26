"""Property-based comparisons against the upstream JavaScript, generated from source.

All configuration lives in comments.

Global settings are the single line below (defaults apply to missing keys):

# differential-settings: source=orogen/v2/core max_examples=250 deadline_ms=1000 seed=20261126 backends=node,jspybridge pynguin_seconds=60 pynguin_seed=1

A source module opts in with a comment line near the top of its file:

# differential-module: js_module=sphere-geom.mjs mode=approx rel_tol=1e-9 abs_tol=1e-9 params=q:quaternion,omega:vec3

Each public function of an opted-in module gets a test when a JavaScript module
of the same name exports its camelCase name. A comment directly above a function
refines that test: js=<export>, strategies=<a>,<b>, mode=<exact|approx>, or skip.
"""

from __future__ import annotations

import ast
import copy
import importlib
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

hypothesis = pytest.importorskip(
    "hypothesis",
    reason="install Hypothesis with 'python -m pip install hypothesis'",
)
given = hypothesis.given
seed = hypothesis.seed
settings = hypothesis.settings
st = hypothesis.strategies

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
JS_CORE_DIR = ROOT / "tests" / "reference" / "v2" / "core"
JS_RUNNER = ROOT / "tests" / "reference" / "v2" / "property-runner.mjs"

_SETTING_DEFAULTS = {
    "source": "orogen/v2/core",
    "max_examples": "250",
    "deadline_ms": "1000",
    "seed": "20261126",
    "backends": "node,jspybridge",
    "pynguin_seconds": "60",
    "pynguin_seed": "1",
}


def _read_settings() -> dict[str, Any]:
    """Read the '# differential-settings:' line of this file."""
    values = dict(_SETTING_DEFAULTS)
    for line in THIS_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("# differential-settings:"):
            for token in line.split(":", 1)[1].split():
                key, _, value = token.partition("=")
                if key not in values:
                    raise AssertionError(f"Unknown differential setting: {key!r}")
                values[key] = value
            break
    return {
        "source": values["source"],
        "max_examples": int(values["max_examples"]),
        "deadline_ms": int(values["deadline_ms"]),
        "seed": None if values["seed"] == "none" else int(values["seed"]),
        "backends": [item for item in values["backends"].split(",") if item],
        "pynguin_seconds": int(values["pynguin_seconds"]),
        "pynguin_seed": int(values["pynguin_seed"]),
    }


SETTINGS = _read_settings()
SOURCE_DIR = ROOT / SETTINGS["source"]
PACKAGE = ".".join(SOURCE_DIR.relative_to(ROOT).parts)

# Strategies are the only hand-written part of the input domains. A parameter
# gets one of these from its name (params= in the module comment) or, failing
# that, from its annotation (_ANNOTATION_STRATEGIES).
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

_MODULE_KEYS = {"js_module", "mode", "rel_tol", "abs_tol", "params"}


def _js_name(snake: str) -> str:
    """Map a snake_case Python name to the camelCase JavaScript export name."""
    return re.sub(r"_([a-z0-9])", lambda match: match.group(1).upper(), snake)


def _js_module_file(stem: str, explicit: str | None) -> str | None:
    """Return the JavaScript module file for a Python module stem, if one exists."""
    if explicit is not None:
        if not (JS_CORE_DIR / explicit).is_file():
            raise AssertionError(f"{stem}: js_module {explicit!r} does not exist")
        return explicit
    for name in (f"{stem}.mjs", f"{stem.replace('_', '-')}.mjs"):
        if (JS_CORE_DIR / name).is_file():
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


def _module_settings(path: Path) -> dict[str, Any] | None:
    """Read the '# differential-module:' comment of a source file, if present."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("# differential-module:"):
            continue
        settings: dict[str, Any] = {}
        for token in stripped.split(":", 1)[1].split():
            key, _, value = token.partition("=")
            if key not in _MODULE_KEYS:
                raise AssertionError(f"{path.name}: unknown module setting {key!r}")
            if key == "params":
                settings[key] = dict(
                    item.split(":", 1) for item in value.split(",") if item
                )
            elif key in ("rel_tol", "abs_tol"):
                settings[key] = float(value)
            else:
                settings[key] = value
        return settings
    return None


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
        _PY_MODULES[stem] = importlib.import_module(f"{PACKAGE}.{stem}")
    return _PY_MODULES[stem]


def _module_targets(path: Path) -> list[dict[str, Any]]:
    """Generate the comparisons for one opted-in source module, or none."""
    module_settings = _module_settings(path)
    if module_settings is None:
        return []
    stem = path.stem
    js_file = _js_module_file(stem, module_settings.get("js_module"))
    if js_file is None:
        raise AssertionError(f"{stem}: opted in but no JavaScript module exists")
    js_exports = _js_export_names(js_file)
    overrides = module_settings.get("params", {})
    comparison: dict[str, Any] = {"mode": module_settings.get("mode", "approx")}
    for key in ("rel_tol", "abs_tol"):
        if key in module_settings:
            comparison[key] = module_settings[key]

    module = _import_python_module(stem)
    markers = _markers(path)
    targets: list[dict[str, Any]] = []
    for name, function in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_") or function.__module__ != module.__name__:
            continue
        marker = markers.get(name, {})
        if marker.get("skip"):
            continue
        js_name = marker.get("js", _js_name(name))
        if js_name not in js_exports:
            if "js" in marker:
                raise AssertionError(
                    f"{stem}.{name} names JavaScript export {js_name!r}, "
                    f"which {js_file} does not export"
                )
            continue
        strategies = marker.get("strategies") or _strategies_for(function, overrides)
        if strategies is None:
            continue
        function_comparison = dict(comparison)
        if "mode" in marker:
            function_comparison["mode"] = marker["mode"]
        targets.append(
            {
                "id": name,
                "module": stem,
                "js": js_name,
                "js_file": js_file,
                "strategies": strategies,
                "comparison": function_comparison,
            }
        )
    return targets


def _generated_targets() -> list[dict[str, Any]]:
    """Generate one comparison per public function of each opted-in source module."""
    targets: list[dict[str, Any]] = []
    for path in sorted(SOURCE_DIR.glob("*.py")):
        if not path.stem.startswith("_"):
            targets.extend(_module_targets(path))
    return targets


def _target_label(target_config: dict[str, Any]) -> str:
    return f"{target_config['module']}.{target_config['id']}"


def _resolve_python_target(target_config: dict[str, Any]) -> Any:
    name = target_config["id"]
    if name.startswith("_"):
        raise AssertionError(f"Differential target is private: {name}")
    function = getattr(_import_python_module(target_config["module"]), name, None)
    if not callable(function):
        raise AssertionError(f"Differential target is not a function: {name}")
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


class JSTargetError(Exception):
    """The JavaScript target itself raised; the runner is still healthy."""


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
            raise JSTargetError(response["error"])
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
        try:
            text = self._loader.invoke(
                js_file, export, json.dumps(args, ensure_ascii=False, allow_nan=False)
            )
        except Exception as exc:
            raise JSTargetError(str(exc)) from exc
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
    for backend_id in SETTINGS["backends"]
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


ALL_TARGETS = _generated_targets()


@pytest.mark.parametrize("target_config", ALL_TARGETS, ids=_target_label)
@pytest.mark.parametrize("javascript", _BACKEND_PARAMETERS, indirect=True)
@settings(
    max_examples=SETTINGS["max_examples"],
    deadline=timedelta(milliseconds=SETTINGS["deadline_ms"]),
    derandomize=SETTINGS["seed"] is None,
)
@given(data=st.data())
def test_generated_differential_cases(
    target_config: dict[str, Any],
    data: Any,
    javascript: Any,
) -> None:
    for strategy_id in target_config["strategies"]:
        assert strategy_id in STRATEGIES, f"Unknown Hypothesis strategy: {strategy_id}"

    python_target = _resolve_python_target(target_config)
    args = [
        data.draw(STRATEGIES[strategy_id](), label=strategy_id)
        for strategy_id in target_config["strategies"]
    ]
    python_result = _normalise(python_target(*args))
    javascript_result = _normalise(
        javascript.call(target_config["js_file"], target_config["js"], args)
    )
    _assert_equivalent(python_result, javascript_result, target_config["comparison"])


# ---------------------------------------------------------------------------
# Pynguin corpus replay. tests/pynguin_corpus.py records the arguments Pynguin
# passes to each function; the replay compares them with the JavaScript
# reference. Pynguin's own assertions are not used. Entries whose types do not
# fit the target's strategies are reported as skipped, with the reason.
# ---------------------------------------------------------------------------

CORPUS_DIR = ROOT / "tests" / "corpus"
_CORPUS_DOCUMENTS: dict[str, dict[str, Any] | None] = {}


class _Opaque:
    """A value Pynguin produced that the JSON corpus could only record as text."""

    def __init__(self, text: str) -> None:
        self.text = text


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if "__number__" in value:
            return float(value["__number__"])
        if "__bytes__" in value:
            return bytes.fromhex(value["__bytes__"])
        if "__repr__" in value:
            return _Opaque(value["__repr__"])
        return {key: _decode(item) for key, item in value.items()}
    return value


def _is_list_of_numbers(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_is_number(item) for item in value)
    )


# Type checks only: the replay tests what Python and JavaScript do with inputs
# of the right type, including values outside the Hypothesis domains.
_TYPE_CHECKS = {
    "unit_float": _is_number,
    "angle": _is_number,
    "time_step": _is_number,
    "quaternion": lambda value: _is_list_of_numbers(value, 4),
    "vec3": lambda value: _is_list_of_numbers(value, 3),
    "seed_text": lambda value: isinstance(value, str),
}


def _corpus_document(module: str) -> dict[str, Any] | None:
    if module not in _CORPUS_DOCUMENTS:
        path = CORPUS_DIR / f"{module}.json"
        _CORPUS_DOCUMENTS[module] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
    return _CORPUS_DOCUMENTS[module]


def _has_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_has_non_finite(item) for item in value)
    return False


def _skip_reason(target: dict[str, Any], args: list[Any]) -> str | None:
    if len(args) != len(target["strategies"]):
        return f"arity {len(args)} != {len(target['strategies'])}"
    for strategy, value in zip(target["strategies"], args, strict=True):
        if isinstance(value, _Opaque) or not _TYPE_CHECKS[strategy](value):
            return f"type mismatch for {strategy}"
    if _has_non_finite(args):
        return "non-finite input cannot be sent on the JSON wire"
    return None


def _corpus_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for target in ALL_TARGETS:
        document = _corpus_document(target["module"])
        if document is None:
            continue
        for index, entry in enumerate(document["entries"].get(target["id"], [])):
            args = _decode(entry["args"])
            cases.append(
                {
                    "target": target,
                    "args": args,
                    "index": index,
                    "skip": _skip_reason(target, args),
                }
            )
    return cases


def _corpus_label(case: dict[str, Any]) -> str:
    target = case["target"]
    return f"{target['module']}.{target['id']}[{case['index']}]"


CORPUS_CASES = _corpus_cases()


@pytest.mark.parametrize("corpus_case", CORPUS_CASES, ids=_corpus_label)
@pytest.mark.parametrize("javascript", _BACKEND_PARAMETERS, indirect=True)
def test_corpus_replay(corpus_case: dict[str, Any], javascript: Any) -> None:
    if corpus_case["skip"] is not None:
        pytest.skip(corpus_case["skip"])
    target = corpus_case["target"]
    args = copy.deepcopy(corpus_case["args"])

    python_error: str | None = None
    try:
        python_result = _resolve_python_target(target)(*args)
    except Exception as exc:
        python_error = type(exc).__name__

    javascript_error: str | None = None
    try:
        javascript_result = javascript.call(target["js_file"], target["js"], args)
    except JSTargetError as exc:
        javascript_error = str(exc)

    if python_error is not None or javascript_error is not None:
        assert python_error is not None and javascript_error is not None, (
            f"Python raised {python_error!r} but JavaScript raised {javascript_error!r}"
        )
        return
    _assert_equivalent(
        _normalise(python_result), _normalise(javascript_result), target["comparison"]
    )


if SETTINGS["seed"] is not None:
    test_generated_differential_cases = seed(SETTINGS["seed"])(
        test_generated_differential_cases
    )
