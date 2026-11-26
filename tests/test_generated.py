"""Replay of Pynguin-generated calls against the upstream JavaScript reference.

The test suite only replays corpora written by tests/pynguin_corpus.py. Each
recorded call is run through the Python function and the JavaScript export, and
the results are compared. Pynguin's own assertions are not used.

Configuration lives in comments.

Global settings are the single line below (defaults apply to missing keys):

# differential-settings: source=orogen backends=node,jspybridge pynguin_seconds=60 pynguin_seed=1

A source module opts in with a comment line near the top of its file:

# differential-module: js_module=sphere-geom.mjs mode=approx rel_tol=1e-9 abs_tol=1e-9 params=q:quaternion,omega:vec3

Only public functions of an opted-in module with a corpus entry are replayed.
A comment directly above a function refines it: js=<export>, mode=<exact|approx>,
or skip.
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
from pathlib import Path
from typing import Any

import pytest
from differential_support import (
    CORPUS_DIR,
    REFERENCE_DIR,
    ROOT,
    SETTINGS,
    dotted_name,
    js_module_file,
    module_settings,
    opted_in_modules,
)

JS_RUNNER = ROOT / "tests" / "reference" / "v2" / "property-runner.mjs"

# Types a parameter can take. Corpus entries whose arguments do not fit are
# reported as skipped, with the reason. Values outside any range still run.
_TYPE_CHECKS = {
    "unit_float": lambda value: _is_number(value),
    "angle": lambda value: _is_number(value),
    "time_step": lambda value: _is_number(value),
    "quaternion": lambda value: _is_list_of_numbers(value, 4),
    "vec3": lambda value: _is_list_of_numbers(value, 3),
    "seed_text": lambda value: isinstance(value, str),
}
_ANNOTATION_TYPES = {"float": "unit_float", "object": "seed_text"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_list_of_numbers(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_is_number(item) for item in value)
    )


def _js_name(snake: str) -> str:
    """Map a snake_case Python name to the camelCase JavaScript export name."""
    return re.sub(r"_([a-z0-9])", lambda match: match.group(1).upper(), snake)


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
            cwd=REFERENCE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        _JS_EXPORTS_CACHE[js_file] = set(json.loads(completed.stdout))
    return _JS_EXPORTS_CACHE[js_file]


def _function_markers(path: Path) -> dict[str, dict[str, Any]]:
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
                found[node.name] = _parse_function_marker(
                    comment[len("differential:") :]
                )
    return found


def _parse_function_marker(text: str) -> dict[str, Any]:
    marker: dict[str, Any] = {}
    for token in text.split():
        if token == "skip":
            marker["skip"] = True
            continue
        key, _, value = token.partition("=")
        if key not in ("js", "mode"):
            raise AssertionError(f"Invalid differential marker token: {token!r}")
        marker[key] = value
    return marker


def _parameter_types(function: Any, overrides: dict[str, str]) -> list[str] | None:
    """Return the type of each required parameter, or None if one cannot be inferred."""
    types: list[str] = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return None
        if parameter.name in overrides:
            types.append(overrides[parameter.name])
            continue
        annotation = parameter.annotation
        if not isinstance(annotation, str):
            annotation = getattr(annotation, "__name__", str(annotation))
        if annotation not in _ANNOTATION_TYPES:
            return None
        types.append(_ANNOTATION_TYPES[annotation])
    return types


def _import_python_module(dotted: str) -> Any:
    return importlib.import_module(dotted)


def _module_targets(path: Path) -> list[dict[str, Any]]:
    """Describe the replayable functions of one opted-in source module."""
    settings = module_settings(path)
    if settings is None:
        return []
    dotted = dotted_name(path)
    js_file = js_module_file(path, settings.get("js_module"))
    js_exports = _js_export_names(js_file)
    overrides = settings.get("params", {})
    comparison: dict[str, Any] = {"mode": settings.get("mode", "approx")}
    for key in ("rel_tol", "abs_tol"):
        if key in settings:
            comparison[key] = settings[key]

    module = _import_python_module(dotted)
    markers = _function_markers(path)
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
                    f"{dotted}.{name} names JavaScript export {js_name!r}, "
                    f"which {js_file} does not export"
                )
            continue
        types = _parameter_types(function, overrides)
        if types is None:
            continue
        function_comparison = dict(comparison)
        if "mode" in marker:
            function_comparison["mode"] = marker["mode"]
        targets.append(
            {
                "id": name,
                "module": dotted,
                "js": js_name,
                "js_file": js_file,
                "types": types,
                "comparison": function_comparison,
            }
        )
    return targets


def _all_targets() -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for path in opted_in_modules():
        targets.extend(_module_targets(path))
    return targets


def _resolve_python_target(target: dict[str, Any]) -> Any:
    function = getattr(_import_python_module(target["module"]), target["id"])
    if not callable(function):
        raise AssertionError(f"Differential target is not a function: {target['id']}")
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
    """Keep one Node process alive for the whole replay."""

    def __init__(self) -> None:
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Node.js is required for the differential replay")
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
        # The loader returns JSON text, so results cross the bridge as plain
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
    """Create one persistent JavaScript runtime for the whole module."""
    backend_id = request.param
    if backend_id == "node":
        session: Any = JavaScriptSession()
    elif backend_id == "jspybridge":
        session = JSPyBridgeSession()
    else:
        raise ValueError(f"Unsupported JavaScript backend: {backend_id}")
    yield session
    session.close()


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


def _has_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_has_non_finite(item) for item in value)
    return False


def _skip_reason(target: dict[str, Any], args: list[Any]) -> str | None:
    if len(args) != len(target["types"]):
        return f"arity {len(args)} != {len(target['types'])}"
    for kind, value in zip(target["types"], args, strict=True):
        if isinstance(value, _Opaque) or not _TYPE_CHECKS[kind](value):
            return f"type mismatch for {kind}"
    if _has_non_finite(args):
        return "non-finite input cannot be sent on the JSON wire"
    return None


_CORPUS_DOCUMENTS: dict[str, dict[str, Any] | None] = {}


def _corpus_document(dotted: str) -> dict[str, Any] | None:
    if dotted not in _CORPUS_DOCUMENTS:
        path = CORPUS_DIR / f"{dotted}.json"
        _CORPUS_DOCUMENTS[dotted] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
    return _CORPUS_DOCUMENTS[dotted]


def _corpus_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for target in _all_targets():
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


def _case_label(case: dict[str, Any]) -> str:
    target = case["target"]
    return f"{target['module']}.{target['id']}[{case['index']}]"


CORPUS_CASES = _corpus_cases()


@pytest.mark.parametrize("corpus_case", CORPUS_CASES, ids=_case_label)
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
