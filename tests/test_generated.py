"""Replay of recorded calls against the upstream JavaScript reference.

The test suite only replays corpora written by tests/crosshair_corpus.py. Each
recorded call is run through the Python function and the JavaScript export, and
the results are compared. The corpora hold arguments only, never expected values.

Configuration lives in comments.

Global settings are the single line below (defaults apply to missing keys):

# differential-settings: source=orogen backends=node,jspybridge crosshair_timeout=5

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
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Protocol, TypeAlias, TypedDict, cast

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


class _Opaque:
    """A value the JSON corpus could only record as text."""

    def __init__(self, text: str) -> None:
        self.text = text


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
DecodedValue: TypeAlias = (
    None | bool | int | float | str | bytes | list["DecodedValue"]
    | dict[str, "DecodedValue"] | _Opaque
)
PythonValue: TypeAlias = (
    None | bool | int | float | str | list["PythonValue"]
    | tuple["PythonValue", ...] | dict[str, "PythonValue"]
)


class FunctionMarker(TypedDict, total=False):
    js: str
    mode: str
    skip: bool


class ComparisonSettings(TypedDict, total=False):
    mode: str
    rel_tol: float
    abs_tol: float


class TargetInfo(TypedDict):
    id: str
    module: str
    js: str
    js_file: str
    types: list[str]
    comparison: ComparisonSettings


class MethodTargetInfo(TargetInfo, total=False):
    class_name: str
    method: str
    js_member: str
    constructor_types: list[str]


ReplayTarget: TypeAlias = TargetInfo | MethodTargetInfo


class CorpusEntry(TypedDict, total=False):
    args: list[JsonValue]
    constructor: list[JsonValue]


class CorpusDocument(TypedDict):
    entries: dict[str, list[CorpusEntry]]


class ReplayCase(TypedDict):
    target: ReplayTarget
    args: list[DecodedValue]
    constructor: list[DecodedValue] | None
    index: int
    skip: str | None


class ReplaySession(Protocol):
    def call(
        self,
        js_file: str,
        export: str,
        args: list[JsonValue],
        constructor: list[JsonValue] | None = None,
        method: str | None = None,
    ) -> JsonValue: ...

    def close(self) -> None: ...


class BridgeLoader(Protocol):
    def invoke(
        self,
        js_file: str,
        export: str,
        args_json: str,
        constructor_json: str,
        method: str,
    ) -> str: ...


class BridgeRuntime(Protocol):
    def require(self, path: str, caller: str) -> BridgeLoader: ...

    def terminate(self) -> None: ...

JS_RUNNER = ROOT / "tests" / "reference" / "v2" / "property-runner.mjs"

# Types a parameter can take. Corpus entries whose arguments do not fit are
# reported as skipped, with the reason. Values outside any range still run.
_TYPE_CHECKS: dict[str, Callable[[DecodedValue], bool]] = {
    "unit_float": lambda value: _is_number(value),
    "angle": lambda value: _is_number(value),
    "time_step": lambda value: _is_number(value),
    "quaternion": lambda value: _is_list_of_numbers(value, 4),
    "vec3": lambda value: _is_list_of_numbers(value, 3),
    "seed_text": lambda value: isinstance(value, str),
    "seed_input": lambda value: (
        value is None or isinstance(value, (str, int, float, bool))
    ),
}
_ANNOTATION_TYPES = {
    "float": "unit_float",
    "int | float": "unit_float",
    "SeedInput": "seed_input",
}


def _is_number(value: DecodedValue) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_list_of_numbers(value: DecodedValue, length: int) -> bool:
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


def _function_markers(path: Path) -> dict[str, FunctionMarker]:
    """Read '# differential:' comments above top-level functions and class methods.

    Keys are the function name, or ``Class.method`` for a method.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    found: dict[str, FunctionMarker] = {}

    def visit(nodes: list[ast.stmt], prefix: str) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{node.name}.")
            elif isinstance(node, ast.FunctionDef):
                start = min(
                    [
                        node.lineno,
                        *(decorator.lineno for decorator in node.decorator_list),
                    ]
                )
                index = start - 2
                block: list[str] = []
                while index >= 0 and lines[index].strip().startswith("#"):
                    block.insert(0, lines[index].strip()[1:].strip())
                    index -= 1
                for comment in block:
                    if comment.startswith("differential:"):
                        found[prefix + node.name] = _parse_function_marker(
                            comment[len("differential:") :]
                        )

    visit(ast.parse(source).body, "")
    return found


def _parse_function_marker(text: str) -> FunctionMarker:
    marker: FunctionMarker = {}
    for token in text.split():
        if token == "skip":
            marker["skip"] = True
            continue
        key, _, value = token.partition("=")
        if key not in ("js", "mode"):
            raise AssertionError(f"Invalid differential marker token: {token!r}")
        marker[key] = value
    return marker


def _parameter_types(
    function: FunctionType,
    overrides: dict[str, str],
    skip_first: bool = False,
    include_defaults: bool = False,
) -> list[str] | None:
    """Return the type of each parameter, or None if one cannot be inferred.

    Parameters with defaults are left out unless include_defaults is set, which a
    constructor needs when the corpus records a value for them. skip_first drops
    the instance parameter of a method or constructor.
    """
    types: list[str] = []
    parameters = list(inspect.signature(function).parameters.values())
    if skip_first:
        parameters = parameters[1:]
    for parameter in parameters:
        if parameter.default is not inspect.Parameter.empty and not include_defaults:
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


def _import_python_module(dotted: str) -> ModuleType:
    return importlib.import_module(dotted)


def _module_targets(path: Path) -> list[ReplayTarget]:
    """Describe the replayable functions of one opted-in source module."""
    settings = module_settings(path)
    if settings is None:
        return []
    dotted = dotted_name(path)
    js_file = js_module_file(path, settings.get("js_module"))
    js_exports = _js_export_names(js_file)
    overrides = settings.get("params", {})
    comparison: ComparisonSettings = {"mode": settings.get("mode", "approx")}
    for key in ("rel_tol", "abs_tol"):
        if key in settings:
            comparison[key] = settings[key]

    module = _import_python_module(dotted)
    markers = _function_markers(path)
    targets: list[ReplayTarget] = []
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
        types = _parameter_types(function, overrides, include_defaults=True)
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
    targets.extend(
        _class_targets(
            module, dotted, js_file, js_exports, overrides, comparison, markers
        )
    )
    return targets


def _class_targets(
    module: ModuleType,
    dotted: str,
    js_file: str,
    js_exports: set[str],
    overrides: dict[str, str],
    comparison: ComparisonSettings,
    markers: dict[str, FunctionMarker],
) -> list[MethodTargetInfo]:
    """Describe the public methods of each public class whose JavaScript class is exported.

    A method target builds the instance from the constructor arguments recorded
    in the corpus, then calls the method, in Python and in JavaScript.
    """
    targets: list[MethodTargetInfo] = []
    for class_name, cls in inspect.getmembers(module, inspect.isclass):
        if class_name.startswith("_") or cls.__module__ != module.__name__:
            continue
        js_class = _js_name(class_name)
        if js_class not in js_exports:
            continue
        constructor_types = _parameter_types(
            cls.__init__, overrides, skip_first=True, include_defaults=True
        )
        if constructor_types is None:
            continue
        for method_name, method in cls.__dict__.items():
            if method_name.startswith("_") or not inspect.isfunction(method):
                continue
            marker = markers.get(f"{class_name}.{method_name}", {})
            if marker.get("skip"):
                continue
            types = _parameter_types(
                method, overrides, skip_first=True, include_defaults=True
            )
            if types is None:
                continue
            targets.append(
                {
                    "id": f"{class_name}.{method_name}",
                    "module": dotted,
                    "class_name": class_name,
                    "method": method_name,
                    "js": js_class,
                    "js_member": marker.get("js", _js_name(method_name)),
                    "js_file": js_file,
                    "types": types,
                    "constructor_types": constructor_types,
                    "comparison": dict(comparison),
                }
            )
    return targets


def _all_targets() -> list[ReplayTarget]:
    targets: list[ReplayTarget] = []
    for path in opted_in_modules():
        targets.extend(_module_targets(path))
    return targets


def _resolve_python_target(target: ReplayTarget) -> Callable[..., PythonValue]:
    function = getattr(_import_python_module(target["module"]), target["id"])
    if not callable(function):
        raise AssertionError(f"Differential target is not a function: {target['id']}")
    return cast(Callable[..., PythonValue], function)


def _normalise(value: PythonValue) -> JsonValue:
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

    def call(
        self,
        js_file: str,
        export: str,
        args: list[JsonValue],
        constructor: list[JsonValue] | None = None,
        method: str | None = None,
    ) -> JsonValue:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("JavaScript runner streams are unavailable")
        payload: dict[str, JsonValue] = {"module": js_file, "target": export, "args": args}
        if constructor is not None:
            # Build the instance with `new`, then call the method on it.
            payload["construct"] = constructor
            payload["method"] = method
        request = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(f"JavaScript runner exited unexpectedly: {stderr}")
        response = cast(dict[str, JsonValue], json.loads(line))
        if not response["ok"]:
            raise JSTargetError(cast(str, response["error"]))
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
            bridge = cast(BridgeRuntime, importlib.import_module("javascript"))
        except ImportError as exc:
            raise RuntimeError(
                "JSPyBridge is optional; install it with "
                "'python -m pip install javascript==1!1.2.6'"
            ) from exc

        self._bridge = bridge
        self._loader = bridge.require("./jspybridge-loader.cjs", __file__)

    def call(
        self,
        js_file: str,
        export: str,
        args: list[JsonValue],
        constructor: list[JsonValue] | None = None,
        method: str | None = None,
    ) -> JsonValue:
        # The loader returns JSON text, so results cross the bridge as plain
        # strings, exactly as they do through the Node runner. An empty
        # constructor string means "call the export directly".
        constructor_json = (
            ""
            if constructor is None
            else json.dumps(constructor, ensure_ascii=False, allow_nan=False)
        )
        try:
            text = self._loader.invoke(
                js_file,
                export,
                json.dumps(args, ensure_ascii=False, allow_nan=False),
                constructor_json,
                method or "",
            )
        except Exception as exc:
            raise JSTargetError(str(exc)) from exc
        return cast(JsonValue, json.loads(text))

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
def javascript(request: pytest.FixtureRequest) -> Iterator[ReplaySession]:
    """Create one persistent JavaScript runtime for the whole module."""
    backend_id = request.param
    if backend_id == "node":
        session: ReplaySession = JavaScriptSession()
    elif backend_id == "jspybridge":
        session = JSPyBridgeSession()
    else:
        raise ValueError(f"Unsupported JavaScript backend: {backend_id}")
    yield session
    session.close()


def _assert_node(
    actual: JsonValue, expected: JsonValue, mode: str, comparison: ComparisonSettings, path: str
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


def _assert_equivalent(actual: JsonValue, expected: JsonValue, comparison: ComparisonSettings) -> None:
    mode = comparison.get("mode", "exact")
    if mode not in ("exact", "approx"):
        raise ValueError(f"Unsupported comparison mode: {mode}")
    _assert_node(actual, expected, mode, comparison, "result")


def _decode(value: JsonValue) -> DecodedValue:
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


def _has_non_finite(value: DecodedValue) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_has_non_finite(item) for item in value)
    return False


def _skip_reason(
    target: ReplayTarget, args: list[DecodedValue], constructor: list[DecodedValue] | None
) -> str | None:
    if "constructor_types" in target:
        values = constructor or []
        if len(values) > len(target["constructor_types"]):
            return (
                f"constructor arity {len(values)} > {len(target['constructor_types'])}"
            )
        for kind, value in zip(target["constructor_types"], values, strict=False):
            if isinstance(value, _Opaque) or not _TYPE_CHECKS[kind](value):
                return f"constructor type mismatch for {kind}"
        if _has_non_finite(values):
            return "non-finite constructor input cannot be sent on the JSON wire"
    if len(args) != len(target["types"]):
        return f"arity {len(args)} != {len(target['types'])}"
    for kind, value in zip(target["types"], args, strict=True):
        if isinstance(value, _Opaque) or not _TYPE_CHECKS[kind](value):
            return f"type mismatch for {kind}"
    if _has_non_finite(args):
        return "non-finite input cannot be sent on the JSON wire"
    return None


_CORPUS_DOCUMENTS: dict[str, CorpusDocument | None] = {}


def _corpus_document(dotted: str) -> CorpusDocument | None:
    if dotted not in _CORPUS_DOCUMENTS:
        path = CORPUS_DIR / f"{dotted}.json"
        _CORPUS_DOCUMENTS[dotted] = (
            cast(CorpusDocument, json.loads(path.read_text(encoding="utf-8")))
            if path.is_file()
            else None
        )
    return _CORPUS_DOCUMENTS[dotted]


def _corpus_cases() -> list[ReplayCase]:
    cases: list[ReplayCase] = []
    for target in _all_targets():
        document = _corpus_document(target["module"])
        if document is None:
            continue
        for index, entry in enumerate(document["entries"].get(target["id"], [])):
            args = _decode(entry["args"])
            constructor = (
                _decode(entry["constructor"]) if "constructor" in entry else None
            )
            cases.append(
                {
                    "target": target,
                    "args": args,
                    "constructor": constructor,
                    "index": index,
                    "skip": _skip_reason(target, args, constructor),
                }
            )
    return cases


def _case_label(case: ReplayCase) -> str:
    target = case["target"]
    return f"{target['module']}.{target['id']}[{case['index']}]"


CORPUS_CASES = _corpus_cases()


@pytest.mark.parametrize("corpus_case", CORPUS_CASES, ids=_case_label)
@pytest.mark.parametrize("javascript", _BACKEND_PARAMETERS, indirect=True)
def test_corpus_replay(corpus_case: ReplayCase, javascript: ReplaySession) -> None:
    if corpus_case["skip"] is not None:
        pytest.skip(corpus_case["skip"])
    target = corpus_case["target"]
    args = copy.deepcopy(corpus_case["args"])
    is_method = "class_name" in target
    constructor = copy.deepcopy(corpus_case["constructor"]) if is_method else None

    python_error: str | None = None
    try:
        if is_method:
            cls = getattr(_import_python_module(target["module"]), target["class_name"])
            python_result = getattr(cls(*constructor), target["method"])(*args)
        else:
            python_result = _resolve_python_target(target)(*args)
    except Exception as exc:
        python_error = type(exc).__name__

    javascript_error: str | None = None
    try:
        javascript_result = javascript.call(
            target["js_file"],
            target["js"],
            cast(list[JsonValue], args),
            constructor=cast(list[JsonValue] | None, constructor),
            method=target.get("js_member"),
        )
    except JSTargetError as exc:
        javascript_error = str(exc)

    if python_error is not None or javascript_error is not None:
        assert python_error is not None and javascript_error is not None, (
            f"Python raised {python_error!r} but JavaScript raised {javascript_error!r}"
        )
        return
    _assert_equivalent(
        _normalise(cast(PythonValue, python_result)),
        _normalise(javascript_result), target["comparison"]
    )
