"""Build replay corpora for opted-in modules from CrossHair examples.

CrossHair explores the annotated signature of each public function and each
public method of a class, and reports one example per path as an argument
dictionary. A method's example also reports its instance (``self``), printed as
a constructor call: ``FastNoise(seed=0.0)``. This tool replays each example
through the Python code and records the arguments, the instance's constructor
arguments when there is one, and whether Python raised. tests/test_generated.py
replays them against the JavaScript reference, which is the only oracle.

Every module that opts in under the source directory gets a corpus. The corpus
stores a fingerprint of the module's sources, the orogen modules it depends on,
and the CrossHair budget. A module whose fingerprint is unchanged is skipped.

Usage:
    python tests/crosshair_corpus.py                  # changed modules only
    python tests/crosshair_corpus.py rotation         # one module (short stem)
    python tests/crosshair_corpus.py --dry-run        # report only
    python tests/crosshair_corpus.py --force          # everything

Writes tests/corpus/<dotted module name>.json.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import TypeAlias

sys.path.insert(0, str(Path(__file__).resolve().parent))

from differential_support import (  # noqa: E402
    CORPUS_DIR,
    ROOT,
    SETTINGS,
    corpus_path,
    dotted_name,
    find_module,
    input_digest,
    opted_in_modules,
)

# Side effects CrossHair blocks while importing the orogen package. Each is
# read-only or only registers a folder: it runs no code.
#   os.add_dll_directory: numpy registers its DLL folders on Windows.
#   socket.gethostname: numba reads the host name while it loads.
UNBLOCKED_EVENTS = ("os.add_dll_directory", "socket.gethostname")

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
CrossHairValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | bytes
    | bytearray
    | list["CrossHairValue"]
    | tuple["CrossHairValue", ...]
    | dict[str, "CrossHairValue"]
)

IMPORT_FAILURE = "Could not import your code"

# Corpus layout version: 2 adds class methods and their constructor arguments.
FORMAT = 2


def encode(value: CrossHairValue) -> JsonValue:
    """Turn an argument or result into JSON, keeping what JSON cannot express."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__number__": "NaN"}
        if math.isinf(value):
            return {"__number__": "Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": bytes(value).hex()}
    return {"__repr__": repr(value)}


def literal(node: ast.AST) -> CrossHairValue:
    """Evaluate the literal subset CrossHair prints: constants, containers, float(str)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = literal(node.operand)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value
    if isinstance(node, ast.Tuple):
        return tuple(literal(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            literal(key): literal(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and not node.keywords
    ):
        value = literal(node.args[0])
        if isinstance(value, str):
            return float(value)
    raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")


def instance_state(node: ast.AST) -> dict[str, CrossHairValue]:
    """Read CrossHair's printed instance, ``ClassName(name=value, ...)``, as constructor keywords."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.args
        and all(keyword.arg is not None for keyword in node.keywords)
    ):
        return {
            "__instance__": node.func.id,
            "kwargs": {k.arg: literal(k.value) for k in node.keywords},
        }
    raise ValueError(f"unsupported instance: {ast.dump(node)[:80]}")


def parse_example(line: str) -> dict[str, CrossHairValue]:
    """Parse one printed example line into an argument dictionary."""
    node = ast.parse(line, mode="eval").body
    if not isinstance(node, ast.Dict):
        raise ValueError("example is not a dictionary")
    example: dict[str, CrossHairValue] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        key = literal(key_node)
        example[key] = (
            instance_state(value_node) if key == "self" else literal(value_node)
        )
    return example


def unblock_arguments() -> list[str]:
    """Return the --unblock flags for the events CrossHair blocks during import.

    Events are allowed as a whole rather than per argument: a path-specific
    prefix fails when the path CrossHair sees differs in case or form.
    """
    return [f"--unblock={event}" for event in UNBLOCKED_EVENTS]


def run_crosshair(
    dotted: str, target: str, timeout: int
) -> tuple[list[dict[str, CrossHairValue]], int]:
    """Return the examples CrossHair reports for one function or method, and the count it could not read."""
    command = [
        sys.executable,
        "-m",
        "crosshair",
        "cover",
        f"--per_condition_timeout={timeout}",
        "--example_output_format=arg_dictionary",
        *unblock_arguments(),
        f"{dotted}.{target}",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout * 8 + 120,
        check=False,
        # CrossHair prints example strings that the Windows console codepage cannot
        # encode (for example \u0800), so the child must write UTF-8.
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    output = completed.stdout + "\n" + completed.stderr
    if IMPORT_FAILURE in output or completed.returncode != 0:
        # Never write a corpus from a run that did not import the code: an empty
        # corpus would look like a module with nothing to replay.
        reason = next(
            (
                line.strip()
                for line in output.splitlines()
                if "Error" in line or "detected" in line
            ),
            output.strip()[-300:],
        )
        raise SystemExit(f"CrossHair could not run on {dotted}.{target}: {reason}")
    examples: list[dict[str, CrossHairValue]] = []
    unreadable = 0
    for line in completed.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            examples.append(parse_example(line))
        except (SyntaxError, ValueError):
            unreadable += 1
    return examples, unreadable


def positional_for(
    names: list[str], example: dict[str, CrossHairValue]
) -> list[CrossHairValue] | None:
    """Map an argument dictionary to positional arguments, or None if it cannot be replayed."""
    if set(example) - set(names):
        return None
    provided = [name for name in names if name in example]
    if provided != names[: len(provided)]:
        # A defaulted parameter in the middle is skipped; positional replay cannot express that.
        return None
    return [example[name] for name in provided]


def parameter_names(signature: inspect.Signature, skip_first: bool) -> list[str]:
    names = list(signature.parameters)
    return names[1:] if skip_first else names


def capture(dotted: str, timeout: int) -> dict[str, JsonValue]:
    """Run CrossHair on every public function and method of one module; return its corpus document."""
    module = importlib.import_module(dotted)
    entries: dict[str, list[dict[str, JsonValue]]] = {}
    seen: set[str] = set()
    counts = {"examples": 0, "unsupported_calls": 0, "unreadable_lines": 0}

    def record(
        key: str,
        args: list[CrossHairValue],
        constructor: list[CrossHairValue] | None,
        raised: str | None,
    ) -> None:
        encoded_args = encode(args)
        encoded_constructor = None if constructor is None else encode(constructor)
        signature = json.dumps(
            [key, encoded_constructor, encoded_args, raised], sort_keys=True
        )
        if signature in seen:
            return
        seen.add(signature)
        entry: dict[str, JsonValue] = {"args": encoded_args, "raised": raised}
        if encoded_constructor is not None:
            entry["constructor"] = encoded_constructor
        entries.setdefault(key, []).append(entry)

    for name, function in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_") or function.__module__ != module.__name__:
            continue
        examples, unreadable = run_crosshair(dotted, name, timeout)
        counts["unreadable_lines"] += unreadable
        for example in examples:
            counts["examples"] += 1
            args = positional_for(
                parameter_names(inspect.signature(function), skip_first=False), example
            )
            if args is None:
                counts["unsupported_calls"] += 1
                continue
            try:
                function(*args)
                raised = None
            except Exception as exc:
                raised = type(exc).__name__
            record(name, args, None, raised)

    for class_name, cls in inspect.getmembers(module, inspect.isclass):
        if class_name.startswith("_") or cls.__module__ != module.__name__:
            continue
        for method_name, method in cls.__dict__.items():
            if method_name.startswith("_") or not inspect.isfunction(method):
                continue
            key = f"{class_name}.{method_name}"
            examples, unreadable = run_crosshair(dotted, key, timeout)
            counts["unreadable_lines"] += unreadable
            for example in examples:
                counts["examples"] += 1
                state = example.pop("self", None)
                if (
                    not isinstance(state, dict)
                    or state.get("__instance__") != class_name
                ):
                    counts["unsupported_calls"] += 1
                    continue
                constructor = positional_for(
                    parameter_names(inspect.signature(cls.__init__), skip_first=True),
                    state["kwargs"],
                )
                args = positional_for(
                    parameter_names(inspect.signature(method), skip_first=True), example
                )
                if constructor is None or args is None:
                    counts["unsupported_calls"] += 1
                    continue
                try:
                    instance = cls(*constructor)
                except Exception:
                    counts["unsupported_calls"] += 1
                    continue
                try:
                    getattr(instance, method_name)(*args)
                    raised = None
                except Exception as exc:
                    raised = type(exc).__name__
                record(key, args, constructor, raised)

    return {
        "module": dotted,
        "generator": {
            "tool": "crosshair",
            "version": importlib.metadata.version("crosshair-tool"),
            "per_condition_timeout_s": timeout,
            "output_format": "arg_dictionary",
        },
        "stats": {"unique": len(seen), **counts},
        "entries": entries,
    }


def is_current(dotted: str, digest: str) -> bool:
    path = corpus_path(dotted)
    if not path.is_file():
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get("generator", {}).get("input_sha256") == digest


def stale_corpora() -> list[Path]:
    """Corpus files whose module no longer opts in."""
    live = {dotted_name(path) for path in opted_in_modules()}
    if not CORPUS_DIR.is_dir():
        return []
    return [path for path in sorted(CORPUS_DIR.glob("*.json")) if path.stem not in live]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "modules", nargs="*", help="modules to consider (default: all opted-in)"
    )
    parser.add_argument(
        "--force", action="store_true", help="regenerate even if unchanged"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=SETTINGS["crosshair_timeout"],
        help="CrossHair time per condition, in seconds (default: crosshair_timeout on the settings line)",
    )
    options = parser.parse_args()

    if options.modules:
        paths = []
        for query in options.modules:
            path = find_module(query)
            if path is None:
                raise SystemExit(f"No opted-in module matches {query!r}")
            paths.append(path)
    else:
        paths = opted_in_modules()

    # FORMAT names the corpus layout: bump it when the recorded entries change.
    parameters = (
        f"tool=crosshair format={FORMAT} per_condition_timeout={options.timeout}"
    )
    regenerated = unchanged = 0
    for path in paths:
        dotted = dotted_name(path)
        digest, files = input_digest(path, parameters)
        if not options.force and is_current(dotted, digest):
            print(f"unchanged   {dotted}")
            unchanged += 1
            continue
        if options.dry_run:
            print(f"would regen {dotted} ({len(files)} source files)")
            regenerated += 1
            continue
        document = capture(dotted, options.timeout)
        document["generator"]["input_sha256"] = digest
        document["generator"]["inputs"] = files
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        target = corpus_path(dotted)
        target.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"regenerated {dotted}: {document['stats']}")
        regenerated += 1

    for path in stale_corpora():
        print(
            f"stale       {path.relative_to(ROOT)} (module no longer opts in; delete it)"
        )
    print(f"{regenerated} regenerated, {unchanged} unchanged")


if __name__ == "__main__":
    main()
