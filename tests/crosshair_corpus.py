"""Build replay corpora for opted-in modules from CrossHair examples.

CrossHair explores the annotated signature of each public function and reports
one example per path as an argument dictionary. This tool replays those
examples through the Python function and records the arguments, plus whether
Python raised. tests/test_generated.py replays them against the JavaScript
reference, which is the only oracle.

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
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def encode(value: Any) -> Any:
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


def literal(node: ast.AST) -> Any:
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


def unblock_arguments() -> list[str]:
    """Allow os.add_dll_directory, which numpy calls on Windows when it is imported.

    CrossHair blocks that call as a side effect. Allowing it per folder fails
    when the path CrossHair sees differs in case or form from the one we build,
    so the whole event is allowed. The call only registers a DLL search folder:
    it runs no code.
    """
    return ["--unblock=os.add_dll_directory"]


def run_crosshair(
    dotted: str, name: str, timeout: int
) -> tuple[list[dict[str, Any]], int]:
    """Return the argument dictionaries CrossHair reports for one function, and the count it could not read."""
    command = [
        sys.executable,
        "-m",
        "crosshair",
        "cover",
        f"--per_condition_timeout={timeout}",
        "--example_output_format=arg_dictionary",
        *unblock_arguments(),
        f"{dotted}.{name}",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout * 8 + 120,
        check=False,
    )
    examples: list[dict[str, Any]] = []
    unreadable = 0
    for line in completed.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            parsed = literal(ast.parse(line, mode="eval").body)
        except (SyntaxError, ValueError):
            unreadable += 1
            continue
        if isinstance(parsed, dict):
            examples.append(parsed)
    return examples, unreadable


def positional(function: Any, example: dict[str, Any]) -> list[Any] | None:
    """Map an argument dictionary to positional arguments, or None if it cannot be replayed."""
    names = list(inspect.signature(function).parameters)
    if set(example) - set(names):
        return None
    provided = [name for name in names if name in example]
    if provided != names[: len(provided)]:
        # A defaulted parameter in the middle is skipped; positional replay cannot express that.
        return None
    return [example[name] for name in provided]


def capture(dotted: str, timeout: int) -> dict[str, Any]:
    """Run CrossHair on every public function of one module and return its corpus document."""
    module = importlib.import_module(dotted)
    entries: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    examples_total = unsupported = unreadable_total = 0

    for name, function in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_") or function.__module__ != module.__name__:
            continue
        examples, unreadable = run_crosshair(dotted, name, timeout)
        unreadable_total += unreadable
        for example in examples:
            examples_total += 1
            args = positional(function, example)
            if args is None:
                unsupported += 1
                continue
            try:
                function(*args)
                raised = None
            except Exception as exc:
                raised = type(exc).__name__
            encoded = encode(args)
            key = json.dumps([name, encoded, raised], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            entries.setdefault(name, []).append({"args": encoded, "raised": raised})

    return {
        "module": dotted,
        "generator": {
            "tool": "crosshair",
            "version": importlib.metadata.version("crosshair-tool"),
            "per_condition_timeout_s": timeout,
            "output_format": "arg_dictionary",
        },
        "stats": {
            "examples": examples_total,
            "unique": len(seen),
            "unsupported_calls": unsupported,
            "unreadable_lines": unreadable_total,
        },
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

    parameters = f"tool=crosshair per_condition_timeout={options.timeout}"
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
