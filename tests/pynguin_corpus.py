"""Capture the calls Pynguin makes into one Python module as a replay corpus.

Pynguin executes the project's code, so it refuses to run unless the environment
variable PYNGUIN_DANGER_AWARE=1 is set. Pynguin's own assertions are not kept:
they freeze whatever Python returns. The corpus records only the arguments of
each call and whether Python raised. tests/test_generated.py replays them
against the JavaScript reference, which is the only oracle.

Usage:
    PYNGUIN_DANGER_AWARE=1 python tests/pynguin_corpus.py rotation --seconds 60 --seed 1

Writes tests/corpus/<module>.json.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "orogen.v2.core"
CORPUS_DIR = ROOT / "tests" / "corpus"


def _settings_from_test_file() -> dict[str, str]:
    """Read the '# differential-settings:' line of tests/test_generated.py."""
    path = ROOT / "tests" / "test_generated.py"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# differential-settings:"):
            tokens = line.split(":", 1)[1].split()
            return dict(token.split("=", 1) for token in tokens if "=" in token)
    return {}


def encode(value: Any) -> Any:
    """Turn a call argument or result into JSON, keeping what JSON cannot express."""
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


class Recorder:
    """Wrap module functions so every call's arguments are captured before it runs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any], str | None]] = []
        self.unsupported = 0

    def wrap(self, name: str, function: Any) -> Any:
        signature = inspect.signature(function)
        names = list(signature.parameters)
        recorder = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                recorder.unsupported += 1
                raise
            provided = [item for item in names if item in bound.arguments]
            if provided != names[: len(provided)]:
                # The replay passes positional arguments, so a call that skips a
                # defaulted parameter in the middle cannot be replayed faithfully.
                recorder.unsupported += 1
                return function(*args, **kwargs)
            snapshot = encode([bound.arguments[item] for item in provided])
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                recorder.calls.append((name, snapshot, type(exc).__name__))
                raise
            recorder.calls.append((name, snapshot, None))
            return result

        wrapper.__name__ = name
        return wrapper


def run_pynguin(module: str, seconds: int, seed: int, out_dir: Path) -> Path:
    if os.environ.get("PYNGUIN_DANGER_AWARE") != "1":
        raise SystemExit(
            "Set PYNGUIN_DANGER_AWARE=1: Pynguin executes the project's code."
        )
    command = [
        sys.executable,
        "-m",
        "pynguin",
        "--project-path",
        str(ROOT),
        "--module-name",
        f"{PACKAGE}.{module}",
        "--output-path",
        str(out_dir),
        "--maximum-search-time",
        str(seconds),
        "--seed",
        str(seed),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    generated = out_dir / f"test_{module}.py"
    if not generated.is_file():
        raise SystemExit(f"Pynguin did not produce {generated}")
    return generated


def replay(generated: Path, recorder: Recorder) -> None:
    """Run each Pynguin test. Exceptions are expected; the calls are already recorded."""
    spec = importlib.util.spec_from_file_location("pynguin_generated", generated)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in sorted(dir(module)):
        if name.startswith("test_") and callable(getattr(module, name)):
            with contextlib.suppress(Exception):
                getattr(module, name)()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("module", help="Python module stem under orogen/v2/core")
    settings = _settings_from_test_file()
    parser.add_argument(
        "--seconds",
        type=int,
        default=int(settings.get("pynguin_seconds", 60)),
        help="Pynguin search budget (default: pynguin_seconds in test_generated.py)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(settings.get("pynguin_seed", 1)),
        help="Pynguin seed (default: pynguin_seed in test_generated.py)",
    )
    options = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    out_dir = Path(tempfile.mkdtemp(prefix=f"pynguin-{options.module}-"))
    generated = run_pynguin(options.module, options.seconds, options.seed, out_dir)

    module = importlib.import_module(f"{PACKAGE}.{options.module}")
    recorder = Recorder()
    for name, function in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_") or function.__module__ != module.__name__:
            continue
        setattr(module, name, recorder.wrap(name, function))
    replay(generated, recorder)

    entries: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for name, args, raised in recorder.calls:
        key = json.dumps([name, args, raised], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        entries.setdefault(name, []).append({"args": args, "raised": raised})

    document = {
        "module": options.module,
        "generator": {
            "tool": "pynguin",
            "version": importlib.metadata.version("pynguin"),
            "seed": options.seed,
            "maximum_search_time_s": options.seconds,
        },
        "stats": {
            "calls": len(recorder.calls),
            "unique": len(seen),
            "unsupported_calls": recorder.unsupported,
        },
        "entries": entries,
    }
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    target = CORPUS_DIR / f"{options.module}.json"
    target.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{target.relative_to(ROOT)}: {document['stats']}")


if __name__ == "__main__":
    main()
