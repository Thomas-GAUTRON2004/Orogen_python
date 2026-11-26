"""Capture the calls Pynguin makes into opted-in modules as replay corpora.

Pynguin executes the project's code, so it refuses to run unless the environment
variable PYNGUIN_DANGER_AWARE=1 is set. Its own assertions are not kept: they
would freeze whatever Python returns. A corpus records only the arguments of each
call and whether Python raised. tests/test_generated.py replays them against the
JavaScript reference, which is the only oracle.

Every module that opts in under the source directory gets a corpus. The corpus
file stores a fingerprint of the module's sources and of the orogen modules it
depends on, plus the Pynguin budget and seed. A module whose fingerprint is
unchanged is skipped, so only added or edited modules are regenerated.

Usage:
    PYNGUIN_DANGER_AWARE=1 python tests/pynguin_corpus.py              # changed modules
    PYNGUIN_DANGER_AWARE=1 python tests/pynguin_corpus.py rotation     # one module
    python tests/pynguin_corpus.py --dry-run                           # report only
    PYNGUIN_DANGER_AWARE=1 python tests/pynguin_corpus.py --force      # everything

Writes tests/corpus/<dotted module name>.json.
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


def run_pynguin(dotted: str, seconds: int, seed: int, out_dir: Path) -> Path:
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
        dotted,
        "--output-path",
        str(out_dir),
        "--maximum-search-time",
        str(seconds),
        "--seed",
        str(seed),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    generated = sorted(out_dir.glob("test_*.py"))
    if len(generated) != 1:
        raise SystemExit(f"Pynguin produced {len(generated)} test files for {dotted}")
    return generated[0]


def replay(generated: Path) -> None:
    """Run each Pynguin test. Exceptions are expected; the calls are already recorded."""
    spec = importlib.util.spec_from_file_location("pynguin_generated", generated)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in sorted(dir(module)):
        if name.startswith("test_") and callable(getattr(module, name)):
            with contextlib.suppress(Exception):
                getattr(module, name)()


def capture(dotted: str, seconds: int, seed: int) -> dict[str, Any]:
    """Run Pynguin on one module and return its corpus document."""
    out_dir = Path(tempfile.mkdtemp(prefix=f"pynguin-{dotted}-"))
    generated = run_pynguin(dotted, seconds, seed, out_dir)

    module = importlib.import_module(dotted)
    recorder = Recorder()
    for name, function in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_") or function.__module__ != module.__name__:
            continue
        setattr(module, name, recorder.wrap(name, function))
    replay(generated)

    entries: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for name, args, raised in recorder.calls:
        key = json.dumps([name, args, raised], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        entries.setdefault(name, []).append({"args": args, "raised": raised})

    return {
        "module": dotted,
        "generator": {
            "tool": "pynguin",
            "version": importlib.metadata.version("pynguin"),
            "seed": seed,
            "maximum_search_time_s": seconds,
        },
        "stats": {
            "calls": len(recorder.calls),
            "unique": len(seen),
            "unsupported_calls": recorder.unsupported,
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
    parser.add_argument("--seconds", type=int, default=SETTINGS["pynguin_seconds"])
    parser.add_argument("--seed", type=int, default=SETTINGS["pynguin_seed"])
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

    regenerated = unchanged = 0
    for path in paths:
        dotted = dotted_name(path)
        digest, files = input_digest(path, options.seconds, options.seed)
        if not options.force and is_current(dotted, digest):
            print(f"unchanged   {dotted}")
            unchanged += 1
            continue
        if options.dry_run:
            print(f"would regen {dotted} ({len(files)} source files)")
            regenerated += 1
            continue
        document = capture(dotted, options.seconds, options.seed)
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
