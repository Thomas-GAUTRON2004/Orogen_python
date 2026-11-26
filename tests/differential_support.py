"""Discovery shared by the differential replay and the Pynguin corpus tool.

Everything is read from comments: the settings line in tests/test_generated.py,
and '# differential-module:' lines in the Python sources under orogen/.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
TEST_FILE = TESTS_DIR / "test_generated.py"
REFERENCE_DIR = TESTS_DIR / "reference" / "v2"
CORPUS_DIR = TESTS_DIR / "corpus"
V2_DIR = ROOT / "orogen" / "v2"

SETTING_DEFAULTS = {
    "source": "orogen",
    "backends": "node,jspybridge",
    "pynguin_seconds": "60",
    "pynguin_seed": "1",
}
MODULE_KEYS = {"js_module", "mode", "rel_tol", "abs_tol", "params"}


def read_settings() -> dict[str, Any]:
    """Read the '# differential-settings:' line of tests/test_generated.py."""
    values = dict(SETTING_DEFAULTS)
    for line in TEST_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("# differential-settings:"):
            for token in line.split(":", 1)[1].split():
                key, _, value = token.partition("=")
                if key not in values:
                    raise AssertionError(f"Unknown differential setting: {key!r}")
                values[key] = value
            break
    return {
        "source": values["source"],
        "backends": [item for item in values["backends"].split(",") if item],
        "pynguin_seconds": int(values["pynguin_seconds"]),
        "pynguin_seed": int(values["pynguin_seed"]),
    }


SETTINGS = read_settings()
SOURCE_DIR = ROOT / SETTINGS["source"]


def module_settings(path: Path) -> dict[str, Any] | None:
    """Read the '# differential-module:' comment of a source file, if present."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("# differential-module:"):
            continue
        settings: dict[str, Any] = {}
        for token in stripped.split(":", 1)[1].split():
            key, _, value = token.partition("=")
            if key not in MODULE_KEYS:
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


def dotted_name(path: Path) -> str:
    """Return the import name of a source file, for example orogen.v2.core.rotation."""
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def opted_in_modules() -> list[Path]:
    """Return every source file under the source directory that opts in."""
    return [
        path
        for path in sorted(SOURCE_DIR.rglob("*.py"))
        if not path.name.startswith("_") and module_settings(path) is not None
    ]


def js_module_file(path: Path, explicit: str | None) -> str:
    """Return the reference module path, relative to tests/reference/v2, for a source file.

    An explicit js_module with a slash is a path from tests/reference/v2, so a
    reference outside the module's own directory can be named, for example
    js/simplex-noise.mjs. Without a slash it is relative to the module's directory.
    """
    try:
        directory = path.parent.relative_to(V2_DIR).as_posix()
    except ValueError as exc:
        raise AssertionError(
            f"{path.name}: opted-in modules must live under orogen/v2"
        ) from exc

    def join(name: str) -> str:
        if "/" in name:
            return name
        return name if directory == "." else f"{directory}/{name}"

    if explicit is not None:
        if not (REFERENCE_DIR / join(explicit)).is_file():
            raise AssertionError(f"{path.name}: js_module {explicit!r} does not exist")
        return join(explicit)
    stem = path.stem
    for name in (f"{stem}.mjs", f"{stem.replace('_', '-')}.mjs"):
        if (REFERENCE_DIR / join(name)).is_file():
            return join(name)
    same_name = sorted(
        str(found.relative_to(REFERENCE_DIR).as_posix())
        for found in REFERENCE_DIR.rglob(f"{stem.replace('_', '-')}.mjs")
    ) + sorted(
        str(found.relative_to(REFERENCE_DIR).as_posix())
        for found in REFERENCE_DIR.rglob(f"{stem}.mjs")
    )
    hint = (
        f" A reference with this name exists at {', '.join(same_name)}: add "
        f"js_module={same_name[0]} to the module comment."
        if same_name
        else ""
    )
    raise AssertionError(
        f"{path.name}: opted in but no JavaScript module exists.{hint}"
    )


def corpus_path(dotted: str) -> Path:
    return CORPUS_DIR / f"{dotted}.json"


def _resolve(dotted: str) -> Path | None:
    """Return the source file of an orogen module, or None if it is not one."""
    parts = dotted.split(".")
    if parts[0] != "orogen":
        return None
    base = ROOT.joinpath(*parts)
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imported_names(path: Path, dotted: str) -> set[str]:
    """Return the dotted names a source file imports, relative imports resolved."""
    package = (
        dotted.split(".") if path.name == "__init__.py" else dotted.split(".")[:-1]
    )
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - (node.level - 1)] if node.level else []
            prefix = ".".join(base + ([node.module] if node.module else []))
            if not prefix:
                continue
            names.add(prefix)
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def _with_ancestors(dotted: str) -> list[str]:
    parts = dotted.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts) + 1)]


def dependency_files(path: Path) -> list[Path]:
    """Return a module's source file and the orogen files it depends on.

    Direct imports are followed transitively. Parent packages are included as
    leaves: importing a module runs their __init__.py, so an edit there matters,
    but their own imports are not followed, so editing one core module does not
    invalidate its siblings.
    """
    start = dotted_name(path)
    found: dict[str, Path] = {}
    expanded: set[str] = set()
    stack: list[tuple[str, Path, bool]] = []
    for ancestor in _with_ancestors(start)[:-1]:
        parent = _resolve(ancestor)
        if parent is not None:
            stack.append((ancestor, parent, False))
    stack.append((start, path, True))
    while stack:
        dotted, file, expand = stack.pop()
        found.setdefault(dotted, file)
        if not expand or dotted in expanded:
            continue
        expanded.add(dotted)
        for name in _imported_names(file, dotted):
            for ancestor in _with_ancestors(name):
                dependency = _resolve(ancestor)
                if dependency is not None:
                    stack.append((ancestor, dependency, ancestor == name))
    return sorted(found.values())


def input_digest(path: Path, seconds: int, seed: int) -> tuple[str, list[str]]:
    """Fingerprint everything a corpus depends on: its sources, budget and seed."""
    digest = hashlib.sha256(f"seconds={seconds} seed={seed}\n".encode())
    files: list[str] = []
    for file in dependency_files(path):
        relative = file.relative_to(ROOT).as_posix()
        digest.update(
            relative.encode() + b"\0" + hashlib.sha256(file.read_bytes()).digest()
        )
        files.append(relative)
    return digest.hexdigest(), files


def find_module(query: str) -> Path | None:
    """Resolve a full dotted name or a short stem to one opted-in module."""
    matches = [
        path
        for path in opted_in_modules()
        if dotted_name(path) == query or dotted_name(path).endswith("." + query)
    ]
    if len(matches) > 1:
        raise AssertionError(
            f"{query!r} is ambiguous: {[dotted_name(m) for m in matches]}"
        )
    return matches[0] if matches else None
