#!/usr/bin/env python3
"""Find uses of typing.Any in Python annotations under a directory.

Usage:
    python scripts/find_any_types.py orogen
    python scripts/find_any_types.py orogen/v2/tectonics
    python scripts/find_any_types.py orogen --json

The command exits with status 1 when at least one Any annotation is found,
which makes it suitable for use in CI.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    annotation: str


def _import_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return local names for Any and typing module imports."""
    any_names = {"Any"}
    typing_modules = {"typing"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "typing",
            "typing_extensions",
        }:
            for alias in node.names:
                if alias.name == "Any":
                    any_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "typing":
                    typing_modules.add(alias.asname or alias.name)

    return any_names, typing_modules


def _annotation_nodes(tree: ast.Module) -> list[ast.expr]:
    annotations: list[ast.expr] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
            annotations.extend(
                arg.annotation
                for arg in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
                if arg.annotation is not None
            )
            if node.args.vararg and node.args.vararg.annotation:
                annotations.append(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                annotations.append(node.args.kwarg.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)

    return annotations


def find_in_file(path: Path, root: Path) -> tuple[list[Finding], str | None]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        return [], str(error)

    any_names, typing_modules = _import_aliases(tree)
    findings: list[Finding] = []

    for annotation in _annotation_nodes(tree):
        for node in ast.walk(annotation):
            is_any_name = isinstance(node, ast.Name) and node.id in any_names
            is_typing_any = (
                isinstance(node, ast.Attribute)
                and node.attr == "Any"
                and isinstance(node.value, ast.Name)
                and node.value.id in typing_modules
            )
            if is_any_name or is_typing_any:
                findings.append(
                    Finding(
                        path=path.relative_to(root).as_posix(),
                        line=node.lineno,
                        column=node.col_offset + 1,
                        annotation=ast.unparse(annotation),
                    )
                )

    return findings, None


def iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if not any(part in IGNORED_DIRS for part in path.parts):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List typing.Any occurrences in Python type annotations."
    )
    parser.add_argument("directory", type=Path, help="Directory to scan recursively")
    parser.add_argument(
        "--json", action="store_true", help="Print findings as JSON"
    )
    args = parser.parse_args()

    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {args.directory}")

    findings: list[Finding] = []
    errors: list[str] = []

    for path in iter_python_files(root):
        file_findings, error = find_in_file(path, root)
        findings.extend(file_findings)
        if error:
            errors.append(f"{path}: {error}")

    if args.json:
        print(json.dumps([asdict(item) for item in findings], indent=2))
    else:
        for item in findings:
            print(
                f"{item.path}:{item.line}:{item.column}: "
                f"Any in annotation: {item.annotation}"
            )
        print(f"\n{len(findings)} Any annotation(s) found.")

    for error in errors:
        print(f"Warning: {error}", file=sys.stderr)

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
