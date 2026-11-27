"""Structural checks for the pinned upstream tectonics reference tree."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "reference" / "v2"
TECTONICS_DIR = REFERENCE_DIR / "tectonics"
POINTER = re.compile(r'^export \* from "([^"]+)"\s*;?$', re.MULTILINE)


def _pointer_files() -> list[Path]:
    return sorted(TECTONICS_DIR.rglob("*.mjs"))


@pytest.mark.parametrize(
    "pointer_file",
    _pointer_files(),
    ids=lambda path: path.relative_to(REFERENCE_DIR).as_posix(),
)
def test_tectonics_reference_points_to_pinned_upstream(pointer_file: Path) -> None:
    source = pointer_file.read_text(encoding="utf-8")
    match = POINTER.search(source)
    assert match is not None, f"{pointer_file} must remain a one-line upstream pointer"
    target = (pointer_file.parent / match.group(1)).resolve()
    upstream_root = (ROOT / "external" / "planet_heightmap_generation").resolve()
    assert target.is_relative_to(upstream_root), (
        f"{pointer_file} escapes the pinned submodule"
    )
    assert target.is_file(), f"{pointer_file} points to missing upstream file: {target}"
