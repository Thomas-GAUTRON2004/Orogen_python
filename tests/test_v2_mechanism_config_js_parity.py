"""Check that mechanism helper overrides are applied by the upstream JS world."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from orogen.v2.tectonics.config import (
    MECHANISMS,
    TECTONICS,
    make_tectonics_config,
    mechanisms_off,
    mechanisms_on,
    only_mechanisms,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "reference" / "v2" / "scenes" / "dump_mechanism_overrides.mjs"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return "__OROGEN_POSITIVE_INFINITY__" if value > 0 else "__OROGEN_NEGATIVE_INFINITY__"
    return value


def _cases() -> dict[str, dict[str, dict[str, Any]]]:
    names = sorted(MECHANISMS)
    return {
        "mechanisms_off": {name: mechanisms_off(name) for name in names},
        "mechanisms_on": {name: mechanisms_on(name) for name in names},
        "only_mechanisms": {name: only_mechanisms(name) for name in names},
    }


def test_mechanism_helper_overrides_match_upstream_world_constructor():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    cases = _cases()
    payload = json.dumps(_json_safe(cases), separators=(",", ":"), allow_nan=False)
    result = subprocess.run(
        [node, str(RUNNER)],
        input=payload,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (
        "upstream mechanism override runner failed:\n"
        f"{result.stderr}\n{result.stdout}"
    )
    observed = json.loads(result.stdout)

    for helper_name, groups in cases.items():
        for group_name, overrides in groups.items():
            # Every mechanism helper emits top-level override values. The upstream
            # constructor applies the same keys with its actual JS object spread.
            expected_config = make_tectonics_config(overrides)
            expected = {key: _json_safe(expected_config[key]) for key in overrides}
            assert observed[helper_name][group_name] == expected, (
                f"{helper_name}({group_name!r}) differs from upstream constructor"
            )


def test_mechanism_enable_groups_are_derived_from_upstream_defaults():
    # This makes the relationship explicit: a helper may restore only fields
    # represented in the upstream TECTONICS defaults.
    for name in MECHANISMS:
        expected = {
            key: TECTONICS[key]
            for key in MECHANISMS[name]
            if key in TECTONICS
        }
        assert mechanisms_on(name) == expected
