"""Python TECTONICS defaults match the pinned upstream config (ed466f9).

The fixture is the upstream ``v2/js/tectonics/config.js`` TECTONICS object, dumped
from the submodule by ``tests/reference/v2/config/dump_config.mjs``. Every key and
value must agree; there is no tolerance.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from orogen.v2.tectonics.config import TECTONICS

REF = Path(__file__).parent / "reference" / "v2" / "config"
EXPECTED = json.loads((REF / "js-tectonics-defaults.json").read_text(encoding="utf-8"))


def _normal(value):
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, dict):
        return {key: _normal(item) for key, item in value.items()}
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def test_key_sets_are_identical():
    assert set(TECTONICS) == set(EXPECTED)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_default_value_matches_upstream(key: str):
    assert _normal(TECTONICS[key]) == _normal(EXPECTED[key])


def test_fixture_is_current_with_the_submodule():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    try:
        result = subprocess.run(
            [node, str(REF / "dump_config.mjs")],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        pytest.skip(f"upstream dump unavailable: {error}")
    assert _normal(json.loads(result.stdout)) == _normal(EXPECTED)
