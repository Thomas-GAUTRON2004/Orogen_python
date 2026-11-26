"""Explicit field-by-field default checks for every nested config group."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orogen.v2.tectonics.config import make_tectonics_config

UPSTREAM_DEFAULTS = json.loads(
    (
        Path(__file__).parent
        / "reference"
        / "v2"
        / "config"
        / "js-tectonics-defaults.json"
    ).read_text(encoding="utf-8")
)
NESTED_FIELDS = sorted(
    (group, key)
    for group, values in UPSTREAM_DEFAULTS.items()
    if isinstance(values, dict)
    for key in values
)


@pytest.mark.parametrize(("group", "key"), NESTED_FIELDS)
def test_each_absent_nested_field_keeps_its_upstream_default(group: str, key: str):
    config = make_tectonics_config({"testOnlyUnknownKey": True})
    assert group in config
    assert key in config[group]
    assert config[group][key] == UPSTREAM_DEFAULTS[group][key]
