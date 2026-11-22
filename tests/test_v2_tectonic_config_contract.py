"""Exhaustive default, belt-catalogue and mechanism-helper contract tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from orogen.v2.tectonics.config import (
    MECHANISMS,
    TECTONICS,
    make_tectonics_config,
    mechanisms_off,
    mechanisms_on,
    only_mechanisms,
)

UPSTREAM_DEFAULTS = json.loads(
    (
        Path(__file__).parent
        / "reference"
        / "v2"
        / "config"
        / "js-tectonics-defaults.json"
    ).read_text(encoding="utf-8")
)
EXPECTED_MECHANISMS = {
    "wings",
    "arcs",
    "rifts",
    "plumes",
    "subductionInit",
    "marginFounder",
    "backgroundRetires",
    "oceanSplits",
    "microplates",
    "flatSlabs",
    "backArcs",
    "marginIsles",
    "marginErosion",
    "underthrust",
    "remnantSeas",
    "arcRecycling",
    "arcMaturation",
    "mantleHeatRifts",
    "mantleFlow",
    "farField",
}


@pytest.mark.parametrize("key", sorted(UPSTREAM_DEFAULTS))
def test_each_omitted_top_level_override_keeps_upstream_default(key: str):
    # The test override deliberately omits every upstream key. Check each key
    # independently so additions to the JS config cannot be hidden by a partial assertion.
    config = make_tectonics_config({"testOnlyUnknownKey": True})
    assert key in config
    assert config[key] == UPSTREAM_DEFAULTS[key]


@pytest.mark.parametrize(
    "group",
    sorted(key for key, value in UPSTREAM_DEFAULTS.items() if isinstance(value, dict)),
)
def test_each_omitted_configuration_group_keeps_upstream_structure(group: str):
    config = make_tectonics_config({"testOnlyUnknownKey": True})
    assert config[group] == UPSTREAM_DEFAULTS[group]


def test_belts_group_has_the_exact_upstream_entries_and_fields():
    config = make_tectonics_config()
    expected = UPSTREAM_DEFAULTS["belts"]
    actual = config["belts"]

    assert list(actual) == list(expected)
    assert set(actual) == set(expected)
    for name, expected_fields in expected.items():
        assert set(actual[name]) == {"baseKm", "perKm", "maxKm"}
        assert actual[name] == expected_fields


def test_mechanism_catalogue_names_are_explicit_and_complete():
    assert set(MECHANISMS) == EXPECTED_MECHANISMS


@pytest.mark.parametrize("name", sorted(EXPECTED_MECHANISMS))
def test_mechanisms_off_covers_every_field_in_the_selected_group(name: str):
    assert mechanisms_off(name) == deepcopy(MECHANISMS[name])


@pytest.mark.parametrize("name", sorted(EXPECTED_MECHANISMS))
def test_mechanisms_on_restores_every_available_upstream_default(name: str):
    expected = {
        key: TECTONICS[key]
        for key in MECHANISMS[name]
        if key in UPSTREAM_DEFAULTS
    }
    assert mechanisms_on(name) == expected


@pytest.mark.parametrize("name", sorted(EXPECTED_MECHANISMS))
def test_only_mechanisms_disables_every_group_except_the_selected_one(name: str):
    expected = {}
    for other_name, group in MECHANISMS.items():
        if other_name != name:
            expected.update(deepcopy(group))
    assert only_mechanisms(name) == expected


def test_only_mechanisms_without_arguments_disables_the_complete_catalogue():
    expected = {}
    for group in MECHANISMS.values():
        expected.update(deepcopy(group))
    assert only_mechanisms() == expected
