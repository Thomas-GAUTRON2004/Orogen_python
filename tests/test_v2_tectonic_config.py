import pytest

from orogen.v2.tectonics.config import (
    CM_PER_YR_TO_KM_PER_MYR,
    CM_PER_YR_TO_RAD_PER_MYR,
    MECHANISMS,
    TECTONICS,
    make_tectonics_config,
    mechanisms_off,
    mechanisms_on,
    only_mechanisms,
    validate_tectonics_config,
)


def test_upstream_unit_conversions_and_default_values():
    assert CM_PER_YR_TO_KM_PER_MYR == 10.0
    assert pytest.approx(10.0 / 6371.0) == CM_PER_YR_TO_RAD_PER_MYR
    config = make_tectonics_config()
    assert config["maxStepMyr"] == 4.0
    assert config["snapshotEveryMyr"] == 10.0
    assert config["belts"]["andean"] == {
        "baseKm": 120.0,
        "perKm": 1.0,
        "maxKm": 400.0,
    }


def test_nested_overrides_are_merged_without_mutating_defaults():
    config = make_tectonics_config({"belts": {"andean": {"maxKm": 450.0}}})
    assert config["belts"]["andean"] == {
        "baseKm": 120.0,
        "perKm": 1.0,
        "maxKm": 450.0,
    }
    assert config["belts"]["himalayan"]["maxKm"] == 1000.0
    config["belts"]["andean"]["baseKm"] = 1.0
    assert TECTONICS["belts"]["andean"]["baseKm"] == 120.0


def test_unknown_upstream_keys_are_preserved_for_incremental_porting():
    config = make_tectonics_config({"futureGroup": {"enabled": True}})
    assert config["futureGroup"] == {"enabled": True}


@pytest.mark.parametrize(
    "overrides",
    [
        {"maxStepMyr": 0},
        {"simCells": -1},
        {"mobility": float("nan")},
        {"obliqueLimit": 1.1},
        {"collisionShearShare": -0.1},
        {"beltSpeedFactorMin": 2.0, "beltSpeedFactorMax": 1.0},
        {"beltSpeedExponent": -0.1},
        {"startArcKm": [0, 0]},
        {"startArcKm": [0, 100]},
        {"startStepKm": [100, 0]},
        {"startMaxLatitudeDeg": 0},
        {"coastUnionSoftnessKm": 0},
        {"startGrowthBias": -0.1},
        {"belts": {"andean": {"baseKm": 500, "maxKm": 400}}},
    ],
)
def test_invalid_configuration_values_are_rejected(overrides):
    with pytest.raises((TypeError, ValueError)):
        make_tectonics_config(overrides)


def test_start_margin_ranges_reject_reversed_bounds():
    for key in ("startArcKm", "startStepKm"):
        with pytest.raises(ValueError, match="lower bound cannot exceed upper bound"):
            make_tectonics_config({key: [500.0, 100.0]})


def test_boolean_is_not_accepted_as_numeric_configuration():
    with pytest.raises(TypeError, match="real number"):
        validate_tectonics_config({"maxStepMyr": True})


def test_missing_optional_fields_and_empty_overrides_use_defaults():
    assert make_tectonics_config({}) == make_tectonics_config()


def test_complete_upstream_configuration_groups_are_present():
    config = make_tectonics_config()
    for group in ("initial", "belts", "geology", "farField", "reserved"):
        if group in TECTONICS:
            assert config[group] == TECTONICS[group]
    assert config["simCells"] == 40000
    assert config["cratonCount"] == 14
    assert config["geology"]["records"] is True
    assert config["farReachKm"] == 1100


def test_mechanism_helpers_match_upstream_disable_and_enable_semantics():
    assert len(MECHANISMS) == 20
    assert mechanisms_off("rifts") == {"riftRatePerMyr": 0}
    assert "riftRatePerMyr" not in only_mechanisms("rifts")
    assert only_mechanisms("rifts")["plumeRatePerMyr"] == 0
    assert mechanisms_on("rifts")["riftRatePerMyr"] == TECTONICS["riftRatePerMyr"]
    with pytest.raises(ValueError, match="unknown mechanism"):
        mechanisms_off("unknown")


def test_full_schema_rejects_invalid_nested_types_and_array_shapes():
    with pytest.raises(TypeError, match="geology.records"):
        make_tectonics_config({"geology": {"records": 1}})
    with pytest.raises(ValueError, match="cratonRiftEdges"):
        make_tectonics_config({"cratonRiftEdges": [1]})
    with pytest.raises(ValueError, match="farReachKm"):
        make_tectonics_config({"farReachKm": float("inf")})


def test_mechanism_disable_infinity_sentinel_is_accepted_by_world_config():
    config = make_tectonics_config(only_mechanisms("rifts"))
    assert config["backgroundEndMyr"] == float("inf")
    assert config["riftRatePerMyr"] == TECTONICS["riftRatePerMyr"]


def test_mechanism_retirement_infinity_is_validated_from_schema():
    from orogen.v2.tectonics.config import mechanisms_off

    config = make_tectonics_config(mechanisms_off("backgroundRetires"))
    assert config["backgroundEndMyr"] == float("inf")
    with pytest.raises(ValueError, match="must be finite"):
        make_tectonics_config({"backgroundEndMyr": float("-inf")})


def test_positive_infinity_is_only_allowed_for_mechanism_sentinels():
    assert make_tectonics_config({"backgroundEndMyr": float("inf")})[
        "backgroundEndMyr"
    ] == float("inf")
    with pytest.raises(ValueError, match="must be finite"):
        make_tectonics_config({"farReachKm": float("inf")})


def test_config_schema_keeps_tectonics_defaults_for_overlapping_mechanism_keys():
    from orogen.v2.tectonics.config import _CONFIG_SCHEMA

    assert _CONFIG_SCHEMA["riftRatePerMyr"] == TECTONICS["riftRatePerMyr"] == 0.08


def test_belt_speed_exponent_may_be_zero_but_not_negative():
    assert make_tectonics_config({"beltSpeedExponent": 0})["beltSpeedExponent"] == 0
    with pytest.raises(ValueError, match="beltSpeedExponent must be non-negative"):
        make_tectonics_config({"beltSpeedExponent": -0.1})


def test_merge_replaces_non_mapping_values_and_keeps_unknown_nested_values():
    from orogen.v2.tectonics.config import _merge

    source = {"scalar": 1, "nested": {"keep": [1, 2]}, "list": [1, 2]}
    result = _merge(
        source,
        {"scalar": {"new": True}, "nested": {"extra": 3}, "list": {"replace": 4}},
    )
    assert result == {
        "scalar": {"new": True},
        "nested": {"keep": [1, 2], "extra": 3},
        "list": {"replace": 4},
    }
    assert source == {"scalar": 1, "nested": {"keep": [1, 2]}, "list": [1, 2]}


@pytest.mark.parametrize(
    ("value", "default", "error", "message"),
    [
        (1, True, TypeError, "enabled must be a boolean"),
        (True, 1.0, TypeError, "rate must be a real number"),
        (float("-inf"), 1.0, ValueError, "rate must be finite"),
        ("wrong", [1, 2], TypeError, "range must be a list"),
        ([1], [1, 2], ValueError, "range must contain 2 values"),
        ([], {"enabled": True}, TypeError, "group must be a mapping"),
        (1, "text", TypeError, "label must be str"),
    ],
)
def test_known_shape_validation_rejects_invalid_scalar_and_container_shapes(
    value, default, error, message
):
    from orogen.v2.tectonics.config import _validate_known_shape

    with pytest.raises(error, match=message):
        _validate_known_shape(value, default, message.split()[0])


def test_known_shape_validation_ignores_unknown_nested_keys():
    from orogen.v2.tectonics.config import _validate_known_shape

    _validate_known_shape(
        {"known": True, "future": object()},
        {"known": True},
        "group",
    )


def test_validate_config_rejects_non_mapping_input():
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_tectonics_config([])


@pytest.mark.parametrize(
    ("key", "value", "error", "message"),
    [
        ("maxStepMyr", True, TypeError, "real number"),
        ("maxStepMyr", float("inf"), ValueError, "finite"),
        ("maxStepMyr", 0, ValueError, "positive"),
        ("slabPull", -0.1, ValueError, "non-negative"),
        ("obliqueLimit", 1.1, ValueError, r"in \[0, 1\]"),
        ("beltSpeedFactorMin", -0.1, ValueError, "non-negative"),
    ],
)
def test_validate_config_enforces_numeric_bounds(key, value, error, message):
    with pytest.raises(error, match=message):
        validate_tectonics_config({key: value})


def test_validate_config_accepts_boundaries_and_omits_optional_belts():
    validate_tectonics_config(
        {
            "beltSpeedExponent": 0,
            "beltSpeedFactorMin": 0,
            "beltSpeedFactorMax": 0,
            "belts": None,
        }
    )


@pytest.mark.parametrize(
    ("belts", "error", "message"),
    [
        ([], TypeError, "belts must be a mapping"),
        ({"custom": 1}, TypeError, "belts.custom must be a mapping"),
        ({"custom": {"baseKm": True}}, TypeError, "real number"),
        ({"custom": {"perKm": float("inf")}}, ValueError, "finite and non-negative"),
        ({"custom": {"maxKm": -1}}, ValueError, "finite and non-negative"),
        (
            {"custom": {"baseKm": 2, "maxKm": 1}},
            ValueError,
            "cannot exceed maxKm",
        ),
    ],
)
def test_validate_config_checks_belt_shapes_and_bounds(belts, error, message):
    with pytest.raises(error, match=message):
        validate_tectonics_config({"belts": belts})


def test_validate_config_accepts_partial_belt_records():
    validate_tectonics_config({"belts": {"custom": {"perKm": 0.5}}})


def test_make_config_rejects_non_mapping_overrides_and_base():
    with pytest.raises(TypeError, match="overrides must be a mapping"):
        make_tectonics_config([])
    with pytest.raises(TypeError, match="base config must be a mapping"):
        make_tectonics_config(base=[])


def test_make_config_merges_into_an_explicit_partial_base():
    base = {"custom": {"nested": [1]}, "simCells": 10}
    config = make_tectonics_config(
        {"custom": {"extra": True}, "simCells": 20},
        base=base,
    )
    assert config == {
        "custom": {"nested": [1], "extra": True},
        "simCells": 20,
    }
    assert base == {"custom": {"nested": [1]}, "simCells": 10}


def test_mechanism_helpers_reject_unknown_names_and_copy_nested_values():
    with pytest.raises(ValueError, match="unknown mechanism"):
        mechanisms_on("not-a-mechanism")
    with pytest.raises(ValueError, match="unknown mechanism"):
        only_mechanisms("not-a-mechanism")

    disabled = mechanisms_off("mantleHeatRifts")
    disabled["riftHeatGain"] = 999
    assert MECHANISMS["mantleHeatRifts"]["riftHeatGain"] == 0


def test_validate_config_rejects_latitude_above_ninety_degrees():
    with pytest.raises(ValueError, match="startMaxLatitudeDeg cannot exceed 90"):
        validate_tectonics_config({"startMaxLatitudeDeg": 91.0})


def test_known_shape_validation_accepts_matching_scalar_type():
    from orogen.v2.tectonics.config import _validate_known_shape

    _validate_known_shape("continental", "oceanic", "label")


@pytest.mark.parametrize("key", sorted(TECTONICS))
def test_omitted_top_level_override_keeps_upstream_default(key):
    """An absent override must leave that upstream default unchanged."""
    config = make_tectonics_config({})
    assert config[key] == TECTONICS[key]


def test_mechanism_helpers_cover_every_upstream_mechanism_field():
    """Exercise every mechanism rather than only the rifts representative."""
    for name, disabled in MECHANISMS.items():
        off = mechanisms_off(name)
        assert off == disabled
        only = only_mechanisms(name)
        for other_name, other_fields in MECHANISMS.items():
            if other_name == name:
                continue
            for key, value in other_fields.items():
                assert only[key] == value
        enabled = mechanisms_on(name)
        for key in disabled:
            assert enabled[key] == TECTONICS[key]


def test_mechanism_helpers_compose_without_losing_unrelated_fields():
    for first in MECHANISMS:
        for second in MECHANISMS:
            if first == second:
                continue
            combined = mechanisms_off(first, second)
            assert combined == {**MECHANISMS[first], **MECHANISMS[second]}
