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
        {"belts": {"andean": {"baseKm": 500, "maxKm": 400}}},
    ],
)
def test_invalid_configuration_values_are_rejected(overrides):
    with pytest.raises((TypeError, ValueError)):
        make_tectonics_config(overrides)


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
