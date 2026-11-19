import pytest

from orogen.v2.tectonics.config import (
    CM_PER_YR_TO_KM_PER_MYR,
    CM_PER_YR_TO_RAD_PER_MYR,
    TECTONICS,
    make_tectonics_config,
    validate_tectonics_config,
)


def test_upstream_unit_conversions_and_default_values():
    assert CM_PER_YR_TO_KM_PER_MYR == 10.0
    assert CM_PER_YR_TO_RAD_PER_MYR == pytest.approx(10.0 / 6371.0)
    config = make_tectonics_config()
    assert config["maxStepMyr"] == 4.0
    assert config["snapshotEveryMyr"] == 10.0
    assert config["belts"]["andean"] == {
        "baseKm": 120.0, "perKm": 1.0, "maxKm": 400.0
    }


def test_nested_overrides_are_merged_without_mutating_defaults():
    config = make_tectonics_config({"belts": {"andean": {"maxKm": 450.0}}})
    assert config["belts"]["andean"] == {
        "baseKm": 120.0, "perKm": 1.0, "maxKm": 450.0
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
