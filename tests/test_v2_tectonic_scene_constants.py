import pytest

from orogen.v2.tectonics.scene_constants import (
    ARC_TERRANE_ID,
    FAR_TERRANES,
    MARIANA_ARC_WIDTH_KM,
)


def test_scene_terrane_identifiers_are_centralized_and_unique():
    identifiers = (ARC_TERRANE_ID, *FAR_TERRANES.values())

    assert identifiers == tuple(range(900, 906))
    with pytest.raises(TypeError):
        FAR_TERRANES["tarim"] = 999


def test_mariana_arc_width_is_a_positive_physical_distance():
    assert MARIANA_ARC_WIDTH_KM > 0.0
