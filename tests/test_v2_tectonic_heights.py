import math

import pytest

from orogen.v2.tectonics.crust import BeltSide, CrustRaster, CrustType, OrogenyType
from orogen.v2.tectonics.heights import DEFAULT_HEIGHTS, belt_relief_km, broad_profile, build_heights
from orogen.v2.tectonics.geology import thicken_cover


def test_relief_maturity_and_profiles_cover_every_belt_kind():
    assert belt_relief_km(OrogenyType["ANDEAN"], 450, DEFAULT_HEIGHTS) == pytest.approx(3.4)
    assert belt_relief_km(OrogenyType["ANDEAN"], 0, DEFAULT_HEIGHTS) == pytest.approx(3.4 * 0.55)
    for kind in (
        OrogenyType["ANDEAN"], OrogenyType["LARAMIDE"], OrogenyType["URAL"],
        OrogenyType["HIMALAYAN"], OrogenyType["ARC"], OrogenyType["RIFT"],
        OrogenyType["TRANSPRESSION"], OrogenyType["INTRAPLATE"],
    ):
        assert math.isfinite(broad_profile(kind, BeltSide["UPPER"], 200, 1000))
        assert math.isfinite(broad_profile(kind, BeltSide["LOWER"], 200, 1000))
    assert math.isfinite(broad_profile(OrogenyType["RIFT"], BeltSide["UPPER"], 200, 1000, wear_g=5000))
    assert math.isfinite(broad_profile(OrogenyType["ANDEAN"], BeltSide["UPPER"], 200, 1000, wear_g=5000))


def test_build_heights_root_uplift_rugged_and_arc_scaling():
    c = CrustRaster(2)
    c.create(0, CrustType["CONTINENT"], 0)
    c.create(1, CrustType["ARC"], 0)
    build_heights(c, 0, OrogenyType["ANDEAN"], BeltSide["UPPER"], 0.3, 450, 20, 2, {})
    build_heights(c, 1, OrogenyType["ANDEAN"], BeltSide["UPPER"], 0.3, 450, 20, 0, {})
    assert c.rootKm[0] == pytest.approx(2.5)
    assert c.upliftKm[0] > c.upliftKm[1] > 0
    assert c.rugged[0] > 0
    # A custom gain and build scale override the built-in upstream constants.
    custom = {"heights": {
        **DEFAULT_HEIGHTS,
        "rootGain": {},
        "upliftBuildKm": {**DEFAULT_HEIGHTS["upliftBuildKm"], OrogenyType["ANDEAN"]: 10},
    }}
    previous = float(c.upliftKm[0])
    build_heights(c, 0, OrogenyType["ANDEAN"], BeltSide["UPPER"], 0.3, 450, 20, 0, custom)
    assert c.upliftKm[0] > previous


def test_build_heights_can_sink_a_foreland_and_unknown_profiles_are_flat():
    c = CrustRaster(1)
    c.create(0, CrustType["CONTINENT"], 0)
    c.upliftKm[0] = 0.1
    build_heights(c, 0, OrogenyType["HIMALAYAN"], BeltSide["LOWER"], 0.86, 1000, 50, 0, {})
    assert c.upliftKm[0] < 0.1
    assert broad_profile(255, BeltSide["UPPER"], 100, 500) == 0
    c.rugged[0] = 100
    stable = float(c.upliftKm[0])
    build_heights(c, 0, OrogenyType["HIMALAYAN"], BeltSide["LOWER"], 0.86, 1000, 50, 0, {})
    assert c.upliftKm[0] < stable
    assert c.rugged[0] == 100


def test_thicken_cover_updates_maximum_and_burial():
    c = CrustRaster(2)
    c.create(0, CrustType["CONTINENT"], 0)
    c.create(1, CrustType["CONTINENT"], 0)
    c.coverKm[0] = 2
    assert thicken_cover(c, 0, 3, 12, {"crustKm": 30}) == pytest.approx(0.2)
    assert c.coverKm[0] == pytest.approx(2.2)
    assert c.coverMaxKm[0] == pytest.approx(2.2)
    assert c.coverMaxAt[0] == 12
    assert c.unroofKm[0] == pytest.approx(-0.2)
    c.coverMaxKm[0] = 10
    assert thicken_cover(c, 0, 1, 13, {"crustKm": 30}) == pytest.approx(2.2 / 30)
    assert c.coverMaxAt[0] == 12
    assert thicken_cover(c, 1, 3, 12, {"crustKm": 30}) == 0
    with pytest.raises(ValueError):
        thicken_cover(c, 0, 1, 12, {"crustKm": 0})
