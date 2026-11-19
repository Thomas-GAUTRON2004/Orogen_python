import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from orogen.v2.tectonics.geology import (
    GEO_FRAME_FIELDS,
    GEOLOGY_READS,
    GeoFlag,
    decode_age,
    decode_arc_myr,
    decode_cover_km,
    decode_exhum_km,
    decode_facies,
    decode_flood_age,
    decode_flood_myr,
    encode_age,
    encode_arc_myr,
    encode_cover_km,
    encode_exhum_km,
    encode_facies,
    encode_flood_age,
    encode_flood_myr,
    encode_unroof_km,
)


def test_shared_geology_model_metadata_matches_upstream_contract():
    assert GeoFlag == {
        "WAS_ARC": 1,
        "FORELAND": 2,
        "RIFT_BASIN": 4,
        "OPHIOLITE": 8,
        "SLAB_CHANGE": 16,
        "SHELF": 32,
        "EPEIRIC": 64,
    }
    assert GEO_FRAME_FIELDS == [
        "geoFlags",
        "coverKm",
        "coverMaxKm",
        "coverMaxAge",
        "facies",
        "exhumKm",
        "unroofKm",
        "floodMyr",
        "floodAge",
        "arcMyr",
    ]
    assert GEOLOGY_READS == ["oroStart"]


@pytest.mark.parametrize(
    ("encode", "decode", "value", "upper"),
    [
        (encode_cover_km, decode_cover_km, 2.0, 16.0),
        (encode_exhum_km, decode_exhum_km, 3.25, None),
        (encode_flood_myr, decode_flood_myr, 24.0, None),
        (encode_arc_myr, decode_arc_myr, 18.0, None),
        (encode_age, decode_age, 120.0, None),
    ],
)
def test_geology_scalar_encodings_round_trip_with_quantization(
    encode, decode, value, upper
):
    packed = encode(value)
    assert 0 <= packed <= 255
    restored = decode(packed)
    if upper is not None:
        assert 0 <= restored <= upper
    assert restored == pytest.approx(value, abs=upper / 255 if upper else 1.0)


def test_geology_encodings_clamp_and_use_javascript_half_up_rounding():
    assert encode_cover_km(-1) == 0
    assert encode_exhum_km(0.125) == 1
    assert encode_unroof_km(-1000) == -127
    assert encode_unroof_km(1000) == 127
    assert encode_flood_myr(1.0) == 1
    assert encode_arc_myr(1.0) == 1
    assert encode_age(4.0) == 1


def test_flood_age_uses_zero_as_never_and_infinity_on_decode():
    assert encode_flood_age(0, 100, False) == 0
    assert math.isinf(decode_flood_age(0))
    assert encode_flood_age(90, 100, True) == 3
    assert decode_flood_age(3) == 10


def test_facies_packs_four_nibbles_and_can_reuse_output():
    packed = encode_facies(1.0, 0.5, 0.25, 0.0)
    out = {"carb": 0.0, "salt": 0.0, "coal": 0.0, "organic": 1.0}
    assert decode_facies(packed, out) is out
    assert out == {
        "carb": 1.0,
        "salt": 8 / 15,
        "coal": 4 / 15,
        "organic": 0.0,
    }


def test_geology_encodings_match_upstream_javascript():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for upstream JavaScript differential tests")
    runner = Path(__file__).parent / "reference" / "v2" / "geology-encoding.mjs"
    result = subprocess.run(
        [node, str(runner)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    expected = json.loads(result.stdout)

    assert expected["flags"] == GeoFlag
    assert expected["fields"] == GEO_FRAME_FIELDS
    assert expected["reads"] == GEOLOGY_READS
    values = [-1, 0, 0.125, 0.25, 1, 2, 3.25, 8, 16, 32, 1000]
    for name, encode, decode in (
        ("cover", encode_cover_km, decode_cover_km),
        ("exhum", encode_exhum_km, decode_exhum_km),
        ("unroof", encode_unroof_km, lambda x: x * 0.25),
        ("flood", encode_flood_myr, decode_flood_myr),
        ("arc", encode_arc_myr, decode_arc_myr),
        ("age", encode_age, decode_age),
    ):
        actual = [[encode(value), decode(encode(value))] for value in values]
        assert actual == expected[name]

    times = [(0, 100, False), (90, 100, True), (0, 1275, True), (100, 0, True)]
    actual_flood_age = [
        [
            encode_flood_age(last, time, ever),
            (
                "Infinity"
                if math.isinf(decode_flood_age(encode_flood_age(last, time, ever)))
                else decode_flood_age(encode_flood_age(last, time, ever))
            ),
        ]
        for last, time, ever in times
    ]
    assert actual_flood_age == expected["floodAge"]

    facies_values = [
        (0, 0, 0, 0),
        (1, 0.5, 0.25, 0),
        (0.1, 0.3, 0.7, 1),
        (2, -1, 0.5, 0.5),
    ]
    actual_facies = [
        [encode_facies(*values), decode_facies(encode_facies(*values))]
        for values in facies_values
    ]
    assert actual_facies == expected["facies"]
