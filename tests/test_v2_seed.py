"""Focused regression tests for Orogen V2 seed parsing and random helpers."""

import pytest

from orogen.v2.core import SEEDS, is_word_seed, make_rand_int, random_seed, seed_of


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("   ", False),
        ("0", False),
        ("-17", False),
        ("3.14159", False),
        ("1e308", False),
        ("0xFF", False),
        ("0b101", False),
        ("0o17", False),
        ("Infinity", True),
        ("+Infinity", True),
        ("-Infinity", True),
        ("NaN", True),
        ("0xGG", True),
        ("1_000", True),
        ("Pangaea", True),
    ],
)
def test_is_word_seed_matches_javascript_number_classification(value, expected):
    assert is_word_seed(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        ("", 0),
        ("  ", 0),
        ("0", 0),
        ("-0", 0),
        ("+0", 0),
        (" 42 ", 42),
        ("3.125", 3.125),
        ("1e3", 1000.0),
        ("0x10", 16),
        ("0b101", 5),
        ("0o17", 15),
    ],
)
def test_seed_of_preserves_javascript_numeric_seed_forms(value, expected):
    assert seed_of(value) == expected


@pytest.mark.parametrize("value", ["Pangaea", "0xGG", "Infinity", "東京"])
def test_word_seed_hash_is_stable_and_trims_only_outer_whitespace(value):
    assert seed_of(f"  {value}  ") == seed_of(value)


def test_make_rand_int_uses_the_seeded_rng_and_rejects_non_positive_bounds():
    rand_int = make_rand_int(42)
    assert [rand_int(10) for _ in range(5)] == [4, 7, 0, 5, 6]

    with pytest.raises(ValueError, match="n must be positive"):
        rand_int(0)
    with pytest.raises(ValueError, match="n must be positive"):
        rand_int(-1)


def test_random_seed_uses_the_supported_exclusive_upper_bound(monkeypatch):
    calls = []

    def fake_randbelow(upper_bound):
        calls.append(upper_bound)
        return upper_bound - 1

    monkeypatch.setattr("orogen.v2.core.seed.secrets.randbelow", fake_randbelow)

    assert random_seed() == SEEDS - 1
    assert calls == [SEEDS]
