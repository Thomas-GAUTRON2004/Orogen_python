"""Deterministic seed parsing and Park–Miller RNG from Orogen V2.

Constants and arithmetic are compatible with the upstream JavaScript files.
Changing them changes generated worlds.
"""
from __future__ import annotations

import math
import secrets
from collections.abc import Callable

SEEDS = 2_147_483_646
_RNG_MODULUS = 2_147_483_647
_RNG_MULTIPLIER = 16_807


def make_rng(seed: float | int = 0) -> Callable[[], float]:
    """Return the upstream Park–Miller generator for seed."""
    state = (abs(math.floor(seed * 9301 + 49297)) % 2_147_483_646) + 1

    def random() -> float:
        nonlocal state
        state = (state * _RNG_MULTIPLIER) % _RNG_MODULUS
        return (state - 1) / 2_147_483_646

    return random


def make_rand_int(seed: float | int = 0) -> Callable[[int], int]:
    """Return a deterministic integer sampler in [0, n)."""
    random = make_rng(seed)

    def rand_int(n: int) -> int:
        if n <= 0:
            raise ValueError("n must be positive")
        return math.floor(random() * n)

    return rand_int


def is_word_seed(text: object) -> bool:
    """Whether a seed string is non-empty and does not parse as a number."""
    value = "" if text is None else str(text)
    value = value.strip()
    if not value:
        return False
    try:
        return not math.isfinite(float(value))
    except ValueError:
        return True


def seed_of(text: object) -> int | float:
    """Convert numeric text directly or hash words with upstream FNV-1a."""
    value = "" if text is None else str(text)
    value = value.strip()
    if not is_word_seed(value):
        try:
            number = float(value)
        except ValueError:
            return 0
        return number if number else 0

    h = 0x811C9DC5
    for byte in value.encode("utf-8"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h % SEEDS


def random_seed() -> int:
    """Return a seed in the supported range."""
    return secrets.randbelow(SEEDS)
