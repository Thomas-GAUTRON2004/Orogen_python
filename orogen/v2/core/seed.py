"""Deterministic seed parsing and Park–Miller RNG from Orogen V2.

Constants and arithmetic are compatible with the upstream JavaScript files.
Changing them changes generated worlds.
"""

# differential-module: mode=exact

from __future__ import annotations

import math
import re
import secrets
from collections.abc import Callable

SeedInput = str | int | float | bool | None

SEEDS = 2_147_483_646
_RNG_MODULUS = 2_147_483_647
_RNG_MULTIPLIER = 16_807
_DECIMAL_NUMBER = re.compile(r"^[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?$")


def _js_number(text: str) -> float | None:
    """Parse the numeric string forms accepted by JavaScript Number()."""
    if text == "":
        return 0.0
    if text in ("Infinity", "+Infinity", "-Infinity"):
        return math.inf if not text.startswith("-") else -math.inf
    try:
        if text.startswith(("0x", "0X")):
            return float(int(text[2:], 16))
        if text.startswith(("0b", "0B")):
            return float(int(text[2:], 2))
        if text.startswith(("0o", "0O")):
            return float(int(text[2:], 8))
        if _DECIMAL_NUMBER.fullmatch(text):
            return float(text)
    except ValueError:
        return None
    return None


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


def is_word_seed(text: SeedInput) -> bool:
    """Determine whether a seed string is non-empty and not finite under JavaScript Number()."""
    value = "" if text is None else str(text)
    value = value.strip()
    if not value:
        return False
    number = _js_number(value)
    return number is None or not math.isfinite(number)


def seed_of(text: SeedInput) -> int | float:
    """Convert numeric text directly or hash words with upstream FNV-1a."""
    value = "" if text is None else str(text)
    value = value.strip()
    if not is_word_seed(value):
        number = _js_number(value)
        return number if number else 0

    h = 0x811C9DC5
    for byte in value.encode("utf-8"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h % SEEDS


# differential: skip
def random_seed() -> int:
    """Return a seed in the supported range."""
    return secrets.randbelow(SEEDS)
