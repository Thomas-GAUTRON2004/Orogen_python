"""Registry of Python functions compared with the pinned JavaScript source."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from orogen.v2.core.seed import is_word_seed, make_rng, seed_of


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class DifferentialCase:
    """One deterministic input shared by the Python and JavaScript sides."""

    id: str
    input: JsonObject


@dataclass(frozen=True)
class DifferentialPair:
    """An explicit Python/JavaScript function pair and its test cases."""

    name: str
    javascript_id: str
    python_call: Callable[[JsonObject], Any]
    cases: tuple[DifferentialCase, ...]
    rtol: float = 2e-7
    atol: float = 2e-9


def _seed_and_rng(data: JsonObject) -> JsonObject:
    """Call the production seed helpers with the case's input."""
    rng = make_rng(data["rngSeed"])
    return {
        "seeds": [
            {
                "text": value,
                "word": is_word_seed(value),
                "seed": seed_of(value),
            }
            for value in data["texts"]
        ],
        "random": [rng() for _ in range(data["randomCount"])],
    }


PAIRS = (
    DifferentialPair(
        name="Seed parsing and Park-Miller RNG",
        javascript_id="seed.seedAndRng",
        python_call=_seed_and_rng,
        cases=(
            DifferentialCase(
                id="representative-seeds",
                input={
                    "texts": [
                        "",
                        "0",
                        "42",
                        "-17",
                        "3.14159",
                        "0x10",
                        "hello",
                        "Orogen 3.0.1",
                        "é",
                    ],
                    "rngSeed": 123456,
                    "randomCount": 12,
                },
            ),
        ),
        rtol=0.0,
        atol=0.0,
    ),
)
