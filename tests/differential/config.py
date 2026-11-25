"""Registry of Python functions compared with the pinned JavaScript source."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from orogen.v2.core.rotation import (
    quat_from_axis_angle,
    quat_integrate,
    quat_mul,
    quat_normalize,
    quat_to_euler_pole,
    quat_to_mat3,
)
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


def _quaternion_operations(data: JsonObject) -> JsonObject:
    """Call the production quaternion operations for the shared input."""
    q1 = quat_from_axis_angle(*data["q1Axis"], data["q1Angle"])
    q2 = quat_from_axis_angle(*data["q2Axis"], data["q2Angle"])
    quaternion = quat_normalize(quat_mul(q1, q2))
    integrated = quat_integrate(quaternion, data["omega"], data["dt"])
    return {
        "quaternion": quaternion,
        "integrated": integrated,
        "matrix": quat_to_mat3(integrated),
    }


def _quaternion_euler_pole(data: JsonObject) -> JsonObject:
    """Return the production Euler-pole representation for the same rotation."""
    q1 = quat_from_axis_angle(*data["q1Axis"], data["q1Angle"])
    q2 = quat_from_axis_angle(*data["q2Axis"], data["q2Angle"])
    quaternion = quat_normalize(quat_mul(q1, q2))
    integrated = quat_integrate(quaternion, data["omega"], data["dt"])
    return quat_to_euler_pole(integrated)


_QUATERNION_INPUT = {
    "q1Axis": [0, 0, 1],
    "q1Angle": 3.141592653589793 / 3,
    "q2Axis": [1, 0, 0],
    "q2Angle": -3.141592653589793 / 7,
    "omega": [0.001, -0.002, 0.003],
    "dt": 12.5,
}


PAIRS = (
    DifferentialPair(
        name="Quaternion composition and integration",
        javascript_id="rotation.quaternionOperations",
        python_call=_quaternion_operations,
        cases=(DifferentialCase(id="composition-and-integration", input=_QUATERNION_INPUT),),
        atol=1e-15,
    ),
    DifferentialPair(
        name="Quaternion Euler pole conversion",
        javascript_id="rotation.eulerPole",
        python_call=_quaternion_euler_pole,
        cases=(DifferentialCase(id="integrated-rotation", input=_QUATERNION_INPUT),),
        atol=1e-13,
    ),
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
