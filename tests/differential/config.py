"""Registry of Python functions compared with the pinned JavaScript source."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
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
from orogen.v2.tectonics.crust import BeltSide, CrustType, OrogenyType
from orogen.v2.tectonics.dynamics import solve3
from orogen.v2.tectonics.geology import thicken_cover
from orogen.v2.tectonics.heights import (
    DEFAULT_HEIGHTS,
    belt_relief_km,
    broad_profile,
)
from orogen.v2.tectonics.segments import fracture_coordinate
from orogen.v2.tectonics.simulation import fastest_speed

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
    "q1Angle": math.pi / 3,
    "q2Axis": [1, 0, 0],
    "q2Angle": -math.pi / 7,
    "omega": [0.001, -0.002, 0.003],
    "dt": 12.5,
}


PAIRS = (
    DifferentialPair(
        name="Quaternion composition and integration",
        javascript_id="rotation.quaternionOperations",
        python_call=_quaternion_operations,
        cases=(
            DifferentialCase(
                id="composition-and-integration",
                input=_QUATERNION_INPUT,
            ),
        ),
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


def _fracture_coordinate(data: JsonObject) -> Any:
    return fracture_coordinate(*data["point"], data["omega"])


_SEGMENT_CASES = (
    {"point": [1.0, 0.0, 0.0], "omega": [0.0, 0.0, 0.0]},
    {"point": [1.0, 0.0, 0.0], "omega": [1.0, 0.0, 0.0]},
    {"point": [0.0, 1.0, 0.0], "omega": [0.0, 0.0, 2.0]},
    {"point": [-1.0, 0.0, 0.0], "omega": [1.0, 0.0, 0.0]},
    {"point": [0.0, 0.0, 1.0], "omega": [0.2, 0.3, 0.4]},
)

PAIRS = PAIRS + (
    DifferentialPair(
        name="Fracture-zone coordinates",
        javascript_id="tectonics.fractureCoordinate",
        python_call=_fracture_coordinate,
        cases=tuple(
            DifferentialCase(id=f"segment-{index}", input=case)
            for index, case in enumerate(_SEGMENT_CASES)
        ),
        atol=1e-12,
    ),
)


def _solve3(data: JsonObject) -> Any:
    return solve3(data["matrix"], data["vector"])


_LINEAR_SYSTEM_CASES = (
    {"matrix": [3, 1, 0, 1, 4, 1, 0, 1, 2], "vector": [7, 8, 3]},
    {"matrix": [2, 0, 0, 0, 5, 0, 0, 0, 4], "vector": [6, 10, -8]},
    {"matrix": [1, 2, 3, 2, 4, 6, 0, 0, 0], "vector": [1, 2, 0]},
)

PAIRS = PAIRS + (
    DifferentialPair(
        name="Force-balance linear solver",
        javascript_id="tectonics.solve3",
        python_call=_solve3,
        cases=tuple(
            DifferentialCase(id=f"system-{index}", input=case)
            for index, case in enumerate(_LINEAR_SYSTEM_CASES)
        ),
        atol=1e-12,
    ),
)


def _fastest_speed(data: JsonObject) -> Any:
    plate = SimpleNamespace(capCos=data["capCos"])
    return fastest_speed(data["omega"], plate, data["centre"])


_SPEED_CASES = (
    {"omega": [0.0, 0.0, 0.0], "capCos": 0.8, "centre": [0, 0, 1]},
    {"omega": [0.2, -0.3, 0.4], "capCos": 0.98, "centre": [0, 0, 1]},
    {"omega": [0.7, 0.0, 0.1], "capCos": -0.2, "centre": [0, 1, 0]},
)

PAIRS = PAIRS + (
    DifferentialPair(
        name="Fastest plate speed",
        javascript_id="tectonics.fastestSpeed",
        python_call=_fastest_speed,
        cases=tuple(
            DifferentialCase(id=f"plate-speed-{index}", input=case)
            for index, case in enumerate(_SPEED_CASES)
        ),
        atol=1e-12,
    ),
)


def _thicken_cover(data: JsonObject) -> JsonObject:
    import numpy as np

    crust = SimpleNamespace(
        coverKm=np.asarray(data["coverKm"], dtype=np.float32),
        coverMaxKm=np.asarray(data["coverMaxKm"], dtype=np.float32),
        coverMaxAt=np.asarray(data["coverMaxAt"], dtype=np.float32),
        unroofKm=np.asarray(data["unroofKm"], dtype=np.float32),
    )
    thicken_cover(
        crust, 0, data["thickenedKm"], data["time"], data["geology"]
    )
    return {
        "coverKm": float(crust.coverKm[0]),
        "coverMaxKm": float(crust.coverMaxKm[0]),
        "coverMaxAt": float(crust.coverMaxAt[0]),
        "unroofKm": float(crust.unroofKm[0]),
    }


_COVER_CASES = (
    {
        "coverKm": [2.0], "coverMaxKm": [2.5], "coverMaxAt": [4.0],
        "unroofKm": [-1.0], "thickenedKm": 3.0, "time": 30.0,
        "geology": {"crustKm": 30.0},
    },
    {
        "coverKm": [8.0], "coverMaxKm": [4.0], "coverMaxAt": [10.0],
        "unroofKm": [0.5], "thickenedKm": 2.0, "time": 42.0,
        "geology": {"crustKm": 25.0},
    },
    {
        "coverKm": [0.0], "coverMaxKm": [0.0], "coverMaxAt": [0.0],
        "unroofKm": [0.0], "thickenedKm": 4.0, "time": 50.0,
        "geology": {"crustKm": 30.0},
    },
)

PAIRS = PAIRS + (
    DifferentialPair(
        name="Geology cover thickening",
        javascript_id="tectonics.thickenCover",
        python_call=_thicken_cover,
        cases=tuple(
            DifferentialCase(id=f"cover-{index}", input=case)
            for index, case in enumerate(_COVER_CASES)
        ),
        atol=2e-6,
    ),
)


def _broad_profile(data: JsonObject) -> Any:
    return broad_profile(
        data["type"], data["side"], data["distance"], data["reach"],
        data["jitter"], data["wear"],
    )


def _belt_relief(data: JsonObject) -> Any:
    return belt_relief_km(data["type"], data["reach"], data["heights"])


_PROFILE_CASES = tuple(
    {
        "type": oro_type,
        "side": side,
        "distance": distance,
        "reach": reach,
        "jitter": 0.08 if (oro_type + side) % 2 else 0.0,
        "wear": 180.0 if side == BeltSide["LOWER"] else 0.0,
    }
    for oro_type in range(1, 9)
    for side, distance, reach in (
        (BeltSide["UPPER"], 0.18 * 500.0, 500.0),
        (BeltSide["LOWER"], 0.76 * 850.0, 850.0),
    )
)
_RELIEF_CASES = tuple(
    {"type": oro_type, "reach": reach, "heights": DEFAULT_HEIGHTS}
    for oro_type in range(1, 9)
    for reach in (
        0.0,
        0.5 * DEFAULT_HEIGHTS["fullReachKm"][oro_type],
        1.5 * DEFAULT_HEIGHTS["fullReachKm"][oro_type],
    )
)

PAIRS = PAIRS + (
    DifferentialPair(
        name="Orogenic belt profile",
        javascript_id="tectonics.broadProfile",
        python_call=_broad_profile,
        cases=tuple(
            DifferentialCase(id=f"profile-{index}", input=case)
            for index, case in enumerate(_PROFILE_CASES)
        ),
        atol=2e-12,
    ),
    DifferentialPair(
        name="Orogenic belt relief",
        javascript_id="tectonics.beltReliefKm",
        python_call=_belt_relief,
        cases=tuple(
            DifferentialCase(id=f"relief-{index}", input=case)
            for index, case in enumerate(_RELIEF_CASES)
        ),
        atol=2e-12,
    ),
)


from orogen.v2.tectonics.heights import build_heights


def _build_heights(data: JsonObject) -> JsonObject:
    import numpy as np

    crust = SimpleNamespace(
        type=np.asarray([data["crustType"]], dtype=np.uint8),
        rootKm=np.asarray([data["rootKm"]], dtype=np.float32),
        upliftKm=np.asarray([data["upliftKm"]], dtype=np.float32),
        rugged=np.asarray([data["rugged"]], dtype=np.float32),
    )
    build_heights(
        crust, 0, data["type"], data["side"], data["u"], data["reach"],
        data["work"], data["thickened"], {"heights": DEFAULT_HEIGHTS}, data["amp"],
    )
    return {
        "rootKm": float(crust.rootKm[0]),
        "upliftKm": float(crust.upliftKm[0]),
        "rugged": float(crust.rugged[0]),
    }


_HEIGHT_CASES = (
    {
        "type": OrogenyType["ANDEAN"], "side": BeltSide["LOWER"], "u": 0.28, "reach": 620.0,
        "work": 65.0, "thickened": 2.4, "amp": 1.0,
        "crustType": CrustType["CONTINENT"], "rootKm": 1.2, "upliftKm": 0.1, "rugged": 0.2,
    },
    {
        "type": OrogenyType["URAL"], "side": BeltSide["LOWER"], "u": 0.82, "reach": 320.0,
        "work": 90.0, "thickened": 1.1, "amp": 0.85,
        "crustType": CrustType["CONTINENT"], "rootKm": 0.5, "upliftKm": -0.02, "rugged": 0.05,
    },
    {
        "type": OrogenyType["ARC"], "side": BeltSide["UPPER"], "u": 0.2, "reach": 240.0,
        "work": 30.0, "thickened": 0.7, "amp": 0.75,
        "crustType": CrustType["ARC"], "rootKm": 0.0, "upliftKm": 0.0, "rugged": 0.0,
    },
    {
        "type": OrogenyType["RIFT"], "side": BeltSide["UPPER"], "u": 0.48, "reach": 260.0,
        "work": 18.0, "thickened": 0.9, "amp": 1.1,
        "crustType": CrustType["CONTINENT"], "rootKm": 0.2, "upliftKm": 0.0, "rugged": 0.1,
    },
    {
        "type": OrogenyType["INTRAPLATE"], "side": BeltSide["LOWER"], "u": 0.64, "reach": 410.0,
        "work": 52.0, "thickened": 1.6, "amp": 0.9,
        "crustType": CrustType["CRATON"], "rootKm": 2.0, "upliftKm": 0.4, "rugged": 0.6,
    },
)

PAIRS = PAIRS + (
    DifferentialPair(
        name="Belt height accumulation",
        javascript_id="tectonics.buildHeights",
        python_call=_build_heights,
        cases=tuple(
            DifferentialCase(id=f"height-{index}", input=case)
            for index, case in enumerate(_HEIGHT_CASES)
        ),
        atol=2e-6,
    ),
)
