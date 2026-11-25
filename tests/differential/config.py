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
from orogen.v2.tectonics.crust import (
    CELL_FIELDS,
    BeltSide,
    CrustFlag,
    CrustRaster,
    CrustType,
    OrogenyType,
)
from orogen.v2.tectonics.dynamics import solve3
from orogen.v2.tectonics.geology import (
    GEO_FRAME_FIELDS,
    GEOLOGY_READS,
    decode_age,
    decode_arc_myr,
    decode_cover_km,
    decode_exhum_km,
    decode_facies,
    decode_flood_age,
    decode_flood_myr,
    decode_unroof_km,
    encode_age,
    encode_arc_myr,
    encode_cover_km,
    encode_exhum_km,
    encode_facies,
    encode_flood_age,
    encode_flood_myr,
    encode_unroof_km,
    thicken_cover,
)
from orogen.v2.tectonics.heights import (
    DEFAULT_HEIGHTS,
    belt_relief_km,
    broad_profile,
    build_heights,
)
from orogen.v2.tectonics.kinematics import (
    advance_plates,
    lat_lon_to_xyz,
    omega_from_euler_pole,
    pose_at_constant_omega,
    speed_cm_yr_at,
    surface_velocity,
    xyz_to_lat_lon,
)
from orogen.v2.tectonics.occupancy import bin_index, mark_bin, may_have_crust_near
from orogen.v2.tectonics.plate import Plate, frame_rotation
from orogen.v2.tectonics.segments import fracture_coordinate
from orogen.v2.tectonics.simulation import choose_step, fastest_speed

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
        data["work"], data["thickened"], {"heights": data["heights"]}, data["amp"],
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
            DifferentialCase(id=f"height-{index}", input={**case, "heights": DEFAULT_HEIGHTS})
            for index, case in enumerate(_HEIGHT_CASES)
        ),
        atol=2e-6,
    ),
)


def _geology_codecs(data: JsonObject) -> JsonObject:
    def pairs(values, encode, decode):
        return [[encode(value), decode(encode(value))] for value in values]

    flood_age = []
    for last, time, ever in data["floodAge"]:
        encoded = encode_flood_age(last, time, ever)
        decoded = decode_flood_age(encoded)
        flood_age.append([encoded, None if math.isinf(decoded) else decoded])

    facies = []
    for values in data["facies"]:
        encoded = encode_facies(*values)
        facies.append([encoded, decode_facies(encoded)])

    return {
        "cover": pairs(data["cover"], encode_cover_km, decode_cover_km),
        "exhum": pairs(data["exhum"], encode_exhum_km, decode_exhum_km),
        "unroof": pairs(data["unroof"], encode_unroof_km, decode_unroof_km),
        "flood": pairs(data["flood"], encode_flood_myr, decode_flood_myr),
        "arc": pairs(data["arc"], encode_arc_myr, decode_arc_myr),
        "age": pairs(data["age"], encode_age, decode_age),
        "floodAge": flood_age,
        "facies": facies,
        "decodeFacies": [decode_facies(value) for value in data["decodeFacies"]],
        "fields": GEO_FRAME_FIELDS,
        "reads": GEOLOGY_READS,
    }


_CODEC_INPUT = {
    "cover": [-1.0, 0.0, 0.0001, 0.25, 1.0, 4.0, 16.0, 40.0],
    "exhum": [-1.0, 0.0, 0.1, 0.25, 1.0, 20.0, 100.0],
    "unroof": [-100.0, -31.125, -0.125, 0.0, 0.125, 31.125, 100.0],
    "flood": [0.0, 0.5, 2.0, 15.0, 120.0, 800.0],
    "arc": [0.0, 0.5, 2.0, 22.0, 130.0, 800.0],
    "age": [0.0, 1.0, 4.0, 160.0, 2048.0, 4000.0],
    "floodAge": [
        [-10.0, 10.0, False], [10.0, 10.0, True],
        [2.5, 12.5, True], [100.0, 10.0, True],
        [-100.0, 10000.0, True],
    ],
    "facies": [
        [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.25, 0.5],
        [0.33, 0.66, 0.12, 0.91], [-1.0, 2.0, 0.5, 0.5],
    ],
    "decodeFacies": [0, 0x1234, 0xFFFF, 0x8001],
}

PAIRS = PAIRS + (
    DifferentialPair(
        name="Geology frame encodings",
        javascript_id="tectonics.geologyCodecs",
        python_call=_geology_codecs,
        cases=(DifferentialCase(id="codec-boundaries", input=_CODEC_INPUT),),
        atol=1e-12,
    ),
)


def _adaptive_step(data: JsonObject) -> Any:
    plates = [
        SimpleNamespace(
            alive=case["alive"],
            omega=case["omega"],
            capCos=case["capCos"],
            capCenter=case["capCenter"],
            crust=SimpleNamespace(count=case["count"]),
        )
        for case in data["plates"]
    ]
    for plate in plates:
        plate.to_world = lambda x, y, z: (x, y, z)
    world = SimpleNamespace(
        config=data["config"],
        mesh=SimpleNamespace(edge_rad=data["edgeRad"]),
        plates=plates,
    )
    return choose_step(world)


_ADAPTIVE_STEP_INPUT = {
    "edgeRad": 0.012,
    "config": {
        "maxStepMyr": 4.0,
        "maxCellsPerStep": 1.5,
        "maxSpeedOceanic": 15.0,
    },
    "plates": [
        {
            "alive": True, "count": 4, "omega": [0.002, 0.001, -0.003],
            "capCos": 0.85, "capCenter": [0.0, 0.0, 1.0],
        },
        {
            "alive": True, "count": 2, "omega": [-0.001, 0.003, 0.002],
            "capCos": 0.72, "capCenter": [0.0, 1.0, 0.0],
        },
        {
            "alive": True, "count": 0, "omega": [0.5, 0.5, 0.5],
            "capCos": 0.9, "capCenter": [1.0, 0.0, 0.0],
        },
        {
            "alive": False, "count": 5, "omega": [1.0, 1.0, 1.0],
            "capCos": 0.9, "capCenter": [0.0, 0.0, 1.0],
        },
    ],
}

PAIRS = PAIRS + (
    DifferentialPair(
        name="Adaptive tectonic simulation step",
        javascript_id="tectonics.chooseStep",
        python_call=_adaptive_step,
        cases=(DifferentialCase(id="mixed-plate-states", input=_ADAPTIVE_STEP_INPUT),),
        atol=1e-12,
    ),
)


def _occupancy_bins(data: JsonObject) -> JsonObject:
    import numpy as np

    mesh = SimpleNamespace(
        num_cells=data["numCells"],
        xyz=np.asarray(data["xyz"], dtype=np.float32),
        edge_km=data["edgeKm"],
    )
    bins = bin_index(mesh)
    plate = SimpleNamespace(bins=None)
    mark_bin(plate, int(bins.bin_of[0]), data["terrane"], bins.count)
    return {
        "count": bins.count,
        "binOf": bins.bin_of.tolist(),
        "counts": [len(values) for values in bins.neighbours],
        "near": [
            may_have_crust_near(plate, index, data["terrane"], bins)
            for index in range(data["numCells"])
        ],
        "absent": may_have_crust_near(
            SimpleNamespace(bins=None), 0, data["terrane"], bins
        ),
    }


PAIRS = PAIRS + (
    DifferentialPair(
        name="Tectonic occupancy bins",
        javascript_id="tectonics.occupancyBins",
        python_call=_occupancy_bins,
        cases=(
            DifferentialCase(
                id="small-spherical-mesh",
                input={
                    "numCells": 4,
                    "xyz": [1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0],
                    "edgeKm": 500.0,
                    "terrane": 19,
                },
            ),
        ),
        atol=0.0,
        rtol=0.0,
    ),
)


def _plate_kinematics(data: JsonObject) -> JsonObject:
    origin = lat_lon_to_xyz(*data["origin"])
    velocity_point = lat_lon_to_xyz(*data["velocityPoint"])
    omega = omega_from_euler_pole(origin, data["eulerSpeed"])
    velocity = surface_velocity(omega, *velocity_point)
    q0 = quat_from_axis_angle(*data["poseAxis"], data["poseAngle"])
    pose = pose_at_constant_omega(q0, omega, data["poseDt"])

    class PlateState:
        def __init__(self, alive, quaternion, angular_velocity):
            self.alive = alive
            self.quat = quaternion
            self.omega = angular_velocity

        def set_quat(self, quaternion):
            self.quat = quaternion

    world = SimpleNamespace(
        plates=[
            PlateState(True, q0, omega),
            PlateState(False, q0, [1, 2, 3]),
        ],
        time=data["worldTime"],
        step=data["worldStep"],
    )
    advance_plates(world, data["advanceDt"])
    return {
        "xyz": origin,
        "latlon": xyz_to_lat_lon(*velocity_point),
        "omega": omega,
        "velocity": velocity,
        "speed": speed_cm_yr_at(omega, *velocity_point),
        "pose": pose,
        "advanced": world.plates[0].quat,
        "time": world.time,
        "step": world.step,
        "dead": world.plates[1].quat,
    }


PAIRS = PAIRS + (
    DifferentialPair(
        name="Plate kinematics",
        javascript_id="tectonics.plateKinematics",
        python_call=_plate_kinematics,
        cases=(
            DifferentialCase(
                id="euler-pole-and-time-integration",
                input={
                    "origin": [35, -120],
                    "velocityPoint": [-12, 77],
                    "eulerSpeed": 4.5,
                    "poseAxis": [0, 1, 0],
                    "poseAngle": 0.2,
                    "poseDt": 3.25,
                    "worldTime": 7,
                    "worldStep": 2,
                    "advanceDt": 1.75,
                },
            ),
        ),
        atol=1e-15,
    ),
)


def _crust_snapshot(data: JsonObject) -> JsonObject:
    crust = CrustRaster(data["size"])
    cell = data["cell"]
    crust.create(cell, data["type"], data["longitude"], data["terrane"])
    crust.arm_trench(cell, data["trenchDepth"])
    crust.record_subduction(cell, data["subduction"])
    for oro_type, work, start, last in data["orogeny"]:
        crust.record_orogeny(cell, oro_type, work, start, last)
    return {
        "fields": CELL_FIELDS,
        "count": crust.count,
        "mask": crust.mask.tolist(),
        "snapshot": crust.snapshot(cell),
        "flags": int(crust.flags[cell]),
        "slabKm": float(crust.slabKm[cell]),
        "oroType": int(crust.oroType[cell]),
        "oroWork": float(crust.oroWork[cell]),
        "oroStart": float(crust.oroStart[cell]),
        "oroLast": float(crust.oroLast[cell]),
        "flagTrench": CrustFlag["TRENCH"],
    }


_CRUST_INPUT = {
    "size": 6,
    "cell": 2,
    "type": CrustType["CONTINENT"],
    "longitude": -120.25,
    "terrane": 17,
    "trenchDepth": 35.5,
    "subduction": 2.75,
    "orogeny": [
        [OrogenyType["ANDEAN"], 8.125, 20, 15],
        [OrogenyType["HIMALAYAN"], 1.5, 25, 15],
    ],
}


def _crust_branches(data: JsonObject) -> Any:
    crust = CrustRaster(data["size"])
    for index in range(data["size"]):
        crust.create(
            index, data["baseType"], data["longitude"], data["terraneStart"] + index
        )
    for cell, oro_type, work, start, last in data["operations"]:
        crust.record_orogeny(cell, oro_type, work, start, last)
    return [crust.snapshot(index) for index in range(data["size"])]


_CRUST_BRANCH_INPUT = {
    "size": 5,
    "baseType": CrustType["CONTINENT"],
    "longitude": 0,
    "terraneStart": 1,
    "operations": [
        [0, OrogenyType["INTRAPLATE"], 2, 10, 5],
        [0, OrogenyType["ANDEAN"], 3, 12, 5],
        [1, OrogenyType["HIMALAYAN"], 4, 10, 5],
        [1, OrogenyType["ARC"], 1, 11, 5],
        [2, OrogenyType["ANDEAN"], 4, 10, 5],
        [2, OrogenyType["RIFT"], 1, 30, 5],
        [3, OrogenyType["TRANSPRESSION"], 4, 10, 5],
        [3, OrogenyType["URAL"], 1, 11, 5],
        [4, OrogenyType["ANDEAN"], 4, 10, 5],
        [4, OrogenyType["HIMALAYAN"], 1, 11, 5],
    ],
}

PAIRS = PAIRS + (
    DifferentialPair(
        name="Crust raster snapshot",
        javascript_id="tectonics.crustSnapshot",
        python_call=_crust_snapshot,
        cases=(DifferentialCase(id="trench-and-orogeny", input=_CRUST_INPUT),),
        atol=0.0,
        rtol=0.0,
    ),
    DifferentialPair(
        name="Crust orogeny branch transitions",
        javascript_id="tectonics.crustBranches",
        python_call=_crust_branches,
        cases=(DifferentialCase(id="successive-orogeny-types", input=_CRUST_BRANCH_INPUT),),
        atol=0.0,
        rtol=0.0,
    ),
)


def _plate_extents(data: JsonObject) -> JsonObject:
    import numpy as np

    q = quat_from_axis_angle(*data["qAxis"], data["qAngle"])
    results = {}
    for case in data["cases"]:
        plate = Plate(data["id"], case["numCells"], data["radius"])
        plate.set_quat(q)
        for cell in case["occupied"]:
            plate.crust.create(
                cell, case["types"][cell], case["longitude"],
                case["terraneStart"] + cell,
            )
        plate.dirty = True
        plate.refresh_extent(np.asarray(case["xyz"], dtype=np.float32))
        results[case["name"]] = {
            "stats": plate.stats,
            "count": plate.crust.count,
            "dirty": plate.dirty,
            "capCenter": plate.capCenter,
            "capCos": plate.capCos,
            "cap": plate.cap_in_world(SimpleNamespace(edge_rad=data["edgeRad"])),
        }
    return results


_EXTENT_CASES = (
    {
        "name": "empty", "occupied": [], "xyz": [1, 0, 0, 0, 1, 0],
        "types": [0, 0], "numCells": 0, "longitude": -1, "terraneStart": 1,
    },
    {
        "name": "antipodal", "occupied": [0, 1],
        "xyz": [1, 0, 0, -1, 0, 0], "types": [2, 0],
        "numCells": 2, "longitude": -1, "terraneStart": 1,
    },
    {
        "name": "mixed", "occupied": [0, 1, 2],
        "xyz": [1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0],
        "types": [2, 0, 1, 0], "numCells": 3,
        "longitude": -1, "terraneStart": 1,
    },
)

PAIRS = PAIRS + (
    DifferentialPair(
        name="Plate extent and occupied-cell statistics",
        javascript_id="tectonics.plateExtents",
        python_call=_plate_extents,
        cases=(
            DifferentialCase(
                id="empty-antipodal-and-mixed",
                input={
                    "id": 8, "radius": 2, "edgeRad": 0.015,
                    "qAxis": [0, 0, 1], "qAngle": math.pi / 3,
                    "cases": _EXTENT_CASES,
                },
            ),
        ),
        atol=2e-7,
    ),
)


def _plate_frame_transform(data: JsonObject) -> JsonObject:
    import numpy as np

    q1 = quat_from_axis_angle(*data["q1Axis"], data["q1Angle"])
    q2 = quat_from_axis_angle(*data["q2Axis"], data["q2Angle"])
    integrated = quat_integrate(
        quat_normalize(quat_mul(q1, q2)), data["omega"], data["dt"]
    )
    plate_data = data["plate"]
    plate = Plate(plate_data["id"], plate_data["numCells"], plate_data["radius"])
    plate.set_quat(integrated)
    world = plate.to_world(*plate_data["worldPoint"])
    local = plate.to_plate_frame(*world)
    xyz = np.asarray(plate_data["xyz"], dtype=np.float32)
    for cell, crust_type, longitude, terrane in plate_data["crustCells"]:
        plate.crust.create(cell, crust_type, longitude, terrane)
    plate.refresh_extent(xyz)
    other = Plate(plate_data["otherId"], plate_data["otherNumCells"])
    other.set_quat(q1)
    rotation = frame_rotation(plate, other)
    return {
        "world": world,
        "local": local,
        "matrix": plate.mat.reshape(-1),
        "stats": plate.stats,
        "capCenter": plate.capCenter,
        "capCos": plate.capCos,
        "cap": plate.cap_in_world(SimpleNamespace(edge_rad=plate_data["capEdgeRad"])),
        "frameRotation": rotation.reshape(-1),
    }


_PLATE_FRAME_INPUT = {
    "q1Axis": [0, 0, 1], "q1Angle": math.pi / 3,
    "q2Axis": [1, 0, 0], "q2Angle": -math.pi / 7,
    "omega": [0.001, -0.002, 0.003], "dt": 12.5,
    "plate": {
        "id": 3, "numCells": 6, "radius": 4.5,
        "worldPoint": [0.25, -0.5, 0.75],
        "xyz": [1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1],
        "crustCells": [[0, CrustType["CRATON"], -10, 1], [1, CrustType["OCEAN"], -5, 2]],
        "otherId": 4, "otherNumCells": 6, "capEdgeRad": 0.02,
    },
}

PAIRS = PAIRS + (
    DifferentialPair(
        name="Plate frame transform and extent",
        javascript_id="tectonics.plateFrameTransform",
        python_call=_plate_frame_transform,
        cases=(DifferentialCase(id="integrated-rotation", input=_PLATE_FRAME_INPUT),),
        atol=2e-7,
    ),
)
