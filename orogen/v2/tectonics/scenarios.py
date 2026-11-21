"""Small deterministic tectonic scenes used by tests and development tools.

These builders deliberately keep the fixtures smaller than the interactive
JavaScript scenario catalogue. They provide controlled continental, oceanic
and mixed states while preserving the TectonicWorld and Plate contracts.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from ..core.fast_noise import FastNoise
from ..core.array_selection import stable_top_k as _stable_top_k
from ..core.mesh_bfs import touches
from ..core.rotation import quat_identity
from ..core.seed import make_rng
from ..core.sphere_geometry import (
    angle_between,
    cross,
    dot,
    normalize,
    polar_frame,
    random_point,
    tangent_basis,
)
from .crust import CrustFlag, CrustType
from .initial_world import _plate_outline_with_apron, build_start_world
from .kinematics import (
    lat_lon_to_xyz,
    omega_from_euler_pole,
    pose_at_constant_omega,
)
from .world import EventKind, TectonicWorld


def _xyz(sim_mesh: Any) -> np.ndarray:
    values = np.asarray(sim_mesh.xyz, dtype=np.float64).reshape((-1, 3))
    if values.shape[0] != int(sim_mesh.num_cells):
        raise ValueError("mesh xyz does not match num_cells")
    return values


def heading(point: list[float] | np.ndarray, bearing_deg: float) -> list[float]:
    """Return the unit tangent at a point toward a compass bearing."""
    position = np.asarray(point, dtype=np.float64)
    north = normalize(
        [
            -position[2] * position[0],
            -position[2] * position[1],
            1.0 - position[2] * position[2],
        ]
    )
    east = cross(north, position.tolist())
    bearing = math.radians(bearing_deg)
    return normalize(
        [
            north[index] * math.cos(bearing) + east[index] * math.sin(bearing)
            for index in range(3)
        ]
    )


def omega_for(
    point: list[float] | np.ndarray,
    direction: list[float] | np.ndarray,
    speed_cm_yr: float,
) -> list[float]:
    """Angular velocity for moving a point along a tangent at a given speed."""
    pole = normalize(
        cross(
            np.asarray(point, dtype=np.float64).tolist(),
            np.asarray(direction, dtype=np.float64).tolist(),
        )
    )
    return omega_from_euler_pole(pole, speed_cm_yr)




def build_continental_scene(
    sim_mesh: Any,
    seed: int = 0,
    fraction: float = 0.30,
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Build a single continental plate against the passive background ocean."""
    if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be finite and strictly between 0 and 1")
    world = TectonicWorld(sim_mesh, seed, overrides=overrides)
    xyz = _xyz(sim_mesh)
    centre = np.asarray(random_point(make_rng(seed * 31 + 211)), dtype=np.float64)
    count = max(
        1,
        min(
            sim_mesh.num_cells - 1, int(math.floor(sim_mesh.num_cells * fraction + 0.5))
        ),
    )
    cosine = xyz @ centre
    # Stable secondary index ordering makes ties reproducible across NumPy
    # versions and platforms.
    land = _stable_top_k(cosine, count).astype(np.int32)
    land_mask = np.zeros(sim_mesh.num_cells, dtype=bool)
    land_mask[land] = True

    continent = world.add_plate("Continental scene")
    ocean = world.add_plate("Background ocean")
    ocean.isBackground = True
    continent.crust.create_many(
        land,
        CrustType["CONTINENT"],
        -float(world.config.get("initialLandAgeMyr", 2000.0)),
        1,
    )
    water = np.flatnonzero(~land_mask)
    ocean.crust.create_many(
        water, CrustType["OCEAN"], -float(world.config.get("initialOceanAgeMyr", 150.0))
    )
    for plate in world.plates:
        plate.refresh_extent(sim_mesh.xyz)
    world.terraneLetters = {1: "A"}
    world.log(
        EventKind["START"],
        "Continental fixture",
        {"plate": continent.id, "continentalCells": int(land.size)},
    )
    return {
        "world": world,
        "start": {
            "scenario": "continental",
            "centre": centre.tolist(),
            "continentId": continent.id,
            "oceanId": ocean.id,
            "landCells": int(land.size),
            "continentalFraction": land.size / sim_mesh.num_cells,
        },
    }


def build_oceanic_scene(
    sim_mesh: Any, seed: int = 0, overrides: dict | None = None
) -> dict[str, Any]:
    """Build two oceanic plates separated by a deterministic great-circle boundary."""
    world = TectonicWorld(sim_mesh, seed, overrides=overrides)
    xyz = _xyz(sim_mesh)
    pole = np.asarray(random_point(make_rng(seed * 31 + 307)), dtype=np.float64)
    signed = xyz @ pole
    first = np.flatnonzero(signed >= 0.0).astype(np.int32)
    second = np.flatnonzero(signed < 0.0).astype(np.int32)
    if not first.size or not second.size:
        order = np.argsort(signed, kind="stable")
        midpoint = max(1, len(order) // 2)
        second, first = (
            order[:midpoint].astype(np.int32),
            order[midpoint:].astype(np.int32),
        )

    age = -float(world.config.get("initialOceanAgeMyr", 150.0))
    plate_a = world.add_plate("Oceanic plate A")
    plate_b = world.add_plate("Oceanic plate B")
    plate_a.crust.create_many(first, CrustType["OCEAN"], age)
    plate_b.crust.create_many(second, CrustType["OCEAN"], age)
    for plate in world.plates:
        plate.refresh_extent(sim_mesh.xyz)
    world.log(
        EventKind["START"],
        "Two-plate oceanic fixture",
        {"plates": [plate_a.id, plate_b.id]},
    )
    return {
        "world": world,
        "start": {
            "scenario": "oceanic",
            "boundaryPole": pole.tolist(),
            "plateA": plate_a.id,
            "plateB": plate_b.id,
            "plateACells": int(first.size),
            "plateBCells": int(second.size),
        },
    }


def build_mixed_scene(
    sim_mesh: Any, seed: int = 0, overrides: dict | None = None
) -> dict[str, Any]:
    """Build the full mixed supercontinent, apron and background-ocean fixture."""
    result = build_start_world(sim_mesh, seed, overrides)
    result["start"]["scenario"] = "mixed"
    return result


def build_scenario_scene(
    sim_mesh: Any,
    seed: int = 0,
    scenario: str = "mixed",
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Dispatch one of the three supported deterministic fixture builders."""
    key = str(scenario).strip().lower()
    if key == "continental":
        return build_continental_scene(sim_mesh, seed, overrides=overrides)
    if key == "oceanic":
        return build_oceanic_scene(sim_mesh, seed, overrides=overrides)
    if key == "mixed":
        return build_mixed_scene(sim_mesh, seed, overrides)
    raise ValueError("scenario must be one of: continental, oceanic, mixed")


# Public camel-case aliases for mechanically ported callers.
buildContinentalScene = build_continental_scene  # noqa: N816 -- preserve upstream V2 public spelling
buildOceanicScene = build_oceanic_scene  # noqa: N816 -- preserve upstream V2 public spelling
buildMixedScene = build_mixed_scene  # noqa: N816 -- preserve upstream V2 public spelling
buildScenarioScene = build_scenario_scene  # noqa: N816 -- preserve upstream V2 public spelling


def scene_canvas(sim_mesh: Any, config: dict) -> dict[str, Any]:
    """Create the cell-ownership canvas used by hand-built tectonic scenes."""
    return {
        "mesh": sim_mesh,
        "taken": np.zeros(int(sim_mesh.num_cells), dtype=np.uint8),
        "landAgeMyr": float(config["initialLandAgeMyr"]),
        "lastTerrane": 0,
    }


def scene_terrane(canvas: dict[str, Any]) -> int:
    """Allocate a terrane identifier unique within a hand-built scene."""
    canvas["lastTerrane"] = int(canvas["lastTerrane"]) + 1
    return int(canvas["lastTerrane"])


def _in_sectors(azimuth: float, sectors: str | list[list[float]]) -> bool:
    if isinstance(sectors, str):
        return sectors == "all"
    position = float(azimuth) % 360.0
    return any(
        (position - float(start)) % 360.0 <= (float(end) - float(start)) % 360.0
        for start, end in sectors
    )


def add_continent(
    canvas: dict[str, Any],
    noise: FastNoise,
    *,
    plate: Any,
    centre: tuple[float, float] | list[float],
    radius_km: float,
    craton_km: float | None = None,
    wobble: float = 0.22,
) -> dict[str, Any]:
    """Paint a noise-shaped continent and its central craton onto a scene plate."""
    if not math.isfinite(radius_km) or radius_km <= 0.0:
        raise ValueError("radius_km must be finite and positive")
    if not math.isfinite(wobble) or wobble < 0.0:
        raise ValueError("wobble must be finite and non-negative")
    craton_radius = radius_km * 0.4 if craton_km is None else float(craton_km)
    if not math.isfinite(craton_radius) or not 0.0 <= craton_radius <= radius_km:
        raise ValueError("craton_km must be finite and between zero and radius_km")
    mesh = canvas["mesh"]
    xyz = _xyz(mesh)
    taken = canvas["taken"]
    centre_xyz = np.asarray(lat_lon_to_xyz(float(centre[0]), float(centre[1])))
    offset = (float(centre[0]) * 0.37 + float(centre[1]) * 0.11) % 17.0
    terrane = scene_terrane(canvas)
    is_land = np.zeros(int(mesh.num_cells), dtype=np.uint8)
    age = -float(canvas["landAgeMyr"])
    for cell in range(int(mesh.num_cells)):
        if taken[cell]:
            continue
        x, y, z = (float(value) for value in xyz[cell])
        dot = max(
            -1.0, min(1.0, x * centre_xyz[0] + y * centre_xyz[1] + z * centre_xyz[2])
        )
        distance = math.acos(dot) * 6371.0
        shaped = distance * (
            1.0
            + wobble
            * noise.fbm(x * 2.3 + offset, y * 2.3 + offset, z * 2.3 + offset, 5)
        )
        if shaped > radius_km:
            continue
        kind = CrustType["CRATON"] if shaped < craton_radius else CrustType["CONTINENT"]
        plate.crust.create(cell, kind, age, terrane)
        is_land[cell] = 1
        taken[cell] = 1
    return {"centre": centre_xyz.tolist(), "isLand": is_land, "terrane": terrane}


def add_apron_and_trench(
    canvas: dict[str, Any],
    plate: Any,
    continent: dict[str, Any],
    sectors: str | list[list[float]],
    config: dict,
) -> None:
    """Add the captured ocean apron and mature trench along selected rim sectors."""
    mesh = canvas["mesh"]
    xyz = _xyz(mesh)
    taken = canvas["taken"]
    in_plate, rim = _plate_outline_with_apron(
        mesh, continent["isLand"].astype(bool), config
    )
    for cell in range(int(mesh.num_cells)):
        if not in_plate[cell] or taken[cell]:
            continue
        plate.crust.create(
            cell, CrustType["OCEAN"], -float(config["initialOceanAgeMyr"])
        )
        plate.crust.flags[cell] |= CrustFlag["APRON"]
        taken[cell] = 1
    reference = tangent_basis(continent["centre"])[0]
    to_polar = polar_frame(continent["centre"], reference)
    for cell_value in rim:
        cell = int(cell_value)
        if not plate.crust.mask[cell]:
            continue
        x, y, z = (float(value) for value in xyz[cell])
        azimuth = float(to_polar(x, y, z)["azimuth"])
        if _in_sectors(azimuth, sectors):
            plate.crust.arm_trench(cell, float(config["slabMatureKm"]))


def flag_coast_trench(
    sim_mesh: Any,
    plate: Any,
    centre_xyz: list[float] | np.ndarray,
    sectors: str | list[list[float]],
    config: dict,
) -> None:
    """Arm exposed continental coast cells as mature trench in selected sectors."""
    xyz = _xyz(sim_mesh)
    centre = tuple(float(value) for value in centre_xyz)
    reference = tangent_basis(centre)[0]
    to_polar = polar_frame(centre, reference)
    crust = plate.crust
    for cell in range(int(sim_mesh.num_cells)):
        if not crust.mask[cell] or not touches(
            sim_mesh, cell, lambda neighbour: not bool(crust.mask[neighbour])
        ):
            continue
        x, y, z = (float(value) for value in xyz[cell])
        azimuth = float(to_polar(x, y, z)["azimuth"])
        if _in_sectors(azimuth, sectors):
            crust.arm_trench(cell, float(config["slabMatureKm"]))


def add_ocean(
    canvas: dict[str, Any],
    plate: Any,
    wanted: Any,
    formed_at: float,
) -> None:
    """Attach available ocean cells selected by a world-coordinate predicate."""
    mesh = canvas["mesh"]
    xyz = _xyz(mesh)
    taken = canvas["taken"]
    for cell in range(int(mesh.num_cells)):
        if taken[cell]:
            continue
        x, y, z = (float(value) for value in xyz[cell])
        if not wanted(x, y, z):
            continue
        plate.crust.create(cell, CrustType["OCEAN"], float(formed_at))
        taken[cell] = 1


def fill_background(canvas: dict[str, Any], world: TectonicWorld, config: dict):
    """Fill every unclaimed scene cell with the passive pre-history ocean."""
    mesh = canvas["mesh"]
    ocean = world.add_plate("Pre-history ocean")
    ocean.isBackground = True
    formed_at = -float(config["initialOceanAgeMyr"])
    for cell in range(int(mesh.num_cells)):
        if not canvas["taken"][cell]:
            ocean.crust.create(cell, CrustType["OCEAN"], formed_at)
            canvas["taken"][cell] = 1
    for plate in world.plates:
        plate.refresh_extent(mesh.xyz)
    return ocean


# Mechanical camel-case aliases matching upstream scene-kit.js.
sceneCanvas = scene_canvas  # noqa: N816
sceneTerrane = scene_terrane  # noqa: N816
addContinent = add_continent  # noqa: N816
addApronAndTrench = add_apron_and_trench  # noqa: N816
flagCoastTrench = flag_coast_trench  # noqa: N816
addOcean = add_ocean  # noqa: N816
fillBackground = fill_background  # noqa: N816

RIGID_DRIFT_MAX_MYR = 300.0
WILSON_MAX_MYR = 150.0
WEST_CM_YR = 1.0
EAST_CM_YR = 6.0
WEST_REVERSES_MYR = 100.0
NUDGE_MAX_MYR = 60.0
NUDGE_CM_YR = 2.0

_RIGID_PLATES: tuple[dict[str, Any], ...] = (
    {
        "name": "Drifter",
        "centre": (20.0, -60.0),
        "poleOffsetDeg": 90.0,
        "speedCmYr": 5.0,
        "note": "translates at 5 cm/yr",
    },
    {
        "name": "Spinner",
        "centre": (-25.0, 40.0),
        "poleOffsetDeg": 0.0,
        "speedCmYr": 4.0,
        "note": "spins in place",
    },
    {
        "name": "Hinge",
        "centre": (45.0, 140.0),
        "poleOffsetDeg": 35.0,
        "speedCmYr": 3.0,
        "note": "swings like a hinge",
    },
    {
        "name": "Anchor",
        "centre": (-50.0, -150.0),
        "poleOffsetDeg": 90.0,
        "speedCmYr": 0.0,
        "note": "stationary reference continent",
    },
)
_RIGID_OCEAN: dict[str, Any] = {
    "name": "Pre-history ocean",
    "centre": (90.0, 0.0),
    "poleOffsetDeg": 90.0,
    "speedCmYr": 0.0,
    "note": "passive background",
}
_WILSON_HOME = (8.0, -40.0)
_WILSON_HINGE_POLE = (72.0, -40.0)
_WILSON_FAR_CENTRE = (2.0, 72.0)


def _clamp_dot(point: np.ndarray, vector: list[float]) -> float:
    return max(-1.0, min(1.0, float(np.dot(point, np.asarray(vector)))))


def _pole_for(centre: list[float], offset_deg: float) -> list[float]:
    if offset_deg == 0.0:
        return list(centre)
    up = [1.0, 0.0, 0.0] if abs(centre[2]) > 0.95 else [0.0, 0.0, 1.0]
    projection = dot(up, centre)
    tangent = normalize([up[i] - projection * centre[i] for i in range(3)])
    angle = math.radians(offset_deg)
    return normalize(
        [centre[i] * math.cos(angle) + tangent[i] * math.sin(angle) for i in range(3)]
    )


def build_rigid_drift_world(sim_mesh: Any, seed: int) -> TectonicWorld:
    """Build four continent-only plates with prescribed Euler-pole motion."""
    world = TectonicWorld(sim_mesh, seed)
    noise = FastNoise(seed)
    xyz = _xyz(sim_mesh)
    centres = [
        lat_lon_to_xyz(*cast(tuple[float, float], definition["centre"]))
        for definition in _RIGID_PLATES
    ]
    plates = []
    for definition, centre in zip(_RIGID_PLATES, centres, strict=True):
        plate = world.add_plate(str(definition["name"]))
        plate.scene = {
            "q0": quat_identity(),
            "def": dict(definition),
            "centre": centre,
            "pole": _pole_for(centre, float(definition["poleOffsetDeg"])),
        }
        plate.omega = omega_from_euler_pole(
            plate.scene["pole"], float(definition["speedCmYr"])
        )
        plates.append(plate)
    ocean = world.add_plate(str(_RIGID_OCEAN["name"]))
    ocean.scene = {
        "q0": quat_identity(),
        "def": dict(_RIGID_OCEAN),
        "centre": lat_lon_to_xyz(*_RIGID_OCEAN["centre"]),
        "pole": lat_lon_to_xyz(*_RIGID_OCEAN["centre"]),
    }
    ocean.isBackground = True
    for cell, point in enumerate(xyz):
        best = 0
        own_distance = math.inf
        for index, centre in enumerate(centres):
            distance = math.acos(_clamp_dot(point, centre))
            if distance < own_distance:
                own_distance, best = distance, index
        x, y, z = (float(value) for value in point)
        shape = own_distance + 0.30 * noise.fbm(
            x * 2.3 + 9.0, y * 2.3 + 9.0, z * 2.3 + 9.0, 5
        )
        if shape < 0.20:
            kind = CrustType["CRATON"]
        elif shape < 0.46:
            kind = CrustType["CONTINENT"]
        else:
            kind = CrustType["OCEAN"]
        if kind == CrustType["OCEAN"]:
            ocean.crust.create(cell, kind, -float(world.config["initialOceanAgeMyr"]))
        else:
            plates[best].crust.create(
                cell, kind, -float(world.config["initialLandAgeMyr"])
            )
    for plate in world.plates:
        plate.refresh_extent(sim_mesh.xyz)
    set_rigid_drift_time(world, 0.0)
    return world


def set_rigid_drift_time(world: TectonicWorld, time_myr: float) -> None:
    """Set every rigid-drift plate pose at an absolute time."""
    for plate in world.plates:
        scene = plate.scene
        plate.set_quat(
            pose_at_constant_omega(scene["q0"], plate.omega, float(time_myr))
        )
    world.time = float(time_myr)


def build_wilson_world(
    sim_mesh: Any,
    seed: int,
    *,
    far_centre: tuple[float, float] = _WILSON_FAR_CENTRE,
) -> TectonicWorld:
    """Build the prescribed-motion mini Wilson cycle (rift, ocean, collision)."""
    world = TectonicWorld(sim_mesh, seed)
    noise = FastNoise(seed)
    xyz = _xyz(sim_mesh)
    west = world.add_plate("West")
    east = world.add_plate("East")
    far = world.add_plate("Far")
    ocean = world.add_plate("Pre-history ocean")
    ocean.isBackground = True
    pole = lat_lon_to_xyz(*_WILSON_HINGE_POLE)
    west.scene = {
        "motion": {
            "pole": pole,
            "speedCmYr": -WEST_CM_YR,
            "note": f"hinges west at {WEST_CM_YR:g} cm/yr, then reverses at {WEST_REVERSES_MYR:g} Myr",
        }
    }
    east.scene = {
        "motion": {
            "pole": pole,
            "speedCmYr": EAST_CM_YR,
            "note": f"hinges east at {EAST_CM_YR:g} cm/yr toward Far",
        }
    }
    far.scene = {
        "motion": {"pole": pole, "speedCmYr": 0.0, "note": "stationary continent"}
    }
    ocean.scene = {
        "motion": {"pole": pole, "speedCmYr": 0.0, "note": "passive background"}
    }
    home = lat_lon_to_xyz(*_WILSON_HOME)
    far_xyz = lat_lon_to_xyz(*far_centre)
    east_axis = normalize(cross([0.0, 0.0, 1.0], home))
    north_axis = cross(home, east_axis)
    craton_w = lat_lon_to_xyz(_WILSON_HOME[0] + 4.0, _WILSON_HOME[1] - 15.0)
    craton_e = lat_lon_to_xyz(_WILSON_HOME[0] - 5.0, _WILSON_HOME[1] + 16.0)
    edge_rad = float(sim_mesh.edge_rad)
    for cell, point in enumerate(xyz):
        x, y, z = (float(value) for value in point)
        wobble = 0.26 * noise.fbm(x * 2.3 + 9.0, y * 2.3 + 9.0, z * 2.3 + 9.0, 5)
        home_distance = angle_between(point, home)
        far_distance = angle_between(point, far_xyz)
        if home_distance + wobble < 0.52:
            east_coordinate = dot(point, east_axis)
            north_coordinate = dot(point, north_axis)
            phase = north_coordinate * 3.2 + 0.4
            triangle = 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0
            zig = 0.07 * triangle
            plate = west if east_coordinate < zig else east
            craton_centre = craton_w if plate is west else craton_e
            craton = angle_between(point, craton_centre) + 0.5 * wobble < 0.17
            plate.crust.create(
                cell,
                CrustType["CRATON"] if craton else CrustType["CONTINENT"],
                -float(world.config["initialLandAgeMyr"]),
            )
            if abs(east_coordinate - zig) < 1.2 * edge_rad:
                plate.crust.flags[cell] |= CrustFlag["RIFT_MARGIN"]
        elif far_distance + wobble < 0.34:
            far.crust.create(
                cell,
                CrustType["CRATON"]
                if far_distance + 0.5 * wobble < 0.15
                else CrustType["CONTINENT"],
                -float(world.config["initialLandAgeMyr"]),
            )
        else:
            ocean.crust.create(
                cell, CrustType["OCEAN"], -float(world.config["initialOceanAgeMyr"])
            )
    return world


def drive_wilson(world: TectonicWorld) -> None:
    """Apply Wilson-scene prescribed Euler-pole speeds at the current world time."""
    for plate in world.plates:
        motion = plate.scene["motion"]
        speed = float(motion["speedCmYr"])
        if plate.name == "West" and world.time >= WEST_REVERSES_MYR:
            speed = WEST_CM_YR
        plate.omega = omega_from_euler_pole(motion["pole"], speed)


def build_nudge_scene(
    sim_mesh: Any, seed: int, overrides: dict | None = None
) -> dict[str, Any]:
    """Build the generated start world and prescribe a short whole-continent drift."""
    result = build_start_world(sim_mesh, seed, overrides)
    world, start = result["world"], result["start"]
    azimuth = math.radians(
        (float(start["passiveFrom"]) + float(start["passiveTo"])) / 2.0 + 180.0
    )
    reference = start["reference"]
    transverse = cross(start["centre"], reference)
    heading = normalize(
        [
            math.cos(azimuth) * reference[index] + math.sin(azimuth) * transverse[index]
            for index in range(3)
        ]
    )
    pole = normalize(cross(start["centre"], heading))
    continent = world.plates[int(start["continentId"])]

    def drive(current_world: TectonicWorld) -> None:
        continent.omega = omega_from_euler_pole(pole, NUDGE_CM_YR)
        _ = current_world

    return {**result, "drive": drive, "heading": heading, "pole": pole}


# Public camel-case names used by upstream scenario callers.
buildRigidDriftWorld = build_rigid_drift_world  # noqa: N816
setRigidDriftTime = set_rigid_drift_time  # noqa: N816
buildWilsonWorld = build_wilson_world  # noqa: N816
driveWilson = drive_wilson  # noqa: N816
buildNudgeScene = build_nudge_scene  # noqa: N816
