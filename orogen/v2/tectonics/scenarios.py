"""Small deterministic tectonic scenes used by tests and development tools.

These builders deliberately keep the fixtures smaller than the interactive
JavaScript scenario catalogue. They provide controlled continental, oceanic
and mixed states while preserving the TectonicWorld and Plate contracts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.fast_noise import FastNoise
from ..core.mesh_bfs import touches
from ..core.seed import make_rng
from ..core.sphere_geometry import polar_frame, random_point, tangent_basis
from .crust import CrustFlag, CrustType
from .initial_world import _plate_outline_with_apron, build_start_world
from .kinematics import lat_lon_to_xyz
from .world import EventKind, TectonicWorld


def _xyz(sim_mesh: Any) -> np.ndarray:
    values = np.asarray(sim_mesh.xyz, dtype=np.float64).reshape((-1, 3))
    if values.shape[0] != int(sim_mesh.num_cells):
        raise ValueError("mesh xyz does not match num_cells")
    return values


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
    land = np.argpartition(cosine, -count)[-count:]
    land = land[np.argsort(-cosine[land], kind="stable")].astype(np.int32)
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
    if sectors == "all":
        return True
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
        dot = max(-1.0, min(1.0, x * centre_xyz[0] + y * centre_xyz[1] + z * centre_xyz[2]))
        distance = math.acos(dot) * 6371.0
        shaped = distance * (
            1.0 + wobble * noise.fbm(x * 2.3 + offset, y * 2.3 + offset, z * 2.3 + offset, 5)
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
    in_plate, rim = _plate_outline_with_apron(mesh, continent["isLand"].astype(bool), config)
    for cell in range(int(mesh.num_cells)):
        if not in_plate[cell] or taken[cell]:
            continue
        plate.crust.create(cell, CrustType["OCEAN"], -float(config["initialOceanAgeMyr"]))
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
    reference = tangent_basis(centre_xyz)[0]
    to_polar = polar_frame(centre_xyz, reference)
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
