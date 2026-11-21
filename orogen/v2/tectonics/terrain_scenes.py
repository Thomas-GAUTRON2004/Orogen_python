"""Phase 8 terrain scene catalogue ported from V2 terrain-scenes.js."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from ..core.sphere_geometry import PLANET_RADIUS_KM, cross, dot, normalize
from .budget_scenes import BUDGET_SCENES
from .config import mechanisms_on, only_mechanisms
from .crust import CrustFlag, CrustType
from .dynamics import update_plate_motion
from .hotspot_scenes import HOTSPOT_SCENES
from .kinematics import lat_lon_to_xyz, xyz_to_lat_lon
from .margin_scenes import MARGIN_SCENES, heading, omega_for
from .rift_scenes import RIFT_SCENES
from .scenarios import (
    add_apron_and_trench,
    add_continent,
    fill_background,
)
from .scene_constants import FAR_TERRANES
from .scene_support import add_ocean_corridor, begin_scene
from .world import TectonicWorld

WHOLE_PLANET_PACE = MappingProxyType({"maxCellsPerStep": 2.25, "slowRulesEveryMyr": 3})


def flagged(
    flag: int,
) -> Callable[[Mapping[str, Any], Any], tuple[float, float] | None]:
    """Return a focus callback for the centroid of land carrying a crust flag."""

    def focus(frame: Mapping[str, Any], sim_mesh: Any) -> tuple[float, float] | None:
        owner = np.asarray(frame["owner"])
        cell_type = np.asarray(frame["type"])
        flags = np.asarray(frame["flags"])
        selected = (
            (owner >= 0)
            & (cell_type != CrustType["OCEAN"])
            & ((flags & int(flag)) != 0)
        )
        if not np.any(selected):
            return None
        xyz = np.asarray(sim_mesh.xyz, dtype=np.float64).reshape((-1, 3))
        centre = xyz[selected].sum(axis=0)
        length = float(np.linalg.norm(centre))
        if length <= 0.0:
            return None
        ll = xyz_to_lat_lon(*(centre / length).tolist())
        return float(ll["lat"]), float(ll["lon"])

    return focus


def _from_catalogue(
    catalogue: list[dict[str, Any]], scene_id: str, **overrides: Any
) -> dict[str, Any]:
    scene = next(item for item in catalogue if item["id"] == scene_id)
    return {**scene, "drive": update_plate_motion, **overrides}


def _build_cratons(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = begin_scene(sim_mesh, seed)
    asia = world.add_plate("Asia")
    india = world.add_plate("India")
    asia_land = add_continent(
        canvas,
        noise,
        plate=asia,
        centre=(40.0, 72.0),
        radius_km=2800.0,
        craton_km=1900.0,
    )
    add_apron_and_trench(canvas, asia, asia_land, [[215.0, 325.0]], world.config)
    india_land = add_continent(
        canvas,
        noise,
        plate=india,
        centre=(-28.0, 72.0),
        radius_km=1300.0,
        craton_km=950.0,
    )
    add_ocean_corridor(
        canvas,
        india,
        india_land["centre"],
        asia_land["centre"],
        1900.0,
        1900.0,
    )
    fill_background(canvas, world, world.config)
    world.config.update(only_mechanisms("underthrust"))
    world.config.update(mechanisms_on("underthrust"))
    return world


def _build_farfield(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = begin_scene(sim_mesh, seed)
    asia = world.add_plate("Asia")
    india = world.add_plate("India")
    asia_land = add_continent(
        canvas,
        noise,
        plate=asia,
        centre=(40.0, 72.0),
        radius_km=3000.0,
        craton_km=0.0,
    )
    xyz = np.asarray(sim_mesh.xyz, dtype=np.float64).reshape((-1, 3))
    crust = asia.crust

    def within(latitude: float, longitude: float, radius_km: float):
        centre = lat_lon_to_xyz(latitude, longitude)

        def contains(cell: int) -> bool:
            distance = math.acos(max(-1.0, min(1.0, float(np.dot(xyz[cell], centre)))))
            return distance * PLANET_RADIUS_KM < radius_km

        return contains

    def band(
        first_ll: tuple[float, float], second_ll: tuple[float, float], half_km: float
    ):
        first = lat_lon_to_xyz(*first_ll)
        second = lat_lon_to_xyz(*second_ll)
        axis = normalize(cross(first, second))
        span = math.acos(max(-1.0, min(1.0, dot(first, second))))
        half = half_km / PLANET_RADIUS_KM

        def contains(cell: int) -> bool:
            point = xyz[cell].tolist()
            lateral = (
                abs(math.asin(max(-1.0, min(1.0, dot(point, axis))))) * PLANET_RADIUS_KM
            )
            if lateral > half_km:
                return False
            along = math.atan2(dot(cross(first, point), axis), dot(first, point))
            return -half < along < span + half

        return contains

    tarim = within(25.0, 72.0, 550.0)
    siberia = within(53.0, 72.0, 800.0)
    across_suture = band((35.0, 58.0), (35.5, 86.0), 80.0)
    along_suture = band((22.0, 94.0), (48.0, 94.0), 80.0)
    aulacogen = band((27.0, 50.0), (37.0, 61.0), 80.0)

    for cell in range(int(sim_mesh.num_cells)):
        if not crust.mask[cell]:
            continue
        if tarim(cell):
            crust.type[cell] = CrustType["CRATON"]
            crust.terrane[cell] = FAR_TERRANES["tarim"]
            continue
        if siberia(cell):
            crust.type[cell] = CrustType["CRATON"]
            crust.terrane[cell] = FAR_TERRANES["siberia"]
            continue
        if across_suture(cell):
            crust.flags[cell] |= CrustFlag["SUTURE"]
            crust.terrane[cell] = FAR_TERRANES["across"]
        elif along_suture(cell):
            crust.flags[cell] |= CrustFlag["SUTURE"]
            crust.terrane[cell] = FAR_TERRANES["along"]
        elif aulacogen(cell):
            crust.flags[cell] |= CrustFlag["FAILED_RIFT"]
            crust.terrane[cell] = FAR_TERRANES["aulacogen"]

    add_apron_and_trench(canvas, asia, asia_land, [[215.0, 325.0]], world.config)
    india_land = add_continent(
        canvas,
        noise,
        plate=india,
        centre=(-28.0, 72.0),
        radius_km=1300.0,
        craton_km=950.0,
    )
    add_ocean_corridor(
        canvas,
        india,
        india_land["centre"],
        asia_land["centre"],
        1900.0,
        1900.0,
    )
    fill_background(canvas, world, world.config)
    world.config.update(only_mechanisms("underthrust", "farField"))
    world.config.update(mechanisms_on("underthrust", "farField"))
    return world


def _build_traps(sim_mesh: Any, seed: int) -> TectonicWorld:
    world = next(scene for scene in HOTSPOT_SCENES if scene["id"] == "deccan")["build"](
        sim_mesh, seed
    )
    for plume in world.plumes:
        plume["lifeMyr"] = 15.0
    return world


def _drive_traps(world: TectonicWorld) -> None:
    india = next(plate for plate in world.plates if plate.name == "Indica")
    point = lat_lon_to_xyz(-30.0, 75.0)
    india.omega = omega_for(point, heading(point, 0.0), 1.5)


def _whole_planet(scene: dict[str, Any]) -> dict[str, Any]:
    def build(sim_mesh: Any, seed: int) -> TectonicWorld:
        world = scene["build"](sim_mesh, seed)
        world.config.update(WHOLE_PLANET_PACE)
        return world

    return {**scene, "build": build}


TERRAIN_SCENES: list[dict[str, Any]] = [
    _from_catalogue(
        MARGIN_SCENES,
        "margins",
        id="andes",
        label="An Andean margin",
        about=(
            "An ocean plate subducts along a continent’s leading coast: a coastal "
            "range, a valley behind it, the main volcanic range and inland "
            "fold-and-thrust ridges, all parallel to the coast."
        ),
    ),
    _from_catalogue(
        MARGIN_SCENES,
        "laramide",
        id="laramide",
        label="A Laramide belt",
        about=(
            "A continent drives over a young, flat slab. Deformation reaches far "
            "inland, producing a high interior with blocky basement ranges and a "
            "front range but no volcanic chain."
        ),
    ),
    _from_catalogue(
        BUDGET_SCENES,
        "india",
        id="india",
        label="A continental collision",
    ),
    _from_catalogue(
        MARGIN_SCENES,
        "mariana",
        id="mariana",
        label="An ocean island arc",
    ),
    _from_catalogue(
        MARGIN_SCENES,
        "japan",
        id="japan",
        label="A strip of coast rifts off",
    ),
    _from_catalogue(
        HOTSPOT_SCENES,
        "hawaii",
        id="hawaii",
        label="A hotspot track",
    ),
    {
        "id": "traps",
        "label": "Flood basalts through time",
        "maxMyr": 312,
        "view": (-15.0, 75.0),
        "focus": flagged(CrustFlag["LIP"]),
        "build": _build_traps,
        "drive": _drive_traps,
        "about": (
            "A plume head floods a continent with basalt at 12 Myr; the continent "
            "drifts slowly north for 300 Myr. The province erodes from a plateau "
            "to escarpments, mesas and buttes while a short-lived tail avoids "
            "leaving a long volcanic chain."
        ),
    },
    _from_catalogue(
        RIFT_SCENES,
        "plume",
        id="rift",
        label="A continent splits",
        maxMyr=40,
        focus=flagged(CrustFlag["RIFT_MARGIN"]),
    ),
    {
        "id": "cratons",
        "label": "Ranges against cratons",
        "maxMyr": 90,
        "view": (8.0, 72.0),
        "build": _build_cratons,
        "drive": update_plate_motion,
        "about": (
            "India collides with Asia, whose shields lie behind the southern "
            "collision front. The plateau reaches the Asian shield while the "
            "lower-plate foothills ride onto India’s cratonic rim."
        ),
    },
    {
        "id": "farfield",
        "label": "A collision felt far inland",
        "maxMyr": 140,
        "view": (30.0, 72.0),
        "build": _build_farfield,
        "drive": update_plate_motion,
        "about": (
            "A collision pushes against two cratonic blocks, transverse and "
            "longitudinal sutures and a failed rift. The far-field stress reaches "
            "beyond the strong block and builds a distant range."
        ),
    },
    _whole_planet(
        _from_catalogue(
            BUDGET_SCENES,
            "history",
            id="history",
            label="A whole planet (600 Myr)",
            maxMyr=600,
            endings=True,
        )
    ),
    _whole_planet(
        _from_catalogue(
            BUDGET_SCENES,
            "history",
            id="cycle",
            label="The supercontinent cycle (1200 Myr)",
            maxMyr=1200,
            endings=True,
        )
    ),
]


def build_terrain_scene(
    sim_mesh: Any, seed: int = 0, scenario: str = "andes"
) -> TectonicWorld:
    """Build a named deterministic terrain scene from the upstream catalogue."""
    key = str(scenario).strip().lower()
    for definition in TERRAIN_SCENES:
        if definition["id"] == key:
            return definition["build"](sim_mesh, int(seed))
    available = ", ".join(definition["id"] for definition in TERRAIN_SCENES)
    raise ValueError(
        f"unknown terrain scenario {scenario!r}; expected one of: {available}"
    )


# Mechanical alias matching the upstream catalogue naming.
buildTerrainScene = build_terrain_scene  # noqa: N816
