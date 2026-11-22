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
            "A continent overrides an ocean plate along its leading coast: a coastal range, a valley behind it, the main volcanic range, fold-and-thrust ridges inland — all parallel to the coast. Offshore, the trench and its outer rise; the trailing coast behind has a wide shelf."
        ),
    ),
    _from_catalogue(
        MARGIN_SCENES,
        "laramide",
        id="laramide",
        label="A Laramide belt",
        about=(
            "A continent drives over a young slab, which goes in flat: deformation reaches far inland — a high interior with blocky basement ranges and a front range bracketing it, and no volcanoes (the magmatic gap)."
        ),
    ),
    _from_catalogue(
        BUDGET_SCENES,
        "india",
        id="india",
        label="A continental collision",
        about="“India” reaches “Asia” and slides under it. On Asia’s side (the upper plate): a range along the suture, the plateau of thickened crust, a back range and a fall line. On India’s side: the highest range just off the suture, fold hills, then the foreland basin. Ridges run parallel to the suture.",
    ),
    _from_catalogue(
        MARGIN_SCENES,
        "mariana",
        id="mariana",
        label="An ocean island arc",
        about="An arc on an ocean plate, rifting off behind: a chain of volcanic islands on a submarine ridge, the trench in front, a young back-arc basin behind.",
    ),
    _from_catalogue(
        MARGIN_SCENES,
        "japan",
        id="japan",
        label="A strip of coast rifts off",
        about="A strip of an active continental margin rifts off as a back-arc basin opens behind it: a Japan — a mountainous arc of continental crust with volcanoes, a sea behind it.",
    ),
    _from_catalogue(
        HOTSPOT_SCENES,
        "hawaii",
        id="hawaii",
        label="A hotspot track",
        about="A plate slides over a fixed plume: volcanic islands near the plume, older ones worn down and drowned downstream as the floor under them ages and sinks (seamounts, guyots).",
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
            "A plume head floods a continent with basalt at 12 Myr; the continent drifts slowly north for 300 Myr. Young, the province is a plateau with a lobed edge; as it wears, canyons with stepped walls cut into it and its escarpment retreats from the edge, leaving flat-topped mesas before it (Ethiopia’s ambas, the Western Ghats); old, it is mesas and buttes on a low, stepped remnant. Around it a faint grain of dykes radiates from the plume’s centre. (The Deccan, Siberian and Karoo traps.)"
        ),
    },
    _from_catalogue(
        RIFT_SCENES,
        "plume",
        id="rift",
        label="A continent splits",
        maxMyr=40,
        focus=flagged(CrustFlag["RIFT_MARGIN"]),
        about="A plume splits a continent: the new coasts are lifted into shoulders — a great escarpment facing the young ocean, sloping gently inland — and the arm that failed is a low corridor.",
    ),
    {
        "id": "cratons",
        "label": "Ranges against cratons",
        "maxMyr": 90,
        "view": (8.0, 72.0),
        "build": _build_cratons,
        "drive": update_plate_motion,
        "about": (
            "India, with a big shield close behind its leading coast, collides with Asia, whose shield lies ~900 km behind its south coast. On India’s side (the lower plate) the range’s foothills ride onto the shield’s rim, and in front of them a lowland bends the shield down; on Asia’s side the plateau reaches the shield and its back meets it in a steep front. Neither shield’s outline shows as the sim’s cells."
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
            "India collides with an Asia that has a Tarim-like shield ~1100 km behind its south coast and a Siberian one far to the north; between them an old suture runs east–west, square across the push, another north–south along it, and a failed rift strikes north-east. The push passes through the stiff shield and is taken up beyond it: the suture across it rises as a range far from the front (the Tian Shan), the one along it barely stirs, the failed rift is squeezed up into a ridge (an inverted aulacogen) — and broad, low swells and basins run parallel to the front across the continent. Far field layer: how much push each place has felt, and the axis it was squeezed along."
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
            about="The start world run for 600 Myr with every rule on (building takes about a minute). Young belts stand high and sharp, old ones low but still grained; cratons flat and low; wide shelves on passive margins, narrow ones on active margins; mid-ocean ridges and trenches in the sea.",
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
            about="The start world run for 1200 Myr with every rule on (building takes two minutes or so). The supercontinent breaks up in a burst of rifts, its pieces drift apart, and over a few hundred Myr they gather again into a new one — about 90° from the last, where the mantle goes down — which in its turn breaks up (chart: the largest landmass’s share of the continental crust, how many landmasses, the rifts).",
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
