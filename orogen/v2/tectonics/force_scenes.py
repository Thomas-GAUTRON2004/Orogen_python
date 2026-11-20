"""Interactive force-model scenario catalogue ported from V2 force-scenes.js."""

from __future__ import annotations

import math
from typing import Any

from ..core.fast_noise import FastNoise
from ..core.sphere_geometry import (
    PLANET_RADIUS_KM,
    cross,
    dot,
    normalize,
)
from .initial_world import build_start_world
from .kinematics import lat_lon_to_xyz
from .scenarios import (
    add_apron_and_trench,
    add_continent,
    add_ocean,
    build_wilson_world,
    fill_background,
    flag_coast_trench,
    scene_canvas,
)
from .world import TectonicWorld

WEST = [[125.0, 235.0]]
EAST = [[-55.0, 55.0]]
WILSON_HOME = (8.0, -40.0)

def _begin(sim_mesh: Any, seed: int):
    world = TectonicWorld(sim_mesh, seed)
    return world, scene_canvas(sim_mesh, world.config), FastNoise(seed)


def _add_ocean_between(
    canvas: dict[str, Any],
    plate: Any,
    first: list[float],
    second: list[float],
    skirt_km: float,
    half_width_km: float,
) -> None:
    """Paint an old-ocean skirt around the first centre and a corridor to the second."""
    axis = normalize(cross(first, second))
    span = math.acos(max(-1.0, min(1.0, dot(first, second))))

    def wanted(x: float, y: float, z: float) -> bool:
        point = [x, y, z]
        distance = math.acos(max(-1.0, min(1.0, dot(point, first))))
        if distance * PLANET_RADIUS_KM < skirt_km:
            return True
        off_track = abs(
            math.asin(max(-1.0, min(1.0, dot(point, axis))))
        ) * PLANET_RADIUS_KM
        along = math.atan2(dot(cross(first, point), axis), dot(first, point))
        return off_track < half_width_km and 0.0 < along < span

    add_ocean(canvas, plate, wanted, -90.0)


def _one_sided(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = _begin(sim_mesh, seed)
    solo = world.add_plate("Solo")
    continent = add_continent(
        canvas,
        noise,
        plate=solo,
        centre=(5.0, -20.0),
        radius_km=2300.0,
    )
    add_apron_and_trench(canvas, solo, continent, WEST, world.config)
    fill_background(canvas, world, world.config)
    return world


def _tug(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = _begin(sim_mesh, seed)
    balanced = world.add_plate("Balanced")
    lopsided = world.add_plate("Lopsided")
    balanced_land = add_continent(
        canvas,
        noise,
        plate=balanced,
        centre=(25.0, -65.0),
        radius_km=1900.0,
        wobble=0.05,
    )
    add_apron_and_trench(
        canvas, balanced, balanced_land, [[140.0, 220.0], [-40.0, 40.0]], world.config
    )
    lopsided_land = add_continent(
        canvas,
        noise,
        plate=lopsided,
        centre=(-20.0, 55.0),
        radius_km=1900.0,
    )
    add_apron_and_trench(
        canvas, lopsided, lopsided_land, [[125.0, 235.0], [-18.0, 18.0]], world.config
    )
    fill_background(canvas, world, world.config)
    return world


def _wilson(sim_mesh: Any, seed: int) -> TectonicWorld:
    world = build_wilson_world(sim_mesh, seed, far_centre=(2.0, 26.0))
    west = next(plate for plate in world.plates if plate.name == "West")
    east = next(plate for plate in world.plates if plate.name == "East")
    home = lat_lon_to_xyz(*WILSON_HOME)
    flag_coast_trench(sim_mesh, west, home, WEST, world.config)
    flag_coast_trench(sim_mesh, east, home, EAST, world.config)
    for plate in world.plates:
        plate.refresh_extent(sim_mesh.xyz)
    return world


def _india(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = _begin(sim_mesh, seed)
    asia = world.add_plate("Asia")
    india = world.add_plate("India")
    asia_land = add_continent(
        canvas,
        noise,
        plate=asia,
        centre=(40.0, 72.0),
        radius_km=2800.0,
    )
    add_apron_and_trench(
        canvas, asia, asia_land, [[215.0, 325.0]], world.config
    )
    india_land = add_continent(
        canvas,
        noise,
        plate=india,
        centre=(-28.0, 72.0),
        radius_km=850.0,
        craton_km=400.0,
    )
    first = india_land["centre"]
    second = asia_land["centre"]
    _add_ocean_between(canvas, india, first, second, 1500.0, 1700.0)
    fill_background(canvas, world, world.config)
    return world


def _start_world(sim_mesh: Any, seed: int) -> TectonicWorld:
    return build_start_world(sim_mesh, seed)["world"]


FORCE_SCENES: list[dict[str, Any]] = [
    {
        "id": "one-sided",
        "label": "One-sided trench",
        "maxMyr": 80,
        "view": (5, -35),
        "about": (
            "A lone continent with a trench along its west coast only. Slab suction "
            "should pull it west, toward its trench, and it should settle at an "
            "active-margin speed (2–5 cm/yr)."
        ),
        "expect": {"Solo": ["active"]},
        "checklist": [
            "Solo heads west, toward its trench: the cyan suction arrow points the "
            "way it goes, and the white velocity arrows follow it.",
            "It settles at an active-margin speed (WBP: 2–5 cm/yr) and its land "
            "keeps its shape; new ocean opens behind it, in the east.",
            "Try the sliders: zero suction freezes it; more continental drag slows it.",
        ],
        "build": _one_sided,
    },
    {
        "id": "tug",
        "label": "Tug of war",
        "maxMyr": 80,
        "view": (5, -5),
        "about": (
            "Balanced has equal trenches on opposite coasts: the pulls cancel, it "
            "barely moves, and its tension reads high. Lopsided has a long trench "
            "in the west and a short one in the east, so it drifts slowly west."
        ),
        "expect": {"Balanced": ["ringed", "active"], "Lopsided": ["active", "ringed"]},
        "checklist": [
            "Balanced sits almost still for tens of Myr with its two pulls cancelling: "
            "two long cyan arrows pointing opposite ways, and tension close to 1.",
            "It is an unstable balance, so it eventually tips one way and slowly "
            "speeds up; a new trench has little slab at first, which keeps tipping slow.",
            "Lopsided drifts steadily west toward its longer trench, slower than a "
            "plate with one trench would.",
        ],
        "build": _tug,
    },
    {
        "id": "wilson",
        "label": "Self-driven Wilson cycle",
        "maxMyr": 160,
        "view": (10, -20),
        "about": (
            "The Phase 2 landmass moves under its own forces. Trenches on the two "
            "outer coasts pull West and East apart; an ocean opens between them and "
            "ridge push joins in. East then meets Far and the collision slows it."
        ),
        "expect": {
            "West": ["active"],
            "East": ["active", "colliding"],
            "Far": ["passive", "colliding", "active", "ringed"],
        },
        "checklist": [
            "The two halves separate on their own, each toward its outer trench; a "
            "ridge and new ocean open between them and ridge-push arrows appear.",
            "When East reaches Far it slows, and the two lock and move together; "
            "the overlap stays a thin seam instead of growing.",
            "Far, pushed east, starts to override the old ocean ahead of it and "
            "grows a trench of its own there.",
        ],
        "build": _wilson,
    },
    {
        "id": "india",
        "label": "India: a subducting ocean plate",
        "maxMyr": 70,
        "view": (5, 72),
        "about": (
            "Asia is a big continent with a trench along its south coast. India is "
            "a small continent on a plate that is mostly ocean, and that ocean is "
            "going down Asia's trench: slab pull should make India race north, then "
            "slow sharply when the two landmasses meet."
        ),
        "expect": {"India": ["subducting", "colliding"], "Asia": ["active", "colliding"]},
        "checklist": [
            "India, whose own ocean floor is going down Asia's trench, should be "
            "several times faster than a suction-driven continent (yellow slab-pull arrow).",
            "Its speed is set by slab pull against mantle resistance, not by a speed cap.",
            "On contact it drops sharply and the two plates lock.",
        ],
        "build": _india,
    },
    {
        "id": "start",
        "label": "The Phase 3 start world",
        "maxMyr": 60,
        "view": None,
        "about": (
            "A generated supercontinent with a trench round most of its rim. Pulled "
            "from almost every side at once, it should hardly move, but its tension "
            "should be high. That stored tension is what Phase 6 uses to decide when "
            "and where it rifts."
        ),
        "expect": {"*": ["ringed"]},
        "checklist": [
            "Pulled from nearly all sides, the supercontinent barely moves "
            "(well under 1 cm/yr), but its tension reads close to 1.",
            "Phase 6 uses that stored tension to decide when and where a "
            "supercontinent rifts.",
        ],
        "build": _start_world,
    },
]


def build_force_scene(
    sim_mesh: Any, seed: int = 0, scenario: str = "one-sided"
) -> TectonicWorld:
    """Build a named deterministic force-model scene from the upstream catalogue."""
    key = str(scenario).strip().lower()
    for definition in FORCE_SCENES:
        if definition["id"] == key:
            return definition["build"](sim_mesh, int(seed))
    available = ", ".join(definition["id"] for definition in FORCE_SCENES)
    raise ValueError(f"unknown force scenario {scenario!r}; expected one of: {available}")


# Mechanical alias matching the upstream catalogue naming.
buildForceScene = build_force_scene  # noqa: N816
