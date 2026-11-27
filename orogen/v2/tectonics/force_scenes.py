"""Interactive force-model scenario catalogue ported from V2 force-scenes.js."""

from __future__ import annotations

from typing import Any

from ..core.sim_mesh import SimMesh
from .initial_world import build_start_world
from .kinematics import lat_lon_to_xyz
from .scenarios import (
    add_apron_and_trench,
    add_continent,
    build_wilson_world,
    fill_background,
    flag_coast_trench,
)
from .scene_support import add_ocean_corridor, begin_scene
from .world import TectonicWorld

WEST: tuple[tuple[float, float], ...] = ((125.0, 235.0),)
EAST: tuple[tuple[float, float], ...] = ((-55.0, 55.0),)
WILSON_HOME = (8.0, -40.0)


# Compatibility names; new scenario modules import the shared public helpers.
_begin = begin_scene
_add_ocean_between = add_ocean_corridor


def _one_sided(sim_mesh: SimMesh, seed: int) -> TectonicWorld:
    world, canvas, noise = begin_scene(sim_mesh, seed)
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


def _tug(sim_mesh: SimMesh, seed: int) -> TectonicWorld:
    world, canvas, noise = begin_scene(sim_mesh, seed)
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


def _wilson(sim_mesh: SimMesh, seed: int) -> TectonicWorld:
    world = build_wilson_world(sim_mesh, seed, far_centre=(2.0, 26.0))
    west = next(plate for plate in world.plates if plate.name == "West")
    east = next(plate for plate in world.plates if plate.name == "East")
    home = lat_lon_to_xyz(*WILSON_HOME)
    flag_coast_trench(sim_mesh, west, home, WEST, world.config)
    flag_coast_trench(sim_mesh, east, home, EAST, world.config)
    for plate in world.plates:
        plate.refresh_extent(sim_mesh.xyz)
    return world


def _india(sim_mesh: SimMesh, seed: int) -> TectonicWorld:
    world, canvas, noise = begin_scene(sim_mesh, seed)
    asia = world.add_plate("Asia")
    india = world.add_plate("India")
    asia_land = add_continent(
        canvas,
        noise,
        plate=asia,
        centre=(40.0, 72.0),
        radius_km=2800.0,
    )
    add_apron_and_trench(canvas, asia, asia_land, [[215.0, 325.0]], world.config)
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
    add_ocean_corridor(canvas, india, first, second, 1500.0, 1700.0)
    fill_background(canvas, world, world.config)
    return world


def _start_world(sim_mesh: SimMesh, seed: int) -> TectonicWorld:
    """Build a fresh mutable start world instead of reusing simulation state."""
    return build_start_world(sim_mesh, seed)["world"]


FORCE_SCENES: list[dict[str, Any]] = [
    {
        "id": "one-sided",
        "label": "One-sided trench",
        "maxMyr": 80,
        "view": (5, -35),
        "about": (
            "A lone continent with a trench along its west coast only. Slab suction should pull it west, toward its trench, and it should settle at an active-margin speed (2–5 cm/yr)."
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
            "“Balanced” has equal trenches on opposite coasts: the pulls cancel, it barely moves, and its tension reads high — the state a ringed supercontinent is in before it rifts. “Lopsided” has a long trench in the west and a short one in the east, so it drifts slowly west."
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
            "The Phase 2 landmass, but nobody tells it how to move. Trenches on the two outer coasts pull West and East apart; an ocean opens between them and ridge push joins in. East then meets Far — and this time the collision slows it and locks the two together instead of letting them plough through each other."
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
            "“Asia” is a big continent with a trench along its south coast. “India” is a small continent on a plate that is mostly ocean, and that ocean is going down Asia’s trench: slab pull acts on India’s own plate, so it should race north at subducting-ocean speeds (8–20 cm/yr), several times faster than any suction-driven continent, then slow sharply when the two landmasses meet."
        ),
        "expect": {
            "India": ["subducting", "colliding"],
            "Asia": ["active", "colliding"],
        },
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
            "A generated supercontinent with a trench round most of its rim. Pulled from almost every side at once, it should hardly move — but its tension should be high. That stored tension is what Phase 6 will use to decide when and where it rifts."
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
    sim_mesh: SimMesh, seed: int = 0, scenario: str = "one-sided"
) -> TectonicWorld:
    """Build a named deterministic force-model scene from the upstream catalogue."""
    key = str(scenario).strip().lower()
    for definition in FORCE_SCENES:
        if definition["id"] == key:
            return definition["build"](sim_mesh, int(seed))
    available = ", ".join(definition["id"] for definition in FORCE_SCENES)
    raise ValueError(
        f"unknown force scenario {scenario!r}; expected one of: {available}"
    )


# Mechanical alias matching the upstream catalogue naming.
buildForceScene = build_force_scene  # noqa: N816
