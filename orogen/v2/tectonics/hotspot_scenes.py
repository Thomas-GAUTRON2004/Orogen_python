"""Interactive plume/hotspot scenario catalogue ported from V2 hotspot-scenes.js."""

from __future__ import annotations

import math
from typing import Any

from ..core.sphere_geometry import PLANET_RADIUS_KM
from .config import only_mechanisms
from .dynamics import update_plate_motion
from .initial_world import build_start_world
from .kinematics import lat_lon_to_xyz
from .scenarios import (
    add_apron_and_trench,
    add_continent,
    add_ocean,
    fill_background,
    heading,
    omega_for,
)
from .scene_support import begin_scene
from .world import TectonicWorld

HAWAII: dict[str, Any] = {
    "plume": (19.0, -155.0),
    "speedCmYr": 8.0,
    "bendMyr": 45.0,
    "legs": (0.0, -60.0),
}
DECCAN: dict[str, Any] = {"plume": (-30.0, 75.0), "speedCmYr": 12.0, "arrivesMyr": 12.0}
PLUMES_ONLY = only_mechanisms()


def _hawaii(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, _noise = begin_scene(sim_mesh, seed)
    world.config.update(PLUMES_ONLY)
    pacifica = world.add_plate("Pacifica")
    centre = lat_lon_to_xyz(-6.0, -148.0)
    reach = math.cos(5200.0 / PLANET_RADIUS_KM)
    add_ocean(
        canvas,
        pacifica,
        lambda x, y, z: x * centre[0] + y * centre[1] + z * centre[2] >= reach,
        -60.0,
    )
    fill_background(canvas, world, world.config)
    world.plumes.append(
        {
            "time": 0.0,
            "xyz": lat_lon_to_xyz(*HAWAII["plume"]),
            "head": False,
            "rift": False,
            "lifeMyr": 1000.0,
            "strength": 1.0,
        }
    )
    return world


def _deccan(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = begin_scene(sim_mesh, seed)
    world.config.update(PLUMES_ONLY)
    indica = world.add_plate("Indica")
    land = add_continent(
        canvas,
        noise,
        plate=indica,
        centre=(-32.0, 75.0),
        radius_km=1800.0,
        craton_km=600.0,
    )
    add_apron_and_trench(canvas, indica, land, [], world.config)
    fill_background(canvas, world, world.config)
    world.plumes.append(
        {
            "time": DECCAN["arrivesMyr"],
            "xyz": lat_lon_to_xyz(*DECCAN["plume"]),
            "head": True,
            "rift": False,
            "lifeMyr": 1000.0,
            "strength": 1.0,
        }
    )
    return world


def _history(sim_mesh: Any, seed: int) -> TectonicWorld:
    """Build a fresh mutable history world; sharing cached worlds is unsafe."""
    return build_start_world(sim_mesh, seed)["world"]


def _drive_hawaii(world: TectonicWorld) -> None:
    pacifica = next(plate for plate in world.plates if plate.name == "Pacifica")
    point = lat_lon_to_xyz(*HAWAII["plume"])
    bearing = HAWAII["legs"][0] if world.time < HAWAII["bendMyr"] else HAWAII["legs"][1]
    pacifica.omega = omega_for(point, heading(point, bearing), HAWAII["speedCmYr"])


def _drive_deccan(world: TectonicWorld) -> None:
    indica = next(plate for plate in world.plates if plate.name == "Indica")
    point = lat_lon_to_xyz(*DECCAN["plume"])
    indica.omega = omega_for(point, heading(point, 0.0), DECCAN["speedCmYr"])


HOTSPOT_SCENES: list[dict[str, Any]] = [
    {
        "id": "hawaii",
        "label": "A plate slides over a fixed plume",
        "maxMyr": 90,
        "view": (20, -165),
        "about": (
            "Pacifica slides over a plume fixed in the mantle. The plate carries "
            "away an age-progressive volcanic chain that bends when its heading "
            "changes at 45 Myr."
        ),
        **HAWAII,
        "build": _hawaii,
        "drive": _drive_hawaii,
    },
    {
        "id": "deccan",
        "label": "A plume head reaches a drifting continent",
        "maxMyr": 70,
        "view": (-15, 75),
        "about": (
            "Indica drifts north. At 12 Myr a plume head reaches its southern half "
            "and floods a broad province with basalt; its tail later draws a track "
            "across the ocean floor left behind by the moving plate."
        ),
        **DECCAN,
        "build": _deccan,
        "drive": _drive_deccan,
    },
    {
        "id": "history",
        "label": "Plumes through a whole history",
        "maxMyr": 300,
        "view": None,
        "about": (
            "The generated supercontinent runs with the full rule set. Mantle heat "
            "builds beneath long-lived landmasses, plumes rise and their heads and "
            "tails leave large igneous provinces and volcanic tracks."
        ),
        "build": _history,
        "drive": update_plate_motion,
    },
]


def build_hotspot_scene(
    sim_mesh: Any, seed: int = 0, scenario: str = "hawaii"
) -> TectonicWorld:
    """Build a named deterministic plume/hotspot scene from the upstream catalogue."""
    key = str(scenario).strip().lower()
    for definition in HOTSPOT_SCENES:
        if definition["id"] == key:
            return definition["build"](sim_mesh, int(seed))
    available = ", ".join(definition["id"] for definition in HOTSPOT_SCENES)
    raise ValueError(
        f"unknown hotspot scenario {scenario!r}; expected one of: {available}"
    )


# Mechanical alias matching the upstream catalogue naming.
buildHotspotScene = build_hotspot_scene  # noqa: N816
