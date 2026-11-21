"""Interactive continental-rifting scenario catalogue ported from V2 rift-scenes.js."""

from __future__ import annotations

import math
from typing import Any

from .config import mechanisms_off
from .scene_support import begin_scene
from .initial_world import build_start_world
from .kinematics import lat_lon_to_xyz
from .scenarios import add_apron_and_trench, add_continent, add_ocean, fill_background
from .world import TectonicWorld


def _share_basin(
    canvas: dict[str, Any],
    plates: list[Any],
    centres: list[list[float]],
    config: dict[str, Any],
) -> None:
    """Split a shared old-ocean basin between two plates along its midline."""
    separation = math.acos(
        max(-1.0, min(1.0, sum(centres[0][i] * centres[1][i] for i in range(3))))
    )
    reach = separation * 0.72
    for index, plate in enumerate(plates):
        own_centre = centres[index]
        other_centre = centres[1 - index]

        def wanted(
            x: float,
            y: float,
            z: float,
            own: list[float] = own_centre,
            other: list[float] = other_centre,
        ) -> bool:
            point = (x, y, z)
            own_dot = sum(point[i] * own[i] for i in range(3))
            other_dot = sum(point[i] * other[i] for i in range(3))
            distance = math.acos(max(-1.0, min(1.0, own_dot)))
            if distance > reach:
                return False
            return own_dot >= other_dot

        add_ocean(canvas, plate, wanted, -float(config["basinOceanAgeMyr"]))


def _plume(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = begin_scene(sim_mesh, seed)
    pangaea = world.add_plate("Pangaea")
    pangaea.born = -float(world.config["riftTenureMyr"])
    land = add_continent(
        canvas,
        noise,
        plate=pangaea,
        centre=(5.0, -20.0),
        radius_km=2600.0,
        craton_km=800.0,
        wobble=0.12,
    )
    add_apron_and_trench(
        canvas, pangaea, land, [[140.0, 220.0], [-40.0, 40.0]], world.config
    )
    fill_background(canvas, world, world.config)
    world.config.update(
        mechanisms_off(
            "plumes",
            "backArcs",
            "marginIsles",
            "marginErosion",
            "underthrust",
            "arcRecycling",
        )
    )
    plume_point = lat_lon_to_xyz(-3.0, -14.0)
    world.plumes.append(
        {
            "time": 20.0,
            "plate": pangaea.id,
            "cell": int(sim_mesh.index.nearest(*plume_point)),
            "done": False,
        }
    )
    return world


def _breakup(sim_mesh: Any, seed: int) -> TectonicWorld:
    return build_start_world(sim_mesh, seed)["world"]


def _founder(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = begin_scene(sim_mesh, seed)
    west = world.add_plate("Westland")
    east = world.add_plate("Eastland")
    west_land = add_continent(
        canvas,
        noise,
        plate=west,
        centre=(0.0, -26.0),
        radius_km=2200.0,
        craton_km=700.0,
    )
    east_land = add_continent(
        canvas,
        noise,
        plate=east,
        centre=(0.0, 26.0),
        radius_km=2200.0,
        craton_km=700.0,
    )
    add_apron_and_trench(canvas, west, west_land, [], world.config)
    add_apron_and_trench(canvas, east, east_land, [], world.config)
    _share_basin(
        canvas,
        [west, east],
        [west_land["centre"], east_land["centre"]],
        world.config,
    )
    fill_background(canvas, world, world.config)
    world.config.update(
        mechanisms_off(
            "rifts",
            "plumes",
            "backArcs",
            "marginIsles",
            "marginErosion",
            "underthrust",
            "arcRecycling",
        )
    )
    world.config["oceanPlateMinAreaKm2"] = math.inf
    return world


RIFT_SCENES: list[dict[str, Any]] = [
    {
        "id": "plume",
        "label": "A plume splits a continent",
        "maxMyr": 160,
        "view": (5, -20),
        "about": (
            "Pangaea is a single continent with trenches on its west and east "
            "coasts. At 20 Myr a mantle plume rises beneath it: a rift follows a "
            "low-cost path around the craton, leaves a failed arm and pulls the "
            "two halves apart."
        ),
        "build": _plume,
    },
    {
        "id": "breakup",
        "label": "The start world breaks up",
        "maxMyr": 300,
        "view": None,
        "about": (
            "The generated supercontinent, ringed by subduction and under tension, "
            "rifts on its own. The seed determines when it starts; old sutures "
            "between cratons guide the breakup."
        ),
        "build": _breakup,
    },
    {
        "id": "founder",
        "label": "A passive margin founders",
        "maxMyr": 300,
        "view": (0, 0),
        "about": (
            "Two continents share an old ocean basin without a trench. As the "
            "passive margin founders, a new trench is born on their shared plate "
            "boundary and the basin begins to close."
        ),
        "build": _founder,
    },
]


def build_rift_scene(
    sim_mesh: Any, seed: int = 0, scenario: str = "plume"
) -> TectonicWorld:
    """Build a named deterministic continental-rifting scene from the upstream catalogue."""
    key = str(scenario).strip().lower()
    for definition in RIFT_SCENES:
        if definition["id"] == key:
            return definition["build"](sim_mesh, int(seed))
    available = ", ".join(definition["id"] for definition in RIFT_SCENES)
    raise ValueError(
        f"unknown rift scenario {scenario!r}; expected one of: {available}"
    )


# Mechanical alias matching the upstream catalogue naming.
buildRiftScene = build_rift_scene  # noqa: N816
