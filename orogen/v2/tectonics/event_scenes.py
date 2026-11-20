"""Interactive tectonic-event scenario catalogue ported from V2 event-scenes.js."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.sphere_geometry import PLANET_RADIUS_KM, cross, dot, normalize
from .crust import CrustFlag, CrustType
from .force_scenes import FORCE_SCENES, _add_ocean_between, _begin
from .kinematics import lat_lon_to_xyz
from .scenarios import (
    add_apron_and_trench,
    add_continent,
    fill_background,
    scene_terrane,
)
from .world import TectonicWorld


def _add_arc_ridge(
    canvas: dict[str, Any],
    plate: Any,
    first: list[float],
    second: list[float],
    distance_km: float,
    half_width_km: float,
    config: dict[str, Any],
    terrane_id: int,
) -> None:
    """Paint an already-grown island arc as a narrow great-circle ridge."""
    mesh = canvas["mesh"]
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape((-1, 3))
    axis = normalize(cross(first, second))
    band = distance_km / PLANET_RADIUS_KM
    half = half_width_km / PLANET_RADIUS_KM
    crust = plate.crust
    plate.arcTerraneId = terrane_id
    for cell in range(int(mesh.num_cells)):
        if not crust.mask[cell] or crust.type[cell] != CrustType["OCEAN"]:
            continue
        point = xyz[cell]
        lateral = math.asin(max(-1.0, min(1.0, dot(point, axis))))
        if abs(lateral) > half * 2.2:
            continue
        along = math.acos(max(-1.0, min(1.0, dot(point, first))))
        if abs(along - band) > half * 0.5:
            continue
        crust.type[cell] = CrustType["ARC"]
        crust.terrane[cell] = terrane_id
        crust.formedAt[cell] = -float(config["initialOceanAgeMyr"])
        crust.flags[cell] = np.uint8(
            int(crust.flags[cell]) & (0xFF ^ int(CrustFlag["TRENCH"]))
        )


def _add_arc_plate(
    canvas: dict[str, Any],
    plate: Any,
    *,
    latitude: float,
    longitude: float,
    half_width_deg: float,
    depth_deg: float,
    config: dict[str, Any],
) -> None:
    """Build an oceanic plate patch with a mature southern trench and an inland arc."""
    mesh = canvas["mesh"]
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape((-1, 3))
    taken = canvas["taken"]
    count = int(mesh.num_cells)
    inside = np.zeros(count, dtype=bool)
    for cell in range(count):
        if taken[cell]:
            continue
        x, y, z = (float(value) for value in xyz[cell])
        cell_latitude = math.degrees(math.asin(max(-1.0, min(1.0, z))))
        cell_longitude = math.degrees(math.atan2(y, x))
        delta_longitude = (cell_longitude - longitude + 540.0) % 360.0 - 180.0
        if (
            cell_latitude < latitude
            or cell_latitude > latitude + depth_deg
            or abs(delta_longitude) > half_width_deg
        ):
            continue
        plate.crust.create(
            cell, CrustType["OCEAN"], -float(config["initialOceanAgeMyr"])
        )
        inside[cell] = True
        taken[cell] = 1

    trench_rows = max(1, math.floor(243.0 / float(mesh.edge_km) + 0.5))
    arc_rows = max(1, math.floor(121.0 / float(mesh.edge_km) + 0.5))
    row = np.full(count, -1, dtype=np.int16)
    ring: list[int] = []
    for cell in range(count):
        if not inside[cell]:
            continue
        lo, hi = int(mesh.adj_offset[cell]), int(mesh.adj_offset[cell + 1])
        for neighbour_value in mesh.adj_list[lo:hi]:
            neighbour = int(neighbour_value)
            if not inside[neighbour] and xyz[neighbour, 2] < xyz[cell, 2]:
                row[cell] = 0
                ring.append(cell)
                break

    for row_index in range(1, trench_rows + arc_rows):
        following: list[int] = []
        for cell in ring:
            lo, hi = int(mesh.adj_offset[cell]), int(mesh.adj_offset[cell + 1])
            for neighbour_value in mesh.adj_list[lo:hi]:
                neighbour = int(neighbour_value)
                if (
                    inside[neighbour]
                    and row[neighbour] < 0
                    and xyz[neighbour, 2] > xyz[cell, 2]
                ):
                    row[neighbour] = row_index
                    following.append(neighbour)
        ring = following

    terrane_id = scene_terrane(canvas)
    plate.arcTerraneId = terrane_id
    for cell in range(count):
        if row[cell] < 0:
            continue
        if row[cell] < trench_rows:
            plate.crust.arm_trench(cell, float(config["arcMatureConsumedKm"]))
        else:
            plate.crust.type[cell] = CrustType["ARC"]
            plate.crust.formedAt[cell] = -40.0
            plate.crust.terrane[cell] = terrane_id


def _wings(sim_mesh: Any, seed: int) -> TectonicWorld:
    definition = next(scene for scene in FORCE_SCENES if scene["id"] == "one-sided")
    return definition["build"](sim_mesh, seed)


def _terrane(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = _begin(sim_mesh, seed)
    asia = world.add_plate("Asia")
    rider = world.add_plate("Rider")
    asia_land = add_continent(
        canvas,
        noise,
        plate=asia,
        centre=(40.0, 72.0),
        radius_km=2800.0,
    )
    add_apron_and_trench(canvas, asia, asia_land, [[215.0, 325.0]], world.config)
    microcontinent = add_continent(
        canvas,
        noise,
        plate=rider,
        centre=(-26.0, 72.0),
        radius_km=450.0,
        craton_km=0.0,
    )
    _add_ocean_between(
        canvas, rider, microcontinent["centre"], asia_land["centre"], 1500.0, 1700.0
    )
    fill_background(canvas, world, world.config)
    return world


def _arc(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = _begin(sim_mesh, seed)
    austral = world.add_plate("Austral")
    oceanica = world.add_plate("Oceanica")
    land = add_continent(
        canvas,
        noise,
        plate=austral,
        centre=(-28.0, 130.0),
        radius_km=2000.0,
    )
    _add_arc_plate(
        canvas,
        oceanica,
        latitude=14.0,
        longitude=130.0,
        half_width_deg=17.0,
        depth_deg=22.0,
        config=world.config,
    )
    _add_ocean_between(
        canvas,
        austral,
        land["centre"],
        lat_lon_to_xyz(12.0, 130.0),
        2200.0,
        2600.0,
    )
    fill_background(canvas, world, world.config)
    return world


def _arc_collision(sim_mesh: Any, seed: int) -> TectonicWorld:
    world, canvas, noise = _begin(sim_mesh, seed)
    west = world.add_plate("Westland")
    east = world.add_plate("Eastland")
    west_land = add_continent(
        canvas,
        noise,
        plate=west,
        centre=(0.0, -26.0),
        radius_km=2300.0,
    )
    east_land = add_continent(
        canvas,
        noise,
        plate=east,
        centre=(0.0, 26.0),
        radius_km=2300.0,
    )
    add_apron_and_trench(canvas, west, west_land, [[240.0, 300.0]], world.config)
    add_apron_and_trench(canvas, east, east_land, [[60.0, 120.0]], world.config)
    _add_ocean_between(
        canvas, west, west_land["centre"], east_land["centre"], 400.0, 2200.0
    )
    _add_ocean_between(
        canvas, east, east_land["centre"], west_land["centre"], 400.0, 2200.0
    )
    _add_arc_ridge(
        canvas,
        west,
        west_land["centre"],
        east_land["centre"],
        2000.0,
        700.0,
        world.config,
        scene_terrane(canvas),
    )
    _add_arc_ridge(
        canvas,
        east,
        east_land["centre"],
        west_land["centre"],
        2000.0,
        700.0,
        world.config,
        scene_terrane(canvas),
    )
    fill_background(canvas, world, world.config)
    return world


EVENT_SCENES: list[dict[str, Any]] = [
    {
        "id": "wings",
        "label": "Trench wings & island arcs",
        "maxMyr": 300,
        "view": (5, -35),
        "about": (
            "The lone continent has a trench on its west coast. Its ends propagate "
            "out to sea, capturing old ocean into the plate as a fore-arc; where a "
            "wing has swallowed enough slab, an island arc forms behind it."
        ),
        "expect": {"wings": True, "arc": True},
        "build": _wings,
    },
    {
        "id": "terrane",
        "label": "Microcontinent docks",
        "maxMyr": 90,
        "view": (5, 72),
        "about": (
            "A microcontinent rides north on a mostly oceanic plate toward Asia. "
            "On arrival it docks with a narrow belt and subduction jumps to its "
            "outboard side."
        ),
        "expect": {"docks": True},
        "build": _terrane,
    },
    {
        "id": "arc",
        "label": "Arc meets a passive margin",
        "maxMyr": 90,
        "view": (0, 130),
        "about": (
            "Austral is a continent with no trench of its own. Oceanica carries a "
            "mature island arc toward Austral; the arc accretes and subduction "
            "polarity reverses without a special-case rule."
        ),
        "expect": {"docks": True, "reversal": True},
        "build": _arc,
    },
    {
        "id": "arc-collision",
        "label": "Arcs between closing continents",
        "maxMyr": 220,
        "view": (0, 0),
        "about": (
            "Two continents close on each other with island arcs on their facing "
            "margins. The thin buoyant arcs are scraped onto the continents before "
            "the continental cores meet and form a suture."
        ),
        "expect": {"docks": True},
        "build": _arc_collision,
    },
]


def build_event_scene(
    sim_mesh: Any, seed: int = 0, scenario: str = "wings"
) -> TectonicWorld:
    """Build a named deterministic event-rule scene from the upstream catalogue."""
    key = str(scenario).strip().lower()
    for definition in EVENT_SCENES:
        if definition["id"] == key:
            return definition["build"](sim_mesh, int(seed))
    available = ", ".join(definition["id"] for definition in EVENT_SCENES)
    raise ValueError(
        f"unknown event scenario {scenario!r}; expected one of: {available}"
    )


# Mechanical alias matching the upstream catalogue naming.
buildEventScene = build_event_scene  # noqa: N816
