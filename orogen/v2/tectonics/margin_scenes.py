"""Interactive passive/active-margin scenario catalogue ported from V2 margin-scenes.js."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any
from ..core.sim_mesh import SimMesh

import numpy as np

from .config import only_mechanisms
from .crust import CrustType
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
from .scene_constants import ARC_TERRANE_ID, MARIANA_ARC_WIDTH_KM
from .scene_support import begin_scene

ROLLBACK_ONLY = MappingProxyType(
    {
        **only_mechanisms("backArcs", "flatSlabs"),
        "backArcRatePerMyr": 0.5,
        "backArcContinentalShare": 1.0,
    }
)
QUIET = MappingProxyType(only_mechanisms("flatSlabs", "marginIsles"))
LARAMIDE = MappingProxyType({"speedCmYr": 5.0, "trenchLon": -118.0})


def _set_motion(
    world: Any,
    name: str,
    point: list[float],
    bearing_deg: float,
    speed_cm_yr: float,
) -> None:
    plate = next(
        (item for item in world.plates if item.name == name and item.alive),
        None,
    )
    if plate is not None:
        plate.omega = omega_for(point, heading(point, bearing_deg), speed_cm_yr)


def _arm_against(mesh: Any, plate: Any, other: Any, config: dict) -> list[int]:
    """Arm a mature trench along every cell of one plate touching another."""
    trench: list[int] = []
    for cell in range(int(mesh.num_cells)):
        if not plate.crust.mask[cell]:
            continue
        lo, hi = int(mesh.adj_offset[cell]), int(mesh.adj_offset[cell + 1])
        for neighbour_value in mesh.adj_list[lo:hi]:
            neighbour = int(neighbour_value)
            if other.crust.mask[neighbour]:
                plate.crust.arm_trench(cell, float(config["slabMatureKm"]))
                trench.append(cell)
                break
    return trench


def _arm_facing(
    mesh: Any,
    plate: Any,
    centre: list[float],
    bearing_deg: float,
    config: dict,
) -> None:
    """Arm cells on a plate rim whose outward normal faces a compass sector."""
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape((-1, 3))
    direction = np.asarray(heading(centre, bearing_deg), dtype=np.float64)
    for cell in range(int(mesh.num_cells)):
        if not plate.crust.mask[cell]:
            continue
        normal = np.zeros(3, dtype=np.float64)
        rim = False
        lo, hi = int(mesh.adj_offset[cell]), int(mesh.adj_offset[cell + 1])
        for neighbour_value in mesh.adj_list[lo:hi]:
            neighbour = int(neighbour_value)
            if plate.crust.mask[neighbour]:
                continue
            rim = True
            normal += xyz[neighbour] - xyz[cell]
        length = float(np.linalg.norm(normal))
        if (
            rim
            and length > 0.0
            and float(np.dot(normal, direction)) / length > math.cos(math.radians(50.0))
        ):
            plate.crust.arm_trench(cell, float(config["slabMatureKm"]))


def _arc_behind(mesh: Any, plate: Any, trench: list[int], hops: int) -> None:
    """Paint only the ring exactly hops cells inland of the trench."""
    seen = set(trench)
    ring = list(trench)
    for _ in range(hops):
        following: list[int] = []
        for cell in ring:
            lo, hi = int(mesh.adj_offset[cell]), int(mesh.adj_offset[cell + 1])
            for neighbour_value in mesh.adj_list[lo:hi]:
                neighbour = int(neighbour_value)
                if plate.crust.mask[neighbour] and neighbour not in seen:
                    seen.add(neighbour)
                    following.append(neighbour)
        ring = following
    for cell in ring:
        plate.crust.type[cell] = CrustType["ARC"]
        plate.crust.terrane[cell] = ARC_TERRANE_ID
    plate.arcTerraneId = ARC_TERRANE_ID


def _margins(sim_mesh: SimMesh, seed: int):
    world, canvas, noise = begin_scene(sim_mesh, seed)
    world.config.update(QUIET)
    drift = world.add_plate("Driftia")
    land = add_continent(
        canvas,
        noise,
        plate=drift,
        centre=(0.0, 0.0),
        radius_km=2200.0,
        craton_km=700.0,
    )
    add_apron_and_trench(canvas, drift, land, [], world.config)
    fill_background(canvas, world, world.config)
    centre = lat_lon_to_xyz(0.0, 0.0)
    _arm_facing(sim_mesh, drift, centre, -90.0, world.config)
    return world


def _laramide(sim_mesh: SimMesh, seed: int):
    world, canvas, noise = begin_scene(sim_mesh, seed)
    world.config.update(QUIET)
    laurentia = world.add_plate("Laurentia")
    farallona = world.add_plate("Farallona")
    land = add_continent(
        canvas,
        noise,
        plate=laurentia,
        centre=(42.0, -98.0),
        radius_km=2300.0,
        craton_km=400.0,
    )
    add_apron_and_trench(canvas, laurentia, land, [], world.config)

    def young_ocean(x: float, y: float, z: float) -> bool:
        latitude = math.degrees(math.asin(max(-1.0, min(1.0, z))))
        longitude = math.degrees(math.atan2(y, x))
        return (
            12.0 < latitude < 72.0
            and LARAMIDE["trenchLon"] - 30.0 < longitude < LARAMIDE["trenchLon"] + 20.0
        )

    add_ocean(canvas, farallona, young_ocean, 0.0)
    fill_background(canvas, world, world.config)
    _arm_against(sim_mesh, laurentia, farallona, world.config)
    return world


def _mariana(sim_mesh: SimMesh, seed: int):
    world, canvas, noise = begin_scene(sim_mesh, seed)
    world.config.update(ROLLBACK_ONLY)
    asiana = world.add_plate("Asiana")
    philippina = world.add_plate("Philippina")
    pacifica = world.add_plate("Pacifica")
    land = add_continent(
        canvas,
        noise,
        plate=asiana,
        centre=(18.0, 100.0),
        radius_km=2200.0,
        craton_km=800.0,
    )
    add_apron_and_trench(canvas, asiana, land, [], world.config)

    philippina_centre = lat_lon_to_xyz(15.0, 138.0)
    pacifica_centre = lat_lon_to_xyz(15.0, 190.0)
    philippina_reach = math.cos(1900.0 / 6371.0)
    pacifica_reach = math.cos(5000.0 / 6371.0)
    add_ocean(
        canvas,
        philippina,
        lambda x, y, z: (
            x * philippina_centre[0]
            + y * philippina_centre[1]
            + z * philippina_centre[2]
            >= philippina_reach
        ),
        -10.0,
    )
    add_ocean(
        canvas,
        pacifica,
        lambda x, y, z: (
            x * pacifica_centre[0] + y * pacifica_centre[1] + z * pacifica_centre[2]
            >= pacifica_reach
        ),
        -150.0,
    )
    fill_background(canvas, world, world.config)
    _arm_against(sim_mesh, asiana, philippina, world.config)
    trench = _arm_against(sim_mesh, philippina, pacifica, world.config)
    hops = max(1, math.floor(MARIANA_ARC_WIDTH_KM / float(sim_mesh.edge_km) + 0.5))
    _arc_behind(sim_mesh, philippina, trench, hops)
    return world


def _drive_mariana(world: Any) -> None:
    point = lat_lon_to_xyz(15.0, 150.0)
    _set_motion(world, "Philippina", point, -90.0, 3.0)
    _set_motion(world, "Pacifica", point, -90.0, 8.0)


def _drive_japan(world: Any) -> None:
    point = lat_lon_to_xyz(35.0, 140.0)
    _set_motion(world, "Asiana", point, 90.0, 1.0)
    _set_motion(world, "Pacifica", point, -90.0, 8.0)


def _japan(sim_mesh: SimMesh, seed: int):
    world, canvas, noise = begin_scene(sim_mesh, seed)
    world.config.update(ROLLBACK_ONLY)
    asiana = world.add_plate("Asiana")
    pacifica = world.add_plate("Pacifica")
    land = add_continent(
        canvas,
        noise,
        plate=asiana,
        centre=(35.0, 110.0),
        radius_km=2700.0,
        craton_km=800.0,
    )
    add_apron_and_trench(canvas, asiana, land, [], world.config)
    pacifica_centre = lat_lon_to_xyz(32.0, 185.0)
    reach = math.cos(5000.0 / 6371.0)
    add_ocean(
        canvas,
        pacifica,
        lambda x, y, z: (
            x * pacifica_centre[0] + y * pacifica_centre[1] + z * pacifica_centre[2]
            >= reach
        ),
        -150.0,
    )
    fill_background(canvas, world, world.config)
    _arm_against(sim_mesh, asiana, pacifica, world.config)
    return world


def _history(sim_mesh: SimMesh, seed: int):
    """Build a fresh mutable history world; sharing cached worlds is unsafe."""
    return build_start_world(sim_mesh, seed)["world"]


MARGIN_SCENES: list[dict[str, Any]] = [
    {
        "id": "margins",
        "label": "A continent’s leading and trailing coasts",
        "maxMyr": 60,
        "view": (0, 0),
        "about": (
            "“Driftia” has a trench along its western coast and none elsewhere; slab suction pulls it west. Its western coast is ACTIVE and LEADING — a narrow shelf and a coastal range to come — and its eastern coast PASSIVE and TRAILING, a wide shelf; along its flanks it slides past the old ocean (Margins layer)."
        ),
        "build": _margins,
        "drive": update_plate_motion,
    },
    {
        "id": "laramide",
        "label": "A continent drives over a young slab",
        "maxMyr": 50,
        "view": (40, -110),
        "about": (
            "“Laurentia” drives west at 5 cm/yr into a narrow, YOUNG ocean plate, “Farallona”, which is being made at a ridge only a short way offshore — so the slab going down is never older than a few tens of Myr. Young slab is warm and buoyant, and under a plate that drives at it this hard it goes in flat: the belt is Laramide — it reaches far inland, like the Rockies — and there are no arc volcanoes above it (Orogeny layer). Compare the Andean belt a steep slab makes (tunable: flat slabs up to …)."
        ),
        "speedCmYr": 5.0,
        "trenchLon": -118.0,
        "build": _laramide,
        "drive": lambda world: _set_motion(
            world, "Laurentia", lat_lon_to_xyz(42.0, -118.0), -90.0, 5.0
        ),
    },
    {
        "id": "mariana",
        "label": "An arc rifts off its ocean plate",
        "maxMyr": 50,
        "view": (15, 140),
        "about": (
            "“Philippina” is an ocean plate caught between two trenches. On its west it goes down under the continent “Asiana”, drifting west at 3 cm/yr; on its east it overrides “Pacifica”, very old ocean coming in at 8 cm/yr — and there it carries an island arc. It does not keep up with its own eastern trench, and the old, cold Pacifica slab sinks steeply and rolls back: Philippina is stretched behind its arc until the arc rifts away after the trench, and a back-arc sea opens behind it (the Mariana Trough, the Philippine Sea)."
        ),
        "build": _mariana,
        "drive": _drive_mariana,
    },
    {
        "id": "japan",
        "label": "A strip of coast rifts off a continent",
        "maxMyr": 60,
        "view": (35, 135),
        "about": (
            "“Asiana” overrides very old ocean, “Pacifica”, coming in at 8 cm/yr along its eastern coast; Asiana itself creeps east at only 1 cm/yr. The cold slab sinks steeply and rolls back faster than the continent follows: Asiana’s eastern margin is stretched until a strip of the coast — hundreds of km wide, accreted crust, never a craton — rifts away after the trench as a chain of islands, and a sea opens behind it (Japan and the Sea of Japan)."
        ),
        "build": _japan,
        "drive": _drive_japan,
    },
    {
        "id": "history",
        "label": "Margins and back-arcs through a whole history",
        "maxMyr": 300,
        "view": None,
        "about": (
            "The generated supercontinent of Phase 3 with every rule on: every coast is classified as it goes (Margins layer), young slabs go in flat under plates driving at them, and old, rolling-back trenches peel arcs — now and then a strip of a continent — off their plates and open seas behind them."
        ),
        "build": _history,
        "drive": update_plate_motion,
    },
]


def build_margin_scene(sim_mesh: SimMesh, seed: int = 0, scenario: str = "margins"):
    """Build a named deterministic margin-rule scene from the upstream catalogue."""
    key = str(scenario).strip().lower()
    for definition in MARGIN_SCENES:
        if definition["id"] == key:
            return definition["build"](sim_mesh, int(seed))
    available = ", ".join(definition["id"] for definition in MARGIN_SCENES)
    raise ValueError(
        f"unknown margin scenario {scenario!r}; expected one of: {available}"
    )


# Mechanical aliases matching upstream helper/catalogue naming.
buildMarginScene = build_margin_scene  # noqa: N816
headingForMargin = heading  # noqa: N816
omegaForMargin = omega_for  # noqa: N816
