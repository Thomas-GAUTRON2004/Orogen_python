"""Interactive collision scenario catalogue ported from V2 collision-scenes.js."""

from __future__ import annotations

import numpy as np

from ..core.sim_mesh import SimMesh
from ..core.sphere_geometry import normalize
from .scenarios import add_apron_and_trench, add_continent, fill_background
from .scene_support import add_ocean_corridor, begin_scene
from .scene_types import CompleteSceneDefinition, build_from_catalogue

def _zipper(sim_mesh: SimMesh, seed: int):
    world, canvas, noise = begin_scene(sim_mesh, seed)
    borea = world.add_plate("Borea")
    austra = world.add_plate("Austra")

    core_centres = ((36.0, 20.0), (33.0, 45.0), (33.0, 70.0), (36.0, 95.0))
    cores = [
        add_continent(
            canvas,
            noise,
            plate=borea,
            centre=centre,
            radius_km=1700.0,
            craton_km=550.0,
        )
        for centre in core_centres
    ]
    is_land = np.zeros(int(sim_mesh.num_cells), dtype=np.uint8)
    for core in cores:
        is_land |= core["isLand"]
    middle = normalize(
        np.sum(
            np.asarray([core["centre"] for core in cores], dtype=np.float64), axis=0
        ).tolist()
    )
    borea_land = {"centre": middle, "isLand": is_land}
    add_apron_and_trench(canvas, borea, borea_land, [[180.0, 360.0]], world.config)

    parts = [
        add_continent(
            canvas,
            noise,
            plate=austra,
            centre=centre,
            radius_km=1250.0,
            craton_km=450.0,
        )
        for centre in ((2.0, 92.0), (-8.0, 75.0), (-18.0, 58.0))
    ]
    austra_land = parts[1]
    add_ocean_corridor(
        canvas,
        austra,
        austra_land["centre"],
        cores[2]["centre"],
        3400.0,
        4200.0,
    )
    fill_background(canvas, world, world.config)
    return world


COLLISION_SCENES: list[CompleteSceneDefinition] = [
    {
        "id": "zipper",
        "label": "Two continents meet at an angle",
        "maxMyr": 160,
        "view": (18, 72),
        "about": (
            "“Borea”, a long continent, has a trench along its whole south coast. “Austra” rides north on a mostly oceanic plate, its long chain of land set at an angle to that coast: its north-east end strikes Borea first, while the ocean to the west of the contact is still going down the trench. The pull of that slab turns Austra about the first contact, so the rest of the gap closes like a zipper — and only then do the two weld, with little or no sea left between them."
        ),
        "build": _zipper,
    },
]


def build_collision_scene(sim_mesh: SimMesh, seed: int = 0, scenario: str = "zipper"):
    """Build a named deterministic collision-rule scene from the upstream catalogue."""
    key = str(scenario).strip().lower()
    for definition in COLLISION_SCENES:
        if definition["id"] == key:
            return build_from_catalogue(definition, sim_mesh, int(seed))
    available = ", ".join(str(definition["id"]) for definition in COLLISION_SCENES)
    raise ValueError(
        f"unknown collision scenario {scenario!r}; expected one of: {available}"
    )


# Mechanical alias matching the upstream catalogue naming.
buildCollisionScene = build_collision_scene  # noqa: N816
