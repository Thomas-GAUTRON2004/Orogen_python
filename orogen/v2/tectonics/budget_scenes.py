"""Interactive crust-budget scenario catalogue ported from V2 budget-scenes.js."""

from __future__ import annotations

from typing import Any

from .config import mechanisms_on, only_mechanisms
from .dynamics import update_plate_motion
from .event_scenes import build_event_scene
from .force_scenes import build_force_scene
from .initial_world import build_start_world
from .margin_scenes import build_margin_scene
from .world import TectonicWorld


def _erosion(sim_mesh: Any, seed: int) -> TectonicWorld:
    world = build_margin_scene(sim_mesh, seed, "margins")
    world.config.update(only_mechanisms("marginIsles", "marginErosion"))
    world.config.update(mechanisms_on("marginIsles", "marginErosion"))
    world.config.update({"marginErosionPatchKm": 1500.0, "marginErosionRateBias": 0.0})
    return world


def _india(sim_mesh: Any, seed: int) -> TectonicWorld:
    world = build_force_scene(sim_mesh, seed, "india")
    world.config.update(only_mechanisms("underthrust"))
    world.config.update(mechanisms_on("underthrust"))
    return world


def _arcs(sim_mesh: Any, seed: int) -> TectonicWorld:
    world = build_event_scene(sim_mesh, seed, "arc-collision")
    world.config.update(only_mechanisms("wings", "arcs", "arcRecycling"))
    world.config.update(mechanisms_on("wings", "arcs", "arcRecycling"))
    return world


def _history(sim_mesh: Any, seed: int) -> TectonicWorld:
    return build_start_world(sim_mesh, seed)["world"]


BUDGET_SCENES: list[dict[str, Any]] = [
    {
        "id": "erosion",
        "label": "Erosive and accretionary margins",
        "maxMyr": 150,
        "view": (0, -20),
        "about": (
            "Driftia is pulled west by its western trench. Some margin stretches "
            "are erosive and lose fore-arc crust; others are accretionary and grow "
            "islands into the continent. Their character stays fixed to the plate."
        ),
        "build": _erosion,
        "drive": update_plate_motion,
    },
    {
        "id": "india",
        "label": "A continent slides under another",
        "maxMyr": 90,
        "view": (5, 72),
        "about": (
            "India reaches Asia. The leading edge of the lower continent slides "
            "under the upper one: India’s land in the overlap is consumed and "
            "Asia’s crust thickens into a plateau before the continents weld."
        ),
        "build": _india,
        "drive": update_plate_motion,
    },
    {
        "id": "arcs",
        "label": "Small arcs go down, big arcs accrete",
        "maxMyr": 220,
        "view": (0, 0),
        "about": (
            "Two continents close with island arcs on their facing margins. Small "
            "arcs are recycled with their slab; larger arcs accrete onto the "
            "continents, so growth happens in fewer, larger pieces."
        ),
        "build": _arcs,
        "drive": update_plate_motion,
    },
    {
        "id": "history",
        "label": "The crust budget over 600 Myr",
        "maxMyr": 600,
        "view": None,
        "about": (
            "The generated supercontinent runs with the full mechanism set. "
            "Continental area grows through arc and fore-arc accretion, while "
            "erosive margins, underthrusting and recycled arcs remove crust."
        ),
        "build": _history,
        "drive": update_plate_motion,
    },
]


def build_budget_scene(
    sim_mesh: Any, seed: int = 0, scenario: str = "erosion"
) -> TectonicWorld:
    """Build a named deterministic crust-budget scene from the upstream catalogue."""
    key = str(scenario).strip().lower()
    for definition in BUDGET_SCENES:
        if definition["id"] == key:
            return definition["build"](sim_mesh, int(seed))
    available = ", ".join(definition["id"] for definition in BUDGET_SCENES)
    raise ValueError(
        f"unknown budget scenario {scenario!r}; expected one of: {available}"
    )


# Mechanical alias matching the upstream catalogue naming.
buildBudgetScene = build_budget_scene  # noqa: N816
