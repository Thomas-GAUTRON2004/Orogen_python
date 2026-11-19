"""Small deterministic tectonic scenes used by tests and development tools.

These builders deliberately keep the fixtures smaller than the interactive
JavaScript scenario catalogue. They provide controlled continental, oceanic
and mixed states while preserving the TectonicWorld and Plate contracts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.seed import make_rng
from ..core.sphere_geometry import random_point
from .crust import CrustType
from .initial_world import build_start_world
from .world import EventKind, TectonicWorld


def _xyz(sim_mesh: Any) -> np.ndarray:
    values = np.asarray(sim_mesh.xyz, dtype=np.float64).reshape((-1, 3))
    if values.shape[0] != int(sim_mesh.num_cells):
        raise ValueError("mesh xyz does not match num_cells")
    return values


def build_continental_scene(sim_mesh: Any, seed: int = 0, fraction: float = 0.30) -> dict[str, Any]:
    """Build a single continental plate against the passive background ocean."""
    if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be finite and strictly between 0 and 1")
    world = TectonicWorld(sim_mesh, seed)
    xyz = _xyz(sim_mesh)
    centre = np.asarray(random_point(make_rng(seed * 31 + 211)), dtype=np.float64)
    count = max(1, min(sim_mesh.num_cells - 1, int(math.floor(sim_mesh.num_cells * fraction + 0.5))))
    cosine = xyz @ centre
    land = np.argpartition(cosine, -count)[-count:]
    land = land[np.argsort(-cosine[land], kind="stable")].astype(np.int32)
    land_mask = np.zeros(sim_mesh.num_cells, dtype=bool)
    land_mask[land] = True

    continent = world.add_plate("Continental scene")
    ocean = world.add_plate("Background ocean")
    ocean.isBackground = True
    continent.crust.create_many(
        land, CrustType["CONTINENT"], -float(world.config.get("initialLandAgeMyr", 2000.0)), 1
    )
    water = np.flatnonzero(~land_mask)
    ocean.crust.create_many(
        water, CrustType["OCEAN"], -float(world.config.get("initialOceanAgeMyr", 150.0))
    )
    for plate in world.plates:
        plate.refresh_extent(sim_mesh.xyz)
    world.terraneLetters = {1: "A"}
    world.log(
        EventKind["START"],
        "Continental fixture",
        {"plate": continent.id, "continentalCells": int(land.size)},
    )
    return {
        "world": world,
        "start": {
            "scenario": "continental",
            "centre": centre.tolist(),
            "continentId": continent.id,
            "oceanId": ocean.id,
            "landCells": int(land.size),
            "continentalFraction": land.size / sim_mesh.num_cells,
        },
    }


def build_oceanic_scene(sim_mesh: Any, seed: int = 0) -> dict[str, Any]:
    """Build two oceanic plates separated by a deterministic great-circle boundary."""
    world = TectonicWorld(sim_mesh, seed)
    xyz = _xyz(sim_mesh)
    pole = np.asarray(random_point(make_rng(seed * 31 + 307)), dtype=np.float64)
    signed = xyz @ pole
    first = np.flatnonzero(signed >= 0.0).astype(np.int32)
    second = np.flatnonzero(signed < 0.0).astype(np.int32)
    if not first.size or not second.size:
        order = np.argsort(signed, kind="stable")
        midpoint = max(1, len(order) // 2)
        second, first = order[:midpoint].astype(np.int32), order[midpoint:].astype(np.int32)

    age = -float(world.config.get("initialOceanAgeMyr", 150.0))
    plate_a = world.add_plate("Oceanic plate A")
    plate_b = world.add_plate("Oceanic plate B")
    plate_a.crust.create_many(first, CrustType["OCEAN"], age)
    plate_b.crust.create_many(second, CrustType["OCEAN"], age)
    for plate in world.plates:
        plate.refresh_extent(sim_mesh.xyz)
    world.log(
        EventKind["START"],
        "Two-plate oceanic fixture",
        {"plates": [plate_a.id, plate_b.id]},
    )
    return {
        "world": world,
        "start": {
            "scenario": "oceanic",
            "boundaryPole": pole.tolist(),
            "plateA": plate_a.id,
            "plateB": plate_b.id,
            "plateACells": int(first.size),
            "plateBCells": int(second.size),
        },
    }


def build_mixed_scene(sim_mesh: Any, seed: int = 0, overrides: dict | None = None) -> dict[str, Any]:
    """Build the full mixed supercontinent, apron and background-ocean fixture."""
    result = build_start_world(sim_mesh, seed, overrides)
    result["start"]["scenario"] = "mixed"
    return result


def build_scenario_scene(
    sim_mesh: Any,
    seed: int = 0,
    scenario: str = "mixed",
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Dispatch one of the three supported deterministic fixture builders."""
    key = str(scenario).strip().lower()
    if key == "continental":
        return build_continental_scene(sim_mesh, seed)
    if key == "oceanic":
        return build_oceanic_scene(sim_mesh, seed)
    if key == "mixed":
        return build_mixed_scene(sim_mesh, seed, overrides)
    raise ValueError("scenario must be one of: continental, oceanic, mixed")


# Public camel-case aliases for mechanically ported callers.
buildContinentalScene = build_continental_scene  # noqa: N816 -- preserve upstream V2 public spelling
buildOceanicScene = build_oceanic_scene  # noqa: N816 -- preserve upstream V2 public spelling
buildMixedScene = build_mixed_scene  # noqa: N816 -- preserve upstream V2 public spelling
buildScenarioScene = build_scenario_scene  # noqa: N816 -- preserve upstream V2 public spelling
