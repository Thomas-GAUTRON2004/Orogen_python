"""Differential coverage for the deterministic tectonic scene catalogues.

Each Python catalogue builder and the corresponding original upstream JavaScript
builder receives the same mesh arrays and seed. The resulting world, plate and
crust states are compared field by field.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.budget_scenes import BUDGET_SCENES
from orogen.v2.tectonics.collision_scenes import COLLISION_SCENES
from orogen.v2.tectonics.crust import CELL_FIELDS
from orogen.v2.tectonics.event_scenes import EVENT_SCENES
from orogen.v2.tectonics.force_scenes import FORCE_SCENES
from orogen.v2.tectonics.hotspot_scenes import HOTSPOT_SCENES
from orogen.v2.tectonics.margin_scenes import MARGIN_SCENES
from orogen.v2.tectonics.rift_scenes import RIFT_SCENES
from orogen.v2.tectonics.terrain_scenes import TERRAIN_SCENES

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/scene-catalogue-differential-runner.mjs"
CATALOGUES = {
    "budget": BUDGET_SCENES,
    "collision": COLLISION_SCENES,
    "event": EVENT_SCENES,
    "force": FORCE_SCENES,
    "hotspot": HOTSPOT_SCENES,
    "margin": MARGIN_SCENES,
    "rift": RIFT_SCENES,
    "terrain": TERRAIN_SCENES,
}
SKIP_SCENES = {"history", "cycle", "farfield"}


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mesh_payload(mesh) -> dict[str, Any]:
    return {
        "numCells": int(mesh.num_cells),
        "xyz": np.asarray(mesh.xyz, dtype=np.float32).reshape(-1).tolist(),
        "adjOffset": np.asarray(mesh.adj_offset, dtype=np.int32).tolist(),
        "adjList": np.asarray(mesh.adj_list, dtype=np.int32).tolist(),
    }


def _cases() -> list[dict[str, Any]]:
    return [
        {"catalogue": catalogue, "id": scene["id"], "drive": callable(scene.get("drive"))}
        for catalogue, scenes in CATALOGUES.items()
        for scene in scenes
        if scene["id"] not in SKIP_SCENES and callable(scene.get("build"))
    ]


def _snapshot(world) -> dict[str, Any]:
    plates = []
    for plate in world.plates:
        plates.append(
            {
                "id": plate.id,
                "name": plate.name,
                "alive": plate.alive,
                "born": plate.born,
                "isBackground": bool(plate.isBackground),
                "omega": plate.omega,
                "quat": plate.quat,
                "stats": plate.stats,
                "crust": {
                    "mask": plate.crust.mask,
                    **{field: getattr(plate.crust, field) for field in CELL_FIELDS},
                },
            }
        )
    state = {
        "time": world.time,
        "step": world.step,
        "config": world.config,
        "budget": world.budget,
        "events": world.events,
        "owner": world.owner,
        "ownerCell": world.ownerCell,
        "claimCount": world.claimCount,
        "landClaims": world.landClaims,
        "collider": world.collider,
        "oceanOverrider": world.oceanOverrider,
        "consumedAt": world.consumedAt,
        "provisional": world.provisional,
        "provisionalLike": world.provisionalLike,
        "boundaryKind": world.boundaryKind,
        "convergence": world.convergence,
        "mantleHeat": world.mantleHeat,
        "plumes": getattr(world, "plumes", []),
        "hotspots": getattr(world, "hotspots", []),
        "plates": plates,
    }
    return _safe(state)


def test_scene_catalogues_match_upstream_javascript_full_world_state() -> None:
    mesh = build_sim_mesh(128, 941)
    seed = 942
    cases = _cases()
    javascript = run_js_reference(
        RUNNER,
        {"seed": seed, "mesh": _mesh_payload(mesh), "cases": cases},
        timeout=600,
    )
    python = []
    for item in cases:
        scene = next(
            scene
            for scene in CATALOGUES[item["catalogue"]]
            if scene["id"] == item["id"]
        )
        world = scene["build"](mesh, seed)
        if item["drive"] and callable(scene.get("drive")):
            scene["drive"](world)
        python.append(
            {
                "catalogue": item["catalogue"],
                "id": item["id"],
                "state": _snapshot(world),
            }
        )
    assert_js_python_equal(python, javascript, rtol=3e-6, atol=3e-6)
