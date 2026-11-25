"""Full-state differential test for initial-world generation.

The JS side is the original init/start-world.js implementation from the pinned
upstream submodule. Both engines receive exactly the same mesh arrays, seed and
configuration; all observable start-world and crust arrays are compared.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from _differential import assert_js_python_equal, run_js_reference
from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.config import TECTONICS
from orogen.v2.tectonics.initial_world import build_start_world
from orogen.v2.tectonics.crust import CELL_FIELDS

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/initial-world-differential-runner.mjs"


def _safe(value: Any) -> Any:
    """Match JSON.stringify semantics for typed arrays and non-finite numbers."""
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


def _python_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    world, start = result["world"], result["start"]
    plates = []
    for plate in world.plates:
        plates.append(
            {
                "id": plate.id,
                "name": plate.name,
                "born": plate.born,
                "alive": plate.alive,
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
    start_state = {
        "cratons": start["cratons"],
        "centre": start["centre"],
        "rimCentre": start["rimCentre"],
        "reference": start["reference"],
        "elongation": start["elongation"],
        "seams": start["seams"],
        "owner": start["owner"],
        "skirtKm": start["skirtKm"],
        "rim": start["rim"],
        "passiveFrom": start["passiveFrom"],
        "passiveTo": start["passiveTo"],
        "attempts": start["attempts"],
        "cratonAreaScale": start["cratonAreaScale"],
        "fractionMet": start["fractionMet"],
        "continentId": start["continentId"],
        "oceanId": start["oceanId"],
        "landCells": start["landCells"],
        "continentalFraction": start["continentalFraction"],
        "apronCells": start["apronCells"],
        "trench": start["trench"],
    }
    world_state = {
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
        "startSeamKm": world.startSeamKm,
    }
    return _safe({"start": start_state, "world": world_state, "plates": plates})


def test_production_start_world_matches_upstream_full_state() -> None:
    # 512 is the smallest mesh that selects the production craton/landmass pipeline.
    mesh = build_sim_mesh(512, 911)
    seed = 912
    payload = {
        "seed": seed,
        "config": _safe(TECTONICS),
        "mesh": _mesh_payload(mesh),
    }
    javascript = run_js_reference(RUNNER, payload, timeout=600)
    python = _python_snapshot(build_start_world(mesh, seed, overrides=TECTONICS))
    assert_js_python_equal(python, javascript, rtol=2e-5, atol=2e-5)
