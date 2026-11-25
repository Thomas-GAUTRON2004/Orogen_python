"""Differential coverage for far-field collection, Dijkstra spread and records."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.crust import CELL_FIELDS, CrustFlag, CrustType
from orogen.v2.tectonics.far_field import (
    evolve_far_field,
    far_take,
    feed_far_field,
    line_strike,
    recent_push,
    record_push,
    spread_push,
)
from orogen.v2.tectonics.world import TectonicWorld

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/far-field-differential-runner.mjs"
CONFIG = {
    "farFieldEveryMyr": 2.0,
    "farReachKm": 1000.0,
    "farFrontRefKm": 1000.0,
    "farMaxKm": 1000.0,
    "farCratonCost": 0.3,
    "farWeakCost": 1.6,
    "farCratonTake": 0.15,
    "farWeakTake": 1.6,
    "farFoldMyr": 100.0,
    "farReactivate": 0.0,
}


def _mesh() -> SimpleNamespace:
    angle = np.arange(5, dtype=np.float64) * 0.01
    xyz = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(5))).astype(
        np.float32
    )
    return SimpleNamespace(
        num_cells=5,
        xyz=xyz.reshape(-1),
        adj_offset=np.asarray([0, 1, 3, 5, 7, 8], dtype=np.int32),
        adj_list=np.asarray([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int32),
    )


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


@pytest.mark.parametrize("numerical_backend", ("numba", "python"))
def test_far_field_collection_spread_and_cell_records_match_upstream(
    numerical_backend: str, monkeypatch
) -> None:
    from orogen.v2.tectonics import far_field as far_field_module

    if numerical_backend == "python":
        monkeypatch.setattr(far_field_module, "NUMBA_AVAILABLE", False)
    mesh = _mesh()
    payload_mesh = {
        "numCells": mesh.num_cells,
        "xyz": mesh.xyz.tolist(),
        "adjOffset": mesh.adj_offset.tolist(),
        "adjList": mesh.adj_list.tolist(),
    }
    javascript = run_js_reference(
        RUNNER,
        {"seed": 771, "mesh": payload_mesh, "config": CONFIG},
    )

    world = TectonicWorld(mesh, 771, overrides=CONFIG)
    plate = world.add_plate("Reference")
    types = [
        CrustType["CONTINENT"],
        CrustType["CRATON"],
        CrustType["CONTINENT"],
        CrustType["CONTINENT"],
        CrustType["OCEAN"],
    ]
    for cell, crust_type in enumerate(types):
        plate.crust.create(cell, crust_type, -20, cell + 1)
    for cell in (1, 2, 3):
        plate.crust.flags[cell] |= CrustFlag["SUTURE"]
    plate.crust.flags[4] |= CrustFlag["FAILED_RIFT"]
    world.time = 10.0

    direct = spread_push(world, plate, {0: 100.0}, 1000.0, world.time)
    record_push(plate.crust, 3, 4.5, 200.0, mesh.xyz, 0, 15.0, 100.0)
    strike = line_strike(plate.crust, mesh, 2)
    no_strike = line_strike(plate.crust, mesh, 0)
    factors = {
        "craton": far_take(world.config, CrustType["CRATON"], 0),
        "weak": far_take(world.config, CrustType["CONTINENT"], CrustFlag["SUTURE"]),
        "normal": far_take(world.config, CrustType["CONTINENT"], 0),
        "recent": recent_push(10, 5, 15, 100),
        "future": recent_push(10, 15, 5, 100),
        "zero": recent_push(0, 0, 100, 100),
    }

    dead = world.add_plate("Dead")
    dead.alive = False
    feed_far_field(world, plate, "front", [0, 1], [10, 20], 500, 0.5)
    feed_far_field(world, plate, "front", [1, 2], [2, 4], 700, 0.25)
    feed_far_field(world, plate, "ignored-zero", [0], 10, 900, 0)
    feed_far_field(world, plate, "ignored-empty", [], 10, 900, 1)
    feed_far_field(world, dead, "dead-front", [1], 30, 800, 1)
    pending = [
        {
            "key": key,
            "plate": value["plate"].id,
            "cells": sorted(value["cells"].items()),
            "frontKm": value["frontKm"],
        }
        for key, value in world.farPending.items()
    ]
    early = evolve_far_field(world, 1)
    due = evolve_far_field(world, 1)
    crust = {field: getattr(plate.crust, field) for field in CELL_FIELDS}
    crust["mask"] = plate.crust.mask
    python = _safe(
        {
            "direct": {
                "cells": direct["cells"],
                "inc": direct["inc"],
                "from": direct["from"],
                "dist": direct["dist"],
            },
            "strike": strike,
            "noStrike": no_strike,
            "factors": factors,
            "pending": pending,
            "early": early,
            "due": due,
            "farClock": world.farClock,
            "pendingAfter": list(world.farPending),
            "crust": crust,
        }
    )
    assert_js_python_equal(python, javascript, rtol=3e-5, atol=3e-6)
