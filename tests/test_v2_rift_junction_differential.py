"""Differential checks for weighted rift-junction selection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from _differential import assert_js_python_equal, run_js_reference
from orogen.v2.tectonics.ops.rift import choose_junction

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/rift-junction-differential-runner.mjs"


def _line_case(
    *,
    rng: float,
    min_arm_km: float = 2.0,
    craton_cells: tuple[int, ...] = (),
    weak_cells: tuple[int, ...] = (3, 5),
    land_cells: tuple[int, ...] = tuple(range(1, 8)),
) -> dict[str, Any]:
    num_cells = 9
    adjacency = [[cell - 1] if cell else [] for cell in range(num_cells)]
    for cell in range(num_cells - 1):
        adjacency[cell].append(cell + 1)
    offsets = [0]
    flattened: list[int] = []
    for neighbours in adjacency:
        flattened.extend(neighbours)
        offsets.append(len(flattened))

    mask = [0] * num_cells
    crust_type = [0] * num_cells
    flags = [0] * num_cells
    for cell in land_cells:
        mask[cell] = 1
        crust_type[cell] = 1
    for cell in craton_cells:
        crust_type[cell] = 2
    for cell in weak_cells:
        flags[cell] = 2

    return {
        "adjOffset": offsets,
        "adjList": flattened,
        "mask": mask,
        "type": crust_type,
        "flags": flags,
        "edgeKm": 1.0,
        "riftMinArmKm": min_arm_km,
        "riftWeaknessCost": 0.2,
        "rng": rng,
    }


def _python_results(cases: list[dict[str, Any]]) -> list[int]:
    results = []
    for spec in cases:
        mesh = SimpleNamespace(
            adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
            adj_list=np.asarray(spec["adjList"], dtype=np.int32),
            num_cells=len(spec["mask"]),
            edge_km=spec["edgeKm"],
        )
        crust = SimpleNamespace(
            mask=np.asarray(spec["mask"], dtype=np.uint8),
            type=np.asarray(spec["type"], dtype=np.uint8),
            flags=np.asarray(spec["flags"], dtype=np.uint8),
        )
        world = SimpleNamespace(
            mesh=mesh,
            config={
                "riftMinArmKm": spec["riftMinArmKm"],
                "riftWeaknessCost": spec["riftWeaknessCost"],
            },
        )
        plate = SimpleNamespace(crust=crust)
        results.append(choose_junction(world, plate, lambda value=float(spec["rng"]): value))
    return results


def test_weighted_rift_junction_selection_matches_upstream_javascript():
    cases = [
        _line_case(rng=0.0),
        _line_case(rng=0.5),
        _line_case(rng=0.999),
        _line_case(rng=0.999, craton_cells=(4,)),
    ]
    python_result = _python_results(cases)
    javascript = run_js_reference(RUNNER, {"cases": cases})

    assert_js_python_equal(python_result, javascript)


def test_rift_junction_returns_minus_one_when_no_interior_candidate_exists():
    cases = [
        _line_case(rng=0.5, land_cells=()),
        _line_case(rng=0.5, min_arm_km=4.0),
        _line_case(rng=0.5, craton_cells=(3, 4, 5)),
    ]
    python_result = _python_results(cases)
    javascript = run_js_reference(RUNNER, {"cases": cases})

    assert_js_python_equal(python_result, javascript)
