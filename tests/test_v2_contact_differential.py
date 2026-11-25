"""Differential checks for world-frame contacts and occupancy-stamp caching."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.contact import contact_at

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/contact-differential-runner.mjs"


def _payload() -> dict[str, Any]:
    return {
        "cases": [
            {
                "numCells": 2,
                "xyz": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "adjOffset": [0, 1, 2],
                "adjList": [1, 0],
                "owner": [0, 0],
                "queries": [{"cell": 0}, {"cell": 1}],
            },
            {
                "numCells": 2,
                "xyz": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "adjOffset": [0, 1, 2],
                "adjList": [1, 0],
                "owner": [0, 1],
                "queries": [{"cell": 0}, {"cell": 1}],
            },
            {
                "numCells": 3,
                "xyz": [
                    1.0, 0.0, 0.0,
                    0.8, 0.6, 0.0,
                    0.8, -0.6, 0.0,
                ],
                "adjOffset": [0, 2, 3, 4],
                "adjList": [1, 2, 0, 0],
                "owner": [0, 1, 1],
                "queries": [{"cell": 0}],
            },
            {
                "numCells": 3,
                "xyz": [
                    0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0,
                    -1.0, 0.0, 0.0,
                ],
                "adjOffset": [0, 2, 3, 4],
                "adjList": [1, 2, 0, 0],
                "owner": [0, 1, 2],
                "queries": [{"cell": 0}],
            },
            {
                "numCells": 3,
                "xyz": [
                    1.0, 0.0, 0.0,
                    0.8, 0.6, 0.0,
                    0.8, -0.6, 0.0,
                ],
                "adjOffset": [0, 2, 3, 4],
                "adjList": [1, 2, 0, 0],
                "owner": [0, 1, 0],
                "occupancyStamp": 4,
                "queries": [
                    {"cell": 0},
                    {"cell": 0, "owner": [0, 0, 1]},
                    {"cell": 0, "occupancyStamp": 5},
                ],
            },
        ]
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return run_js_reference(RUNNER, payload)


def _run_case(spec: dict[str, Any]) -> list[dict[str, Any]]:
    mesh = SimpleNamespace(
        num_cells=spec["numCells"],
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
        adj_list=np.asarray(spec["adjList"], dtype=np.int32),
    )
    world = SimpleNamespace(
        mesh=mesh,
        owner=np.asarray(spec["owner"], dtype=np.int16),
        occupancyStamp=spec.get("occupancyStamp", 0),
    )
    results = []
    for query in spec["queries"]:
        if "owner" in query:
            world.owner[:] = query["owner"]
        if "occupancyStamp" in query:
            world.occupancyStamp = query["occupancyStamp"]
        out = np.asarray([9.0, 9.0, 9.0, 9.0], dtype=np.float64)
        across = contact_at(world, query["cell"], out)
        results.append({"across": across, "normal": out.tolist()})
    return results


def test_contact_cells_normals_and_cache_invalidation_match_javascript(
    javascript, payload
):
    expected = [_run_case(spec) for spec in payload["cases"]]
    assert_js_python_equal(expected, javascript, atol=1e-12)

    assert expected[0] == [
        {"across": -1, "normal": [9.0, 9.0, 9.0, 9.0]},
        {"across": -1, "normal": [9.0, 9.0, 9.0, 9.0]},
    ]
    assert expected[1][0]["across"] == 1
    assert expected[1][0]["normal"][3] == 1
    assert expected[2][0]["normal"] == pytest.approx([-1.0, 0.0, 0.0, 2.0])
    assert expected[3][0]["normal"] == pytest.approx([1.0, 0.0, 0.0, 2.0])
    assert [query["across"] for query in expected[4]] == [1, 1, 2]
