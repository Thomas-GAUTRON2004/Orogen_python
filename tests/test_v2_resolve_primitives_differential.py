"""Differential checks for divergence candidate selection and gap ownership rules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.resolve import candidate_cells, decide_birth

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/resolve-primitives-differential-runner.mjs"


def _mesh_spec() -> dict[str, Any]:
    xyz = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.7],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    neighbours = ([1], [0, 2], [1, 3], [2, 4], [3, 5], [4])
    adjacency: list[int] = []
    offsets = [0]
    for cells in neighbours:
        adjacency.extend(cells)
        offsets.append(len(adjacency))
    return {
        "numCells": 6,
        "xyz": [value for point in xyz for value in point],
        "adjOffset": offsets,
        "adjList": adjacency,
        "edgeRad": 0.02,
    }


def _payload() -> dict[str, Any]:
    mesh = _mesh_spec()
    candidate = {
        **mesh,
        "plate": {
            "id": 0,
            "mask": [1, 0, 1, 0, 0, 1],
            "capFloor": 0.5,
            "underCell": 0,
        },
        "gaps": [0, 1, 2],
    }
    decisions = [
        {
            **mesh,
            "cell": 0,
            "grower": {
                "id": 0,
                "mask": [0, 0, 0, 0, 0, 0],
                "capFloor": -1.0,
                "underCell": 0,
            },
            "rival": None,
        },
        {
            **mesh,
            "cell": 0,
            "grower": {
                "id": 0,
                "mask": [1, 0, 0, 0, 0, 0],
                "capFloor": -1.0,
                "underCell": 0,
            },
            "rival": {
                "id": 1,
                "mask": [0, 1, 0, 0, 0, 0],
                "capFloor": -1.0,
                "underCell": 1,
            },
        },
        {
            **mesh,
            "cell": 0,
            "grower": {
                "id": 0,
                "mask": [1, 0, 0, 0, 0, 0],
                "flags": [1, 0, 0, 0, 0, 0],
                "wingKm": [0.0] * 6,
                "capFloor": -1.0,
                "underCell": 0,
            },
            "rival": None,
        },
        {
            **mesh,
            "xyz": [
                1.0, 0.0, 0.0,
                0.9999995, 0.001, 0.0,
                0.9999995, 0.001, 0.0,
                0.0, 0.0, 1.0,
                0.0, 1.0, 0.0,
                -1.0, 0.0, 0.0,
            ],
            "edgeRad": 0.02,
            "cell": 0,
            "grower": {
                "id": 0,
                "mask": [1, 0, 0, 0, 0, 0],
                "capFloor": -1.0,
                "underCell": 0,
            },
            "rival": {
                "id": 1,
                "mask": [0, 0, 1, 0, 0, 0],
                "capFloor": -1.0,
                "underCell": 1,
            },
        },
        {
            **mesh,
            "xyz": [
                1.0, 0.0, 0.0,
                0.9995, 0.01, 0.0,
                0.9992, 0.04, 0.0,
                0.0, 0.0, 1.0,
                0.0, 1.0, 0.0,
                -1.0, 0.0, 0.0,
            ],
            "edgeRad": 0.02,
            "cell": 0,
            "grower": {
                "id": 0,
                "mask": [1, 0, 0, 0, 0, 0],
                "omega": [0.0, 0.0, 1.0],
                "capFloor": -1.0,
                "underCell": 0,
            },
            "rival": {
                "id": 1,
                "mask": [0, 0, 1, 0, 0, 0],
                "capFloor": -1.0,
                "underCell": 1,
            },
        },
        {
            **mesh,
            "xyz": [
                1.0, 0.0, 0.0,
                0.9995, 0.01, 0.0,
                0.9992, 0.04, 0.0,
                0.0, 0.0, 1.0,
                0.0, 1.0, 0.0,
                -1.0, 0.0, 0.0,
            ],
            "edgeRad": 0.02,
            "cell": 0,
            "grower": {
                "id": 0,
                "mask": [1, 0, 0, 0, 0, 0],
                "omega": [0.0, 0.0, 1.0],
                "capFloor": -1.0,
                "underCell": 0,
            },
            "rival": {
                "id": 1,
                "mask": [0, 0, 1, 0, 0, 0],
                "capFloor": -1.0,
                "underCell": 1,
            },
        },
    ]
    return {"candidate": candidate, "decisions": decisions}


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> dict[str, Any]:
    return run_js_reference(RUNNER, payload)


def _mesh(spec: dict[str, Any]):
    mesh = SimpleNamespace(
        num_cells=spec["numCells"],
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
        adj_list=np.asarray(spec["adjList"], dtype=np.int32),
        edge_rad=spec["edgeRad"],
        index=SimpleNamespace(nearest=lambda *_point: 0),
    )
    mesh._bins = SimpleNamespace(
        bin_of=np.zeros(spec["numCells"], dtype=np.uint16),
        neighbours=(np.asarray([0], dtype=np.int32),),
        count=1,
        at=lambda _x, _y, _z: 0,
    )
    return mesh


def _plate(spec: dict[str, Any]):
    return SimpleNamespace(
        id=spec["id"],
        isBackground=False,
        omega=np.asarray(spec.get("omega", [0.0, 0.0, 0.0]), dtype=np.float64),
        bins=np.asarray([7], dtype=np.uint32),
        crust=SimpleNamespace(
            mask=np.asarray(spec["mask"], dtype=np.uint8),
            type=np.zeros(len(spec["mask"]), dtype=np.uint8),
            flags=np.asarray(spec.get("flags", [0] * len(spec["mask"])), dtype=np.uint16),
            wingKm=np.asarray(spec.get("wingKm", [0.0] * len(spec["mask"])), dtype=np.float32),
        ),
        cap_in_world=lambda _mesh: np.asarray(
            [0.0, 0.0, 1.0, spec["capFloor"]]
        ),
        cell_at=lambda *_args: spec["underCell"],
        to_world=lambda x, y, z: np.asarray(
            [x, y - spec.get("worldYOffset", 0.0), z]
        ),
        to_plate_frame=lambda x, y, z: np.asarray([x, y, z]),
    )


def test_divergence_candidate_cells_match_javascript(javascript, payload):
    spec = payload["candidate"]
    mesh = _mesh(spec)
    grower = _plate(spec["plate"])
    expected = candidate_cells(
        SimpleNamespace(mesh=mesh, occupancyStamp=7),
        grower,
        spec["gaps"],
    )
    assert_js_python_equal(expected, javascript["candidates"], atol=0)
    assert expected == [1]


def test_divergence_birth_rejection_rules_match_javascript(javascript, payload):
    expected = []
    for spec in payload["decisions"]:
        mesh = _mesh(spec)
        grower = _plate(spec["grower"])
        rival = _plate(spec["rival"]) if spec["rival"] else None
        everyone = [(grower, [0.0, 0.0, 1.0, -1.0])]
        if rival is not None:
            everyone.append((rival, [0.0, 0.0, 1.0, -1.0]))
        result = decide_birth(
            SimpleNamespace(
                mesh=mesh,
                occupancyStamp=7,
                config={"closingCmYr": 0.0},
            ),
            grower,
            spec["cell"],
            everyone,
        )
        why = result.get("why")
        if why and why.startswith("not attached"):
            why = "not-attached"
        expected.append(
            {
                "born": result["born"],
                "why": why,
                "trenchParent": result.get("trenchParent"),
            }
        )
    assert_js_python_equal(expected, javascript["decisions"], atol=0)
    assert [item["why"] for item in expected] == [
        "not-attached",
        "not a gap: another plate has crust here",
        None,
        "too close to the mid-line",
        "closing boundary",
    ]
    assert expected[2]["trenchParent"] == 0
    assert expected[3]["born"] is False
    assert expected[4]["born"] is False
