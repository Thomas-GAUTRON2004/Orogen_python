"""Differential checks for rift-side partitioning and exterior-ocean tracing."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.ops.rift_partition import (
    elongation_of,
    exterior_sea,
    masked_cells,
    partition,
    sides_of,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/rift-partition-differential-runner.mjs"


def _xyz(latitude: float, longitude: float) -> list[float]:
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    return [
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ]


def _base_case() -> dict[str, Any]:
    rows, columns = 6, 6
    xyz = [
        _xyz(-25.0 + row * 10.0, -25.0 + column * 10.0)
        for row in range(rows)
        for column in range(columns)
    ]
    adjacency = []
    offsets = [0]
    for row in range(rows):
        for column in range(columns):
            neighbours = []
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                nr, nc = row + dr, column + dc
                if 0 <= nr < rows and 0 <= nc < columns:
                    neighbours.append(nr * columns + nc)
            adjacency.extend(neighbours)
            offsets.append(len(adjacency))

    types = []
    terrane = []
    for _row in range(rows):
        for column in range(columns):
            if column == 3:
                types.append(CrustType["OCEAN"])
                terrane.append(0)
            elif column in (0, 4):
                types.append(CrustType["CRATON"])
                terrane.append(1 if column == 0 else 2)
            else:
                types.append(CrustType["CONTINENT"])
                terrane.append(0)

    return {
        "numCells": rows * columns,
        "xyz": [value for point in xyz for value in point],
        "adjOffset": offsets,
        "adjList": adjacency,
        "edgeKm": 2500.0,
        "mask": [1] * (rows * columns),
        "type": types,
        "terrane": terrane,
        "path": [9, 15, 21, 27],
        "cells": None,
    }


def _payload() -> dict[str, Any]:
    base = _base_case()
    outside_sea = {**base, "mask": list(base["mask"])}
    outside_sea["mask"][3] = 0
    outside_sea["mask"][33] = 0

    all_ocean = {
        **base,
        "type": [CrustType["OCEAN"]] * base["numCells"],
        "terrane": [0] * base["numCells"],
    }
    empty = {
        **base,
        "mask": [0] * base["numCells"],
        "type": [CrustType["OCEAN"]] * base["numCells"],
        "terrane": [0] * base["numCells"],
        "path": [],
        "cells": [],
    }
    single_cell_path = {**base, "path": [9]}
    return {
        "cases": [base, outside_sea, all_ocean, empty, single_cell_path]
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return run_js_reference(RUNNER, payload)


def _mesh(spec: dict[str, Any]):
    return SimpleNamespace(
        num_cells=spec["numCells"],
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
        adj_list=np.asarray(spec["adjList"], dtype=np.int32),
        edge_km=spec["edgeKm"],
    )


def _crust(spec: dict[str, Any]):
    mask = np.asarray(spec["mask"], dtype=np.uint8)
    return SimpleNamespace(
        count=int(mask.sum()),
        mask=mask,
        type=np.asarray(spec["type"], dtype=np.uint8),
        terrane=np.asarray(spec["terrane"], dtype=np.int32),
    )


def _evaluate(spec: dict[str, Any]) -> dict[str, Any]:
    mesh = _mesh(spec)
    crust = _crust(spec)
    cells = masked_cells(crust) if spec["cells"] is None else spec["cells"]
    side, land, cratons = partition(mesh, crust, spec["path"], cells)
    return {
        "masked": masked_cells(crust).tolist(),
        "partition": {
            "side": side.tolist(),
            "land": land,
            "cratons": cratons,
        },
        "summary": sides_of(mesh, crust, spec["path"], cells),
        "exterior": exterior_sea(mesh, crust).tolist(),
        "elongation": [
            elongation_of(mesh, side, 1, cells),
            elongation_of(mesh, side, 2, cells),
        ],
    }


def test_rift_partition_and_exterior_sea_match_pinned_javascript(javascript, payload):
    expected = [_evaluate(case) for case in payload["cases"]]
    assert_js_python_equal(expected, javascript, atol=3e-6)
