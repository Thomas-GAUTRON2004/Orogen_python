"""Differential checks for continental cores and the supercontinent clock."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.supercontinent import landmasses, track_cycle
from orogen.v2.tectonics.world import EventKind

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/supercontinent-differential-runner.mjs"


def _xyz(latitude: float, longitude: float) -> list[float]:
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    return [
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ]


def _grid() -> dict[str, Any]:
    rows, columns = 5, 9
    xyz = [
        _xyz((2 - row) * 5.0, (column - 4) * 5.0)
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

    owners = [3] * (rows * columns)
    owner_cells = [0] * (rows * columns)
    types: list[list[int]] = [[], [], [], [], []]
    counts = [0] * len(types)
    for row in range(1, 4):
        for column in range(1, 4):
            cell = row * columns + column
            owners[cell] = 0
            owner_cells[cell] = counts[0]
            counts[0] += 1
            types[0].append(CrustType["CONTINENT"])
        for column in range(5, 8):
            cell = row * columns + column
            owners[cell] = 1
            owner_cells[cell] = counts[1]
            counts[1] += 1
            types[1].append(CrustType["CRATON"])

    bridge = 2 * columns + 4
    owners[bridge] = 2
    owner_cells[bridge] = counts[2]
    counts[2] += 1
    types[2].append(CrustType["CONTINENT"])

    island = 0
    owners[island] = 4
    owner_cells[island] = counts[4]
    counts[4] += 1
    types[4].append(CrustType["CONTINENT"])

    for cell, owner in enumerate(owners):
        if owner == 3:
            owner_cells[cell] = counts[3]
            counts[3] += 1
            types[3].append(CrustType["OCEAN"])

    return {
        "xyz": [value for point in xyz for value in point],
        "adjOffset": offsets,
        "adjList": adjacency,
        "owners": owners,
        "ownerCells": owner_cells,
        "plates": [{"types": values} for values in types],
        "cellAreaKm2": 120000.0,
        "mantleHeat": [0.1 + (cell % 7) * 0.1 for cell in range(rows * columns)],
    }


def _config() -> dict[str, Any]:
    return {
        "mantleOutflow": 1.0,
        "mantleInflow": 1.0,
        "riftHeatGain": 0.1,
        "cycleNeckKm": 1500.0,
        "superShare": 0.45,
        "dispersedShare": 0.2,
        "cycleHoldMyr": 0.0,
        "cycleCheckMyr": 10.0,
        "cycleGatherMyr": 80.0,
        "cycleDisperseMinMyr": 20.0,
        "cyclePeriodMinMyr": 300.0,
        "cyclePeriodMaxMyr": 500.0,
        "superHoldMarginKm": 300.0,
        "cycleUpwellingKm": 7000.0,
        "cycleRampMyr": 50.0,
        "dispersalMaxShare": 0.4,
    }


def _payload() -> dict[str, Any]:
    grid = _grid()
    config = _config()
    empty = {
        **grid,
        "plates": [
            {"types": [CrustType["OCEAN"]] * len(plate["types"])}
            for plate in grid["plates"]
        ],
        "config": config,
    }

    initial_cycle = {
        "up": None,
        "down": None,
        "upHeat": 0.0,
        "gather": 0.0,
        "share": 1.0,
        "landmasses": 1,
        "state": "dispersed",
        "since": 0.0,
        "phase": "none",
        "formedAt": 0.0,
        "period": 0.0,
        "breakupAt": None,
        "gatherAt": 0.0,
        "hold": 0.0,
        "holdKm": 0.0,
        "checkedAt": 0.0,
        "near": None,
        "highSince": None,
        "lowSince": None,
        "seenEvents": 0,
    }
    dispersed_cycle = {
        "up": [0.0, 0.0, 1.0],
        "down": [1.0, 0.0, 0.0],
        "upHeat": 0.0,
        "gather": 0.0,
        "share": 0.5,
        "landmasses": 1,
        "largestPlates": [0],
        "state": "assembled",
        "since": 5.0,
        "phase": "assembled",
        "formedAt": 5.0,
        "period": 300.0,
        "breakupAt": None,
        "gatherAt": None,
        "hold": 1.0,
        "holdKm": 1000.0,
        "checkedAt": 0.0,
        "near": None,
        "highSince": 0.0,
        "lowSince": None,
        "seenEvents": 0,
    }
    not_yet_formed = {**config, "superShare": 0.9}

    return {
        "massCases": [
            {**grid, "config": config},
            empty,
        ],
        "cycleCases": [
            {
                **grid,
                "config": config,
                "time": 100.0,
                "events": [],
                "cycle": initial_cycle,
                "rng": [0.25, 0.5],
            },
            {
                **grid,
                "config": config,
                "time": 40.0,
                "events": [{"kind": EventKind["RIFT"], "time": 20.0}],
                "cycle": dispersed_cycle,
                "rng": [0.25, 0.5],
            },
            {
                **grid,
                "config": not_yet_formed,
                "time": 100.0,
                "events": [],
                "cycle": initial_cycle,
                "rng": [0.25, 0.5],
            },
        ],
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> dict[str, Any]:
    return run_js_reference(RUNNER, payload)


@dataclass(eq=False)
class _Mesh:
    num_cells: int
    adj_offset: np.ndarray
    adj_list: np.ndarray
    cell_area_km2: float
    xyz: np.ndarray


def _world(spec: dict[str, Any]):
    plates = [
        SimpleNamespace(
            isBackground=False,
            crust=SimpleNamespace(type=np.asarray(plate["types"], dtype=np.uint8)),
        )
        for plate in spec["plates"]
    ]
    mesh = _Mesh(
        num_cells=len(spec["owners"]),
        adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
        adj_list=np.asarray(spec["adjList"], dtype=np.int32),
        cell_area_km2=spec["cellAreaKm2"],
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
    )
    draws = iter(spec.get("rng", []))
    return SimpleNamespace(
        config=dict(spec["config"]),
        time=spec.get("time", 0.0),
        events=[dict(event) for event in spec.get("events", [])],
        cycle=(
            None
            if spec.get("cycle") is None
            else dict(spec["cycle"])
        ),
        mesh=mesh,
        owner=np.asarray(spec["owners"], dtype=np.int16),
        ownerCell=np.asarray(spec["ownerCells"], dtype=np.int32),
        plates=plates,
        mantleHeat=np.asarray(spec["mantleHeat"], dtype=np.float32),
        rngCycle=lambda: next(draws, 0.0),
        log=lambda *args: None,
    )


def _cycle_snapshot(cycle):
    if cycle is None:
        return None
    snapshot = dict(cycle)
    if isinstance(snapshot.get("near"), np.ndarray):
        snapshot["near"] = snapshot["near"].tolist()
    if not math.isfinite(float(snapshot["gatherAt"])):
        snapshot["gatherAt"] = None
    return snapshot


def test_continental_landmasses_and_narrow_neck_match_upstream(javascript, payload):
    expected = [landmasses(_world(spec)) for spec in payload["massCases"]]
    assert_js_python_equal(expected, javascript["masses"], atol=2e-6)


def test_supercontinent_formation_and_breakup_clock_match_upstream(javascript, payload):
    expected = []
    for scenario in payload["cycleCases"]:
        world = _world(scenario)
        cycle = track_cycle(world)
        expected.append(_cycle_snapshot(cycle))
    assert_js_python_equal(expected, javascript["cycles"], atol=2e-6)
