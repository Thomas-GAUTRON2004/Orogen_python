"""Differential checks for passive-margin subduction initiation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.subduction_init import initiate_subduction

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/subduction-init-differential-runner.mjs"


@dataclass(eq=False)
class _Mesh:
    num_cells: int
    adj_offset: np.ndarray
    adj_list: np.ndarray
    xyz: np.ndarray
    edge_km: float
    index: object


class _Crust:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.mask = np.asarray(spec["mask"], dtype=np.uint8)
        self.type = np.asarray(spec["type"], dtype=np.uint8)
        self.flags = np.asarray(spec["flags"], dtype=np.uint16)
        self.formedAt = np.asarray(spec["formedAt"], dtype=np.float64)
        self.slabKm = np.asarray(spec["slabKm"], dtype=np.float32)

    def arm_trench(self, cell: int, depth: float) -> None:
        self.flags[cell] |= CrustFlag["TRENCH"]
        self.slabKm[cell] = max(float(self.slabKm[cell]), depth)


class _Plate:
    def __init__(self, plate_id: int, spec: dict[str, Any], num_cells: int) -> None:
        self.id = plate_id
        self.name = spec["name"]
        self.alive = spec["alive"]
        self.isBackground = False
        self.reshaped = spec["reshaped"]
        self.dirty = False
        self.offset = spec["offset"]
        self.world_y_offset = spec["worldYOffset"]
        self.crust = _Crust(spec)

    def cell_at(self, _index, x: float, _y: float, _z: float) -> int:
        return max(0, min(len(self.crust.mask) - 1, round(x) + self.offset))

    def to_world(self, x: float, y: float, z: float):
        return (x, y - self.world_y_offset, z)


def _grid() -> dict[str, Any]:
    neighbours = (
        (1, 3),
        (0, 2, 4),
        (1, 5),
        (0, 4),
        (1, 3, 5),
        (2, 4),
    )
    adjacency: list[int] = []
    offsets = [0]
    for cells in neighbours:
        adjacency.extend(cells)
        offsets.append(len(adjacency))
    xyz = []
    for cell in range(6):
        xyz.extend([float(cell % 3), float(cell // 3), 1.0])
    return {
        "numCells": 6,
        "adjOffset": offsets,
        "adjList": adjacency,
        "xyz": xyz,
        "edgeKm": 100.0,
        "owner": [0, 0, 0, 1, 1, 1],
        "ownerCell": [0, 1, 2, 3, 4, 5],
    }


def _config() -> dict[str, Any]:
    return {
        "subductionInitRatePerMyr": 0.5,
        "subductionInitMaxOpeningCmYr": 0.5,
        "subductionInitMinLengthKm": 200.0,
        "subductionInitAgeMyr": 100.0,
        "subductionInitAgeCap": 2.0,
        "subductionInitSlabKm": 30.0,
        "disperseFounderFactor": 0.2,
        "oceanPlateReachKm": 500.0,
    }


def _case(
    *,
    time: float = 200.0,
    formed_a: float = 50.0,
    formed_b: float = 60.0,
    convergence: float = 0.0,
    draws: tuple[float, ...] = (0.0,),
    cycle: dict[str, Any] | None = None,
    rate: float = 0.5,
    alive_b: bool = True,
    reshaped_a: bool = False,
    trench_a: bool = False,
) -> dict[str, Any]:
    grid = _grid()
    config = _config()
    config["subductionInitRatePerMyr"] = rate
    crust_a = {
        "mask": [1, 1, 1, 0, 0, 0],
        "type": [CrustType["OCEAN"]] * 6,
        "flags": [int(CrustFlag["TRENCH"]) if trench_a else 0] * 6,
        "formedAt": [formed_a] * 6,
        "slabKm": [0.0] * 6,
        "name": "Oceanic plate",
        "alive": True,
        "reshaped": False,
        "offset": 0,
        "worldYOffset": 0.0,
    }
    crust_b = {
        "mask": [0, 0, 0, 1, 1, 1],
        "type": [CrustType["CONTINENT"]] * 6,
        "flags": [0] * 6,
        "formedAt": [formed_b] * 6,
        "slabKm": [0.0] * 6,
        "name": "Continental plate",
        "alive": alive_b,
        "reshaped": reshaped_a,
        "offset": 3,
        "worldYOffset": 1.0,
    }
    return {
        **grid,
        "plates": [crust_a, crust_b],
        "config": config,
        "time": time,
        "cycle": cycle,
        "convergence": [convergence] * 6,
        "draws": list(draws),
        "dt": 1.0,
    }


def _payload() -> dict[str, Any]:
    return {
        "cases": [
            _case(),
            _case(formed_a=180.0, formed_b=180.0),
            _case(convergence=-1.0),
            _case(draws=(0.99,)),
            _case(alive_b=False),
            _case(reshaped_a=True),
            _case(trench_a=True),
            _case(rate=0.0),
            _case(
                time=250.0,
                formed_a=120.0,
                formed_b=130.0,
                draws=(0.01,),
                cycle={"phase": "dispersing", "breakupAt": 100.0},
            ),
            _case(
                time=250.0,
                formed_a=120.0,
                formed_b=130.0,
                draws=(0.05,),
                cycle={"phase": "dispersing", "breakupAt": 100.0},
            ),
        ]
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return run_js_reference(RUNNER, payload)["results"]


def _world(spec: dict[str, Any]):
    mesh = _Mesh(
        num_cells=spec["numCells"],
        adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
        adj_list=np.asarray(spec["adjList"], dtype=np.int32),
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        edge_km=spec["edgeKm"],
        index=object(),
    )
    plates = [
        _Plate(plate_id, plate, spec["numCells"])
        for plate_id, plate in enumerate(spec["plates"])
    ]
    events = []
    draws = iter(spec["draws"])
    world = SimpleNamespace(
        mesh=mesh,
        owner=np.asarray(spec["owner"], dtype=np.int16),
        ownerCell=np.asarray(spec["ownerCell"], dtype=np.int32),
        convergence=np.asarray(spec["convergence"], dtype=np.float64),
        plates=plates,
        config=spec["config"],
        time=spec["time"],
        cycle=spec["cycle"],
        rngSubduction=lambda: next(draws, 0.0),
        log=lambda kind, message, data: events.append(
            {"kind": int(kind), "message": message, "data": data}
        ),
    )
    return world, events


def _run_python(spec: dict[str, Any]) -> dict[str, Any]:
    world, events = _world(spec)
    started = initiate_subduction(world, spec["dt"])
    return {
        "started": started,
        "flags": [plate.crust.flags.tolist() for plate in world.plates],
        "slabKm": [plate.crust.slabKm.tolist() for plate in world.plates],
        "dirty": [plate.dirty for plate in world.plates],
        "events": events,
    }


def test_passive_margin_foundering_matches_pinned_javascript(javascript, payload):
    expected = [_run_python(case) for case in payload["cases"]]
    assert_js_python_equal(expected, javascript, atol=2e-6)
