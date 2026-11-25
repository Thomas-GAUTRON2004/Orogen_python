"""Differential checks for full resolve gather, subduction, growth and gap ownership."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.resolve import resolve

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/resolve-orchestration-differential-runner.mjs"


def _spec(
    xyz: list[float],
    adj_offset: list[int],
    adj_list: list[int],
    plates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "xyz": xyz,
        "adjOffset": adj_offset,
        "adjList": adj_list,
        "plates": plates,
    }


def _payload() -> dict[str, Any]:
    one_cell = {
        "xyz": [1.0, 0.0, 0.0],
        "adjOffset": [0, 0],
        "adjList": [],
    }
    two_cells = {
        "xyz": [1.0, 0.0, 0.0, 0.999, 0.045, 0.0],
        "adjOffset": [0, 1, 2],
        "adjList": [1, 0],
    }
    return {
        "cases": [
            _spec(
                one_cell["xyz"],
                one_cell["adjOffset"],
                one_cell["adjList"],
                [
                    {
                        "name": "Ocean",
                        "mask": [1],
                        "type": [CrustType["OCEAN"]],
                    }
                ],
            ),
            _spec(
                one_cell["xyz"],
                one_cell["adjOffset"],
                one_cell["adjList"],
                [
                    {
                        "name": "North",
                        "mask": [1],
                        "type": [CrustType["CONTINENT"]],
                    },
                    {
                        "name": "South",
                        "mask": [1],
                        "type": [CrustType["CRATON"]],
                    },
                ],
            ),
            _spec(
                one_cell["xyz"],
                one_cell["adjOffset"],
                one_cell["adjList"],
                [
                    {
                        "name": "Oceanic",
                        "mask": [1],
                        "type": [CrustType["OCEAN"]],
                    },
                    {
                        "name": "Continental",
                        "mask": [1],
                        "type": [CrustType["CONTINENT"]],
                    },
                ],
            ),
            _spec(
                one_cell["xyz"],
                one_cell["adjOffset"],
                one_cell["adjList"],
                [
                    {
                        "name": "Continental",
                        "mask": [1],
                        "type": [CrustType["CONTINENT"]],
                    },
                    {
                        "name": "Oceanic",
                        "mask": [1],
                        "type": [CrustType["OCEAN"]],
                    },
                ],
            ),
            _spec(
                two_cells["xyz"],
                two_cells["adjOffset"],
                two_cells["adjList"],
                [
                    {
                        "name": "Growing plate",
                        "mask": [0, 1],
                        "type": [CrustType["OCEAN"], CrustType["OCEAN"]],
                    }
                ],
            ),
            _spec(
                two_cells["xyz"],
                two_cells["adjOffset"],
                two_cells["adjList"],
                [
                    {
                        "name": "Passive background",
                        "mask": [0, 1],
                        "type": [CrustType["OCEAN"], CrustType["OCEAN"]],
                        "isBackground": True,
                    }
                ],
            ),
        ]
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return run_js_reference(RUNNER, payload)


def _world(spec: dict[str, Any]):
    num_cells = len(spec["xyz"]) // 3
    mesh = SimpleNamespace(
        num_cells=num_cells,
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
        adj_list=np.asarray(spec["adjList"], dtype=np.int32),
        edge_rad=0.02,
        edge_km=100.0,
        cell_area_km2=10000.0,
        index=SimpleNamespace(nearest=lambda *_point: 0),
        _bins=SimpleNamespace(
            bin_of=np.zeros(num_cells, dtype=np.uint16),
            neighbours=(np.asarray([0], dtype=np.int32),),
            count=1,
            at=lambda *_point: 0,
        ),
    )
    plates = []
    for plate_id, source in enumerate(spec["plates"]):
        plate = Plate(plate_id, num_cells)
        plate.name = source["name"]
        plate.alive = source.get("alive", True)
        plate.isBackground = source.get("isBackground", False)
        for cell, occupied in enumerate(source["mask"]):
            if occupied:
                plate.crust.create(
                    cell,
                    source["type"][cell],
                    source.get("formedAt", [0.0] * num_cells)[cell],
                )
        plates.append(plate)

    world = SimpleNamespace(
        mesh=mesh,
        plates=plates,
        owner=np.full(num_cells, -1, dtype=np.int16),
        ownerCell=np.full(num_cells, -1, dtype=np.int32),
        claimCount=np.zeros(num_cells, dtype=np.uint16),
        landClaims=np.zeros(num_cells, dtype=np.uint8),
        collider=np.full(num_cells, -1, dtype=np.int16),
        consumedAt=np.zeros(num_cells, dtype=np.uint8),
        provisional=np.zeros(num_cells, dtype=np.uint8),
        oceanOverrider=np.full(num_cells, -1, dtype=np.int16),
        budget={"created": 0, "consumed": 0},
        time=12.0,
        occupancyStamp=0,
        config={"closingCmYr": 0.0},
        log=lambda *_args: None,
    )
    return world


def _snapshot(world, result) -> dict[str, Any]:
    return {
        "result": result,
        "owner": world.owner.tolist(),
        "ownerCell": world.ownerCell.tolist(),
        "claimCount": world.claimCount.tolist(),
        "landClaims": world.landClaims.tolist(),
        "collider": world.collider.tolist(),
        "consumedAt": world.consumedAt.tolist(),
        "provisional": world.provisional.tolist(),
        "budget": dict(world.budget),
        "plates": [
            {
                "alive": plate.alive,
                "isBackground": plate.isBackground,
                "mask": plate.crust.mask.tolist(),
                "type": plate.crust.type.tolist(),
                "count": plate.crust.count,
                "dirty": plate.dirty,
                "reshaped": plate.reshaped,
                "bins": None if plate.bins is None else plate.bins.tolist(),
                "fzU": plate.crust.fzU.tolist(),
            }
            for plate in world.plates
        ],
    }


def test_full_resolve_gather_growth_and_gap_ownership_match_javascript(
    javascript, payload
):
    expected = []
    for spec in payload["cases"]:
        world = _world(spec)
        result = resolve(world)
        expected.append(_snapshot(world, result))

    assert_js_python_equal(expected, javascript, atol=2e-7)
    assert expected[0]["result"] == {
        "created": 0,
        "consumed": 0,
        "collisionCells": 0,
    }
    assert expected[1]["result"]["collisionCells"] == 1
    assert expected[2]["result"]["consumed"] == 1
    assert expected[3]["result"]["consumed"] == 1
    assert expected[4]["result"]["created"] == 1
    assert expected[4]["owner"] == [0, 0]
    assert expected[5]["result"]["created"] == 0
    assert expected[5]["owner"] == [-1, 0]
