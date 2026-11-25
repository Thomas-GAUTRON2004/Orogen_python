"""Differential checks for a newly tracked, non-convergent continent collision."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.collision import track_collisions
from orogen.v2.tectonics.crust import CrustType

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/collision-record-differential-runner.mjs"

RECORD_FIELDS = (
    "upper",
    "lower",
    "startedAt",
    "lastActive",
    "shorteningKm",
    "frontKm",
    "closingCmYr",
    "meanRateCmYr",
    "stalledMyr",
    "contactMyr",
    "cells",
    "sutured",
    "underthrustDue",
    "underthrustCells",
    "frontMaxKm",
)


def _payload() -> dict[str, Any]:
    return {
        "cases": [
            {
                "plates": [
                    {
                        "name": "Plate A",
                        "type": CrustType["CONTINENT"],
                        "capCenter": [0.0, 0.8, 0.6],
                    },
                    {
                        "name": "Plate B",
                        "type": CrustType["CONTINENT"],
                        "capCenter": [0.0, -0.8, 0.6],
                    },
                ],
                "config": {
                    "collisionForgetMyr": 50.0,
                    "riftJitterFrontKm": 0.0,
                    "riftPushMyr": 10.0,
                    "collisionJitterFrontKm": 0.0,
                    "arcAccreteMinKm2": 100.0,
                    "terraneMinAreaKm2": 1.0,
                    "arcScrapeLogKm2": 1.0,
                    "terraneMaxAreaKm2": 1.0,
                    "terraneForeArcKm": 100.0,
                    "underthrustShare": 0.0,
                    "farCollisionShare": 0.0,
                    "sutureShorteningKm": 1000.0,
                    "sutureStallCmYr": 0.4,
                    "sutureStallMyr": 100.0,
                    "sutureStallMinKm": 100.0,
                    "sutureContactMyr": 100.0,
                    "sutureOverlapKm2": 1_000_000_000.0,
                    "sutureMinContactKm": 100.0,
                    "sutureCheckMyr": 100.0,
                    "sutureGapWidthKm": 10.0,
                    "sutureStallGiveUpMyr": 100.0,
                    "sutureGapStallMyr": 100.0,
                    "sutureLockMaxMyr": 100.0,
                    "sutureStackKm": 1000.0,
                    "ringSeaKm2": 1_000_000.0,
                },
            }
        ]
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return run_js_reference(RUNNER, payload)


def _world(spec: dict[str, Any]):
    mesh = SimpleNamespace(
        num_cells=1,
        xyz=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        adj_offset=np.asarray([0, 0], dtype=np.int32),
        adj_list=np.asarray([], dtype=np.int32),
        edge_rad=0.02,
        edge_km=100.0,
        cell_area_km2=10000.0,
        index=SimpleNamespace(nearest=lambda *_point: 0),
    )
    plates = []
    for plate_id, source in enumerate(spec["plates"]):
        crust = SimpleNamespace(
            mask=np.asarray([1], dtype=np.uint8),
            type=np.asarray([source["type"]], dtype=np.uint8),
            flags=np.asarray([0], dtype=np.uint16),
            terrane=np.asarray([plate_id + 1], dtype=np.int32),
            formedAt=np.asarray([0.0], dtype=np.float64),
            slabKm=np.asarray([0.0], dtype=np.float32),
            coverKm=np.asarray([0.0], dtype=np.float32),
            count=1,
        )
        plates.append(
            SimpleNamespace(
                id=plate_id,
                name=source["name"],
                alive=True,
                isBackground=False,
                born=0.0,
                riftPartner=-1,
                stats={"continentalCells": 1},
                omega=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
                capCenter=np.asarray(source["capCenter"], dtype=np.float64),
                crust=crust,
                cell_at=lambda *_args: 0,
                to_world=lambda x, y, z: np.asarray([x, y, z]),
                to_plate_frame=lambda x, y, z: np.asarray([x, y, z]),
            )
        )
    events = []
    world = SimpleNamespace(
        mesh=mesh,
        owner=np.asarray([0], dtype=np.int16),
        ownerCell=np.asarray([0], dtype=np.int32),
        landClaims=np.asarray([2], dtype=np.uint8),
        collider=np.asarray([1], dtype=np.int16),
        plates=plates,
        collisions={},
        config=spec["config"],
        time=12.0,
        log=lambda kind, text, data: events.append(
            {"kind": kind, "text": text, "data": data}
        ),
    )
    return world, events


def test_new_collision_record_and_non_convergent_progress_match_javascript(
    javascript, payload
):
    expected = []
    for spec in payload["cases"]:
        world, events = _world(spec)
        result = track_collisions(world, 1.0)
        record = world.collisions.get("0:1")
        snapshot = {
            "result": {
                "started": result["started"],
                "tracked": result["tracked"],
                "underthrustCells": result["underthrustCells"],
                "changed": result["changed"],
            },
            "record": (
                None
                if record is None
                else {field: record[field] for field in RECORD_FIELDS}
            ),
            "events": events,
        }
        expected.append(snapshot)

    assert_js_python_equal(expected, javascript, atol=1e-12)
    assert expected[0]["result"] == {
        "started": 1,
        "tracked": 1,
        "underthrustCells": 0,
        "changed": False,
    }
    assert expected[0]["record"]["upper"] == 0
    assert expected[0]["record"]["lower"] == 1
    assert expected[0]["record"]["closingCmYr"] == 0.0
    assert expected[0]["record"]["sutured"] is False
