"""Differential checks for collision tracking and suture sea-ring decisions."""

from __future__ import annotations

import math
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
    "gapKm2",
    "gapWidestKm",
    "checkedAt",
    "gapBestKm2",
    "gapShrankAt",
    "ringKm2",
    "weldCandidate",
    "weldBlockedByRing",
    "mergeEligible",
    "ringLogged",
)


def _base_config() -> dict[str, Any]:
    return {
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
        "sutureGapKm": 200.0,
        "sutureGapWidthKm": 10.0,
        "sutureStallGiveUpMyr": 100.0,
        "sutureGapStallMyr": 100.0,
        "sutureLockMaxMyr": 100.0,
        "sutureStackKm": 1000.0,
        "ringSeaKm2": 1_000_000.0,
    }


def _one_cell_case() -> dict[str, Any]:
    config = _base_config()
    return {
        "xyz": [1.0, 0.0, 0.0],
        "adjOffset": [0, 0],
        "adjList": [],
        "edgeRad": 0.02,
        "edgeKm": 100.0,
        "cellAreaKm2": 10000.0,
        "owners": [0],
        "ownerCell": [0],
        "landClaims": [2],
        "collider": [1],
        "plates": [
            {
                "name": "Plate A",
                "type": [CrustType["CONTINENT"]],
                "mask": [1],
                "capCenter": [0.0, 0.8, 0.6],
            },
            {
                "name": "Plate B",
                "type": [CrustType["CONTINENT"]],
                "mask": [1],
                "capCenter": [0.0, -0.8, 0.6],
            },
        ],
        "config": config,
        "time": 12.0,
        "dt": 1.0,
    }


def _suture_ring_case() -> dict[str, Any]:
    config = _base_config()
    config.update(
        {
            "sutureContactMyr": 1.0,
            "sutureMinContactKm": 50.0,
            "sutureGapKm": 200.0,
            "sutureGapWidthKm": 120.0,
            "sutureStackKm": 500.0,
            "ringSeaKm2": 5000.0,
        }
    )
    # Cells 1 and 2 are opposite continental coasts around enclosed sea cell 3.
    # Cells 4 and 5 form a larger, separate ocean component (the world ocean).
    return {
        "xyz": [
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            0.0, -1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.2, 0.0, -0.9797958971132712,
        ],
        "adjOffset": [0, 2, 4, 6, 8, 9, 10],
        "adjList": [1, 2, 0, 3, 0, 3, 1, 2, 5, 4],
        "edgeRad": 0.02,
        "edgeKm": 100.0,
        "cellAreaKm2": 10000.0,
        "owners": [0, 0, 1, -1, -1, -1],
        "ownerCell": [0, 0, 0, -1, -1, -1],
        "landClaims": [2, 1, 1, 0, 0, 0],
        "collider": [1, -1, -1, -1, -1, -1],
        "plates": [
            {
                "name": "Upper plate",
                "type": [
                    CrustType["CONTINENT"],
                    CrustType["OCEAN"],
                    CrustType["OCEAN"],
                    CrustType["OCEAN"],
                    CrustType["OCEAN"],
                    CrustType["OCEAN"],
                ],
                "mask": [1, 0, 0, 0, 0, 0],
                "capCenter": [0.0, 0.8, 0.6],
            },
            {
                "name": "Lower plate",
                "type": [
                    CrustType["CONTINENT"],
                    CrustType["OCEAN"],
                    CrustType["OCEAN"],
                    CrustType["OCEAN"],
                    CrustType["OCEAN"],
                    CrustType["OCEAN"],
                ],
                "mask": [1, 0, 0, 0, 0, 0],
                "capCenter": [0.0, -0.8, 0.6],
            },
        ],
        "config": config,
        "time": 12.0,
        "dt": 1.0,
    }


def _payload() -> dict[str, Any]:
    return {"cases": [_one_cell_case(), _suture_ring_case()]}


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return run_js_reference(RUNNER, payload)


def _world(spec: dict[str, Any]):
    num_cells = len(spec["owners"])
    mesh = SimpleNamespace(
        num_cells=num_cells,
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
        adj_list=np.asarray(spec["adjList"], dtype=np.int32),
        edge_rad=spec["edgeRad"],
        edge_km=spec["edgeKm"],
        cell_area_km2=spec["cellAreaKm2"],
        index=SimpleNamespace(nearest=lambda *_point: 0),
    )
    plates = []
    for plate_id, source in enumerate(spec["plates"]):
        crust = SimpleNamespace(
            mask=np.asarray(source["mask"], dtype=np.uint8),
            type=np.asarray(source["type"], dtype=np.uint8),
            flags=np.asarray(source.get("flags", [0] * len(source["mask"])), dtype=np.uint16),
            terrane=np.asarray(source.get("terrane", [plate_id + 1] * len(source["mask"])), dtype=np.int32),
            formedAt=np.asarray(source.get("formedAt", [0.0] * len(source["mask"])), dtype=np.float64),
            slabKm=np.zeros(len(source["mask"]), dtype=np.float32),
            coverKm=np.zeros(len(source["mask"]), dtype=np.float32),
            count=int(sum(source["mask"])),
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
                omega=np.zeros(3, dtype=np.float64),
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
        owner=np.asarray(spec["owners"], dtype=np.int16),
        ownerCell=np.asarray(spec["ownerCell"], dtype=np.int32),
        landClaims=np.asarray(spec["landClaims"], dtype=np.uint8),
        collider=np.asarray(spec["collider"], dtype=np.int16),
        plates=plates,
        collisions={},
        arcScrapes={},
        budget={"recycledCells": 0},
        config=spec["config"],
        time=spec["time"],
        log=lambda kind, text, data: events.append(
            {"kind": kind, "text": text, "data": data}
        ),
    )
    return world, events


def _record_snapshot(record):
    if record is None:
        return None
    snapshot = {field: record.get(field) for field in RECORD_FIELDS}
    for key in ("gapKm2", "gapWidestKm", "checkedAt", "gapBestKm2", "gapShrankAt", "ringKm2"):
        value = snapshot[key]
        if value is not None and not math.isfinite(float(value)):
            snapshot[key] = None
    return snapshot


def test_collision_record_progress_and_suture_ring_block_match_javascript(
    javascript, payload
):
    expected = []
    for spec in payload["cases"]:
        world, events = _world(spec)
        result = track_collisions(world, spec["dt"])
        record = world.collisions.get("0:1")
        expected.append(
            {
                "result": {
                    "started": result["started"],
                    "tracked": result["tracked"],
                    "underthrustCells": result["underthrustCells"],
                    "changed": result["changed"],
                },
                "record": _record_snapshot(record),
                "events": events,
            }
        )

    assert_js_python_equal(expected, javascript, atol=1e-10)
    first = expected[0]
    assert first["result"] == {
        "started": 1,
        "tracked": 1,
        "underthrustCells": 0,
        "changed": False,
    }
    assert first["record"]["upper"] == 0
    assert first["record"]["lower"] == 1
    assert first["record"]["closingCmYr"] == 0.0
    assert first["record"]["sutured"] is False

    second = expected[1]
    assert second["result"]["tracked"] == 1
    assert second["record"]["gapKm2"] == pytest.approx(10000.0)
    assert second["record"]["gapWidestKm"] == pytest.approx(100.0)
    assert second["record"]["ringKm2"] == pytest.approx(10000.0)
    assert second["record"]["weldCandidate"] is True
    assert second["record"]["weldBlockedByRing"] is True
    assert second["record"]["mergeEligible"] is False
    assert second["record"]["sutured"] is False
