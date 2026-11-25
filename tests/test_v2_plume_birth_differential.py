"""Differential checks for plume birth, rift eligibility and deterministic naming."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plumes import update_plumes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/plume-birth-differential-runner.mjs"


def _case(
    *,
    owner: int = 0,
    rift: bool | str = False,
    name: str | None = "Kavora",
    draw: float = 0.2,
    draws: list[float] | None = None,
    minimum_area: float = 0.0,
    heat: float = 0.9,
) -> dict[str, Any]:
    plate = None
    if owner >= 0:
        plate = {
            "name": "Teral",
            "mask": [1],
            "type": [CrustType["CONTINENT"]],
            "continentalCells": 1,
        }
    return {
        "time": 25.0,
        "xyz": [2.0, 0.0, 0.0],
        "cellAreaKm2": 1_000_000.0,
        "owner": owner,
        "plate": plate,
        "rift": rift,
        "name": name,
        "draws": draws if draws is not None else [draw],
        "riftMinContinentKm2": minimum_area,
        "plumeRiftChance": 0.5,
        "riftHeatGain": 1.0,
        "riftHeatFloor": 0.1,
        "riftHeatPower": 1.0,
        "heat": heat,
    }


def _payload() -> dict[str, Any]:
    return {
        "cases": [
            _case(rift=True, draw=0.99),
            _case(rift="chance", draw=0.2),
            _case(rift="chance", draw=0.95),
            _case(rift="chance", draw=0.0, minimum_area=2_000_000.0),
            _case(owner=-1, rift="chance", draw=0.0),
            _case(rift=False),
            _case(
                owner=-1,
                rift=False,
                name=None,
                draws=[0.8, 0.2, 0.3, 0.4, 0.5, 0.6],
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
    plate = None
    plates = []
    if spec["plate"] is not None:
        source = spec["plate"]
        plate = SimpleNamespace(
            id=0,
            name=source["name"],
            isBackground=False,
            stats={"continentalCells": source["continentalCells"]},
            crust=SimpleNamespace(
                mask=np.asarray(source["mask"], dtype=np.uint8),
                type=np.asarray(source["type"], dtype=np.uint8),
            ),
        )
        plates.append(plate)

    mesh = SimpleNamespace(
        num_cells=1,
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        index=SimpleNamespace(nearest=lambda *_point: 0),
        cell_area_km2=spec["cellAreaKm2"],
    )
    draws = iter(spec["draws"])
    scheduled = {
        "time": spec["time"],
        "xyz": list(spec["xyz"]),
        "head": False,
        "rift": spec["rift"],
        "name": spec["name"],
        "lifeMyr": 0.0,
        "strength": 1.2,
    }
    events = []
    world = SimpleNamespace(
        config={
            "mantleInsulateMyr": 100.0,
            "mantleCoolMyr": 200.0,
            "plumeRatePerMyr": 0.0,
            "riftMinContinentKm2": spec["riftMinContinentKm2"],
            "plumeRiftChance": spec["plumeRiftChance"],
            "riftHeatGain": spec["riftHeatGain"],
            "riftHeatFloor": spec["riftHeatFloor"],
            "riftHeatPower": spec["riftHeatPower"],
        },
        time=spec["time"],
        mesh=mesh,
        owner=np.asarray([spec["owner"]], dtype=np.int16),
        ownerCell=np.asarray([0], dtype=np.int32),
        plates=plates,
        mantleHeat=np.asarray([spec["heat"]], dtype=np.float32),
        hotspotsReady=True,
        hotspots=[],
        pendingPlumeRifts=[],
        plumes=[scheduled],
        rngHotspot=lambda: next(draws, 0.0),
        log=lambda kind, text, data: events.append(
            {"kind": kind, "text": text, "data": data}
        ),
    )
    return world, scheduled, events


def _snapshot(world, scheduled, events):
    return {
        "summary": {
            "born": len(world.hotspots),
            "expired": sum(event["kind"] == "plume-death" for event in events),
            "alive": sum(plume["alive"] for plume in world.hotspots),
        },
        "hotspots": [
            {
                **plume,
                "pulseUntil": plume["pulseUntil"],
            }
            for plume in world.hotspots
        ],
        "pendingPlumeRifts": world.pendingPlumeRifts,
        "scheduledDone": scheduled["done"],
        "heat": world.mantleHeat.tolist(),
        "events": events,
    }


def test_plume_birth_and_rift_eligibility_match_javascript(javascript, payload):
    expected = []
    for spec in payload["cases"]:
        world, scheduled, events = _world(spec)
        update_plumes(world, 0.0)
        expected.append(_snapshot(world, scheduled, events))

    assert_js_python_equal(expected, javascript, atol=1e-7)
    assert [len(case["pendingPlumeRifts"]) for case in javascript] == [
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]
    assert javascript[-1]["hotspots"][0]["name"] != "Kavora"
