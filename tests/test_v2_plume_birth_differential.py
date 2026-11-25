"""Differential checks for plume birth, rift eligibility and deterministic naming."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.crust import CrustType, VolcanicType
from orogen.v2.tectonics.plumes import update_plumes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/plume-birth-differential-runner.mjs"


def _case(
    *,
    xyz: list[float] | None = None,
    owner: int = 0,
    rift: bool | str = False,
    name: str | None = "Kavora",
    draw: float = 0.2,
    draws: list[float] | None = None,
    minimum_area: float = 0.0,
    heat: float = 0.9,
    head: bool = False,
    life_myr: float = 0.0,
    dt: float = 0.0,
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
        "xyz": xyz if xyz is not None else [2.0, 0.0, 0.0],
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
        "head": head,
        "lifeMyr": life_myr,
        "dt": dt,
        "lipAreaMinMkm2": 0.000001,
        "lipAreaMaxMkm2": 0.000001,
        "lipLobes": 0.0,
        "lipEdgeNoise": 0.0,
        "lipThicknessKm": 10.0,
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
            _case(
                xyz=[1.0, 0.0, 0.0],
                rift=False,
                name="Kavora",
                draws=[0.2],
                head=True,
            ),
            _case(
                xyz=[1.0, 0.0, 0.0],
                rift=False,
                name="Kavora",
                life_myr=100.0,
                dt=10.0,
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
                flags=np.zeros(1, dtype=np.uint16),
                volcType=np.zeros(1, dtype=np.uint8),
                volcAt=np.zeros(1, dtype=np.float64),
                volcKm=np.zeros(1, dtype=np.float32),
                volcSource=np.zeros(1, dtype=np.uint16),
            ),
            cell_at=lambda *_args: 0,
            to_plate_frame=lambda x, y, z: np.asarray([x, y, z]),
        )
        plates.append(plate)

    mesh = SimpleNamespace(
        num_cells=1,
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        adj_offset=np.asarray([0, 0], dtype=np.int32),
        adj_list=np.asarray([], dtype=np.int32),
        index=SimpleNamespace(nearest=lambda *_point: 0),
        cell_area_km2=spec["cellAreaKm2"],
    )
    draws = iter(spec["draws"])
    scheduled = {
        "time": spec["time"],
        "xyz": list(spec["xyz"]),
        "head": spec["head"],
        "rift": spec["rift"],
        "name": spec["name"],
        "lifeMyr": spec["lifeMyr"],
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
            "lipAreaMinMkm2": spec["lipAreaMinMkm2"],
            "lipAreaMaxMkm2": spec["lipAreaMaxMkm2"],
            "lipLobes": spec["lipLobes"],
            "lipEdgeNoise": spec["lipEdgeNoise"],
            "lipThicknessKm": spec["lipThicknessKm"],
            "hotspotTrackKm": 1000.0,
            "hotspotEdificeKm": 100.0,
            "hotspotBuildKmPerMyr": 1.0,
            "hotspotMaxKm": 50.0,
            "hotspotQuietMinMyr": 1.0,
            "hotspotQuietMaxMyr": 4.0,
            "hotspotBurstMinMyr": 1.0,
            "hotspotBurstMaxMyr": 3.0,
        },
        time=spec["time"],
        mesh=mesh,
        owner=np.asarray([spec["owner"]], dtype=np.int16),
        ownerCell=np.asarray([0], dtype=np.int32),
        plates=plates,
        mantleHeat=np.asarray([spec["heat"]], dtype=np.float32),
        provisional=np.zeros(1, dtype=np.uint8),
        rngVolcano=lambda: 0.5,
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
        "crust": [
            {
                "volcType": plate.crust.volcType.tolist(),
                "volcAt": plate.crust.volcAt.tolist(),
                "volcKm": plate.crust.volcKm.tolist(),
                "volcSource": plate.crust.volcSource.tolist(),
                "flags": plate.crust.flags.tolist(),
            }
            for plate in world.plates
        ],
        "events": events,
    }


def test_plume_birth_and_rift_eligibility_match_javascript(javascript, payload):
    expected = []
    for spec in payload["cases"]:
        world, scheduled, events = _world(spec)
        update_plumes(world, spec["dt"])
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
        0,
        0,
    ]
    assert javascript[-3]["hotspots"][0]["name"] != "Kavora"
    lip_case = javascript[-2]
    assert lip_case["hotspots"][0]["lip"]["name"] == "Kavora Traps"
    assert lip_case["hotspots"][0]["lip"]["cells"] == 1
    assert lip_case["crust"][0]["volcType"] == [VolcanicType["LIP"]]

    track_case = javascript[-1]
    assert track_case["summary"] == {"born": 1, "expired": 0, "alive": 1}
    assert track_case["crust"][0]["volcType"] == [VolcanicType["TRACK"]]
    assert track_case["crust"][0]["volcAt"] == [25.0]
    assert track_case["crust"][0]["volcSource"] == [1]
    assert track_case["crust"][0]["volcKm"][0] == pytest.approx(4.56, abs=1e-6)
    assert track_case["hotspots"][0]["pulseUntil"] == pytest.approx(26.0, abs=1e-8)
    assert track_case["hotspots"][0]["bursting"] is False
