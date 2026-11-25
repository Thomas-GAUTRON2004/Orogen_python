"""Differential checks for recycling sub-threshold arc fragments during collision."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.collision import accrete_arcs
from orogen.v2.tectonics.crust import CrustType

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/collision-arc-recycle-differential-runner.mjs"


def _payload() -> dict[str, Any]:
    return {"scenario": {"cellAreaKm2": 100.0, "arcAccreteMinKm2": 250.0}}


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> dict[str, Any]:
    return run_js_reference(RUNNER, payload)


class _Crust:
    def __init__(self, crust_type: int) -> None:
        self.mask = np.ones(1, dtype=np.uint8)
        self.type = np.asarray([crust_type], dtype=np.uint8)
        self.count = 1

    def remove(self, cell: int) -> None:
        if self.mask[cell]:
            self.mask[cell] = 0
            self.count -= 1


class _Plate:
    def __init__(self, plate_id: int, name: str, crust_type: int, land: int) -> None:
        self.id = plate_id
        self.name = name
        self.alive = True
        self.isBackground = False
        self.omega = np.asarray([0.001, 0.002, 0.003])
        self.stats = {"continentalCells": land}
        self.crust = _Crust(crust_type)

    def cell_at(self, *_args) -> int:
        return 0

    def mark_reshaped(self) -> None:
        self.reshaped = True


def test_small_arc_fragment_is_recycled_like_javascript(javascript, payload):
    spec = payload["scenario"]
    mesh = SimpleNamespace(
        xyz=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        index=object(),
        cell_area_km2=spec["cellAreaKm2"],
    )
    donor = _Plate(0, "Arc donor", CrustType["ARC"], 0)
    host = _Plate(1, "Continental host", CrustType["CONTINENT"], 1)
    events = []
    world = SimpleNamespace(
        mesh=mesh,
        time=5.0,
        config={
            "arcAccreteMinKm2": spec["arcAccreteMinKm2"],
            "terraneMinAreaKm2": 100000.0,
        },
        budget={"recycledCells": 0},
        arcScrapes={},
        log=lambda *args: events.append(args),
    )
    result = accrete_arcs(world, donor, host, [0])
    actual = {
        "merged": result["accreted"],
        "donorAlive": donor.alive,
        "donorMask": donor.crust.mask.tolist(),
        "donorCount": donor.crust.count,
        "donorReshaped": bool(getattr(donor, "reshaped", False)),
        "donorOmega": donor.omega.tolist(),
        "recycledCells": world.budget["recycledCells"],
        "hostAlive": host.alive,
    }
    assert_js_python_equal(actual, javascript, atol=0)
    assert actual == {
        "merged": True,
        "donorAlive": False,
        "donorMask": [0],
        "donorCount": 0,
        "donorReshaped": True,
        "donorOmega": [0.0, 0.0, 0.0],
        "recycledCells": 1,
        "hostAlive": True,
    }
