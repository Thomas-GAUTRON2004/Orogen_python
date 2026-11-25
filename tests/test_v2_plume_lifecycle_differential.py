"""Differential checks for scheduled plume birth and immediate expiration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.plumes import update_plumes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/plume-lifecycle-differential-runner.mjs"


def _payload() -> dict[str, Any]:
    return {
        "scenario": {
            "time": 25.0,
            "xyz": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "heat": [0.25, 0.75],
            "plume": {
                "name": "Kavora",
                "lifeMyr": 0.0,
                "strength": 1.2,
            },
        }
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> dict[str, Any]:
    return run_js_reference(RUNNER, payload)


def test_scheduled_plume_birth_and_expiration_match_pinned_javascript(
    javascript, payload
):
    spec = payload["scenario"]
    events = []
    scheduled = {
        "time": spec["time"],
        "xyz": spec["plume"]["xyz"] if "xyz" in spec["plume"] else [2.0, 0.0, 0.0],
        "head": False,
        "rift": False,
        "name": spec["plume"]["name"],
        "lifeMyr": spec["plume"]["lifeMyr"],
        "strength": spec["plume"]["strength"],
    }
    mesh = SimpleNamespace(
        num_cells=len(spec["heat"]),
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        index=SimpleNamespace(nearest=lambda *_point: 0),
    )
    world = SimpleNamespace(
        config={
            "mantleInsulateMyr": 100.0,
            "mantleCoolMyr": 200.0,
            "plumeRatePerMyr": 0.0,
        },
        time=spec["time"],
        mesh=mesh,
        owner=np.full(len(spec["heat"]), -1, dtype=np.int16),
        ownerCell=np.arange(len(spec["heat"]), dtype=np.int32),
        plates=[],
        mantleHeat=np.asarray(spec["heat"], dtype=np.float32),
        hotspotsReady=True,
        hotspots=[],
        pendingPlumeRifts=[],
        plumes=[scheduled],
        rngHotspot=lambda: 0.0,
        log=lambda kind, text, data: events.append(
            {"kind": kind, "text": text, "data": data}
        ),
    )
    summary = update_plumes(world, 0.0)
    actual = {
        "summary": summary,
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
    assert_js_python_equal(actual, javascript, atol=1e-7)
    assert summary == {"born": 1, "expired": 1, "alive": 0}
