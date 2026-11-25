"""Differential checks for collision overlap and touching-contact collection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.collision import collect_contacts
from orogen.v2.tectonics.crust import CrustFlag, CrustType

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/collision-contacts-differential-runner.mjs"


def _case() -> dict[str, Any]:
    neighbours = (
        (1,),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8,),
    )
    adjacency: list[int] = []
    offsets = [0]
    for cells in neighbours:
        adjacency.extend(cells)
        offsets.append(len(adjacency))
    return {
        "adjOffset": offsets,
        "adjList": adjacency,
        "owners": [0, 0, 1, 1, 0, 1, 0, 2, -1, 0],
        "ownerCell": [0, 1, 0, 1, 2, 2, 3, 0, 0, 0],
        "landClaims": [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        "collider": [1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "plates": [
            {
                "isBackground": False,
                "mask": [1, 1, 1, 1],
                "type": [
                    CrustType["CONTINENT"],
                    CrustType["CONTINENT"],
                    CrustType["OCEAN"],
                    CrustType["CONTINENT"],
                ],
                "flags": [0, 0, 0, CrustFlag["RIFT_MARGIN"]],
            },
            {
                "isBackground": False,
                "mask": [1, 1, 1],
                "type": [
                    CrustType["CONTINENT"],
                    CrustType["CONTINENT"],
                    CrustType["CONTINENT"],
                ],
                "flags": [0, 0, 0],
            },
            {
                "isBackground": True,
                "mask": [1],
                "type": [CrustType["CONTINENT"]],
                "flags": [0],
            },
        ],
    }


def _payload() -> dict[str, Any]:
    base = _case()
    blocked_claim = {
        **base,
        "landClaims": [2, 1, 1, 1, 1, 1, 1, 1, 0, 2],
    }
    missing_crust = {
        **base,
        "plates": [
            base["plates"][0],
            {
                **base["plates"][1],
                "mask": [1, 0, 1],
            },
            base["plates"][2],
        ],
    }
    return {"cases": [base, blocked_claim, missing_crust]}


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return run_js_reference(RUNNER, payload)


def _world(spec: dict[str, Any]):
    plates = [
        SimpleNamespace(
            isBackground=plate["isBackground"],
            crust=SimpleNamespace(
                mask=np.asarray(plate["mask"], dtype=np.uint8),
                type=np.asarray(plate["type"], dtype=np.uint8),
                flags=np.asarray(plate["flags"], dtype=np.uint16),
            ),
        )
        for plate in spec["plates"]
    ]
    mesh = SimpleNamespace(
        num_cells=len(spec["owners"]),
        adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
        adj_list=np.asarray(spec["adjList"], dtype=np.int32),
    )
    return SimpleNamespace(
        mesh=mesh,
        owner=np.asarray(spec["owners"], dtype=np.int16),
        ownerCell=np.asarray(spec["ownerCell"], dtype=np.int32),
        landClaims=np.asarray(spec["landClaims"], dtype=np.uint8),
        collider=np.asarray(spec["collider"], dtype=np.int16),
        plates=plates,
    )


def test_collision_overlap_and_touching_contacts_match_javascript(javascript, payload):
    expected = [collect_contacts(_world(spec)) for spec in payload["cases"]]
    assert_js_python_equal(expected, javascript, atol=0)
    assert expected[0] == {
        "overlap": {"0:1": [0]},
        "touching": {"0:1": [1, 2]},
    }
