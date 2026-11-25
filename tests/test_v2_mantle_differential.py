"""Differential checks for mantle traction and continental dispersal."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.mantle import (
    dispersal_field,
    flow_of_cycle,
    mantle_flow_at,
    repel_at,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/mantle-differential-runner.mjs"


def _xyz(latitude: float, longitude: float) -> list[float]:
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    return [
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ]


def _payload() -> dict[str, Any]:
    flow_config = {
        "mantleOutflow": 0.45,
        "mantleInflow": 0.35,
        "mantleInflowPackKm": 1800.0,
    }
    flow_cases = [
        {"cycle": None, "point": [0.6, 0.0, 0.8]},
        {"cycle": {"up": None}, "point": [0.6, 0.0, 0.8]},
        {
            "cycle": {
                "up": [0.0, 0.0, 1.0],
                "down": [1.0, 0.0, 0.0],
                "upHeat": 0.7,
                "gather": 0.6,
                "hold": 0.0,
                "holdKm": 0.0,
            },
            "point": [0.6, 0.0, 0.8],
        },
        {
            "cycle": {
                "up": [0.0, 0.0, 1.0],
                "down": [1.0, 0.0, 0.0],
                "upHeat": 0.25,
                "gather": 0.0,
                "hold": 1.0,
                "holdKm": 3000.0,
            },
            "point": [0.1, 0.0, math.sqrt(0.99)],
        },
        {
            "cycle": {
                "up": [0.0, 0.0, 1.0],
                "down": [-1.0, 0.0, 0.0],
                "upHeat": 0.7,
                "gather": 0.35,
                "hold": 0.5,
                "holdKm": 7000.0,
            },
            "point": [0.8, 0.0, 0.6],
        },
    ]

    groups = [
        (4, (0.0, 0.0), CrustType["CONTINENT"], False),
        (3, (0.0, 20.0), CrustType["CRATON"], False),
        (3, (0.0, 40.0), CrustType["CONTINENT"], False),
        (3, (0.0, 180.0), CrustType["CONTINENT"], True),
        (1, (90.0, 0.0), CrustType["OCEAN"], False),
        (2, (0.0, 100.0), CrustType["CONTINENT"], False),
    ]
    xyz: list[float] = []
    owners: list[int] = []
    owner_cells: list[int] = []
    plates = []
    for plate_id, (count, (lat, lon), crust_type, background) in enumerate(groups):
        xyz.extend(_xyz(lat, lon) * count)
        owners.extend([plate_id] * count)
        owner_cells.extend(range(count))
        plates.append(
            {
                "isBackground": background,
                "types": [crust_type] * count,
            }
        )
    xyz.extend(_xyz(-90.0, 0.0))
    owners.append(-1)
    owner_cells.append(0)

    land_claims = [0] * len(owners)
    collider = [-1] * len(owners)
    land_claims[0] = 2
    collider[0] = 1

    return {
        "mantle": {
            "flowConfig": flow_config,
            "flowCases": flow_cases,
            "dispersal": {
                "config": {
                    "dispersalRepulsion": 0.2,
                    "dispersalRepulsionKm": 5000.0,
                },
                "cycle": {"phase": "dispersing"},
                "xyz": xyz,
                "owners": owners,
                "ownerCells": owner_cells,
                "plates": plates,
                "landClaims": land_claims,
                "collider": collider,
                "cellAreaKm2": 400000.0,
                "repelQueries": [
                    {"plateId": 0, "point": _xyz(0.0, 0.0)},
                    {"plateId": 0, "point": _xyz(0.0, 20.0)},
                    {"plateId": 2, "point": _xyz(0.0, 0.0)},
                    {"plateId": 0, "point": _xyz(0.0, 110.0)},
                    {"plateId": 5, "point": _xyz(0.0, 100.0)},
                    {"plateId": 0, "point": _xyz(0.0, 40.0)},
                ],
            },
        }
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> dict[str, Any]:
    return run_js_reference(RUNNER, payload)


def _world(dispersal: dict[str, Any]):
    plates = [
        SimpleNamespace(
            isBackground=plate["isBackground"],
            crust=SimpleNamespace(
                type=np.asarray(plate["types"], dtype=np.uint8)
            ),
        )
        for plate in dispersal["plates"]
    ]
    return SimpleNamespace(
        config=dispersal["config"],
        cycle=dispersal["cycle"],
        mesh=SimpleNamespace(
            num_cells=len(dispersal["owners"]),
            xyz=np.asarray(dispersal["xyz"], dtype=np.float64),
            cell_area_km2=dispersal["cellAreaKm2"],
        ),
        owner=np.asarray(dispersal["owners"], dtype=np.int16),
        ownerCell=np.asarray(dispersal["ownerCells"], dtype=np.int32),
        landClaims=np.asarray(dispersal["landClaims"], dtype=np.uint8),
        collider=np.asarray(dispersal["collider"], dtype=np.int16),
        plates=plates,
    )


def test_mantle_flow_and_wrapper_match_upstream(javascript, payload):
    mantle = payload["mantle"]
    expected = []
    for scenario in mantle["flowCases"]:
        x, y, z = scenario["point"]
        expected.append(
            {
                "flow": flow_of_cycle(
                    scenario["cycle"], mantle["flowConfig"], x, y, z, [10, 10, 10]
                ),
                "wrapper": mantle_flow_at(
                    SimpleNamespace(
                        cycle=scenario["cycle"], config=mantle["flowConfig"]
                    ),
                    x,
                    y,
                    z,
                    [10, 10, 10],
                ),
            }
        )
    assert_js_python_equal(expected, javascript["flows"], atol=2e-12)


def test_dispersal_field_and_repulsion_match_upstream(javascript, payload):
    dispersal = payload["mantle"]["dispersal"]
    world = _world(dispersal)
    field = dispersal_field(world)
    expected_field = None if field is None else {
        "centres": field["centres"],
        "touching": sorted(field["touching"]),
        "k": field["k"],
        "reach": field["reach"],
    }
    assert_js_python_equal(expected_field, javascript["field"], atol=2e-12)

    expected_pushes = [
        repel_at(field, query["plateId"], *query["point"], [0, 0, 0])
        for query in dispersal["repelQueries"]
    ]
    assert_js_python_equal(expected_pushes, javascript["pushes"], atol=2e-12)

    assert javascript["disabledByConfig"] is True
    assert javascript["disabledByPhase"] is True
    assert dispersal_field(
        SimpleNamespace(
            **{
                **world.__dict__,
                "config": {**world.config, "dispersalRepulsion": 0.0},
            }
        )
    ) is None
    assert dispersal_field(
        SimpleNamespace(
            **{**world.__dict__, "cycle": {"phase": "assembled"}}
        )
    ) is None
