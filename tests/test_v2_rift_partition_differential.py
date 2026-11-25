"""Differential checks for rift-side partitioning and exterior-ocean tracing."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differential import assert_js_python_equal, run_js_reference

from orogen.v2.core.sphere_geometry import cell_distance_km
from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.ops.rift_path import (
    _cheapest_paths,
    _crosses_rather_than_hugs,
    _is_monotone,
    _trace_parent_path,
    coast_distance,
    find_rift,
    grain_of,
    rift_context,
)
from orogen.v2.tectonics.ops.rift_partition import (
    elongation_of,
    exterior_sea,
    masked_cells,
    partition,
    sides_of,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/rift-partition-differential-runner.mjs"


def _xyz(latitude: float, longitude: float) -> list[float]:
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    return [
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ]


def _base_case() -> dict[str, Any]:
    rows, columns = 6, 6
    xyz = [
        _xyz(-25.0 + row * 10.0, -25.0 + column * 10.0)
        for row in range(rows)
        for column in range(columns)
    ]
    adjacency = []
    offsets = [0]
    for row in range(rows):
        for column in range(columns):
            neighbours = []
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                nr, nc = row + dr, column + dc
                if 0 <= nr < rows and 0 <= nc < columns:
                    neighbours.append(nr * columns + nc)
            adjacency.extend(neighbours)
            offsets.append(len(adjacency))

    types = []
    terrane = []
    for _row in range(rows):
        for column in range(columns):
            if column == 3:
                types.append(CrustType["OCEAN"])
                terrane.append(0)
            elif column in (0, 4):
                types.append(CrustType["CRATON"])
                terrane.append(1 if column == 0 else 2)
            else:
                types.append(CrustType["CONTINENT"])
                terrane.append(0)

    return {
        "numCells": rows * columns,
        "xyz": [value for point in xyz for value in point],
        "adjOffset": offsets,
        "adjList": adjacency,
        "edgeKm": 2500.0,
        "mask": [1] * (rows * columns),
        "type": types,
        "terrane": terrane,
        "path": [9, 15, 21, 27],
        "cells": None,
    }


def _payload() -> dict[str, Any]:
    base = _base_case()
    outside_sea = {**base, "mask": list(base["mask"])}
    outside_sea["mask"][3] = 0
    outside_sea["mask"][33] = 0

    all_ocean = {
        **base,
        "type": [CrustType["OCEAN"]] * base["numCells"],
        "terrane": [0] * base["numCells"],
    }
    empty = {
        **base,
        "mask": [0] * base["numCells"],
        "type": [CrustType["OCEAN"]] * base["numCells"],
        "terrane": [0] * base["numCells"],
        "path": [],
        "cells": [],
    }
    single_cell_path = {**base, "path": [9]}
    interior_ocean_hole = {**base, "mask": list(base["mask"])}
    interior_ocean_hole["mask"][14] = 0
    interior_ocean_hole["type"] = list(base["type"])
    interior_ocean_hole["type"][14] = CrustType["OCEAN"]
    disconnected_land = {**base, "mask": [0] * base["numCells"]}
    for cell in (7, 8, 13, 14, 19, 20, 25, 26):
        disconnected_land["mask"][cell] = 1
    explicit_cells = {**base, "cells": [8, 9, 10, 14, 15, 16, 20, 21, 22]}
    return {
        "cases": [
            base,
            outside_sea,
            all_ocean,
            empty,
            single_cell_path,
            interior_ocean_hole,
            disconnected_land,
            explicit_cells,
        ]
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return _payload()


@pytest.fixture(scope="module")
def javascript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return run_js_reference(RUNNER, payload)


def _mesh(spec: dict[str, Any]):
    return SimpleNamespace(
        num_cells=spec["numCells"],
        xyz=np.asarray(spec["xyz"], dtype=np.float64),
        adj_offset=np.asarray(spec["adjOffset"], dtype=np.int32),
        adj_list=np.asarray(spec["adjList"], dtype=np.int32),
        edge_km=spec["edgeKm"],
    )


def _crust(spec: dict[str, Any]):
    mask = np.asarray(spec["mask"], dtype=np.uint8)
    return SimpleNamespace(
        count=int(mask.sum()),
        mask=mask,
        type=np.asarray(spec["type"], dtype=np.uint8),
        terrane=np.asarray(spec["terrane"], dtype=np.int32),
    )


def _evaluate(spec: dict[str, Any]) -> dict[str, Any]:
    mesh = _mesh(spec)
    crust = _crust(spec)
    cells = masked_cells(crust) if spec["cells"] is None else spec["cells"]
    side, land, cratons = partition(mesh, crust, spec["path"], cells)
    return {
        "masked": masked_cells(crust).tolist(),
        "partition": {
            "side": side.tolist(),
            "land": land,
            "cratons": cratons,
        },
        "summary": sides_of(mesh, crust, spec["path"], cells),
        "exterior": exterior_sea(mesh, crust).tolist(),
        "elongation": [
            elongation_of(mesh, side, 1, cells),
            elongation_of(mesh, side, 2, cells),
        ],
    }


def test_rift_partition_and_exterior_sea_match_pinned_javascript(javascript, payload):
    expected = [_evaluate(case) for case in payload["cases"]]
    assert_js_python_equal(expected, javascript, atol=3e-6)



def test_coast_distance_propagates_inland_from_exterior_ocean():
    spec = _base_case()
    mesh = _mesh(spec)
    crust = _crust(spec)
    exterior = exterior_sea(mesh, crust)

    distance = coast_distance(mesh, crust, exterior)

    expected = np.full(spec["numCells"], -1, dtype=np.int16)
    for row in range(6):
        expected[row * 6 + 2] = 0
        expected[row * 6 + 1] = 1
        expected[row * 6] = 2
        expected[row * 6 + 4] = 0
        expected[row * 6 + 5] = 1

    np.testing.assert_array_equal(distance, expected)



def test_coast_distance_leaves_ocean_and_unreachable_land_unassigned():
    spec = _base_case()
    mesh = _mesh(spec)
    crust = _crust(spec)
    crust.mask[0] = 0
    crust.mask[1] = 0
    exterior = np.zeros(spec["numCells"], dtype=np.bool_)

    distance = coast_distance(mesh, crust, exterior)

    assert np.all(distance == -1)


def test_cheapest_paths_never_enters_ocean_cratons_or_unmasked_cells():
    spec = _base_case()
    mesh = _mesh(spec)
    crust = _crust(spec)
    crust.mask[2] = 0

    result = _cheapest_paths(mesh, crust, 8, lambda _cell: 1.0)

    assert result["cost"][8] == 0.0
    assert math.isfinite(result["cost"][7])
    assert math.isinf(result["cost"][2])
    assert math.isinf(result["cost"][10])
    assert math.isinf(result["cost"][3])
    assert result["from"][8] == -1


@pytest.mark.parametrize(
    ("distances", "expected"),
    [
        ([9, 9, 0, 9, 9, 9, 9], True),
        ([9, 9, 0, 1, 9, 9, 9], False),
        ([9, 9, 0, 2, 0, 9, 9], True),
        ([9, 9, -1, 0, 9, 9, 9], True),
    ],
)
def test_rift_path_rejects_long_coastal_hugs(distances, expected):
    mesh = SimpleNamespace(edge_km=10.0)
    path = list(range(len(distances)))
    config = {
        "riftCoastClearanceKm": 20.0,
        "riftCoastHugKm": 10.0,
    }

    assert _crosses_rather_than_hugs(mesh, path, distances, config) is expected



def test_trace_parent_path_reconstructs_root_to_endpoint_order():
    parent = np.asarray([-1, 0, 1, 1, 3, 4], dtype=np.int32)

    assert _trace_parent_path(5, parent) == [0, 1, 3, 4, 5]
    assert _trace_parent_path(2, parent) == [0, 1, 2]


@pytest.mark.parametrize(
    ("longitudes", "expected"),
    [
        ([0.0, 10.0, 20.0, 30.0], True),
        ([0.0, 10.0, 20.0, 15.0, 30.0], False),
    ],
)
def test_rift_path_monotonicity_rejects_large_backtracking(longitudes, expected):
    xyz = [
        coordinate
        for longitude in longitudes
        for coordinate in _xyz(0.0, longitude)
    ]
    mesh = SimpleNamespace(
        xyz=np.asarray(xyz, dtype=np.float64),
        edge_rad=0.001,
    )
    config = {"riftOverhangKm": 1.0}
    path = list(range(len(longitudes)))

    assert _is_monotone(mesh, path, config) is expected


def test_cheapest_paths_prefers_lower_weighted_route_and_records_predecessors():
    positions = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0)]
    xyz = np.asarray(
        [
            coordinate
            for latitude, longitude in positions
            for coordinate in _xyz(latitude, longitude)
        ],
        dtype=np.float64,
    )
    neighbours = [[1, 2], [0, 3], [0, 3], [1, 2, 4], [3]]
    adjacency = [cell for group in neighbours for cell in group]
    offsets = [0]
    for group in neighbours:
        offsets.append(offsets[-1] + len(group))
    mesh = SimpleNamespace(
        num_cells=5,
        xyz=xyz,
        adj_offset=np.asarray(offsets, dtype=np.int32),
        adj_list=np.asarray(adjacency, dtype=np.int32),
        edge_km=100.0,
    )
    crust = SimpleNamespace(
        mask=np.ones(5, dtype=np.uint8),
        type=np.full(5, CrustType["CONTINENT"], dtype=np.uint8),
    )

    result = _cheapest_paths(
        mesh,
        crust,
        0,
        lambda cell: 10.0 if cell == 1 else 1.0,
    )

    assert result["from"].tolist() == [-1, 0, 0, 2, 3]
    expected = sum(
        cell_distance_km(xyz, first, second) / mesh.edge_km
        for first, second in ((0, 2), (2, 3), (3, 4))
    )
    assert result["cost"][4] == pytest.approx(expected)


def test_cheapest_paths_does_not_cross_an_unmasked_gap():
    spec = _base_case()
    mesh = _mesh(spec)
    crust = _crust(spec)
    crust.mask[8] = 0

    result = _cheapest_paths(mesh, crust, 7, lambda _cell: 1.0)

    assert math.isinf(result["cost"][9])
    assert result["from"][9] == -1
    assert math.isfinite(result["cost"][13])


def test_grain_of_returns_unit_weight_when_rift_noise_is_disabled():
    world = SimpleNamespace(
        config={"riftCurveAmp": 0.0, "riftJagAmp": 0.0},
    )

    grain = grain_of(world)

    assert grain(0) == 1.0
    assert not hasattr(world, "riftGrain")


def test_grain_of_caches_seeded_noise_per_cell():
    xyz = np.asarray(
        [
            coordinate
            for longitude in (0.0, 15.0, 30.0)
            for coordinate in _xyz(0.0, longitude)
        ],
        dtype=np.float64,
    )
    world = SimpleNamespace(
        seed=12345,
        mesh=SimpleNamespace(num_cells=3, xyz=xyz),
        config={
            "riftCurveAmp": 0.35,
            "riftCurveKm": 1200.0,
            "riftJagAmp": 0.15,
            "riftJagKm": 450.0,
        },
    )

    grain = grain_of(world)
    first = grain(1)
    second = grain(1)

    assert first == second
    assert 0.2 <= first
    assert np.isnan(world.riftGrainOf[[0, 2]]).all()
    assert world.riftGrainOf[1] == first
    assert world.riftGrain is not None


def test_rift_context_precomputes_coast_and_cell_data_once():
    spec = _base_case()
    mesh = _mesh(spec)
    crust = _crust(spec)
    world = SimpleNamespace(
        mesh=mesh,
        config={"riftCurveAmp": 0.0, "riftJagAmp": 0.0},
    )

    context = rift_context(world, SimpleNamespace(crust=crust))

    np.testing.assert_array_equal(context["cells"], masked_cells(crust))
    np.testing.assert_array_equal(context["exterior"], exterior_sea(mesh, crust))
    np.testing.assert_array_equal(
        context["from_coast"],
        coast_distance(mesh, crust, context["exterior"]),
    )
    assert context["grain"](0) == 1.0


@pytest.mark.parametrize(
    ("junction", "mask"),
    [
        (0, [0, 1, 1]),
        (1, [1, 1, 1]),
        (2, [1, 1, 1]),
    ],
)
def test_find_rift_rejects_unmasked_ocean_and_craton_junctions(junction, mask):
    crust = SimpleNamespace(
        mask=np.asarray(mask, dtype=np.uint8),
        type=np.asarray(
            [
                CrustType["CONTINENT"],
                CrustType["OCEAN"],
                CrustType["CRATON"],
            ],
            dtype=np.uint8,
        ),
    )
    world = SimpleNamespace(
        mesh=SimpleNamespace(xyz=np.empty(0, dtype=np.float64)),
        config={},
    )

    assert find_rift(world, SimpleNamespace(crust=crust), junction) is None
