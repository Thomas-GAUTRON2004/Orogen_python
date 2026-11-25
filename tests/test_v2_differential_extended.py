"""Expanded differential suite against the pinned upstream Orogen V2 JavaScript source."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from orogen.v2.core.fast_noise import FastNoise
from orogen.v2.core.mesh_bfs import (
    flood_from,
    hop_distance,
    hops_from,
    largest_piece_of,
    pieces_of,
    reaches_within,
    touches,
    within_km,
)
from orogen.v2.core.mesh_locate import locate
from orogen.v2.core.mesh_rings import rings_of
from orogen.v2.core.min_heap import MinHeap, spread_km
from orogen.v2.core.seed import make_rng
from orogen.v2.core.sim_mesh import mesh_from_arrays
from orogen.v2.core.simplex_noise import SimplexNoise
from orogen.v2.core.sphere_geometry import (
    angle_between,
    bearing_to,
    cap_fraction,
    cell_distance_km,
    centroid_of_cells,
    cross,
    destination,
    dot,
    normalize,
    polar_frame,
    random_point,
    tangent_basis,
    tangent_toward,
)
from orogen.v2.core.sphere_index import SphereIndex
from orogen.v2.tectonics.crust import CrustRaster, CrustType
from orogen.v2.tectonics.world import (
    BoundaryKind,
    EventKind,
    Provisional,
    TectonicWorld,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/js/differential-extended-runner.mjs"


@pytest.fixture(scope="module")
def extended_payload():
    xyz = [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1]
    rings = [
        [2, 4, 3, 5],
        [2, 5, 3, 4],
        [4, 0, 5, 1],
        [4, 1, 5, 0],
        [3, 0, 2, 1],
        [2, 0, 3, 1],
    ]
    queries = [
        [1, 0.2, 0.3],
        [-0.3, 1, 0.4],
        [0.2, -0.4, 1],
        [-1, -0.3, -0.2],
        [0.2, 0.3, -1],
        [1, 1, 1],
    ]
    return {
        "mesh": {"xyz": xyz, "rings": rings, "edgeRad": 0.02, "indexResolution": 2},
        "random": {"seed": 7821, "count": 5},
        "noise": {
            "seed": 9876,
            "points": [
                [0.1, 0.2, 0.3],
                [-1.25, 2.5, 0.75],
                [12.125, -4.5, 9.75],
                [0, 0, 0],
            ],
            "octaves": 5,
            "persistence": 2 / 3,
        },
        "geometry": {
            "dotA": [1, 2, 3],
            "dotB": [-2, 0.5, 4],
            "crossA": [1, 2, 3],
            "crossB": [-2, 0.5, 4],
            "normalize": [2, -3, 6],
            "angleOrigin": [1, 0, 0],
            "angleVectors": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            "tangent": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            "point": [0.2, -0.3, math.sqrt(0.87)],
            "destinationBearing": 1.234,
            "destinationDistance": 0.42,
            "capFraction": 0.75,
            "polarOrigin": [0.3, 0.4, math.sqrt(0.75)],
            "polarNorth": [0, 0, 1],
            "polarPoints": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]],
            "cellDistance": [0, 4],
            "centroidCells": [0, 2, 4],
        },
        "queries": queries,
        "bfs": {
            "sourceCells": [0, 2],
            "blockedCell": 5,
            "withinKm": 9000,
            "maxHops": 2,
            "floodStart": 0,
            "touchesCell": 0,
            "touchesTarget": 4,
            "reachesStart": 0,
            "reachesTarget": 3,
            "pieces": [0, 2, 4, 1, 3],
        },
        "heap": {
            "capacity": 2,
            "entries": [[7, 70], [1, 10], [4, 40], [1, 11], [3, 30], [0.5, 5], [9, 90]],
            "initialDistances": [
                0,
                "Infinity",
                "Infinity",
                "Infinity",
                "Infinity",
                "Infinity",
            ],
            "maxKm": 12000,
        },
        "crustCopy": {
            "size": 3,
            "cell": 0,
            "type": CrustType["CRATON"],
            "longitude": -1.25,
            "terrane": 17,
            "fields": {
                "flags": 129,
                "slabKm": 23.5,
                "coverKm": 3.25,
                "geoFlags": 3,
                "oroType": 4,
                "farDx": 64,
                "farDy": -32,
                "farDz": 96,
            },
            "rotation": [0, -1, 0, 1, 0, 0, 0, 0, 1],
            "destinationCell": 1,
            "restoreCell": 2,
        },
        "world": {
            "seed": 77,
            "config": {"fixture": {"nested": 3}},
            "plateName": "North",
            "crust": [
                [0, CrustType["CRATON"], -20, 11],
                [2, CrustType["CRATON"], -30, 12],
            ],
            "terraneLetters": {"11": "B", "12": "A"},
            "time": 12.5,
            "eventKind": "START",
            "eventMessage": "fixture event",
            "eventData": {"plate": 0, "area": 2},
            "cellAtPoint": [1, 0, 0],
            "fallbackName": "fallback",
            "streams": [
                "rng",
                "rngSubduction",
                "rngMarginFounder",
                "rngOceanSplit",
                "rngHotspot",
                "rngVolcano",
                "rngBackArc",
                "rngMarginIsle",
                "rngCycle",
            ],
            "streamCount": 4,
        },
    }


@pytest.fixture(scope="module")
def original_extended(extended_payload):
    node = shutil.which("node")
    if node is None:
        message = "Node.js is required for upstream JavaScript differential tests"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    result = subprocess.run(
        [node, str(RUNNER)],
        input=json.dumps(extended_payload),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, (
        f"upstream JS runner failed:\n{result.stderr}\n{result.stdout}"
    )
    return json.loads(result.stdout)


@pytest.fixture(params=("numba", "python"))
def numerical_backend(request, monkeypatch):
    """Run parity scenarios through both Numba and pure-Python implementations."""
    if request.param == "python":
        modules = (
            "orogen.v2.core.fast_noise",
            "orogen.v2.core.mesh_bfs",
            "orogen.v2.core.min_heap",
            "orogen.v2.core.simplex_noise",
            "orogen.v2.core.sphere_geometry",
            "orogen.v2.core.sphere_index",
            "orogen.v2.tectonics.crust",
        )
        for module_name in modules:
            module = import_module(module_name)
            monkeypatch.setattr(module, "NUMBA_AVAILABLE", False)
    return request.param


@dataclass(eq=False)
class _Mesh:
    num_cells: int
    xyz: np.ndarray
    adj_offset: np.ndarray
    adj_list: np.ndarray
    edge_rad: float
    index: object = None


def _mesh(payload):
    scenario = payload["mesh"]
    xyz = np.asarray(scenario["xyz"], dtype=np.float32)
    rings = scenario["rings"]
    offsets = np.asarray([0, *np.cumsum([len(ring) for ring in rings])], dtype=np.int32)
    mesh = _Mesh(
        xyz.size // 3,
        xyz,
        offsets,
        np.asarray([x for ring in rings for x in ring], dtype=np.int32),
        scenario["edgeRad"],
    )
    mesh.index = SphereIndex(xyz, mesh, scenario["indexResolution"])
    return mesh


def _close(a, b, atol=2e-12):
    np.testing.assert_allclose(a, b, rtol=2e-12, atol=atol)


def test_spherical_geometry_matches_original(
    original_extended, numerical_backend, extended_payload
):
    g = original_extended["geom"]
    scenario = extended_payload["geometry"]
    assert dot(scenario["dotA"], scenario["dotB"]) == g["dot"]
    assert cross(scenario["crossA"], scenario["crossB"]) == g["cross"]
    _close(normalize(scenario["normalize"]), g["normalize"])
    _close(
        [angle_between(scenario["angleOrigin"], p) for p in scenario["angleVectors"]],
        g["angles"],
    )
    _close(tangent_toward(*scenario["tangent"]), g["tangent"])
    rng = make_rng(extended_payload["random"]["seed"])
    _close(
        [random_point(rng) for _ in range(extended_payload["random"]["count"])],
        g["randomPoints"],
    )
    p = scenario["point"]
    e, n = tangent_basis(p)
    _close([e, n], g["basis"])
    dest = destination(
        p, scenario["destinationBearing"], scenario["destinationDistance"]
    )
    _close(dest, g["destination"])
    assert bearing_to(p, dest) == pytest.approx(g["bearing"], abs=2e-12)
    assert cap_fraction(scenario["capFraction"]) == pytest.approx(g["cap"], abs=2e-15)
    polar = polar_frame(scenario["polarOrigin"], scenario["polarNorth"])
    for point, expected in zip(
        scenario["polarPoints"],
        g["polar"],
        strict=False,
    ):
        actual = polar(*point)
        assert actual["colat"] == pytest.approx(expected["colat"], abs=2e-12)
        assert actual["azimuth"] == pytest.approx(expected["azimuth"], abs=2e-12)
    m = _mesh(extended_payload)
    assert cell_distance_km(m.xyz, *scenario["cellDistance"]) == pytest.approx(
        g["cellDistance"], abs=0
    )
    _close(centroid_of_cells(m.xyz, scenario["centroidCells"]), g["centroid"])


def test_fast_and_simplex_noise_match_original(
    original_extended, numerical_backend, extended_payload
):
    noise_input = extended_payload["noise"]
    fast = FastNoise(noise_input["seed"])
    simplex = SimplexNoise(noise_input["seed"])
    for point, item in zip(
        noise_input["points"], original_extended["noise"], strict=True
    ):
        assert fast.noise3d(*point) == pytest.approx(item["n"], abs=1e-15)
        assert fast.fbm(
            *point, noise_input["octaves"], noise_input["persistence"]
        ) == pytest.approx(item["f"], abs=1e-15)
    for point, item in zip(
        noise_input["points"], original_extended["simplex"], strict=True
    ):
        assert simplex.noise3D(*point) == pytest.approx(item["n"], abs=1e-15)
        assert simplex.fbm(
            *point, noise_input["octaves"], noise_input["persistence"]
        ) == pytest.approx(item["f"], abs=1e-15)
        assert simplex.ridged_fbm(*point) == pytest.approx(item["r"], abs=1e-15)


def test_sphere_index_cube_table_and_nearest_match_original(
    original_extended, numerical_backend, extended_payload
):
    mesh = _mesh(extended_payload)
    expected = original_extended["index"]
    assert mesh.index.res == expected["res"]
    assert mesh.index.table.tolist() == expected["table"]
    queries = extended_payload["queries"]
    assert [mesh.index.start(*p) for p in queries] == expected["starts"]
    assert [mesh.index.nearest(*p) for p in queries] == expected["nearest"]


def test_mesh_from_arrays_matches_original(
    original_extended, numerical_backend, extended_payload
):
    mesh = _mesh(extended_payload)
    rebuilt = mesh_from_arrays(
        num_cells=mesh.num_cells,
        xyz=mesh.xyz,
        adj_offset=mesh.adj_offset,
        adj_list=mesh.adj_list,
    )
    expected = original_extended["rebuilt"]
    assert rebuilt.num_cells == expected["numCells"]
    assert rebuilt.edge_rad == expected["edgeRad"]
    assert rebuilt.edge_km == pytest.approx(expected["edgeKm"], rel=2e-15)
    assert rebuilt.cell_area_km2 == pytest.approx(expected["cellAreaKm2"], rel=2e-15)
    queries = extended_payload["queries"]
    assert [rebuilt.index.nearest(*p) for p in queries] == expected["nearest"]


def test_crust_copy_restore_remove_match_original(
    original_extended, numerical_backend, extended_payload
):
    scenario = extended_payload["crustCopy"]
    source = CrustRaster(scenario["size"])
    destination = CrustRaster(scenario["size"])
    source.create(
        scenario["cell"], scenario["type"], scenario["longitude"], scenario["terrane"]
    )
    for field, value in scenario["fields"].items():
        getattr(source, field)[scenario["cell"]] = value
    source_snapshot = source.snapshot(scenario["cell"])
    destination.copy_from(
        scenario["destinationCell"], source, scenario["cell"], scenario["rotation"]
    )
    destination.restore(scenario["restoreCell"], source_snapshot)
    destination.remove(scenario["destinationCell"])
    expected = original_extended["crustCopy"]
    assert source_snapshot == expected["source"]
    assert destination.snapshot(scenario["destinationCell"]) == expected["copied"]
    assert destination.snapshot(scenario["restoreCell"]) == expected["restored"]
    assert destination.mask.tolist() == expected["mask"]
    assert destination.count == expected["count"]


def test_barycentric_point_location_matches_original(
    original_extended, extended_payload
):
    mesh = _mesh(extended_payload)
    queries = extended_payload["queries"]
    for point, expected in zip(queries, original_extended["locate"], strict=False):
        cells, weights = locate(mesh, *point)
        assert cells.tolist() == expected["cells"]
        _close(weights, expected["weights"])


def test_breadth_first_mesh_operations_match_original(
    original_extended, numerical_backend, extended_payload
):
    mesh = _mesh(extended_payload)
    scenario = extended_payload["bfs"]

    def source(c):
        return c in scenario["sourceCells"]

    def enter(c):
        return c != scenario["blockedCell"]

    e = original_extended["bfs"]
    assert within_km(mesh, source, scenario["withinKm"]).tolist() == e["within"]
    assert hop_distance(mesh, source, scenario["maxHops"], enter).tolist() == e["hops"]
    assert flood_from(mesh, scenario["floodStart"], enter).tolist() == e["flood"]
    assert (
        touches(mesh, scenario["touchesCell"], lambda c: c == scenario["touchesTarget"])
        is e["touches"]
    )
    d, s = hops_from(mesh, scenario["sourceCells"], enter, scenario["maxHops"])
    assert d.tolist() == e["from"]["dist"]
    assert s.tolist() == e["from"]["source"]
    assert (
        reaches_within(
            mesh,
            scenario["reachesStart"],
            scenario["maxHops"],
            enter,
            lambda c: c == scenario["reachesTarget"],
        )
        is e["reaches"]
    )
    assert pieces_of(mesh, scenario["pieces"]) == e["pieces"]
    assert largest_piece_of(mesh, scenario["pieces"]) == e["largest"]


def test_precomputed_rings_match_original(original_extended, extended_payload):
    r = rings_of(_mesh(extended_payload))
    e = original_extended["rings"]
    assert r["stride"] == e["stride"]
    assert r["start"].tolist() == e["start"]
    assert r["cells"].tolist() == e["cells"]


def test_min_heap_order_and_dijkstra_match_original(
    original_extended, numerical_backend, extended_payload
):
    heap_input = extended_payload["heap"]
    h = MinHeap(heap_input["capacity"])
    for key, i in heap_input["entries"]:
        h.push(key, i)
    actual = []
    while h.size:
        i = h.pop()
        actual.append([h.last_key, i])
    assert actual == original_extended["heap"]
    m = _mesh(extended_payload)
    d = np.asarray(
        [math.inf if x == "Infinity" else x for x in heap_input["initialDistances"]],
        dtype=np.float64,
    )
    spread_km(
        m,
        d,
        max_km=heap_input["maxKm"],
        can_enter=lambda a, b: b != extended_payload["bfs"]["blockedCell"],
    )
    expected = [math.inf if x == "Infinity" else x for x in original_extended["spread"]]
    _close(d, expected, atol=1e-8)


def test_tectonic_world_state_and_rng_streams_match_original(
    original_extended, numerical_backend, extended_payload
):
    scenario = extended_payload["world"]
    m = _mesh(extended_payload)
    w = TectonicWorld(m, scenario["seed"], overrides=scenario["config"])
    p = w.add_plate(scenario["plateName"])
    for cell, crust_type, longitude, terrane in scenario["crust"]:
        p.crust.create(cell, crust_type, longitude, terrane)
    w.terraneLetters = {
        int(key): value for key, value in scenario["terraneLetters"].items()
    }
    w.time = scenario["time"]
    w.log(
        EventKind[scenario["eventKind"]],
        scenario["eventMessage"],
        scenario["eventData"],
    )
    e = original_extended["world"]
    for field in [
        "owner",
        "ownerCell",
        "claimCount",
        "landClaims",
        "collider",
        "oceanOverrider",
        "consumedAt",
        "provisional",
        "provisionalLike",
        "boundaryKind",
        "convergence",
        "mantleHeat",
    ]:
        _close(getattr(w, field).tolist(), e[field], atol=0)
    assert p.cell_at(m.index, *scenario["cellAtPoint"]) == e["cellAt"]
    assert w.config == e["config"]
    assert w.budget == e["budget"]
    assert w.events == e["events"]
    assert {
        "id": p.id,
        "name": p.name,
        "born": p.born,
        "crustCount": p.crust.count,
    } == e["plate"]
    assert w.next_terrane_id() == e["nextTerrane"]
    assert w.name_by_cratons(p, scenario["fallbackName"]) == e["name"]
    for stream, values in e["streams"].items():
        _close(
            [getattr(w, stream)() for _ in range(scenario["streamCount"])],
            values,
            atol=0,
        )
    assert BoundaryKind["COLLISION"] == e["constants"]["collision"]
    assert Provisional["LIKE_NEIGHBOUR"] == e["constants"]["provisional"]
