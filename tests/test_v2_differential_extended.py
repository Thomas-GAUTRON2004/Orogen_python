"""Expanded differential suite against vendored Orogen V2 JavaScript modules."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
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
def original_extended():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for upstream JavaScript differential tests")
    result = subprocess.run(
        [node, str(RUNNER)], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"upstream JS runner failed:\n{result.stderr}\n{result.stdout}"
    )
    return json.loads(result.stdout)


@dataclass(eq=False)
class _Mesh:
    num_cells: int
    xyz: np.ndarray
    adj_offset: np.ndarray
    adj_list: np.ndarray
    edge_rad: float
    index: object = None


def _mesh():
    xyz = np.asarray(
        [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1], dtype=np.float32
    )
    rings = [
        [2, 4, 3, 5],
        [2, 5, 3, 4],
        [4, 0, 5, 1],
        [4, 1, 5, 0],
        [3, 0, 2, 1],
        [2, 0, 3, 1],
    ]
    mesh = _Mesh(
        6,
        xyz,
        np.asarray([0, 4, 8, 12, 16, 20, 24], dtype=np.int32),
        np.asarray([x for ring in rings for x in ring], dtype=np.int32),
        0.02,
    )
    mesh.index = SphereIndex(xyz, mesh, 2)
    return mesh


def _close(a, b, atol=2e-12):
    np.testing.assert_allclose(a, b, rtol=2e-12, atol=atol)


def test_spherical_geometry_matches_original(original_extended):
    g = original_extended["geom"]
    assert dot([1, 2, 3], [-2, 0.5, 4]) == g["dot"]
    assert cross([1, 2, 3], [-2, 0.5, 4]) == g["cross"]
    _close(normalize([2, -3, 6]), g["normalize"])
    _close(
        [angle_between([1, 0, 0], p) for p in [[1, 0, 0], [0, 1, 0], [0, 0, -1]]],
        g["angles"],
    )
    _close(tangent_toward([0, 0, 1], [1, 0, 0], [0, 1, 0]), g["tangent"])
    rng = make_rng(7821)
    _close([random_point(rng) for _ in range(5)], g["randomPoints"])
    p = [0.2, -0.3, math.sqrt(0.87)]
    e, n = tangent_basis(p)
    _close([e, n], g["basis"])
    dest = destination(p, 1.234, 0.42)
    _close(dest, g["destination"])
    assert bearing_to(p, dest) == pytest.approx(g["bearing"], abs=2e-12)
    assert cap_fraction(0.75) == pytest.approx(g["cap"], abs=2e-15)
    polar = polar_frame([0.3, 0.4, math.sqrt(0.75)], [0, 0, 1])
    for point, expected in zip(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]],
        g["polar"],
        strict=False,
    ):
        actual = polar(*point)
        assert actual["colat"] == pytest.approx(expected["colat"], abs=2e-12)
        assert actual["azimuth"] == pytest.approx(expected["azimuth"], abs=2e-12)
    m = _mesh()
    assert cell_distance_km(m.xyz, 0, 4) == pytest.approx(g["cellDistance"], abs=0)
    _close(centroid_of_cells(m.xyz, [0, 2, 4]), g["centroid"])


def test_simplex_noise_and_fbm_match_original(original_extended):
    noise = FastNoise(9876)
    for item in original_extended["noise"]:
        assert noise.noise3d(*item["p"]) == pytest.approx(item["n"], abs=1e-15)
        assert noise.fbm(*item["p"], 5, 2 / 3) == pytest.approx(item["f"], abs=1e-15)


def test_sphere_index_cube_table_and_nearest_match_original(original_extended):
    mesh = _mesh()
    expected = original_extended["index"]
    assert mesh.index.res == expected["res"]
    assert mesh.index.table.tolist() == expected["table"]
    queries = [
        [1, 0.2, 0.3],
        [-0.3, 1, 0.4],
        [0.2, -0.4, 1],
        [-1, -0.3, -0.2],
        [0.2, 0.3, -1],
        [1, 1, 1],
    ]
    assert [mesh.index.start(*p) for p in queries] == expected["starts"]
    assert [mesh.index.nearest(*p) for p in queries] == expected["nearest"]


def test_mesh_from_arrays_matches_original(original_extended):
    mesh = _mesh()
    rebuilt = mesh_from_arrays(
        num_cells=6, xyz=mesh.xyz, adj_offset=mesh.adj_offset, adj_list=mesh.adj_list
    )
    expected = original_extended["rebuilt"]
    assert rebuilt.num_cells == expected["numCells"]
    assert rebuilt.edge_rad == expected["edgeRad"]
    assert rebuilt.edge_km == pytest.approx(expected["edgeKm"], rel=2e-15)
    assert rebuilt.cell_area_km2 == pytest.approx(expected["cellAreaKm2"], rel=2e-15)
    queries = [
        [1, 0.2, 0.3],
        [-0.3, 1, 0.4],
        [0.2, -0.4, 1],
        [-1, -0.3, -0.2],
        [0.2, 0.3, -1],
        [1, 1, 1],
    ]
    assert [rebuilt.index.nearest(*p) for p in queries] == expected["nearest"]


def test_crust_copy_restore_remove_match_original(original_extended):
    source = CrustRaster(3)
    destination = CrustRaster(3)
    source.create(0, CrustType["CRATON"], -1.25, 17)
    source.flags[0] = 129
    source.slabKm[0] = 23.5
    source.coverKm[0] = 3.25
    source.geoFlags[0] = 3
    source.oroType[0] = 4
    source.farDx[0] = 64
    source.farDy[0] = -32
    source.farDz[0] = 96
    source_snapshot = source.snapshot(0)
    rotation = [0, -1, 0, 1, 0, 0, 0, 0, 1]
    destination.copy_from(1, source, 0, rotation)
    destination.restore(2, source_snapshot)
    destination.remove(1)
    expected = original_extended["crustCopy"]
    assert source_snapshot == expected["source"]
    assert destination.snapshot(1) == expected["copied"]
    assert destination.snapshot(2) == expected["restored"]
    assert destination.mask.tolist() == expected["mask"]
    assert destination.count == expected["count"]


def test_barycentric_point_location_matches_original(original_extended):
    mesh = _mesh()
    queries = [
        [1, 0.2, 0.3],
        [-0.3, 1, 0.4],
        [0.2, -0.4, 1],
        [-1, -0.3, -0.2],
        [0.2, 0.3, -1],
        [1, 1, 1],
    ]
    for point, expected in zip(queries, original_extended["locate"], strict=False):
        cells, weights = locate(mesh, *point)
        assert cells.tolist() == expected["cells"]
        _close(weights, expected["weights"])


def test_breadth_first_mesh_operations_match_original(original_extended):
    mesh = _mesh()

    def source(c):
        return c in (0, 2)

    def enter(c):
        return c != 5

    e = original_extended["bfs"]
    assert within_km(mesh, source, 9000).tolist() == e["within"]
    assert hop_distance(mesh, source, 2, enter).tolist() == e["hops"]
    assert flood_from(mesh, 0, enter).tolist() == e["flood"]
    assert touches(mesh, 0, lambda c: c == 4) is e["touches"]
    d, s = hops_from(mesh, [0, 2], enter, 2)
    assert d.tolist() == e["from"]["dist"]
    assert s.tolist() == e["from"]["source"]
    assert reaches_within(mesh, 0, 2, enter, lambda c: c == 3) is e["reaches"]
    assert pieces_of(mesh, [0, 2, 4, 1, 3]) == e["pieces"]
    assert largest_piece_of(mesh, [0, 2, 4, 1, 3]) == e["largest"]


def test_precomputed_rings_match_original(original_extended):
    r = rings_of(_mesh())
    e = original_extended["rings"]
    assert r["stride"] == e["stride"]
    assert r["start"].tolist() == e["start"]
    assert r["cells"].tolist() == e["cells"]


def test_min_heap_order_and_dijkstra_match_original(original_extended):
    h = MinHeap(2)
    for key, i in [[7, 70], [1, 10], [4, 40], [1, 11], [3, 30], [0.5, 5], [9, 90]]:
        h.push(key, i)
    actual = []
    while h.size:
        i = h.pop()
        actual.append([h.last_key, i])
    assert actual == original_extended["heap"]
    m = _mesh()
    d = np.asarray(
        [0, math.inf, math.inf, math.inf, math.inf, math.inf], dtype=np.float64
    )
    spread_km(m, d, max_km=12000, can_enter=lambda a, b: b != 5)
    expected = [math.inf if x == "Infinity" else x for x in original_extended["spread"]]
    _close(d, expected, atol=1e-8)


def test_tectonic_world_state_and_rng_streams_match_original(original_extended):
    m = _mesh()
    w = TectonicWorld(
        m, 77, overrides={"fixture": {"nested": 3}}
    )  # as upstream: defaults merged with overrides
    p = w.add_plate("North")
    p.crust.create(0, CrustType["CRATON"], -20, 11)
    p.crust.create(2, CrustType["CRATON"], -30, 12)
    w.terraneLetters = {11: "B", 12: "A"}
    w.time = 12.5
    w.log(EventKind["START"], "fixture event", {"plate": 0, "area": 2})
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
    assert p.cell_at(m.index, 1, 0, 0) == e["cellAt"]
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
    assert w.name_by_cratons(p, "fallback") == e["name"]
    for stream, values in e["streams"].items():
        _close([getattr(w, stream)() for _ in range(4)], values, atol=0)
    assert BoundaryKind["COLLISION"] == e["constants"]["collision"]
    assert Provisional["LIKE_NEIGHBOUR"] == e["constants"]["provisional"]
