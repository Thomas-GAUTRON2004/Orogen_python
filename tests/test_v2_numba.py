"""The JIT kernels must preserve the pure-Python arithmetic exactly."""
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE
from orogen.v2.core.fast_noise import FastNoise
from orogen.v2.core.mesh_bfs import hop_distance
from orogen.v2.core.min_heap import spread_km


pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="install orogen[speed] to test Numba kernels")


def test_numba_noise_and_fbm_are_bit_identical_to_python_reference():
    noise = FastNoise(9876)
    points = np.asarray([
        (math.sin(i * .013) * 17.0, math.cos(i * .007) * 11.0,
         math.sin(i * .003 + .4) * 23.0)
        for i in range(2048)
    ], dtype=np.float64)
    expected = np.asarray([noise._fbm_python(*p, 5, 2/3) for p in points])
    actual = noise.fbm_many(points, 5, 2/3)
    np.testing.assert_array_equal(actual, expected)
    for p in points[::31]:
        assert noise.noise3d(*p) == noise._noise3d_python(*p)
        assert noise.fbm(*p, 5, 2/3) == noise._fbm_python(*p, 5, 2/3)


def test_numba_bfs_matches_python_fallback_for_multiple_sources_and_cutoffs():
    class Mesh:
        num_cells = 9
        adj_offset = np.asarray([0,2,4,6,8,10,12,14,16,18], dtype=np.int32)
        adj_list = np.asarray([1,8,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,0], dtype=np.int32)

    mesh = Mesh()
    for max_hops in (0, 1, 2, 5, math.inf):
        got = hop_distance(mesh, lambda c: c in (0, 4), max_hops)
        # Explicit reference BFS, independent of the Numba dispatch.
        distance = np.full(mesh.num_cells, -1, dtype=np.int16)
        frontier = []
        for cell in range(mesh.num_cells):
            if cell in (0, 4):
                distance[cell] = 0
                frontier.append(cell)
        depth = 1
        while frontier and depth <= max_hops:
            nxt = []
            for cell in frontier:
                for edge in range(mesh.adj_offset[cell], mesh.adj_offset[cell+1]):
                    neighbour = int(mesh.adj_list[edge])
                    if distance[neighbour] < 0:
                        distance[neighbour] = depth
                        nxt.append(neighbour)
            frontier = nxt
            depth += 1
        np.testing.assert_array_equal(got, distance)


def test_numba_bfs_respects_can_enter_python_predicate():
    class Mesh:
        num_cells = 6
        adj_offset = np.asarray([0,1,3,5,7,9,10], dtype=np.int32)
        adj_list = np.asarray([1,0,2,1,3,2,4,3,5,4], dtype=np.int32)

    got = hop_distance(Mesh(), lambda c: c == 0, 10, lambda c: c != 3)
    np.testing.assert_array_equal(got, np.asarray([0,1,2,-1,-1,-1], dtype=np.int16))



def test_python_fallback_remains_available_without_numba(monkeypatch):
    import orogen.v2.core.fast_noise as fast_noise_module

    monkeypatch.setattr(fast_noise_module, "NUMBA_AVAILABLE", False)
    noise = FastNoise(123)
    point = (0.125, -0.75, 1.5)
    expected = noise._fbm_python(*point, 4, 0.6)
    assert noise.noise3d(*point) == noise._noise3d_python(*point)
    assert noise.fbm(*point, 4, 0.6) == expected
    values = noise.fbm_many(np.asarray([point], dtype=np.float64), 4, 0.6)
    np.testing.assert_array_equal(values, np.asarray([expected]))
    with pytest.raises(ValueError, match="shape"):
        noise.fbm_many(np.zeros((2, 2)), 4, 0.6)



def test_numba_fbm_matches_upstream_javascript_bit_for_bit():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for upstream JavaScript comparison")
    root = Path(__file__).resolve().parents[1]
    runner = root / "tests/reference/v2/noise-probe.mjs"
    points = np.asarray([
        (math.sin(i * .013) * 17.0, math.cos(i * .007) * 11.0,
         math.sin(i * .003 + .4) * 23.0)
        for i in range(2048)
    ], dtype=np.float64)
    result = subprocess.run(
        [node, str(runner)], input=json.dumps(points.tolist()),
        text=True, capture_output=True, check=True,
    )
    expected = np.asarray(json.loads(result.stdout), dtype=np.float64)
    actual = FastNoise(9876).fbm_many(points, 5, 2/3)
    np.testing.assert_array_equal(actual, expected)


def test_numba_dijkstra_matches_python_for_float32_and_float64(monkeypatch):
    import orogen.v2.core.min_heap as min_heap_module

    class Mesh:
        num_cells = 5
        adj_offset = np.asarray([0, 1, 5, 6, 7, 8], dtype=np.int32)
        adj_list = np.asarray([1, 0, 2, 3, 4, 1, 1, 1], dtype=np.int32)
        xyz = np.asarray([
            1.0, 0.0, 0.0,
            math.cos(0.01), math.sin(0.01), 0.0,
            math.cos(0.02), math.sin(0.02), 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ], dtype=np.float64)

    mesh = Mesh()
    for dtype in (np.float64, np.float32):
        for max_km in (100.0, 20000.0):
            initial = np.full(mesh.num_cells, np.inf, dtype=dtype)
            initial[0] = 0.0
            initial[4] = 15.0
            actual = spread_km(mesh, initial.copy(), max_km=max_km)

            monkeypatch.setattr(min_heap_module, "NUMBA_AVAILABLE", False)
            expected = spread_km(mesh, initial.copy(), max_km=max_km)
            monkeypatch.setattr(min_heap_module, "NUMBA_AVAILABLE", True)
            np.testing.assert_array_equal(actual, expected)
