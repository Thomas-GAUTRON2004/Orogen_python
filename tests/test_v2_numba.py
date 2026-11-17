"""The JIT kernels must preserve the pure-Python arithmetic exactly."""
import math

import numpy as np
import pytest

from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE
from orogen.v2.core.fast_noise import FastNoise
from orogen.v2.core.mesh_bfs import hop_distance


pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="install orogen[speed] to test Numba kernels")


def test_numba_noise_and_fbm_are_bit_identical_to_python_reference():
    noise = FastNoise(9876)
    points = np.asarray([
        (math.sin(i * .013) * 17.0, math.cos(i * .007) * 11.0,
         math.sin(i * .003 + .4) * 23.0)
        for i in range(2048)
    ], dtype=np.float64)
    grad = np.asarray((1,1,0,-1,1,0,1,-1,0,-1,-1,0,1,0,1,-1,0,1,1,0,-1,-1,0,-1,0,1,1,0,-1,1,0,1,-1,0,-1,-1), dtype=np.float64)
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
