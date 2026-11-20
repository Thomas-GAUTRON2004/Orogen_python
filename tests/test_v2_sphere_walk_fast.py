"""Parity of the compiled sphere walk with the Python reference."""

from __future__ import annotations

import types

import numpy as np
import pytest

from orogen.v2.core import _numba_kernels as kernels
from orogen.v2.core import sphere_index
from orogen.v2.core.sphere_index import SphereIndex


def _cube_mesh(count: int, seed: int):
    """Random points on the sphere with a k-nearest-neighbour CSR graph."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    gram = points @ points.T
    np.fill_diagonal(gram, -np.inf)
    k = 6
    neighbours = np.argsort(-gram, axis=1)[:, :k]
    offsets = np.arange(count + 1, dtype=np.int32) * k
    adjacency = neighbours.reshape(-1).astype(np.int32)
    mesh = types.SimpleNamespace(
        num_cells=count, adj_offset=offsets, adj_list=adjacency
    )
    return mesh, points.reshape(-1)


def _python_walk(index: SphereIndex, current: int, x: float, y: float, z: float):
    """Reference walk: the original Python loop, with compiled dispatch disabled."""
    original = sphere_index.NUMBA_AVAILABLE
    sphere_index.NUMBA_AVAILABLE = False
    try:
        return index.walk(current, x, y, z)
    finally:
        sphere_index.NUMBA_AVAILABLE = original


@pytest.mark.skipif(not kernels.NUMBA_AVAILABLE, reason="compiled walk needs Numba")
@pytest.mark.parametrize("seed", range(6))
def test_compiled_walk_matches_python_walk(seed):
    mesh, flat = _cube_mesh(180, seed)
    index = SphereIndex(flat, mesh, 2)
    rng = np.random.default_rng(100 + seed)
    queries = rng.normal(size=(200, 3))
    for x, y, z in queries:
        start = index.start(x, y, z)
        assert index.walk(start, x, y, z) == _python_walk(index, start, x, y, z)


@pytest.mark.skipif(not kernels.NUMBA_AVAILABLE, reason="compiled walk needs Numba")
def test_compiled_walk_python_body_matches_reference():
    mesh, flat = _cube_mesh(120, 9)
    index = SphereIndex(flat, mesh, 2)
    kernel = kernels.walk_numba.py_func
    xyz = np.ascontiguousarray(index.xyz)
    off = np.ascontiguousarray(index.adj_offset, dtype=np.int32)
    adj = np.ascontiguousarray(index.adj_list, dtype=np.int32)
    rng = np.random.default_rng(4)
    for x, y, z in rng.normal(size=(60, 3)):
        start = index.start(x, y, z)
        expected = _python_walk(index, start, x, y, z)
        assert kernel(xyz, off, adj, start, float(x), float(y), float(z)) == expected


def test_walk_falls_back_for_non_float64_coordinates(monkeypatch):
    mesh, flat = _cube_mesh(60, 2)
    index = SphereIndex(flat.astype(np.float32), mesh, 2)
    monkeypatch.setattr(sphere_index, "NUMBA_AVAILABLE", True)
    calls = []
    monkeypatch.setattr(
        sphere_index, "walk_numba", lambda *args: calls.append(args) or 0
    )
    index.walk(0, 1.0, 0.0, 0.0)
    assert calls == []  # float32 coordinates keep the Python arithmetic path
