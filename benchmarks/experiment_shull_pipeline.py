"""Full spherical-mesh A/B: SciPy/Qhull vs Rust shull, then Numba consumers.

Run:
    pip install -e ".[speed]" shull==0.4.0
    python benchmarks/experiment_shull_pipeline.py

Uses Orogen's Fibonacci-sphere projection, north-pole closure and cyclic CSR
post-processing. Triangulation and Python post-processing are timed separately.
"""
from __future__ import annotations
import time
from types import SimpleNamespace
import numpy as np
import shull
from scipy.spatial import Delaunay
from orogen.v2.core.mesh_bfs import hop_distance
from orogen.v2.core.sphere_mesh import _cyclic_region_adjacency, generate_fibonacci_sphere, stereographic_projection
from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE

def timed(fn, repeats=3):
    fn()
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)), result

def canonical(simplices):
    rows = np.sort(np.asarray(simplices, dtype=np.int64), axis=1)
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]

def close_planar_simplices(points, simplices, pole_id):
    triangles = np.asarray(simplices, dtype=np.int32).copy()
    for tri in triangles:
        a, b, c = map(int, tri)
        ax, ay = points[a]
        bx, by = points[b]
        cx, cy = points[c]
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
            tri[1], tri[2] = tri[2], tri[1]
    edges = {}
    for tri in triangles:
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            if key in edges:
                old_u, old_v, count = edges[key]
                edges[key] = (old_u, old_v, count + 1)
            else:
                edges[key] = (u, v, 1)
    hull = [(u, v) for u, v, count in edges.values() if count == 1]
    if len(hull) < 3:
        raise AssertionError("planar triangulation has no valid convex hull")
    cap = np.asarray([(v, u, pole_id) for u, v in hull], dtype=np.int32)
    return np.concatenate((triangles, cap), axis=0)

def postprocess(points, simplices, n):
    faces = close_planar_simplices(points, simplices, n)
    offsets, adjacency = _cyclic_region_adjacency(faces, n + 1)
    return faces, offsets, adjacency

def row_sets(offsets, adjacency):
    return [set(map(int, adjacency[offsets[i]:offsets[i + 1]])) for i in range(len(offsets) - 1)]

if not NUMBA_AVAILABLE:
    raise RuntimeError("Install the optional speed extra to run the Numba composition test")

for n in (10_000, 40_000, 80_000):
    xyz = generate_fibonacci_sphere(n, 0.0, lambda: 0.5)
    points = np.ascontiguousarray(stereographic_projection(xyz, n), dtype=np.float64)
    scipy_tri_time, scipy_tri = timed(lambda: Delaunay(points).simplices)
    shull_tri_time, shull_tri = timed(lambda: shull.Delaunay(points).simplices)
    if not np.array_equal(canonical(scipy_tri), canonical(shull_tri)):
        raise AssertionError(f"Qhull and shull simplex sets differ at n={n}")

    scipy_post_time, scipy_mesh = timed(lambda: postprocess(points, scipy_tri, n))
    shull_post_time, shull_mesh = timed(lambda: postprocess(points, shull_tri, n))
    scipy_faces, scipy_offsets, scipy_adj = scipy_mesh
    shull_faces, shull_offsets, shull_adj = shull_mesh
    if not np.array_equal(canonical(scipy_faces), canonical(shull_faces)):
        raise AssertionError("closed spherical face sets differ")
    if row_sets(scipy_offsets, scipy_adj) != row_sets(shull_offsets, shull_adj):
        raise AssertionError("spherical vertex-neighbour sets differ")

    scipy_proxy = SimpleNamespace(num_cells=n + 1, adj_offset=scipy_offsets, adj_list=scipy_adj)
    shull_proxy = SimpleNamespace(num_cells=n + 1, adj_offset=shull_offsets, adj_list=shull_adj)
    source = lambda cell: cell % max(1, n // 100) == 0
    bfs_time, scipy_distance = timed(lambda: hop_distance(scipy_proxy, source))
    _, shull_distance = timed(lambda: hop_distance(shull_proxy, source))
    if not np.array_equal(scipy_distance, shull_distance):
        raise AssertionError("Numba BFS distances differ between triangulators")

    scipy_total = scipy_tri_time + scipy_post_time
    shull_total = shull_tri_time + shull_post_time
    mesh_mib = (shull_faces.nbytes + shull_offsets.nbytes + shull_adj.nbytes) / (1024 * 1024)
    print(
        f"cells={n + 1:>7} scipy_tri={scipy_tri_time:.4f}s shull_tri={shull_tri_time:.4f}s "
        f"tri_speedup={scipy_tri_time / shull_tri_time:.2f}x "
        f"scipy_post={scipy_post_time:.4f}s shull_post={shull_post_time:.4f}s "
        f"scipy_total={scipy_total:.4f}s shull_total={shull_total:.4f}s "
        f"pipeline_speedup={scipy_total / shull_total:.2f}x "
        f"numba_bfs={bfs_time:.4f}s same_topology=True mesh_arrays={mesh_mib:.2f}MiB"
    )
