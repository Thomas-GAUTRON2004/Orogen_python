"""Full spherical-mesh A/B: SciPy/Qhull vs shull, with Numba CSR/BFS."""
from __future__ import annotations
import time
from types import SimpleNamespace
import numpy as np
import shull
from numba import njit
from scipy.spatial import Delaunay
from orogen.v2.core.mesh_bfs import hop_distance
from orogen.v2.core.seed import make_rng
from orogen.v2.core.sphere_index import SphereIndex
from orogen.v2.core.sphere_mesh import (
    _cyclic_region_adjacency, generate_fibonacci_sphere, stereographic_projection
)
from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE

def timed(fn, repeats=3):
    fn()
    samples, result = [], None
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

@njit(cache=False)
def numba_cyclic_adjacency(triangles, num_regions):
    """Numba equivalent of _cyclic_region_adjacency, preserving insertion order."""
    link_count = triangles.shape[0] * 3
    capacity = 1
    while capacity < link_count * 2:
        capacity *= 2
    keys = np.full(capacity, -1, dtype=np.int64)
    values = np.full(capacity, -1, dtype=np.int32)
    degree = np.zeros(num_regions, dtype=np.int32)
    first_neighbour = np.full(num_regions, -1, dtype=np.int32)

    for row in range(triangles.shape[0]):
        a = int(triangles[row, 0])
        b = int(triangles[row, 1])
        c = int(triangles[row, 2])
        for slot_kind in range(3):
            if slot_kind == 0:
                centre, first, second = a, b, c
            elif slot_kind == 1:
                centre, first, second = b, c, a
            else:
                centre, first, second = c, a, b
            key = np.int64(centre) * np.int64(num_regions) + np.int64(first)
            slot = int(key % capacity)
            while keys[slot] != -1 and keys[slot] != key:
                slot = (slot + 1) % capacity
            if keys[slot] == -1:
                keys[slot] = key
                values[slot] = second
                degree[centre] += 1
                if first_neighbour[centre] < 0:
                    first_neighbour[centre] = first
            elif values[slot] != second:
                raise ValueError("non-manifold or inconsistently oriented mesh")

    offsets = np.empty(num_regions + 1, dtype=np.int32)
    offsets[0] = 0
    for cell in range(num_regions):
        offsets[cell + 1] = offsets[cell] + degree[cell]
        if degree[cell] == 0:
            raise ValueError("mesh region has no neighbours")
    adjacency = np.empty(offsets[-1], dtype=np.int32)

    for cell in range(num_regions):
        start = int(first_neighbour[cell])
        current = start
        for position in range(degree[cell]):
            adjacency[offsets[cell] + position] = current
            key = np.int64(cell) * np.int64(num_regions) + np.int64(current)
            slot = int(key % capacity)
            while keys[slot] != key:
                slot = (slot + 1) % capacity
            current = int(values[slot])
        if current != start:
            raise ValueError("mesh adjacency does not form one cyclic ring")
    return offsets, adjacency

def canonicalize_ring_starts(offsets, adjacency):
    result = np.asarray(adjacency, dtype=np.int32).copy()
    for cell in range(len(offsets) - 1):
        start, end = int(offsets[cell]), int(offsets[cell + 1])
        row = result[start:end].copy()
        pivot = int(np.argmin(row))
        result[start:end] = np.concatenate((row[pivot:], row[:pivot]))
    return result

def row_sets(offsets, adjacency):
    return [set(map(int, adjacency[offsets[i]:offsets[i + 1]])) for i in range(len(offsets) - 1)]

if not NUMBA_AVAILABLE:
    raise RuntimeError("Install the project's speed extra to run the Numba composition test")

for n in (10_000, 40_000, 80_000):
    xyz = generate_fibonacci_sphere(n, 0.35, make_rng(12345 + 4242))
    points = np.ascontiguousarray(stereographic_projection(xyz, n), dtype=np.float64)
    scipy_tri_time, scipy_tri = timed(lambda: Delaunay(points).simplices)
    shull_tri_time, shull_tri = timed(lambda: shull.Delaunay(points).simplices)
    if not np.array_equal(canonical(scipy_tri), canonical(shull_tri)):
        raise AssertionError(f"Qhull and shull simplex sets differ at n={n}")

    scipy_faces_time, scipy_faces = timed(lambda: close_planar_simplices(points, scipy_tri, n))
    shull_faces_time, shull_faces = timed(lambda: close_planar_simplices(points, shull_tri, n))
    if not np.array_equal(canonical(scipy_faces), canonical(shull_faces)):
        raise AssertionError("closed spherical face sets differ")

    scipy_csr_time, scipy_csr = timed(lambda: _cyclic_region_adjacency(scipy_faces, n + 1))
    shull_csr_py_time, shull_csr_py = timed(lambda: _cyclic_region_adjacency(shull_faces, n + 1))
    shull_csr_nb_time, shull_csr_nb = timed(lambda: numba_cyclic_adjacency(shull_faces, n + 1))
    scipy_offsets, scipy_adj = scipy_csr
    shull_offsets, shull_adj_py = shull_csr_py
    shull_offsets_nb, shull_adj_nb = shull_csr_nb
    if not np.array_equal(shull_offsets, shull_offsets_nb) or not np.array_equal(shull_adj_py, shull_adj_nb):
        raise AssertionError("Numba CSR does not preserve Orogen's cyclic adjacency order")
    if row_sets(scipy_offsets, scipy_adj) != row_sets(shull_offsets_nb, shull_adj_nb):
        raise AssertionError("spherical vertex-neighbour sets differ")

    # Different simplex enumeration can rotate the starting point of a ring.
    scipy_adj = canonicalize_ring_starts(scipy_offsets, scipy_adj)
    shull_adj_nb = canonicalize_ring_starts(shull_offsets_nb, shull_adj_nb)

    xyz_with_pole = np.empty((n + 1, 3), dtype=np.float32)
    xyz_with_pole[:n] = xyz.reshape(n, 3)
    xyz_with_pole[n] = (0.0, 0.0, 1.0)
    scipy_mesh = SimpleNamespace(num_cells=n + 1, adj_offset=scipy_offsets, adj_list=scipy_adj)
    shull_mesh = SimpleNamespace(num_cells=n + 1, adj_offset=shull_offsets_nb, adj_list=shull_adj_nb)
    scipy_index_time, scipy_index = timed(lambda: SphereIndex(xyz_with_pole.reshape(-1), scipy_mesh))
    shull_index_time, shull_index = timed(lambda: SphereIndex(xyz_with_pole.reshape(-1), shull_mesh))
    if not np.array_equal(scipy_index.table, shull_index.table):
        raise AssertionError("canonical SphereIndex lookup tables differ")

    source = lambda cell: cell % max(1, n // 100) == 0
    bfs_time, scipy_distance = timed(lambda: hop_distance(scipy_mesh, source))
    _, shull_distance = timed(lambda: hop_distance(shull_mesh, source))
    if not np.array_equal(scipy_distance, shull_distance):
        raise AssertionError("Numba BFS distances differ between triangulators")

    scipy_total = scipy_tri_time + scipy_faces_time + scipy_csr_time + scipy_index_time
    shull_python_total = shull_tri_time + shull_faces_time + shull_csr_py_time + shull_index_time
    shull_numba_total = shull_tri_time + shull_faces_time + shull_csr_nb_time + shull_index_time
    mesh_mib = (shull_faces.nbytes + shull_offsets_nb.nbytes + shull_adj_nb.nbytes + shull_index.table.nbytes) / (1024 * 1024)
    print(
        f"cells={n + 1:>7} "
        f"tri_scipy={scipy_tri_time:.4f}s tri_shull={shull_tri_time:.4f}s "
        f"tri_speedup={scipy_tri_time / shull_tri_time:.2f}x "
        f"csr_python={shull_csr_py_time:.4f}s csr_numba={shull_csr_nb_time:.4f}s "
        f"csr_speedup={shull_csr_py_time / shull_csr_nb_time:.2f}x "
        f"index={shull_index_time:.4f}s bfs_numba={bfs_time:.4f}s "
        f"total_scipy_python={scipy_total:.4f}s "
        f"total_shull_python={shull_python_total:.4f}s "
        f"total_shull_numba={shull_numba_total:.4f}s "
        f"pipeline_speedup={scipy_total / shull_numba_total:.2f}x "
        f"same_topology=True same_index_table=True mesh_arrays={mesh_mib:.2f}MiB"
    )
