"""Compare the production shull backend with SciPy on Orogen's mesh pipeline.

Install the optional comparison dependencies with:
    python -m pip install -e ".[benchmark]"
"""
from __future__ import annotations

import platform
import statistics
import time
from types import SimpleNamespace

import numpy as np
import scipy
import shull
from scipy.spatial import Delaunay

from orogen.v2.core.mesh_bfs import hop_distance
from orogen.v2.core.seed import make_rng
from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.core.sphere_index import SphereIndex
from orogen.v2.core.sphere_mesh import (
    _close_simplices,
    _cyclic_region_adjacency,
    generate_fibonacci_sphere,
    stereographic_projection,
)


def timed(fn, repeats: int = 3):
    """Warm a callable and return its median wall time and last result."""
    fn()
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples)), result


def canonical(simplices: np.ndarray) -> np.ndarray:
    rows = np.sort(np.asarray(simplices, dtype=np.int64), axis=1)
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]


def canonicalize_ring_starts(offsets: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Rotate each cyclic row to its smallest neighbour without changing direction."""
    result = np.asarray(adjacency, dtype=np.int32).copy()
    for cell in range(len(offsets) - 1):
        start, end = int(offsets[cell]), int(offsets[cell + 1])
        row = result[start:end].copy()
        pivot = int(np.argmin(row))
        result[start:end] = np.concatenate((row[pivot:], row[:pivot]))
    return result


def measure_pipeline(points, xyz, num_points, triangulate):
    tri_time, simplices = timed(lambda: triangulate(points))
    closure_time, faces = timed(
        lambda: _close_simplices(points, simplices, num_points)
    )
    csr_time, (offsets, adjacency) = timed(
        lambda: _cyclic_region_adjacency(faces, num_points + 1)
    )
    adjacency = canonicalize_ring_starts(offsets, adjacency)
    mesh = SimpleNamespace(
        num_cells=num_points + 1,
        adj_offset=offsets,
        adj_list=adjacency,
    )
    index_time, index = timed(lambda: SphereIndex(xyz, mesh))
    return {
        "simplices": simplices,
        "faces": faces,
        "offsets": offsets,
        "adjacency": adjacency,
        "index": index,
        "triangulation_s": tri_time,
        "pole_closure_s": closure_time,
        "csr_s": csr_time,
        "sphere_index_s": index_time,
        "total_s": tri_time + closure_time + csr_time + index_time,
        "array_mib": (
            faces.nbytes + offsets.nbytes + adjacency.nbytes + index.table.nbytes
        ) / (1024 * 1024),
        "mesh": mesh,
    }


def main():
    print(
        f"python={platform.python_version()} numpy={np.__version__} "
        f"scipy={scipy.__version__} shull={shull.__version__} "
        f"platform={platform.platform()}"
    )
    for num_points in (10_000, 40_000, 80_000):
        xyz = generate_fibonacci_sphere(
            num_points, 0.35, make_rng(12345 + 4242)
        )
        points = np.ascontiguousarray(
            stereographic_projection(xyz, num_points), dtype=np.float64
        )
        xyz_with_pole = np.empty((num_points + 1, 3), dtype=np.float32)
        xyz_with_pole[:num_points] = xyz.reshape(num_points, 3)
        xyz_with_pole[num_points] = (0.0, 0.0, 1.0)
        xyz_flat = xyz_with_pole.reshape(-1)

        scipy_result = measure_pipeline(
            points, xyz_flat, num_points,
            lambda values: Delaunay(values).simplices,
        )
        shull_result = measure_pipeline(
            points, xyz_flat, num_points,
            lambda values: shull.Delaunay(values).simplices,
        )

        if not np.array_equal(
            canonical(scipy_result["simplices"]),
            canonical(shull_result["simplices"]),
        ):
            raise AssertionError(f"simplex topology differs at n={num_points}")
        if not np.array_equal(
            canonical(scipy_result["faces"]),
            canonical(shull_result["faces"]),
        ):
            raise AssertionError(f"closed spherical faces differ at n={num_points}")
        if not np.array_equal(
            scipy_result["offsets"], shull_result["offsets"]
        ) or not np.array_equal(
            scipy_result["adjacency"], shull_result["adjacency"]
        ):
            raise AssertionError(f"cyclic adjacency differs at n={num_points}")
        if not np.array_equal(
            scipy_result["index"].table, shull_result["index"].table
        ):
            raise AssertionError(f"SphereIndex table differs at n={num_points}")

        source = lambda cell: cell % max(1, num_points // 100) == 0
        scipy_distances = hop_distance(scipy_result["mesh"], source)
        shull_distances = hop_distance(shull_result["mesh"], source)
        if not np.array_equal(scipy_distances, shull_distances):
            raise AssertionError(f"BFS distances differ at n={num_points}")

        full_build_time, production_mesh = timed(
            lambda: build_sim_mesh(num_points, 12345)
        )
        if production_mesh.num_cells != num_points + 1:
            raise AssertionError("production build_sim_mesh returned an invalid cell count")

        print(
            f"cells={num_points + 1:>7} "
            f"tri_scipy={scipy_result['triangulation_s']:.4f}s "
            f"tri_shull={shull_result['triangulation_s']:.4f}s "
            f"tri_speedup={scipy_result['triangulation_s'] / shull_result['triangulation_s']:.2f}x "
            f"pipeline_scipy={scipy_result['total_s']:.4f}s "
            f"pipeline_shull={shull_result['total_s']:.4f}s "
            f"pipeline_speedup={scipy_result['total_s'] / shull_result['total_s']:.2f}x "
            f"build_sim_mesh_shull={full_build_time:.4f}s "
            f"arrays={shull_result['array_mib']:.2f}MiB "
            f"same_topology=True same_csr=True same_index=True same_bfs=True"
        )


if __name__ == "__main__":
    main()
