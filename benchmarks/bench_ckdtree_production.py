"""Production-shaped SphereIndex vs cKDTree benchmark.

Uses the real build_sim_mesh path (jitter=0.35, deterministic seed), then
compares index construction, scalar nearest calls, batched nearest calls, and
the complete mesh_locate.locate path. cKDTree is always queried with workers=1
for the scalar comparison; the batch result is reported separately.
"""
# ruff: noqa: I001 - preserve the benchmark's grouped imports.

from __future__ import annotations

import time
from statistics import median
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from orogen.v2.core.mesh_locate import locate
from orogen.v2.core.seed import make_rng
from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.core.sphere_index import SphereIndex
from orogen.v2.core.sphere_mesh import build_sphere


SEED = 20261118
SIZES = (10_000, 40_000, 80_000)
N_QUERIES = 5_000
REPEATS = 3


def measure(fn, repeats: int = REPEATS):
    fn()  # warmup
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return median(samples), result


class KDTreeIndex:
    """Drop-in nearest() adapter for the production locate() function."""

    def __init__(self, tree: cKDTree):
        self.tree = tree

    def nearest(self, x: float, y: float, z: float) -> int:
        return int(self.tree.query((x, y, z), k=1, workers=1)[1])


def build_mesh_with_tree(requested_cells: int, seed: int):
    """Candidate full-build path: production sphere/CSR plus cKDTree."""
    sphere, xyz = build_sphere(requested_cells, 0.35, make_rng(seed + 4242))
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    points /= np.linalg.norm(points, axis=1)[:, None]
    return sphere, xyz, cKDTree(points)


def query_vectors(count: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    points = rng.normal(size=(count, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    return np.ascontiguousarray(points, dtype=np.float64)


def run_locate(view, queries):
    cells = np.empty(3, dtype=np.int32)
    weights = np.empty(3, dtype=np.float64)
    out_cells = np.empty((len(queries), 3), dtype=np.int32)
    out_weights = np.empty((len(queries), 3), dtype=np.float64)
    for i, (x, y, z) in enumerate(queries):
        locate(view, float(x), float(y), float(z), cells, weights)
        out_cells[i] = cells
        out_weights[i] = weights
    return out_cells, out_weights



def benchmark_size(requested_cells: int) -> None:
        print(f"\n=== requested_cells={requested_cells} ===", flush=True)

        build_s, mesh = measure(lambda: build_sim_mesh(requested_cells, SEED), repeats=2)
        candidate_build_s, _candidate_build = measure(
            lambda: build_mesh_with_tree(requested_cells, SEED), repeats=2
        )
        xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape(-1, 3)
        queries = query_vectors(N_QUERIES)

        # Normalize the float32 production vertices in float64. For unit query
        # directions, Euclidean nearest-neighbour order then matches maximum
        # spherical dot-product order, which is the intended geometric criterion.
        norms = np.linalg.norm(xyz, axis=1)
        unit_xyz = xyz / norms[:, None]
        q_unit = queries / np.linalg.norm(queries, axis=1)[:, None]

        sphere_index_s, _ = measure(
            lambda: SphereIndex(mesh.xyz, mesh, 2), repeats=REPEATS
        )
        tree_s, tree = measure(lambda: cKDTree(unit_xyz), repeats=REPEATS)

        prod_view = SimpleNamespace(
            xyz=mesh.xyz, adj_offset=mesh.adj_offset, adj_list=mesh.adj_list,
            index=mesh.index,
        )
        tree_view = SimpleNamespace(
            xyz=mesh.xyz, adj_offset=mesh.adj_offset, adj_list=mesh.adj_list,
            index=KDTreeIndex(tree),
        )

        sphere_scalar_s, sphere_ids = measure(
            lambda: np.fromiter(
                (mesh.index.nearest(*p) for p in queries),
                dtype=np.int32, count=len(queries),
            )
        )
        tree_scalar_s, tree_scalar_ids = measure(
            lambda: np.fromiter(
                (tree_view.index.nearest(*p) for p in queries),
                dtype=np.int32, count=len(queries),
            )
        )
        tree_batch_s, tree_batch_ids = measure(
            lambda: tree.query(q_unit, k=1, workers=1)[1]
        )
        tree_parallel_batch_s, _ = measure(
            lambda: tree.query(q_unit, k=1, workers=-1)[1]
        )

        prod_locate_s, prod_result = measure(lambda: run_locate(prod_view, queries))
        tree_locate_s, tree_result = measure(lambda: run_locate(tree_view, queries))

        exact_scalar = int(np.count_nonzero(sphere_ids == tree_scalar_ids))
        exact_batch = int(np.count_nonzero(sphere_ids == tree_batch_ids))
        locate_same_cells = int(np.count_nonzero(
            np.all(prod_result[0] == tree_result[0], axis=1)
        ))
        weight_max_abs = float(np.max(np.abs(prod_result[1] - tree_result[1])))
        weight_mean_abs = float(np.mean(np.abs(prod_result[1] - tree_result[1])))

        # Angular quality is evaluated from normalized float64 directions, avoiding
        # the negative-loss artifact caused by dot products of float32 coordinates.
        sphere_dot = np.einsum("ij,ij->i", unit_xyz[sphere_ids], q_unit)
        tree_dot = np.einsum("ij,ij->i", unit_xyz[tree_scalar_ids], q_unit)
        sphere_angle = np.degrees(np.arccos(np.clip(sphere_dot, -1.0, 1.0)))
        tree_angle = np.degrees(np.arccos(np.clip(tree_dot, -1.0, 1.0)))
        angular_excess = sphere_angle - tree_angle

        print(
            f"mesh_cells={mesh.num_cells} vertices={len(xyz)} queries={len(queries)}"
        )
        print(
            f"build_sim_mesh_SphereIndex={build_s:.6f}s "
            f"build_mesh_cKDTree={candidate_build_s:.6f}s "
            f"SphereIndex_construct={sphere_index_s:.6f}s "
            f"cKDTree_construct={tree_s:.6f}s"
        )
        print(
            f"nearest_scalar SphereIndex={sphere_scalar_s:.6f}s "
            f"cKDTree_workers1={tree_scalar_s:.6f}s "
            f"speedup={sphere_scalar_s/tree_scalar_s:.2f}x"
        )
        print(
            f"nearest_batch cKDTree_workers1={tree_batch_s:.6f}s "
            f"workers_all={tree_parallel_batch_s:.6f}s "
            f"batch_vs_scalar={tree_scalar_s/tree_batch_s:.2f}x"
        )
        print(
            f"locate_full SphereIndex={prod_locate_s:.6f}s "
            f"cKDTree_workers1={tree_locate_s:.6f}s "
            f"speedup={prod_locate_s/tree_locate_s:.2f}x"
        )
        print(
            f"nearest_exact_matches={exact_scalar}/{len(queries)} "
            f"batch_exact_matches={exact_batch}/{len(queries)} "
            f"locate_same_cell_triplets={locate_same_cells}/{len(queries)}"
        )
        print(
            f"weight_abs_diff_mean={weight_mean_abs:.9g} "
            f"weight_abs_diff_max={weight_max_abs:.9g} "
            f"angular_excess_mean={float(np.mean(angular_excess)):.9g}deg "
            f"angular_excess_max={float(np.max(angular_excess)):.9g}deg"
        )

for requested_cells in SIZES:
    benchmark_size(requested_cells)
