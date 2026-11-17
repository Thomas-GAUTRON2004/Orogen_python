"""A/B spherical nearest-cell queries: SphereIndex vs exact cKDTree.
Run: python benchmarks/experiment_ckdtree.py

The current SphereIndex is a greedy graph walk and is not guaranteed to return
the globally nearest cell. We therefore report exact-match recall and angular
quality instead of treating mismatches as a benchmark crash.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from orogen.v2.core.sphere_mesh import build_sphere
from orogen.v2.core.sphere_index import SphereIndex


def timed(fn, repeats=3):
    fn()
    samples = []
    result = None
    for _ in range(repeats):
        t = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t)
    return float(np.median(samples)), result


for n in (10_000, 40_000, 80_000):
    mesh, xyz = build_sphere(n, 0.0, lambda: 0.5)
    cells = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    rng = np.random.default_rng(1234)
    q = rng.normal(size=(min(20_000, n), 3))
    q /= np.linalg.norm(q, axis=1)[:, None]

    index = SphereIndex(xyz, mesh)
    tree = cKDTree(cells)
    ti, got = timed(
        lambda: np.fromiter(
            (index.nearest(*p) for p in q), dtype=np.int32, count=len(q)
        )
    )
    tk, expected = timed(lambda: tree.query(q, k=1, workers=-1)[1])

    matches = int(np.count_nonzero(got == expected))
    got_dot = np.einsum("ij,ij->i", cells[got], q)
    exact_dot = np.einsum("ij,ij->i", cells[expected], q)
    angular_loss = np.degrees(
        np.arccos(np.clip(got_dot, -1.0, 1.0))
        - np.arccos(np.clip(exact_dot, -1.0, 1.0))
    )
    print(
        f"cells={len(cells):>7} queries={len(q):>6} "
        f"SphereIndex={ti:.4f}s cKDTree={tk:.4f}s "
        f"speedup={ti/tk:.2f}x exact_matches={matches}/{len(q)} "
        f"mean_angular_loss={angular_loss.mean():.6g}deg "
        f"max_angular_loss={angular_loss.max():.6g}deg"
    )
