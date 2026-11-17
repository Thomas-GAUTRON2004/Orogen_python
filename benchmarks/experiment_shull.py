"""A/B Delaunay benchmark: Rust shull vs SciPy/Qhull."""
import time
import numpy as np
from scipy.spatial import Delaunay
import shull
from orogen.v2.core.sphere_mesh import generate_fibonacci_sphere, stereographic_projection

def timed(fn, repeats=5):
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

for n in (10_000, 40_000, 80_000):
    xyz = generate_fibonacci_sphere(n, 0.0, lambda: 0.5)
    points = np.ascontiguousarray(stereographic_projection(xyz, n), dtype=np.float64)
    ts, scipy_tri = timed(lambda: Delaunay(points).simplices)
    tr, rust_tri = timed(lambda: shull.Delaunay(points).simplices)
    scipy_tri = np.asarray(scipy_tri, dtype=np.int32)
    rust_tri = np.asarray(rust_tri, dtype=np.int32)
    if scipy_tri.shape != rust_tri.shape or not np.array_equal(canonical(scipy_tri), canonical(rust_tri)):
        raise AssertionError(f"Qhull and shull simplex topology differs at n={n}")
    if rust_tri.ndim != 2 or rust_tri.shape[1] != 3:
        raise AssertionError("shull must return triangular 2D simplices")
    if rust_tri.min() < 0 or rust_tri.max() >= n:
        raise AssertionError("shull returned an out-of-range vertex index")
    if len(np.unique(rust_tri)) != n:
        raise AssertionError("not every input vertex is represented")
    print(f"n={n:>7} scipy={ts:.4f}s shull={tr:.4f}s speedup={ts/tr:.2f}x triangles={len(rust_tri)} same_topology=True")
