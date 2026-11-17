"""A/B Delaunay benchmark: Rust triangular vs SciPy/Qhull used by Orogen.
Run: pip install triangular && python benchmarks/experiment_triangular.py
"""
import time
import numpy as np
from scipy.spatial import Delaunay
import triangular
from orogen.v2.core.sphere_mesh import generate_fibonacci_sphere, stereographic_projection

def timed(fn, repeats=3):
    fn()
    samples=[]
    for _ in range(repeats):
        t=time.perf_counter(); result=fn(); samples.append(time.perf_counter()-t)
    return float(np.median(samples)), result

for n in (10_000, 40_000, 80_000):
    xyz=generate_fibonacci_sphere(n, 0.0, lambda: 0.5)
    points=np.ascontiguousarray(stereographic_projection(xyz,n), dtype=np.float64)
    ts, ref=timed(lambda: Delaunay(points).simplices)
    tr, rust=timed(lambda: triangular.delaunay(points))
    rust=np.asarray(rust)
    # Delaunay may choose different diagonals for cocircular inputs; validate mesh invariants.
    assert rust.ndim==2 and rust.shape[1]==3
    assert rust.min()>=0 and rust.max()<n
    assert len(np.unique(rust))==n
    print(f"n={n:>7} scipy={ts:.4f}s triangular={tr:.4f}s speedup={ts/tr:.2f}x triangles={len(rust)}")
