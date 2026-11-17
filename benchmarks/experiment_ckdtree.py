"""A/B exact spherical nearest-cell queries: SphereIndex vs cKDTree.
Run: python benchmarks/experiment_ckdtree.py
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from types import SimpleNamespace
from orogen.v2.core.sphere_mesh import build_sphere
from orogen.v2.core.sphere_index import SphereIndex

def timed(fn, repeats=3):
    fn()
    samples=[]
    for _ in range(repeats):
        t=time.perf_counter(); result=fn(); samples.append(time.perf_counter()-t)
    return float(np.median(samples)), result

for n in (10_000, 40_000, 80_000):
    mesh, xyz=build_sphere(n, 0.0, lambda: 0.5)
    cells=np.asarray(xyz, dtype=np.float64).reshape(-1,3)
    rng=np.random.default_rng(1234)
    q=rng.normal(size=(min(20_000,n),3)); q/=np.linalg.norm(q,axis=1)[:,None]
    index=SphereIndex(xyz, mesh)
    tree=cKDTree(cells)
    ti, got=timed(lambda: np.fromiter((index.nearest(*p) for p in q),dtype=np.int32,count=len(q)))
    tk, expected=timed(lambda: tree.query(q,k=1,workers=-1)[1])
    assert np.array_equal(got, expected), "SphereIndex and exact cKDTree nearest-cell results differ"
    print(f"cells={n:>7} queries={len(q):>6} SphereIndex={ti:.4f}s cKDTree={tk:.4f}s speedup={ti/tk:.2f}x")
