"""A/B CSR breadth-first traversal: Orogen hop_distance vs SciPy csgraph.
Run: python benchmarks/experiment_csgraph.py
"""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
from types import SimpleNamespace
from orogen.v2.core.mesh_bfs import hop_distance

def timed(fn, repeats=5):
    fn()
    samples=[]
    for _ in range(repeats):
        t=time.perf_counter(); result=fn(); samples.append(time.perf_counter()-t)
    return float(np.median(samples)), result

for n in (12_000, 40_000, 80_000):
    # deterministic undirected degree-6 ring graph, same CSR for both paths
    neighbours=np.column_stack([(np.arange(n)+d)%n for d in (-3,-2,-1,1,2,3)]).astype(np.int32)
    adjacency=np.sort(neighbours,axis=1).ravel()
    offsets=np.arange(0,6*n+1,6,dtype=np.int32)
    mesh=SimpleNamespace(num_cells=n,adj_offset=offsets,adj_list=adjacency)
    graph=csr_matrix((np.ones(len(adjacency),dtype=np.uint8),adjacency,offsets),shape=(n,n))
    sources=np.zeros(n,dtype=bool); sources[0]=True
    def scipy_bfs():
        order,pred=breadth_first_order(graph,0,directed=False,return_predecessors=True)
        dist=np.full(n,-1,dtype=np.int32); dist[0]=0
        for v in order[1:]: dist[v]=dist[pred[v]]+1
        return dist
    tn, expected=timed(lambda: hop_distance(mesh,lambda c:c==0))
    ts, actual=timed(scipy_bfs)
    assert np.array_equal(expected.astype(np.int32),actual)
    print(f"cells={n:>7} Orogen={tn:.5f}s csgraph={ts:.5f}s speedup={tn/ts:.2f}x")
