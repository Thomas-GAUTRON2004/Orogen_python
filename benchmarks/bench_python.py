import json, math, platform, statistics, sys, time
from types import SimpleNamespace
import numpy as np
from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE
from orogen.v2.core.fast_noise import FastNoise
from orogen.v2.core.sphere_geometry import destination, bearing_to, dot
import orogen.v2.core.mesh_bfs as mesh_bfs_module
from orogen.v2.core.mesh_bfs import hop_distance, pieces_of
from orogen.v2.core.min_heap import spread_km
from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType
from orogen.v2.tectonics import far_field

N_NOISE, N_GEOM, N_CELLS, ROUNDS = 40000, 80000, 12000, 5

def median_ms(fn):
    fn()
    samples=[]
    for _ in range(ROUNDS):
        start=time.perf_counter()
        fn()
        samples.append((time.perf_counter()-start)*1000)
    return statistics.median(samples)

points=np.asarray([(math.sin(i*.013)*17.0, math.cos(i*.007)*11.0, math.sin(i*.003+.4)*23.0) for i in range(N_NOISE)], dtype=np.float64)
vectors=[]
for i in range(N_GEOM):
    z=1-2*(i+.5)/N_GEOM; a=i*2.399963229728653; r=math.sqrt(1-z*z)
    vectors.append((r*math.cos(a),r*math.sin(a),z))
xyz=np.empty(N_CELLS*3,dtype=np.float32)
for i in range(N_CELLS):
    z=1-2*(i+.5)/N_CELLS; a=i*2.399963229728653; r=math.sqrt(1-z*z)
    xyz[3*i:3*i+3]=(r*math.cos(a),r*math.sin(a),z)
offsets=np.arange(N_CELLS+1,dtype=np.int32)*6
adj=np.empty(N_CELLS*6,dtype=np.int32)
for i in range(N_CELLS):
    adj[i*6:i*6+3]=[(i+k+1)%N_CELLS for k in range(3)]
    adj[i*6+3:i*6+6]=[(i-k-1+N_CELLS)%N_CELLS for k in range(3)]
class Mesh: pass
mesh=Mesh(); mesh.num_cells=N_CELLS; mesh.xyz=xyz; mesh.adj_offset=offsets; mesh.adj_list=adj
noise=FastNoise(9876)
sink=0
def bench_noise():
    global sink
    s=0.0
    s=float(np.sum(noise.fbm_many(points,5,2/3)))
    sink=s
def bench_geometry():
    global sink
    s=0.0
    for p in vectors:
        q=destination(p,.71,.013)
        s+=bearing_to(p,q)+dot(p,q)
    sink=s
def bench_bfs():
    global sink
    d=hop_distance(mesh,lambda c:c==0,N_CELLS)
    sink=int(d[-1])
def bench_dijkstra():
    global sink
    d=np.full(N_CELLS,np.inf,dtype=np.float64); d[0]=0.0
    spread_km(mesh,d,max_km=20000)
    sink=float(d[-1])
def bench_crust():
    global sink
    c=CrustRaster(N_CELLS); s=0
    for i in range(0,N_CELLS,2):
        c.create(i,CrustType["CONTINENT"],-100,i+1)
        s+=len(c.snapshot(i))
    sink=s


def bench_mesh_components():
    """Compare selected-cell connected components with and without Numba."""
    selected = []
    block_size, gap = 64, 4
    for start in range(0, N_CELLS, block_size + gap):
        selected.extend(range(start, min(start + block_size, N_CELLS)))
    selected = np.asarray(selected, dtype=np.int32)

    def run(use_numba):
        previous = mesh_bfs_module.NUMBA_AVAILABLE
        mesh_bfs_module.NUMBA_AVAILABLE = bool(use_numba and NUMBA_AVAILABLE)

        def once():
            global sink
            pieces = pieces_of(mesh, selected)
            sink = sum(len(piece) for piece in pieces)

        try:
            return median_ms(once)
        finally:
            mesh_bfs_module.NUMBA_AVAILABLE = previous

    python_ms = run(False)
    numba_ms = run(True) if NUMBA_AVAILABLE else None
    return {
        "workload": "connected components, 12,000-cell degree-6 ring, 64-cell blocks",
        "numba_ms": round(numba_ms, 3) if numba_ms is not None else None,
        "python_fallback_ms": round(python_ms, 3),
        "speedup": round(python_ms / numba_ms, 2) if numba_ms else None,
    }


def bench_far_field_diffusion():
    """Compare the real far-field spread_push path with and without its Numba kernel."""
    angle = np.arange(N_CELLS, dtype=np.float64) * (2.0 * math.pi / N_CELLS)
    far_xyz = np.empty(N_CELLS * 3, dtype=np.float64)
    far_xyz[0::3] = np.cos(angle)
    far_xyz[1::3] = np.sin(angle)
    far_xyz[2::3] = 0.0

    far_offsets = np.arange(N_CELLS + 1, dtype=np.int32) * 6
    far_adjacency = np.empty(N_CELLS * 6, dtype=np.int32)
    for cell in range(N_CELLS):
        far_adjacency[cell * 6:cell * 6 + 3] = [
            (cell + step) % N_CELLS for step in (1, 2, 3)
        ]
        far_adjacency[cell * 6 + 3:cell * 6 + 6] = [
            (cell - step) % N_CELLS for step in (1, 2, 3)
        ]

    far_mesh = SimpleNamespace(
        num_cells=N_CELLS,
        xyz=far_xyz,
        adj_offset=far_offsets,
        adj_list=far_adjacency,
    )
    crust = CrustRaster(N_CELLS)
    crust.mask.fill(1)
    crust.count = N_CELLS
    crust.type.fill(CrustType["CONTINENT"])
    crust.type[::211] = CrustType["CRATON"]
    crust.flags[::97] |= CrustFlag["SUTURE"]

    config = {
        "farReachKm": 21000.0,
        "farFrontRefKm": 1000.0,
        "farMaxKm": 21000.0,
        "farCratonCost": 0.3,
        "farWeakCost": 1.6,
        "farCratonTake": 0.15,
        "farWeakTake": 1.6,
        "farFoldMyr": 100.0,
    }
    world = SimpleNamespace(mesh=far_mesh, config=config, time=10.0)
    plate = SimpleNamespace(crust=crust)
    sources = {0: 100.0, N_CELLS // 2: 80.0}
    state_fields = (
        "farWork", "farDist", "farFold", "farAt",
        "farDx", "farDy", "farDz",
    )

    def run(use_numba):
        previous = far_field.NUMBA_AVAILABLE
        far_field.NUMBA_AVAILABLE = use_numba

        def once():
            for name in state_fields:
                getattr(crust, name).fill(0)
            far_field.spread_push(world, plate, sources, 1000.0, world.time)

        try:
            return median_ms(once)
        finally:
            far_field.NUMBA_AVAILABLE = previous

    numba_ms = run(True)
    python_ms = run(False)
    return {
        "workload": "spread_push, 12,000 cells, degree-6 equatorial ring, two sources",
        "numba_ms": round(numba_ms, 3),
        "python_fallback_ms": round(python_ms, 3),
        "speedup": round(python_ms / numba_ms, 2) if numba_ms else None,
    }


far_field_comparison = bench_far_field_diffusion()
component_comparison = bench_mesh_components()

timings={
"noise_fbm_ms":median_ms(bench_noise),
"spherical_geometry_ms":median_ms(bench_geometry),
"mesh_bfs_ms":median_ms(bench_bfs),
"mesh_dijkstra_ms":median_ms(bench_dijkstra),
"crust_create_snapshot_ms":median_ms(bench_crust),
}
print(json.dumps({"runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"numba_available":NUMBA_AVAILABLE,"platform":platform.platform()},"timings":timings,"accelerator_comparisons":{"far_field_diffusion":far_field_comparison,"mesh_components":component_comparison}}))
