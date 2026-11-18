import json
import math
import platform
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

import orogen.v2.core.mesh_bfs as mesh_bfs_module
import orogen.v2.tectonics.crust as crust_module
from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE
from orogen.v2.core.fast_noise import FastNoise
from orogen.v2.core.mesh_bfs import hop_distance, pieces_of
from orogen.v2.core.min_heap import spread_km
from orogen.v2.core.sphere_geometry import bearing_to_many, destination_many
from orogen.v2.tectonics import far_field
from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType

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
vectors=np.empty((N_GEOM,3),dtype=np.float64)
for i in range(N_GEOM):
    z = 1 - 2 * (i + 0.5) / N_GEOM
    a = i * 2.399963229728653
    r = math.sqrt(1 - z * z)
    vectors[i]=(r*math.cos(a),r*math.sin(a),z)
xyz=np.empty(N_CELLS*3,dtype=np.float32)
for i in range(N_CELLS):
    z = 1 - 2 * (i + 0.5) / N_CELLS
    a = i * 2.399963229728653
    r = math.sqrt(1 - z * z)
    xyz[3*i:3*i+3]=(r*math.cos(a),r*math.sin(a),z)
offsets=np.arange(N_CELLS+1,dtype=np.int32)*6
adj=np.empty(N_CELLS*6,dtype=np.int32)
for i in range(N_CELLS):
    adj[i*6:i*6+3]=[(i+k+1)%N_CELLS for k in range(3)]
    adj[i*6+3:i*6+6]=[(i-k-1+N_CELLS)%N_CELLS for k in range(3)]
class Mesh:
    pass


mesh = Mesh()
mesh.num_cells = N_CELLS
mesh.xyz = xyz
mesh.adj_offset = offsets
mesh.adj_list = adj
noise=FastNoise(9876)
sink=0
def bench_noise():
    global sink
    s=0.0
    s=float(np.sum(noise.fbm_many(points,5,2/3)))
    sink=s
def bench_geometry():
    global sink
    targets=destination_many(vectors,.71,.013)
    angles=bearing_to_many(vectors,targets)
    sink=float(np.sum(angles+np.sum(vectors*targets,axis=1)))
def bench_bfs():
    global sink
    d=hop_distance(mesh,lambda c:c==0,N_CELLS)
    sink=int(d[-1])
def bench_dijkstra():
    global sink
    d = np.full(N_CELLS, np.inf, dtype=np.float64)
    d[0] = 0.0
    spread_km(mesh,d,max_km=20000)
    sink=float(d[-1])
copy_source=CrustRaster(N_CELLS)
copy_cells=np.arange(0,N_CELLS,2,dtype=np.int32)
copy_source.create_many(copy_cells,CrustType["CONTINENT"],-100,copy_cells+1)


def bench_crust():
    global sink
    c=CrustRaster(N_CELLS)
    cells=np.arange(0,N_CELLS,2,dtype=np.int32)
    c.create_many(cells,CrustType["CONTINENT"],-100,cells+1)
    snapshots=c.snapshot_many(cells)
    sink=sum(len(snapshot) for snapshot in snapshots)


def bench_crust_copy():
    global sink
    target=CrustRaster(N_CELLS)
    target.copy_many(copy_cells,copy_source,copy_cells)
    sink=target.count




def bench_crust_scalar_acceleration():
    """A/B the existing per-cell API with Numba enabled and Python fallback."""
    def run(use_numba):
        previous = crust_module.NUMBA_AVAILABLE
        crust_module.NUMBA_AVAILABLE = bool(use_numba and NUMBA_AVAILABLE)

        def once():
            global sink
            raster = CrustRaster(N_CELLS)
            total = 0
            for cell in range(0, N_CELLS, 2):
                raster.create(cell, CrustType["CONTINENT"], -100, cell + 1)
                total += len(raster.snapshot(cell))
            sink = total

        try:
            return median_ms(once)
        finally:
            crust_module.NUMBA_AVAILABLE = previous

    numba_ms = run(True)
    python_ms = run(False)
    return {
        "workload": "6,000 scalar create+snapshot calls",
        "numba_ms": round(numba_ms, 3),
        "python_fallback_ms": round(python_ms, 3),
        "speedup": round(python_ms / numba_ms, 2) if numba_ms else None,
    }



def bench_crust_batch_stages():
    """Isolate raster construction, create_many and snapshot_many costs."""
    cells = np.arange(0, N_CELLS, 2, dtype=np.int32)
    types = CrustType["CONTINENT"]

    def construct():
        global sink
        raster = CrustRaster(N_CELLS)
        sink = raster.numCells

    constructor_ms = median_ms(construct)

    create_raster = CrustRaster(N_CELLS)

    def run_create(use_numba):
        previous = crust_module.NUMBA_AVAILABLE
        crust_module.NUMBA_AVAILABLE = bool(use_numba and NUMBA_AVAILABLE)

        def once():
            global sink
            create_raster.create_many(cells, types, -100, cells + 1)
            sink = create_raster.count

        try:
            return median_ms(once)
        finally:
            crust_module.NUMBA_AVAILABLE = previous

    snapshot_raster = CrustRaster(N_CELLS)
    snapshot_raster.create_many(cells, types, -100, cells + 1)

    def run_snapshot(use_numba):
        previous = crust_module.NUMBA_AVAILABLE
        crust_module.NUMBA_AVAILABLE = bool(use_numba and NUMBA_AVAILABLE)

        def once():
            global sink
            result = snapshot_raster.snapshot_many(cells)
            sink = float(result[-1, -1]) if result.size else 0.0

        try:
            return median_ms(once)
        finally:
            crust_module.NUMBA_AVAILABLE = previous

    create_numba_ms = run_create(True)
    create_python_ms = run_create(False)
    snapshot_numba_ms = run_snapshot(True)
    snapshot_python_ms = run_snapshot(False)
    return {
        "workload": "6,000 cells, 55 crust fields; constructor / batch create / batch snapshot",
        "constructor_ms": round(constructor_ms, 3),
        "create_many_numba_ms": round(create_numba_ms, 3),
        "create_many_python_ms": round(create_python_ms, 3),
        "create_many_speedup": round(create_python_ms / create_numba_ms, 2) if create_numba_ms else None,
        "snapshot_many_numba_ms": round(snapshot_numba_ms, 3),
        "snapshot_many_python_ms": round(snapshot_python_ms, 3),
        "snapshot_many_speedup": round(snapshot_python_ms / snapshot_numba_ms, 2) if snapshot_numba_ms else None,
    }

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
crust_scalar_comparison = bench_crust_scalar_acceleration()
crust_batch_stages = bench_crust_batch_stages()

timings={
"noise_fbm_ms":median_ms(bench_noise),
"spherical_geometry_ms":median_ms(bench_geometry),
"mesh_bfs_ms":median_ms(bench_bfs),
"mesh_components_ms":component_comparison["numba_ms"],
"mesh_dijkstra_ms":median_ms(bench_dijkstra),
"crust_create_snapshot_ms":median_ms(bench_crust),
"crust_copy_ms":median_ms(bench_crust_copy),
}
print(json.dumps({"runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"numba_available":NUMBA_AVAILABLE,"platform":platform.platform()},"timings":timings,"accelerator_comparisons":{"far_field_diffusion":far_field_comparison,"mesh_components":component_comparison,"crust_scalar":crust_scalar_comparison,"crust_batch_stages":crust_batch_stages}}))
