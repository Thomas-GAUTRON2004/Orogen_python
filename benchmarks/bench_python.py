import json, math, platform, statistics, sys, time
import numpy as np
from orogen.v2.core.fast_noise import FastNoise
from orogen.v2.core.sphere_geometry import destination, bearing_to, dot
from orogen.v2.core.mesh_bfs import hop_distance
from orogen.v2.core.min_heap import spread_km
from orogen.v2.tectonics.crust import CrustRaster, CrustType

N_NOISE, N_GEOM, N_CELLS, ROUNDS = 40000, 80000, 12000, 5

def median_ms(fn):
    fn()
    samples=[]
    for _ in range(ROUNDS):
        start=time.perf_counter()
        fn()
        samples.append((time.perf_counter()-start)*1000)
    return statistics.median(samples)

points=[(math.sin(i*.013)*17.0, math.cos(i*.007)*11.0, math.sin(i*.003+.4)*23.0) for i in range(N_NOISE)]
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
    for p in points: s+=noise.fbm(*p,5,2/3)
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
timings={
"noise_fbm_ms":median_ms(bench_noise),
"spherical_geometry_ms":median_ms(bench_geometry),
"mesh_bfs_ms":median_ms(bench_bfs),
"mesh_dijkstra_ms":median_ms(bench_dijkstra),
"crust_create_snapshot_ms":median_ms(bench_crust),
}
print(json.dumps({"runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"platform":platform.platform()},"timings":timings}))
