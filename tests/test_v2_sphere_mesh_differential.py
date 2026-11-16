"""Differential mesh-generation tests against upstream sphere-mesh.js + Delaunator."""
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
import numpy as np
import pytest
from orogen.v2.core.seed import make_rng
from orogen.v2.core.sphere_mesh import build_sphere, generate_fibonacci_sphere, stereographic_projection
from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.core.mesh_bfs import within_km, hop_distance
from orogen.v2.core.min_heap import spread_km
from orogen.v2.core.sphere_geometry import cell_distance_km, centroid_of_cells
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.crust import CrustType

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"tests/reference/v2/sphere-mesh-runner.mjs"

@pytest.fixture(scope="module")
def original_spheres():
    node=shutil.which("node")
    if node is None: pytest.skip("Node.js is required for sphere-mesh differential tests")
    result=subprocess.run([node,str(RUNNER)],cwd=ROOT,check=False,capture_output=True,text=True)
    assert result.returncode==0,f"upstream Delaunator runner failed:\n{result.stderr}\n{result.stdout}"
    return json.loads(result.stdout)

@pytest.mark.parametrize("n,seed",[(12,99),(24,1),(48,17),(96,2026),(128,7),(256,42),(512,12345)])
def test_fibonacci_coordinates_and_projection_match_original(original_spheres,n,seed):
    expected=next(x for x in original_spheres if x["n"]==n and x["seed"]==seed)
    xyz=generate_fibonacci_sphere(n,.35,make_rng(seed+4242))
    np.testing.assert_array_equal(xyz,expected["xyz"][:3*n])
    projected=stereographic_projection(xyz,n)
    np.testing.assert_allclose(projected.reshape(-1),expected["projected"],rtol=0,atol=0)

@pytest.mark.parametrize("n,seed",[(12,99),(24,1),(48,17),(96,2026),(128,7),(256,42),(512,12345)])
def test_spherical_delaunay_topology_matches_original(original_spheres,n,seed):
    expected=next(x for x in original_spheres if x["n"]==n and x["seed"]==seed)
    mesh,xyz=build_sphere(n,.35,make_rng(seed+4242))
    assert mesh.num_regions==expected["numRegions"]
    assert mesh.num_triangles==expected["numTriangles"]
    np.testing.assert_array_equal(xyz,expected["xyz"])
    actual_triangles=mesh.triangles.reshape(-1,3)
    actual_keys=sorted(",".join(map(str,sorted(map(int,t)))) for t in actual_triangles)
    assert actual_keys==expected["triangleKeys"],"Qhull and Delaunator produced different Delaunay connectivity"
    assert mesh.adj_offset.tolist()==expected["adjOffset"]
    actual_neighbours=[sorted(map(int,mesh.adj_list[mesh.adj_offset[i]:mesh.adj_offset[i+1]])) for i in range(mesh.num_regions)]
    assert actual_neighbours==expected["neighbours"]

@pytest.mark.parametrize("n,seed",[(12,99),(24,1),(48,17),(96,2026),(128,7),(256,42),(512,12345)])
def test_sim_mesh_metrics_and_nearest_cells_match_original(original_spheres,n,seed):
    expected=next(x for x in original_spheres if x["n"]==n and x["seed"]==seed)
    mesh=build_sim_mesh(n,seed)
    assert mesh.num_cells==expected["numRegions"]
    assert mesh.edge_rad==pytest.approx(expected["edgeRad"],rel=0,abs=0)
    assert mesh.edge_km==pytest.approx(expected["edgeRad"]*6371,rel=2e-15)
    queries=[[1,.2,.3],[-.3,1,.4],[.2,-.4,1],[-1,-.3,-.2],[.2,.3,-1],[1,1,1]]
    assert [mesh.index.nearest(*p) for p in queries]==expected["nearest"]
    distance=np.full(mesh.num_cells,np.inf,dtype=np.float64)
    distance[0]=0
    spread_km(mesh,distance,max_km=mesh.edge_km*3.5)
    expected_distance=[np.inf if x=="Infinity" else x for x in expected["distance"]]
    np.testing.assert_allclose(distance,expected_distance,rtol=2e-12,atol=1e-8)
    assert within_km(mesh,lambda cell:cell==0,mesh.edge_km*2.25).tolist()==expected["within"]
    assert hop_distance(mesh,lambda cell:cell==0,3).tolist()==expected["hops"]
    assert cell_distance_km(mesh.xyz,0,2)==pytest.approx(expected["cellDistance"],abs=1e-10)
    np.testing.assert_allclose(centroid_of_cells(mesh.xyz,[0,2,4]),expected["centroid"],rtol=2e-12,atol=2e-12)
    plate=Plate(0,mesh.num_cells)
    for cell in [0,1,2,4,7]:
        if cell<mesh.num_cells:
            plate.crust.create(cell,CrustType["CRATON"] if cell==0 else CrustType["OCEAN"],-1,cell+1)
    plate.refresh_extent(mesh.xyz)
    assert plate.stats==expected["plate"]["stats"]
    np.testing.assert_allclose(plate.capCenter,expected["plate"]["capCenter"],rtol=2e-12,atol=2e-12)
    assert plate.capCos==pytest.approx(expected["plate"]["capCos"],abs=2e-7)
