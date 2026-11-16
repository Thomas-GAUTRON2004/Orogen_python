"""Differential mesh-generation tests against upstream sphere-mesh.js + Delaunator."""
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
import numpy as np
import pytest
from orogen.v2.core.seed import make_rng
from orogen.v2.core.sphere_mesh import build_sphere, generate_fibonacci_sphere, stereographic_projection
from orogen.v2.core.sim_mesh import build_sim_mesh

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"tests/reference/v2/sphere-mesh-runner.mjs"

@pytest.fixture(scope="module")
def original_spheres():
    node=shutil.which("node")
    if node is None: pytest.skip("Node.js is required for sphere-mesh differential tests")
    result=subprocess.run([node,str(RUNNER)],cwd=ROOT,check=False,capture_output=True,text=True)
    assert result.returncode==0,f"upstream Delaunator runner failed:\n{result.stderr}\n{result.stdout}"
    return json.loads(result.stdout)

@pytest.mark.parametrize("n,seed",[(24,1),(48,17),(96,2026)])
def test_fibonacci_coordinates_and_projection_match_original(original_spheres,n,seed):
    expected=next(x for x in original_spheres if x["n"]==n and x["seed"]==seed)
    xyz=generate_fibonacci_sphere(n,.35,make_rng(seed+4242))
    np.testing.assert_array_equal(xyz,expected["xyz"][:3*n])
    projected=stereographic_projection(xyz,n)
    np.testing.assert_allclose(projected.reshape(-1),expected["projected"],rtol=0,atol=0)

@pytest.mark.parametrize("n,seed",[(24,1),(48,17),(96,2026)])
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

@pytest.mark.parametrize("n,seed",[(24,1),(48,17),(96,2026)])
def test_sim_mesh_metrics_and_nearest_cells_match_original(original_spheres,n,seed):
    expected=next(x for x in original_spheres if x["n"]==n and x["seed"]==seed)
    mesh=build_sim_mesh(n,seed)
    assert mesh.num_cells==expected["numRegions"]
    assert mesh.edge_rad==pytest.approx(expected["edgeRad"],rel=0,abs=0)
    assert mesh.edge_km==pytest.approx(expected["edgeRad"]*6371,rel=2e-15)
    queries=[[1,.2,.3],[-.3,1,.4],[.2,-.4,1],[-1,-.3,-.2],[.2,.3,-1],[1,1,1]]
    assert [mesh.index.nearest(*p) for p in queries]==expected["nearest"]
