import numpy as np

from orogen.v2.core.sim_mesh import km_to_hops, mesh_from_arrays


def test_sim_mesh_rebuilds_index_from_arrays():
    # Four unit directions with a complete neighbour graph.
    xyz = np.array([
        1, 0, 0,
        -1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ], dtype=np.float32)
    offsets = np.array([0, 3, 6, 9, 12], dtype=np.int32)
    adjacency = np.array([
        1, 2, 3,
        0, 2, 3,
        0, 1, 3,
        0, 1, 2,
    ], dtype=np.int32)
    mesh = mesh_from_arrays(
        num_cells=4, xyz=xyz, adj_offset=offsets, adj_list=adjacency
    )
    assert mesh.num_cells == 4
    assert mesh.index.nearest(0.99, 0.01, 0.0) == 0
    assert mesh.index.nearest(-1.0, 0.0, 0.0) == 1


def test_km_to_hops_rounds_like_javascript_math_round():
    mesh = type("Mesh", (), {"edge_km": 10.0})()
    assert km_to_hops(mesh, 14.9) == 1
    assert km_to_hops(mesh, 15.0) == 2
    assert km_to_hops(mesh, 1.0, minimum=3) == 3
