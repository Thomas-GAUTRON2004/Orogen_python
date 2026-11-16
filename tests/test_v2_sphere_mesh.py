import numpy as np

from orogen.v2.core.seed import make_rng
from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.core.sphere_mesh import (
    build_sphere,
    generate_fibonacci_sphere,
    stereographic_projection,
)


def test_fibonacci_sphere_matches_expected_shape_and_unit_vectors():
    xyz = generate_fibonacci_sphere(32, 0.35, make_rng(4267))
    assert xyz.dtype == np.float32
    assert xyz.shape == (96,)
    lengths = np.linalg.norm(xyz.reshape(-1, 3), axis=1)
    assert np.allclose(lengths, 1.0, atol=2e-7)
    projected = stereographic_projection(xyz, 32)
    assert projected.shape == (32, 2)
    assert np.isfinite(projected).all()


def test_closed_sphere_mesh_has_spherical_euler_topology():
    mesh, xyz = build_sphere(48, 0.35, make_rng(4242))
    vertices = 49
    edges = len(mesh.adj_list) // 2
    assert mesh.num_regions == vertices
    assert xyz.shape == (3 * vertices,)
    assert mesh.num_triangles == 2 * vertices - 4
    assert vertices - edges + mesh.num_triangles == 2
    assert mesh.adj_offset.shape == (vertices + 1,)
    assert mesh.adj_offset[-1] == len(mesh.adj_list)

    for cell in range(vertices):
        neighbours = mesh.adj_list[mesh.adj_offset[cell]:mesh.adj_offset[cell + 1]]
        assert len(neighbours) >= 3
        for neighbour in neighbours:
            reverse = mesh.adj_list[
                mesh.adj_offset[neighbour]:mesh.adj_offset[neighbour + 1]
            ]
            assert cell in reverse


def test_build_sim_mesh_is_seeded_and_includes_north_pole_region():
    first = build_sim_mesh(48, 12345)
    second = build_sim_mesh(48, 12345)
    assert first.num_cells == 49
    assert np.array_equal(first.xyz, second.xyz)
    assert np.array_equal(first.adj_offset, second.adj_offset)
    assert np.array_equal(first.adj_list, second.adj_list)
    assert np.allclose(np.linalg.norm(first.xyz.reshape(-1, 3), axis=1), 1.0, atol=2e-7)
    assert np.array_equal(first.xyz[-3:], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert first.index.nearest(0.0, 0.0, 1.0) == 48
