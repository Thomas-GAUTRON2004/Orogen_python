from types import SimpleNamespace

import numpy as np

from orogen.v2.core.mesh_locate import locate


def test_locate_returns_spherical_triangle_barycentric_weights():
    mesh = SimpleNamespace(
        xyz=np.array([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            -1.0, 0.0, 0.0,
        ], dtype=np.float32),
        adj_offset=np.array([0, 3, 3, 3, 3], dtype=np.int32),
        adj_list=np.array([1, 2, 3], dtype=np.int32),
        index=SimpleNamespace(nearest=lambda x, y, z: 0),
    )
    cells, weights = locate(mesh, 1.0, 1.0, 1.0)
    assert cells.tolist() == [0, 1, 2]
    assert np.allclose(weights, [1 / 3, 1 / 3, 1 / 3])


def test_locate_falls_back_to_nearest_cell_when_no_triangle_exists():
    mesh = SimpleNamespace(
        xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        adj_offset=np.array([0, 0], dtype=np.int32),
        adj_list=np.array([], dtype=np.int32),
        index=SimpleNamespace(nearest=lambda x, y, z: 0),
    )
    cells, weights = locate(mesh, 1.0, 0.0, 0.0)
    assert cells.tolist() == [0, 0, 0]
    assert weights.tolist() == [1.0, 0.0, 0.0]
