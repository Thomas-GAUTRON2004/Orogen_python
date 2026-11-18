"""Tests for cached plate-contact geometry."""
from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.core import build_sim_mesh
from orogen.v2.tectonics.contact import contact_at


def _world(mesh, owner):
    return SimpleNamespace(mesh=mesh, owner=np.asarray(owner, dtype=np.int16), occupancyStamp=0)


def test_contact_returns_interior_sentinel_without_overwriting_output():
    mesh = build_sim_mesh(24, seed=3)
    owner = np.zeros(mesh.num_cells, dtype=np.int16)
    out = np.array([7.0, 8.0, 9.0, 10.0])
    assert contact_at(_world(mesh, owner), 0, out) == -1
    np.testing.assert_array_equal(out, [7.0, 8.0, 9.0, 10.0])


def test_contact_normal_and_across_cell_are_cached_until_stamp_changes():
    mesh = build_sim_mesh(24, seed=9)
    cell = 0
    neighbours = mesh.adj_list[mesh.adj_offset[cell]:mesh.adj_offset[cell + 1]]
    owner = np.zeros(mesh.num_cells, dtype=np.int16)
    owner[neighbours] = 1
    world = _world(mesh, owner)
    out = np.zeros(4)
    across = contact_at(world, cell, out)
    assert across in neighbours
    assert out[3] == len(neighbours)
    assert np.linalg.norm(out[:3]) == pytest.approx(1.0)
    saved = out.copy()
    world.owner[neighbours] = 0
    np.testing.assert_equal(contact_at(world, cell, out), across)
    np.testing.assert_array_equal(out, saved)
    world.occupancyStamp += 1
    assert contact_at(world, cell, out) == -1


def test_contact_uses_first_foreign_direction_for_opposing_sliver():
    class TinyMesh:
        num_cells = 4
        xyz = np.array([0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0], dtype=np.float32)
        adj_offset = np.array([0, 3, 3, 3, 3], dtype=np.int32)
        adj_list = np.array([1, 2, 3], dtype=np.int32)

    world = _world(TinyMesh(), [0, 1, 1, -1])
    out = np.zeros(4)
    assert contact_at(world, 0, out) == 1
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0, 2.0])


def test_contact_validates_output_and_cell_index():
    mesh = build_sim_mesh(24, seed=1)
    world = _world(mesh, np.zeros(mesh.num_cells, dtype=np.int16))
    with pytest.raises(ValueError, match="four writable"):
        contact_at(world, 0, [0.0, 0.0, 0.0])
    with pytest.raises(IndexError, match="out of range"):
        contact_at(world, mesh.num_cells, np.zeros(4))
    with pytest.raises(IndexError, match="out of range"):
        contact_at(world, -1, np.zeros(4))


def test_contact_handles_stamp_wraparound_and_cached_interior_result():
    mesh = build_sim_mesh(24, seed=5)
    world = _world(mesh, np.zeros(mesh.num_cells, dtype=np.int16))
    world.occupancyStamp = 0xFFFFFFFF
    out = np.zeros(4)
    assert contact_at(world, 0, out) == -1
    assert int(world._contact["done"][0]) == 1
    out[:] = 3.0
    assert contact_at(world, 0, out) == -1
    np.testing.assert_array_equal(out, [3.0, 3.0, 3.0, 3.0])

def test_contact_zero_normal_skips_same_owner_neighbour_before_foreign_fallback():
    class TinyMesh:
        num_cells = 4
        xyz = np.array([
            0., 0., 0.,
            0., 1., 0.,
            1., 0., 0.,
            -1., 0., 0.,
        ], dtype=np.float32)
        adj_offset = np.array([0, 3, 3, 3, 3], dtype=np.int32)
        adj_list = np.array([1, 2, 3], dtype=np.int32)

    world = _world(TinyMesh(), [0, 0, 1, 1])
    out = np.zeros(4)
    assert contact_at(world, 0, out) == 2
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0, 2.0])

