from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.core.mesh_bfs import (
    flood_from,
    hop_distance,
    hops_from,
    largest_piece_of,
    pieces_of,
    reaches_within,
    touches,
    within_km,
)


def mesh():
    # Undirected graph: 0—1—2, 1—3, and isolated cell 4.
    return SimpleNamespace(
        num_cells=5,
        adj_offset=np.array([0, 1, 4, 5, 6, 6], dtype=np.int32),
        adj_list=np.array([1, 0, 2, 3, 1, 1], dtype=np.int32),
        xyz=np.array(
            [
                1.0,
                0.0,
                0.0,
                np.cos(0.01),
                np.sin(0.01),
                0.0,
                np.cos(0.02),
                np.sin(0.02),
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        ),
    )


def test_hop_distance_and_source_tracking():
    m = mesh()
    assert hop_distance(m, lambda c: c == 0).tolist() == [0, 1, 2, 2, -1]
    distance, source = hops_from(m, [0, 3])
    assert distance.tolist() == [0, 1, 2, 0, -1]
    assert source.tolist() == [0, 0, 0, 3, -1]


def test_flood_pieces_and_largest_piece():
    m = mesh()
    assert flood_from(m, 0, lambda c: c != 2).tolist() == [1, 1, 0, 1, 0]
    assert pieces_of(m, [0, 1, 2, 4]) == [[0, 1, 2], [4]]
    assert largest_piece_of(m, [0, 1, 2, 4]) == [0, 1, 2]


def test_neighbour_and_reach_queries():
    m = mesh()
    assert touches(m, 0, lambda c: c == 1)
    assert reaches_within(m, 0, 2, lambda c: True, lambda c: c == 3)
    assert not reaches_within(m, 0, 1, lambda c: True, lambda c: c == 3)


def test_within_km_uses_great_circle_distance():
    reached = within_km(mesh(), lambda c: c == 0, 100.0)
    assert reached.tolist() == [1, 1, 0, 0, 0]


def test_numba_mesh_bfs_kernels_match_python_fallback(monkeypatch):
    from orogen.v2.core import mesh_bfs as mesh_bfs_module
    from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        pytest.skip("Numba acceleration is not installed")

    m = mesh()
    for sources, max_hops in (
        ([0, 3], np.inf),
        ([3, 0, 3], 1.5),
        ([], 0),
        ([0, 3], np.nan),
    ):
        monkeypatch.setattr(mesh_bfs_module, "NUMBA_AVAILABLE", False)
        expected_distance, expected_source = mesh_bfs_module.hops_from(
            m, sources, max_hops=max_hops
        )
        if not (isinstance(max_hops, (float, np.floating)) and np.isnan(max_hops)):
            expected_single = mesh_bfs_module.hop_distance(
                m, lambda cell, sources=sources: cell in sources, max_hops=max_hops
            )

        monkeypatch.setattr(mesh_bfs_module, "NUMBA_AVAILABLE", True)
        actual_distance, actual_source = mesh_bfs_module.hops_from(
            m, sources, max_hops=max_hops
        )
        np.testing.assert_array_equal(actual_distance, expected_distance)
        np.testing.assert_array_equal(actual_source, expected_source)

        if not (isinstance(max_hops, (float, np.floating)) and np.isnan(max_hops)):
            actual_single = mesh_bfs_module.hop_distance(
                m, lambda cell, sources=sources: cell in sources, max_hops=max_hops
            )
            np.testing.assert_array_equal(actual_single, expected_single)

    def can_enter(cell):
        return cell != 2

    monkeypatch.setattr(mesh_bfs_module, "NUMBA_AVAILABLE", False)
    expected = mesh_bfs_module.hops_from(m, [0], can_enter=can_enter, max_hops=2.5)
    monkeypatch.setattr(mesh_bfs_module, "NUMBA_AVAILABLE", True)
    actual = mesh_bfs_module.hops_from(m, [0], can_enter=can_enter, max_hops=2.5)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])

    monkeypatch.setattr(mesh_bfs_module, "NUMBA_AVAILABLE", False)
    expected = mesh_bfs_module.hops_from(m, [-1])
    monkeypatch.setattr(mesh_bfs_module, "NUMBA_AVAILABLE", True)
    actual = mesh_bfs_module.hops_from(m, [-1])
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_numba_mesh_radius_matches_python_and_preserves_predicate_order(monkeypatch):
    from orogen.v2.core import mesh_bfs as mesh_bfs_module
    from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        pytest.skip("Numba acceleration is not installed")

    m = mesh()
    calls = []
    counts = np.zeros(m.num_cells, dtype=np.int32)

    def stateful_source(cell):
        calls.append(cell)
        counts[cell] += 1
        return counts[cell] % 3 == 1

    monkeypatch.setattr(mesh_bfs_module, "NUMBA_AVAILABLE", False)
    expected = mesh_bfs_module.within_km(m, stateful_source, 5_000.0)
    expected_calls = calls.copy()

    calls.clear()
    counts.fill(0)
    monkeypatch.setattr(mesh_bfs_module, "NUMBA_AVAILABLE", True)
    actual = mesh_bfs_module.within_km(m, stateful_source, 5_000.0)

    np.testing.assert_array_equal(actual, expected)
    assert calls == expected_calls

    # A source set with no boundary edge still marks the source cells themselves.
    all_sources = mesh_bfs_module.within_km(m, lambda _cell: True, 5_000.0)
    np.testing.assert_array_equal(all_sources, np.ones(m.num_cells, dtype=np.uint8))
