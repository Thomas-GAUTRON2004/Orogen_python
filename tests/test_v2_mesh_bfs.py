from types import SimpleNamespace

import numpy as np

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
