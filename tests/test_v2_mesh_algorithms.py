import math

import numpy as np

from orogen.v2.core.min_heap import MinHeap, spread_km
from orogen.v2.core.mesh_rings import RING_COUNT, rings_of
class MeshFixture:
    pass


def mesh():
    result = MeshFixture()
    result.num_cells = 5
    result.adj_offset = np.array([0, 1, 4, 5, 6, 6], dtype=np.int32)
    result.adj_list = np.array([1, 0, 2, 3, 1, 1], dtype=np.int32)
    result.xyz = np.array([
        1.0, 0.0, 0.0,
        np.cos(0.01), np.sin(0.01), 0.0,
        np.cos(0.02), np.sin(0.02), 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ], dtype=np.float64)
    return result


def test_min_heap_orders_keys_and_preserves_last_key():
    heap = MinHeap(2)
    heap.push(5.0, 50)
    heap.push(3.0, 30)
    heap.push(4.0, 40)  # exercises capacity growth
    assert heap.peek_key() == 3.0
    assert heap.pop() == 30
    assert heap.last_key == 3.0
    assert heap.pop() == 40
    assert heap.pop() == 50
    assert heap.size == 0


def test_spread_km_propagates_distances_and_respects_maximum():
    m = mesh()
    distance = np.full(m.num_cells, np.inf, dtype=np.float64)
    distance[0] = 0.0
    reached = []
    result = spread_km(
        m, distance, max_km=200.0,
        on_reach=lambda source, target: reached.append((source, target)),
    )
    assert math.isclose(result[1], 63.71, rel_tol=2e-4)
    assert math.isclose(result[2], 127.42, rel_tol=2e-4)
    assert math.isinf(result[3])
    assert math.isinf(result[4])
    assert (0, 1) in reached and (1, 2) in reached


def test_rings_of_caches_breadth_first_neighbourhoods():
    m = mesh()
    rings = rings_of(m)
    assert rings["stride"] == RING_COUNT + 2
    start, cells = rings["start"], rings["cells"]
    stride = rings["stride"]
    assert cells[start[0 * stride + 0]:start[0 * stride + 1]].tolist() == [0]
    assert cells[start[0 * stride + 1]:start[0 * stride + 2]].tolist() == [1]
    assert cells[start[0 * stride + 2]:start[0 * stride + 3]].tolist() == [2, 3]
    assert rings_of(m) is rings
