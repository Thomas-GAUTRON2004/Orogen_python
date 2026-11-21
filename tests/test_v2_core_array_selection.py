"""Tests for deterministic partial array selection."""

import numpy as np

from orogen.v2.core.array_selection import stable_top_k


def test_stable_top_k_keeps_index_order_for_equal_values():
    values = np.asarray([0.1, 0.9, 0.9, 0.4, 0.9], dtype=np.float64)

    assert stable_top_k(values, 2).tolist() == [1, 2]
    assert stable_top_k(values, 3).tolist() == [1, 2, 4]


def test_stable_top_k_handles_empty_and_clamped_counts():
    values = np.asarray([2.0, 1.0], dtype=np.float64)

    assert stable_top_k(values, 0).size == 0
    assert stable_top_k(values, -2).size == 0
    assert stable_top_k(values, 9).tolist() == [0, 1]
