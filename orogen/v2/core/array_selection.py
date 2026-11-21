"""Stable partial selection helpers for deterministic array ordering."""

from __future__ import annotations

import numpy as np


def stable_top_k(values: np.ndarray, count: int) -> np.ndarray:
    """Select the highest values with stable index tie-breaking.

    The partition finds the selection threshold in linear time; only the
    selected values are sorted. Equal threshold values retain input-index order.
    """
    count = max(0, min(int(count), int(values.size)))
    if count == 0:
        return np.empty(0, dtype=np.intp)
    threshold_index = values.size - count
    threshold = np.partition(values, threshold_index)[threshold_index]
    above = np.flatnonzero(values > threshold)
    tied = np.flatnonzero(values == threshold)[: count - above.size]
    selected = np.concatenate((above, tied))
    return selected[np.lexsort((selected, -values[selected]))]
