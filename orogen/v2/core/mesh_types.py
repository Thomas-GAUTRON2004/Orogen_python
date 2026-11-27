"""Structural types shared by spherical mesh algorithms."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class MeshGraph(Protocol):
    """Minimum CSR mesh interface consumed by graph and geometry helpers."""

    num_cells: int
    xyz: np.ndarray
    adj_offset: np.ndarray
    adj_list: np.ndarray


class IndexedMeshGraph(MeshGraph, Protocol):
    """Mesh graph with the nearest-cell index required by point location."""

    index: object
