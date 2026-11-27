"""Structural types shared by spherical mesh algorithms."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AdjacencyGraph(Protocol):
    """CSR adjacency interface used to construct nearest-cell indexes."""

    @property
    def num_cells(self) -> int: ...

    adj_offset: np.ndarray
    adj_list: np.ndarray


class NearestCellIndex(Protocol):
    """Nearest-cell lookup interface used by point-location algorithms."""

    def nearest(self, x: float, y: float, z: float) -> int: ...


class MeshGraph(AdjacencyGraph, Protocol):
    """Minimum CSR mesh interface consumed by graph and geometry helpers."""

    xyz: np.ndarray


class IndexedMeshGraph(MeshGraph, Protocol):
    """Mesh graph with the nearest-cell index required by point location."""

    index: NearestCellIndex
