"""Fixed-resolution simulation mesh helpers from V2 sim-mesh.js."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sphere_geometry import PLANET_RADIUS_KM
from .sphere_index import SphereIndex


@dataclass(slots=True)
class SimMesh:
    num_cells: int
    xyz: np.ndarray
    adj_offset: np.ndarray
    adj_list: np.ndarray
    index: SphereIndex
    edge_rad: float
    edge_km: float
    cell_area_km2: float


def texels_for(num_cells: int) -> int:
    """Cube-map table oversampling used by the upstream V2 implementation."""
    return 1 if num_cells > 1_000_000 else 2


def km_to_hops(sim_mesh: SimMesh, distance_km: float, minimum: int = 1) -> int:
    """Convert physical distance to approximate mesh hops (upstream rounding)."""
    return max(minimum, int(distance_km / sim_mesh.edge_km + 0.5))


def mesh_from_arrays(
    *,
    num_cells: int,
    xyz: np.ndarray,
    adj_offset: np.ndarray,
    adj_list: np.ndarray,
) -> SimMesh:
    """Rebuild a SimMesh from transferred arrays, without a Delaunay rebuild."""
    xyz = np.asarray(xyz, dtype=np.float32)
    adj_offset = np.asarray(adj_offset, dtype=np.int32)
    adj_list = np.asarray(adj_list, dtype=np.int32)
    edge_rad = (8 * np.pi / (np.sqrt(3) * num_cells)) ** 0.5
    lightweight = type(
        "_MeshArrays",
        (),
        {
            "num_cells": num_cells,
            "adj_offset": adj_offset,
            "adj_list": adj_list,
        },
    )()
    index = SphereIndex(xyz, lightweight, texels_for(num_cells))
    return SimMesh(
        num_cells=num_cells,
        xyz=xyz,
        adj_offset=adj_offset,
        adj_list=adj_list,
        index=index,
        edge_rad=float(edge_rad),
        edge_km=float(edge_rad * PLANET_RADIUS_KM),
        cell_area_km2=float(4 * np.pi * PLANET_RADIUS_KM**2 / num_cells),
    )
