"""Fixed-resolution simulation mesh helpers from V2 sim-mesh.js."""

# differential-module:

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .seed import make_rng
from .sphere_geometry import PLANET_RADIUS_KM
from .sphere_index import SphereIndex
from .sphere_mesh import build_sphere


@dataclass(eq=False)
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
    """Return the cube-map table oversampling factor used by the upstream V2 implementation."""
    return 1 if num_cells > 1_000_000 else 2


def _mesh_metrics(num_cells: int) -> tuple[float, float, float]:
    edge_rad = float(np.sqrt(8 * np.pi / (np.sqrt(3) * num_cells)))
    edge_km = edge_rad * PLANET_RADIUS_KM
    cell_area = float(4 * np.pi * PLANET_RADIUS_KM**2 / num_cells)
    return edge_rad, edge_km, cell_area


def km_to_hops(sim_mesh: SimMesh, distance_km: float, minimum: int = 1) -> int:
    """Convert a physical distance to hops using JavaScript Math.round semantics."""
    return max(minimum, int(np.floor(distance_km / sim_mesh.edge_km + 0.5)))


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
    if xyz.size != 3 * num_cells:
        raise ValueError("xyz must contain exactly 3 * num_cells values")
    if adj_offset.size != num_cells + 1 or int(adj_offset[-1]) != adj_list.size:
        raise ValueError("invalid CSR adjacency arrays")
    edge_rad, edge_km, cell_area = _mesh_metrics(num_cells)
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
        edge_rad=edge_rad,
        edge_km=edge_km,
        cell_area_km2=cell_area,
    )


# differential: skip
def build_sim_mesh(num_cells: int, seed: int) -> SimMesh:
    """Build the deterministic V2 simulation mesh from a seed."""
    if num_cells < 3:
        raise ValueError("num_cells must be at least 3")
    sphere, xyz = build_sphere(num_cells, 0.35, make_rng(seed + 4242))
    count = sphere.num_regions
    edge_rad, edge_km, cell_area = _mesh_metrics(count)
    index = SphereIndex(xyz, sphere, texels_for(count))
    return SimMesh(
        num_cells=count,
        xyz=xyz,
        adj_offset=sphere.adj_offset,
        adj_list=sphere.adj_list,
        index=index,
        edge_rad=edge_rad,
        edge_km=edge_km,
        cell_area_km2=cell_area,
    )
