"""Fibonacci sphere and Delaunay dual mesh builder from Orogen's V1 sphere-mesh.js.

SciPy/Qhull supplies the planar Delaunay triangulation. The spherical closure
and cyclic CSR region adjacency follow the upstream half-edge construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError


@dataclass(slots=True)
class SphereMesh:
    """Closed triangular mesh dual, with cyclic neighbours for each region."""

    triangles: np.ndarray
    adj_offset: np.ndarray
    adj_list: np.ndarray
    num_regions: int
    num_triangles: int

    @property
    def num_cells(self) -> int:
        return self.num_regions


def generate_fibonacci_sphere(
    num_points: int,
    jitter: float,
    rng,
) -> np.ndarray:
    """Generate the upstream Fibonacci spiral positions (Float32 xyz)."""
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    xyz = np.empty(3 * num_points, dtype=np.float32)
    spacing = 3.6 / math.sqrt(num_points)
    longitude_step = math.pi * (3 - math.sqrt(5))
    dz = 2.0 / num_points

    longitude = 0.0
    z = 1.0 - dz / 2.0
    for k in range(num_points):
        radius = math.sqrt(1.0 - z * z)
        latitude_deg = math.asin(z) * 180.0 / math.pi
        longitude_deg = longitude * 180.0 / math.pi

        if jitter > 0:
            j_lat = rng() - rng()
            j_lon = rng() - rng()
            next_z = max(-1.0, z - dz * 2.0 * math.pi * radius / spacing)
            latitude_deg += jitter * j_lat * (
                latitude_deg - math.asin(next_z) * 180.0 / math.pi
            )
            longitude_deg += jitter * j_lon * (spacing / radius * 180.0 / math.pi)

        latitude = latitude_deg * math.pi / 180.0
        longitude_r = longitude_deg * math.pi / 180.0
        xyz[3 * k] = math.cos(latitude) * math.cos(longitude_r)
        xyz[3 * k + 1] = math.cos(latitude) * math.sin(longitude_r)
        xyz[3 * k + 2] = math.sin(latitude)
        longitude += longitude_step
        z -= dz
    return xyz


def stereographic_projection(xyz: np.ndarray, num_points: int) -> np.ndarray:
    """Project from the north pole onto the plane, as in sphere-mesh.js."""
    points = np.asarray(xyz).reshape(num_points, 3)
    denominator = np.maximum(1e-12, 1.0 - points[:, 2])
    return np.column_stack((points[:, 0] / denominator, points[:, 1] / denominator))


def _closed_triangles(points_2d: np.ndarray, pole_id: int) -> np.ndarray:
    """Triangulate the projected points and close the convex hull at the pole."""
    try:
        delaunay = Delaunay(points_2d)
    except QhullError as exc:
        raise ValueError("Delaunay triangulation failed for the projected sphere") from exc

    triangles = np.asarray(delaunay.simplices, dtype=np.int32).copy()
    # Normalize simplex winding. Qhull normally emits CCW simplices, but make
    # the winding contract explicit before pairing directed half-edges.
    p = points_2d
    for tri in triangles:
        a, b, c = map(int, tri)
        ax, ay = p[a]
        bx, by = p[b]
        cx, cy = p[c]
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
            tri[1], tri[2] = tri[2], tri[1]

    edge_count: dict[tuple[int, int], tuple[int, int, int]] = {}
    for tri in triangles:
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            if key in edge_count:
                old_u, old_v, count = edge_count[key]
                edge_count[key] = (old_u, old_v, count + 1)
            else:
                edge_count[key] = (u, v, 1)

    hull = [(u, v) for u, v, count in edge_count.values() if count == 1]
    if len(hull) < 3:
        raise ValueError("Delaunay triangulation has no closed convex hull")

    # A planar hull edge u→v is closed by the spherical triangle v→u→pole.
    pole_triangles = np.asarray([(v, u, pole_id) for u, v in hull], dtype=np.int32)
    return np.concatenate((triangles, pole_triangles), axis=0)


def _cyclic_region_adjacency(
    triangles: np.ndarray,
    num_regions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build CSR neighbours in the same cyclic order as half-edge circulation."""
    successor: list[dict[int, int]] = [dict() for _ in range(num_regions)]
    for a_raw, b_raw, c_raw in triangles:
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        for centre, first, second in ((a, b, c), (b, c, a), (c, a, b)):
            links = successor[centre]
            previous = links.get(first)
            if previous is not None and previous != second:
                raise ValueError("non-manifold or inconsistently oriented sphere mesh")
            links[first] = second

    offsets = np.zeros(num_regions + 1, dtype=np.int32)
    ordered: list[int] = []
    for region, links in enumerate(successor):
        if not links:
            raise ValueError(f"region {region} has no neighbours")
        start = next(iter(links))
        current = start
        ring: list[int] = []
        for _ in range(len(links)):
            ring.append(current)
            if current not in links:
                raise ValueError("broken cyclic adjacency in sphere mesh")
            current = links[current]
        if current != start or len(set(ring)) != len(links):
            raise ValueError("sphere mesh adjacency does not form one ring")
        ordered.extend(ring)
        offsets[region + 1] = len(ordered)
    return offsets, np.asarray(ordered, dtype=np.int32)


def build_sphere(num_points: int, jitter: float, rng) -> tuple[SphereMesh, np.ndarray]:
    """Build a closed sphere mesh; output includes the extra north-pole region."""
    xyz = generate_fibonacci_sphere(num_points, jitter, rng)
    projected = stereographic_projection(xyz, num_points)
    triangles = _closed_triangles(projected, num_points)
    xyz_with_pole = np.empty(3 * (num_points + 1), dtype=np.float32)
    xyz_with_pole[: 3 * num_points] = xyz
    xyz_with_pole[3 * num_points : 3 * num_points + 3] = (0.0, 0.0, 1.0)

    offsets, adjacency = _cyclic_region_adjacency(triangles, num_points + 1)
    mesh = SphereMesh(
        triangles=triangles.reshape(-1),
        adj_offset=offsets,
        adj_list=adjacency,
        num_regions=num_points + 1,
        num_triangles=len(triangles),
    )
    return mesh, xyz_with_pole
