"""Numerical primitives ported from upstream Orogen V2 core."""

from .device import SMALL_DEVICE, is_small_device
from .fast_noise import FastNoise
from .mesh_bfs import (
    flood_from,
    hop_distance,
    hops_from,
    largest_piece_of,
    pieces_of,
    reaches_within,
    touches,
    within_km,
)
from .mesh_locate import locate
from .mesh_rings import RING_COUNT, rings_of
from .min_heap import MinHeap, forget_edge_km, spread_km
from .rotation import (
    quat_from_axis_angle,
    quat_identity,
    quat_integrate,
    quat_mul,
    quat_normalize,
    quat_to_euler_pole,
    quat_to_mat3,
)
from .seed import SEEDS, is_word_seed, make_rand_int, make_rng, random_seed, seed_of
from .sim_mesh import SimMesh, build_sim_mesh, km_to_hops, mesh_from_arrays
from .sphere_geometry import (
    angle_between,
    bearing_to,
    cap_fraction,
    cell_distance_km,
    centroid_of_cells,
    cross,
    destination,
    dot,
    normalize,
    polar_frame,
    random_point,
    tangent_basis,
    tangent_toward,
)
from .sphere_index import SphereIndex
from .sphere_mesh import (
    SphereMesh,
    build_sphere,
    generate_fibonacci_sphere,
    stereographic_projection,
)

__all__ = [
    "SEEDS",
    "is_word_seed",
    "make_rand_int",
    "make_rng",
    "random_seed",
    "seed_of",
    "SMALL_DEVICE",
    "is_small_device",
    "quat_from_axis_angle",
    "quat_identity",
    "quat_integrate",
    "quat_mul",
    "quat_normalize",
    "quat_to_euler_pole",
    "quat_to_mat3",
    "flood_from",
    "hop_distance",
    "hops_from",
    "largest_piece_of",
    "pieces_of",
    "reaches_within",
    "touches",
    "within_km",
    "FastNoise",
    "MinHeap",
    "forget_edge_km",
    "spread_km",
    "locate",
    "RING_COUNT",
    "rings_of",
    "SphereIndex",
    "SphereMesh",
    "SimMesh",
    "build_sphere",
    "build_sim_mesh",
    "generate_fibonacci_sphere",
    "stereographic_projection",
    "km_to_hops",
    "mesh_from_arrays",
    "angle_between",
    "bearing_to",
    "cap_fraction",
    "cell_distance_km",
    "centroid_of_cells",
    "cross",
    "destination",
    "dot",
    "normalize",
    "polar_frame",
    "random_point",
    "tangent_basis",
    "tangent_toward",
]
