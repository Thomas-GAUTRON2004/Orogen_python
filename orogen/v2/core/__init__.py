"""Numerical primitives ported from upstream Orogen V2 core."""

from .seed import SEEDS, is_word_seed, make_rand_int, make_rng, seed_of
from .rotation import (
    quat_from_axis_angle, quat_identity, quat_integrate, quat_mul,
    quat_normalize, quat_to_euler_pole, quat_to_mat3,
)
from .sphere_geometry import (
    angle_between, bearing_to, cap_fraction, cell_distance_km,
    centroid_of_cells, cross, destination, dot, normalize, random_point,
    tangent_basis, tangent_toward,
)

__all__ = [
    "SEEDS", "is_word_seed", "make_rand_int", "make_rng", "seed_of",
    "quat_from_axis_angle", "quat_identity", "quat_integrate", "quat_mul",
    "quat_normalize", "quat_to_euler_pole", "quat_to_mat3",
    "angle_between", "bearing_to", "cap_fraction", "cell_distance_km",
    "centroid_of_cells", "cross", "destination", "dot", "normalize",
    "random_point", "tangent_basis", "tangent_toward",
]
