"""Regression tests for the first Orogen V2 core primitives."""
import math

from orogen.v2.core import (
    make_rng,
    normalize,
    quat_from_axis_angle,
    quat_identity,
    quat_integrate,
    quat_to_euler_pole,
    quat_to_mat3,
    seed_of,
)
from orogen.v2.core.sphere_geometry import (
    angle_between,
    cell_distance_km,
    destination,
    random_point,
)


def test_park_miller_rng_matches_upstream_javascript_fixture():
    rng = make_rng(42)
    expected = [
        0.4431328917323918,
        0.7345157044329826,
        0.005446482454842406,
        0.5390384020647392,
        0.6184271095492199,
    ]
    assert [rng() for _ in expected] == expected


def test_word_seed_matches_upstream_fnv1a_fixture():
    assert seed_of("Pangaea") == 258625900
    assert seed_of(" 42 ") == 42\n    assert seed_of("0xFF") == 255\n    assert seed_of("0b101") == 5


def test_quaternion_identity_and_axis_angle():
    identity = quat_identity()
    assert quat_to_mat3(identity) == [1, 0, 0, 0, 1, 0, 0, 0, 1]

    quarter_turn = quat_from_axis_angle(0, 0, 1, math.pi / 2)
    pole = quat_to_euler_pole(quarter_turn)
    assert math.isclose(pole["lat"], 90.0, abs_tol=1e-12)
    assert math.isclose(pole["angle"], 90.0, abs_tol=1e-12)


def test_quaternion_integration_preserves_unit_norm():
    result = quat_integrate(quat_identity(), [0.0, 0.0, 0.2], 10.0)
    assert math.isclose(math.hypot(*result), 1.0, abs_tol=1e-12)


def test_spherical_geometry_distance_and_destination():
    north_pole = [0.0, 0.0, 1.0]
    equator_x = [1.0, 0.0, 0.0]
    assert math.isclose(angle_between(north_pole, equator_x), math.pi / 2)
    assert math.isclose(cell_distance_km([*north_pole, *equator_x], 0, 1),
                        math.pi * 6371.0 / 2, rel_tol=1e-12)

    point = normalize([1.0, 0.0, 0.0])
    moved = destination(point, 0.0, math.pi / 6)
    assert math.isclose(math.hypot(*moved), 1.0, abs_tol=1e-12)


def test_random_point_is_on_unit_sphere():
    values = iter([0.75, 0.125])
    point = random_point(lambda: next(values))
    assert math.isclose(math.hypot(*point), 1.0, abs_tol=1e-12)
