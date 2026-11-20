import math

from orogen.v2.core.fast_noise import FastNoise


def test_fast_noise_matches_upstream_javascript_fixtures():
    noise = FastNoise(42)
    assert math.isclose(
        noise.noise3d(0.1, 0.2, 0.3),
        -0.11666551466666661,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        noise.fbm(0.1, 0.2, 0.3),
        -0.006157208265402842,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    noise_zero = FastNoise(0)
    assert math.isclose(
        noise_zero.noise3d(-1.25, 2.5, 0.75),
        -0.45900308641975307,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert noise_zero.noise3d(0.0, 0.0, 0.0) == 0.0
