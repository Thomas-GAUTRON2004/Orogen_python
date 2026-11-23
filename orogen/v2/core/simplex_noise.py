"""Seeded 3-D simplex noise with fBm and ridged fBm, ported from upstream.

Mirrors ``js/simplex-noise.js`` of the upstream JavaScript engine (v3.0.1,
``ed466f9``): the same 12 gradients, the same permutation draw from the
Park-Miller generator of ``js/rng.js``, and the same order of floating-point
operations. Scenes and the start world use this noise, not ``FastNoise``.
"""

from __future__ import annotations

import math

import numpy as np

from ._numba_kernels import (
    NUMBA_AVAILABLE,
    SIMPLEX_GRADIENTS,
    simplex_fbm_numba,
    simplex_fbm_py,
    simplex_noise3d_numba,
    simplex_noise3d_py,
    simplex_ridged_numba,
    simplex_ridged_py,
)
from .seed import make_rng

_G = (
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (1, 0, -1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, -1, 1),
    (0, 1, -1),
    (0, -1, -1),
)
_F = 1.0 / 3.0
_H = 1.0 / 6.0


class SimplexNoise:
    """Seeded simplex noise; method names follow upstream (`noise3D`, `fbm`, `ridgedFbm`).

    The math lives in ``_numba_kernels`` and is compiled with Numba when that extra
    is installed; otherwise the same Python code runs. Both paths are bit-identical.
    """

    def __init__(self, seed: float = 0) -> None:
        rng = make_rng(seed)
        p = list(range(256))
        for i in range(255, 0, -1):
            j = math.floor(rng() * (i + 1))
            p[i], p[j] = p[j], p[i]
        perm = [p[i & 255] for i in range(512)]
        self.perm = np.asarray(perm, dtype=np.int64)
        self.pm12 = np.asarray([v % 12 for v in perm], dtype=np.int64)
        self.grad = SIMPLEX_GRADIENTS

    def noise3D(self, x: float, y: float, z: float) -> float:  # noqa: N802
        if NUMBA_AVAILABLE:
            return float(
                simplex_noise3d_numba(
                    float(x), float(y), float(z), self.perm, self.pm12, self.grad
                )
            )
        return float(
            simplex_noise3d_py(
                float(x), float(y), float(z), self.perm, self.pm12, self.grad
            )
        )

    def fbm(
        self,
        x: float,
        y: float,
        z: float,
        octaves: int = 5,
        persistence: float = 2.0 / 3.0,
    ) -> float:
        kernel = simplex_fbm_numba if NUMBA_AVAILABLE else simplex_fbm_py
        return float(
            kernel(
                float(x),
                float(y),
                float(z),
                int(octaves),
                float(persistence),
                self.perm,
                self.pm12,
                self.grad,
            )
        )

    def ridged_fbm(
        self,
        x: float,
        y: float,
        z: float,
        octaves: int = 6,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        offset: float = 1.0,
    ) -> float:
        kernel = simplex_ridged_numba if NUMBA_AVAILABLE else simplex_ridged_py
        return float(
            kernel(
                float(x),
                float(y),
                float(z),
                int(octaves),
                float(lacunarity),
                float(gain),
                float(offset),
                self.perm,
                self.pm12,
                self.grad,
            )
        )
