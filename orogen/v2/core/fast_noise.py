"""Fast 3-D simplex noise ported from V2 fast-noise.js.

The permutation is derived from the shared Orogen Park–Miller RNG. Arithmetic
order follows the JavaScript implementation to preserve seeded output.
"""

# differential-module:

from __future__ import annotations

import math

import numpy as np

from ._numba_kernels import NUMBA_AVAILABLE, fbm_many_numba, fbm_numba, noise3d_numba
from .seed import make_rng

_GRAD = (
    1,
    1,
    0,
    -1,
    1,
    0,
    1,
    -1,
    0,
    -1,
    -1,
    0,
    1,
    0,
    1,
    -1,
    0,
    1,
    1,
    0,
    -1,
    -1,
    0,
    -1,
    0,
    1,
    1,
    0,
    -1,
    1,
    0,
    1,
    -1,
    0,
    -1,
    -1,
)


class FastNoise:
    """Seeded 3-D simplex noise and fractal Brownian motion."""

    def __init__(self, seed: int | float = 0):
        rng = make_rng(seed)
        permutation = list(range(256))
        for i in range(255, 0, -1):
            j = math.floor(rng() * (i + 1))
            permutation[i], permutation[j] = permutation[j], permutation[i]
        self.perm = np.empty(512, dtype=np.uint8)
        self.pm3 = np.empty(512, dtype=np.uint8)
        self._grad = np.asarray(_GRAD, dtype=np.float64)
        for i in range(512):
            value = permutation[i & 255]
            self.perm[i] = value
            self.pm3[i] = 3 * (value % 12)

    # differential: js=noise3D
    def noise3d(self, x: float, y: float, z: float) -> float:
        if NUMBA_AVAILABLE:
            return float(noise3d_numba(x, y, z, self.perm, self.pm3, self._grad))
        return self._noise3d_python(x, y, z)

    def _noise3d_python(self, x: float, y: float, z: float) -> float:
        p, m, g = self.perm, self.pm3, _GRAD
        f, h = 1 / 3, 1 / 6
        skew = (x + y + z) * f
        i, j, k = math.floor(x + skew), math.floor(y + skew), math.floor(z + skew)
        unskew = (i + j + k) * h
        x0, y0, z0 = x - i + unskew, y - j + unskew, z - k + unskew

        if x0 >= y0:
            if y0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
            elif x0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
        else:
            if y0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
            elif x0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0

        x1, y1, z1 = x0 - i1 + h, y0 - j1 + h, z0 - k1 + h
        x2, y2, z2 = x0 - i2 + 2 * h, y0 - j2 + 2 * h, z0 - k2 + 2 * h
        x3, y3, z3 = x0 - 1 + 3 * h, y0 - 1 + 3 * h, z0 - 1 + 3 * h
        ii, jj, kk = i & 255, j & 255, k & 255
        n0 = n1 = n2 = n3 = 0.0

        a = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
        if a > 0:
            a *= a
            grad = int(m[ii + int(p[jj + int(p[kk])])])
            n0 = a * a * (g[grad] * x0 + g[grad + 1] * y0 + g[grad + 2] * z0)

        b = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
        if b > 0:
            b *= b
            grad = int(m[ii + i1 + int(p[jj + j1 + int(p[kk + k1])])])
            n1 = b * b * (g[grad] * x1 + g[grad + 1] * y1 + g[grad + 2] * z1)

        c = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
        if c > 0:
            c *= c
            grad = int(m[ii + i2 + int(p[jj + j2 + int(p[kk + k2])])])
            n2 = c * c * (g[grad] * x2 + g[grad + 1] * y2 + g[grad + 2] * z2)

        d = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
        if d > 0:
            d *= d
            grad = int(m[ii + 1 + int(p[jj + 1 + int(p[kk + 1])])])
            n3 = d * d * (g[grad] * x3 + g[grad + 1] * y3 + g[grad + 2] * z3)

        return 32 * (n0 + n1 + n2 + n3)

    def fbm(
        self,
        x: float,
        y: float,
        z: float,
        octaves: int = 5,
        persistence: float = 2 / 3,
    ) -> float:
        if NUMBA_AVAILABLE:
            return float(
                fbm_numba(
                    x,
                    y,
                    z,
                    octaves,
                    persistence,
                    self.perm,
                    self.pm3,
                    self._grad,
                )
            )
        return self._fbm_python(x, y, z, octaves, persistence)

    def fbm_many(
        self, points: np.ndarray, octaves: int = 5, persistence: float = 2 / 3
    ) -> np.ndarray:
        """Evaluate an (N, 3) float64 point array in one compiled call when available."""
        values = np.ascontiguousarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if NUMBA_AVAILABLE:
            return fbm_many_numba(
                values,
                octaves,
                persistence,
                self.perm,
                self.pm3,
                np.asarray(_GRAD, dtype=np.float64),
            )
        return np.asarray(
            [self._fbm_python(x, y, z, octaves, persistence) for x, y, z in values],
            dtype=np.float64,
        )

    def _fbm_python(
        self,
        x: float,
        y: float,
        z: float,
        octaves: int = 5,
        persistence: float = 2 / 3,
    ) -> float:
        total = maximum = 0.0
        amplitude = 1.0
        for octave in range(octaves):
            frequency = 1 << octave
            total += amplitude * self.noise3d(
                x * frequency, y * frequency, z * frequency
            )
            maximum += amplitude
            amplitude *= persistence
        return total / maximum
