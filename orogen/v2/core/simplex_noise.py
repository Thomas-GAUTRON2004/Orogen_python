"""Seeded 3-D simplex noise with fBm and ridged fBm, ported from upstream.

Mirrors ``js/simplex-noise.js`` of the upstream JavaScript engine (v3.0.1,
``ed466f9``): the same 12 gradients, the same permutation draw from the
Park-Miller generator of ``js/rng.js``, and the same order of floating-point
operations. Scenes and the start world use this noise, not ``FastNoise``.
"""

from __future__ import annotations

import math

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
    """Seeded simplex noise; ``noise3D`` keeps the upstream method name."""

    def __init__(self, seed: float = 0) -> None:
        rng = make_rng(seed)
        p = list(range(256))
        for i in range(255, 0, -1):
            j = math.floor(rng() * (i + 1))
            p[i], p[j] = p[j], p[i]
        self.perm = [p[i & 255] for i in range(512)]
        self.pm12 = [v % 12 for v in self.perm]

    def noise3D(self, x: float, y: float, z: float) -> float:  # noqa: N802
        s = (x + y + z) * _F
        i = math.floor(x + s)
        j = math.floor(y + s)
        k = math.floor(z + s)
        t = (i + j + k) * _H
        x0 = x - i + t
        y0 = y - j + t
        z0 = z - k + t

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

        x1 = x0 - i1 + _H
        y1 = y0 - j1 + _H
        z1 = z0 - k1 + _H
        x2 = x0 - i2 + 2 * _H
        y2 = y0 - j2 + 2 * _H
        z2 = z0 - k2 + 2 * _H
        x3 = x0 - 1 + 3 * _H
        y3 = y0 - 1 + 3 * _H
        z3 = z0 - 1 + 3 * _H

        ii = i & 255
        jj = j & 255
        kk = k & 255
        perm = self.perm
        m = self.pm12
        g = _G

        n0 = n1 = n2 = n3 = 0.0
        a = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
        if a > 0:
            a *= a
            v = g[m[ii + perm[jj + perm[kk]]]]
            n0 = a * a * (v[0] * x0 + v[1] * y0 + v[2] * z0)
        b = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
        if b > 0:
            b *= b
            v = g[m[ii + i1 + perm[jj + j1 + perm[kk + k1]]]]
            n1 = b * b * (v[0] * x1 + v[1] * y1 + v[2] * z1)
        c = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
        if c > 0:
            c *= c
            v = g[m[ii + i2 + perm[jj + j2 + perm[kk + k2]]]]
            n2 = c * c * (v[0] * x2 + v[1] * y2 + v[2] * z2)
        d = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
        if d > 0:
            d *= d
            v = g[m[ii + 1 + perm[jj + 1 + perm[kk + 1]]]]
            n3 = d * d * (v[0] * x3 + v[1] * y3 + v[2] * z3)
        return 32 * (n0 + n1 + n2 + n3)

    def fbm(
        self,
        x: float,
        y: float,
        z: float,
        octaves: int = 5,
        persistence: float = 2.0 / 3.0,
    ) -> float:
        total = 0.0
        peak = 0.0
        amp = 1.0
        for o in range(octaves):
            f = 1 << o
            total += amp * self.noise3D(x * f, y * f, z * f)
            peak += amp
            amp *= persistence
        return total / peak

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
        total = 0.0
        freq = 1.0
        amp = 1.0
        prev = 1.0
        peak = 0.0
        for _ in range(octaves):
            n = self.noise3D(x * freq, y * freq, z * freq)
            n = offset - abs(n)
            n = n * n
            total += n * amp * prev
            peak += amp
            prev = min(n, 1.0)
            freq *= lacunarity
            amp *= gain
        return total / peak
