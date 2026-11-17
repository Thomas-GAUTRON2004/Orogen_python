"""Optional Numba kernels.

All kernels use float64 arithmetic, preserve the upstream operation order, and
disable fast-math. The pure-Python implementation remains the reference path.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
except ImportError:  # Numba is an optional acceleration extra.
    njit = None

NUMBA_AVAILABLE = njit is not None


def _noise3d_kernel(x, y, z, perm, pm3, grad):
    f, h = 1.0 / 3.0, 1.0 / 6.0
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
    x2, y2, z2 = x0 - i2 + 2.0*h, y0 - j2 + 2.0*h, z0 - k2 + 2.0*h
    x3, y3, z3 = x0 - 1.0 + 3.0*h, y0 - 1.0 + 3.0*h, z0 - 1.0 + 3.0*h
    ii, jj, kk = i & 255, j & 255, k & 255
    n0 = n1 = n2 = n3 = 0.0

    a = 0.6 - x0*x0 - y0*y0 - z0*z0
    if a > 0.0:
        a *= a
        g = int(pm3[ii + int(perm[jj + int(perm[kk])])])
        n0 = a*a*(grad[g]*x0 + grad[g+1]*y0 + grad[g+2]*z0)

    b = 0.6 - x1*x1 - y1*y1 - z1*z1
    if b > 0.0:
        b *= b
        g = int(pm3[ii+i1 + int(perm[jj+j1 + int(perm[kk+k1])])])
        n1 = b*b*(grad[g]*x1 + grad[g+1]*y1 + grad[g+2]*z1)

    c = 0.6 - x2*x2 - y2*y2 - z2*z2
    if c > 0.0:
        c *= c
        g = int(pm3[ii+i2 + int(perm[jj+j2 + int(perm[kk+k2])])])
        n2 = c*c*(grad[g]*x2 + grad[g+1]*y2 + grad[g+2]*z2)

    d = 0.6 - x3*x3 - y3*y3 - z3*z3
    if d > 0.0:
        d *= d
        g = int(pm3[ii+1 + int(perm[jj+1 + int(perm[kk+1])])])
        n3 = d*d*(grad[g]*x3 + grad[g+1]*y3 + grad[g+2]*z3)
    return 32.0*(n0+n1+n2+n3)


def _fbm_kernel(x, y, z, octaves, persistence, perm, pm3, grad):
    total = 0.0
    maximum = 0.0
    amplitude = 1.0
    for octave in range(octaves):
        frequency = 1 << octave
        total += amplitude * _noise3d_kernel(
            x*frequency, y*frequency, z*frequency, perm, pm3, grad
        )
        maximum += amplitude
        amplitude *= persistence
    return total / maximum


def _fbm_many_kernel(points, octaves, persistence, perm, pm3, grad):
    out = np.empty(points.shape[0], dtype=np.float64)
    for i in range(points.shape[0]):
        out[i] = _fbm_kernel(
            points[i, 0], points[i, 1], points[i, 2],
            octaves, persistence, perm, pm3, grad,
        )
    return out



def _destination_kernel(px, py, pz, bearing, distance):
    # Inline tangent_basis, normalize, and destination in upstream JS order.
    if abs(pz) > 0.9:
        ux, uy, uz = 1.0, 0.0, 0.0
    else:
        ux, uy, uz = 0.0, 0.0, 1.0
    ex, ey, ez = uy*pz-uz*py, uz*px-ux*pz, ux*py-uy*px
    el = math.hypot(ex, ey, ez) or 1.0
    ex, ey, ez = ex/el, ey/el, ez/el
    nx, ny, nz = py*ez-pz*ey, pz*ex-px*ez, px*ey-py*ex
    dx, dy = math.cos(bearing), math.sin(bearing)
    sine, cosine = math.sin(distance), math.cos(distance)
    x = px*cosine + (ex*dx + nx*dy)*sine
    y = py*cosine + (ey*dx + ny*dy)*sine
    z = pz*cosine + (ez*dx + nz*dy)*sine
    length = math.hypot(x, y, z) or 1.0
    return x/length, y/length, z/length


def _bearing_kernel(px, py, pz, qx, qy, qz):
    if abs(pz) > 0.9:
        ux, uy, uz = 1.0, 0.0, 0.0
    else:
        ux, uy, uz = 0.0, 0.0, 1.0
    ex, ey, ez = uy*pz-uz*py, uz*px-ux*pz, ux*py-uy*px
    el = math.hypot(ex, ey, ez) or 1.0
    ex, ey, ez = ex/el, ey/el, ez/el
    nx, ny, nz = py*ez-pz*ey, pz*ex-px*ez, px*ey-py*ex
    east_dot = qx*ex + qy*ey + qz*ez
    north_dot = qx*nx + qy*ny + qz*nz
    return math.atan2(north_dot, east_dot)

def _hop_distance_kernel(offsets, adjacency, sources, max_hops):
    n = sources.size
    distance = np.full(n, -1, dtype=np.int16)
    queue = np.empty(n, dtype=np.int32)
    tail = 0
    for cell in range(n):
        if sources[cell]:
            distance[cell] = 0
            queue[tail] = cell
            tail += 1

    head = 0
    while head < tail:
        cell = queue[head]
        head += 1
        next_depth = int(distance[cell]) + 1
        if next_depth > max_hops:
            continue
        for edge in range(offsets[cell], offsets[cell + 1]):
            neighbour = int(adjacency[edge])
            if distance[neighbour] >= 0:
                continue
            distance[neighbour] = next_depth
            queue[tail] = neighbour
            tail += 1
    return distance


if NUMBA_AVAILABLE:
    noise3d_numba = njit(cache=True, fastmath=False)(_noise3d_kernel)
    # Compile dependencies first and expose their dispatchers to the next kernel.
    _noise3d_kernel = noise3d_numba
    fbm_numba = njit(cache=True, fastmath=False)(_fbm_kernel)
    _fbm_kernel = fbm_numba
    fbm_many_numba = njit(cache=True, fastmath=False)(_fbm_many_kernel)
    hop_distance_numba = njit(cache=True, fastmath=False)(_hop_distance_kernel)
    destination_numba = njit(cache=True, fastmath=False)(_destination_kernel)
    bearing_numba = njit(cache=True, fastmath=False)(_bearing_kernel)
else:
    noise3d_numba = fbm_numba = fbm_many_numba = hop_distance_numba = None
    destination_numba = bearing_numba = None
