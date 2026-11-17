"""Optional Numba kernels.

All kernels use float64 arithmetic, preserve the upstream operation order, and
disable fast-math. The pure-Python implementation remains the reference path.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit, literal_unroll
except ImportError:  # Numba is an optional acceleration extra.
    njit = None
    literal_unroll = None

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



def _far_field_distance_kernel(
    offsets, adjacency, xyz, mask, crust_type, flags,
    source_cells, source_works, strongest, reach_length, max_cost,
    craton_type, weak_mask, craton_factor, weak_factor,
):
    """Weighted far-field Dijkstra; side effects remain in the Python caller."""
    n = mask.size
    distance = np.full(n, math.inf, dtype=np.float64)
    source_of = np.full(n, -1, dtype=np.int32)
    reached = np.empty(n, dtype=np.int32)
    reached_count = 0
    capacity = adjacency.size + n + 1
    heap_keys = np.empty(capacity, dtype=np.float64)
    heap_ids = np.empty(capacity, dtype=np.int32)
    heap_size = 0

    # Keep the insertion order of the source mapping, including equal keys.
    for i in range(source_cells.size):
        cell = int(source_cells[i])
        initial = -reach_length * math.log(source_works[i] / strongest)
        if initial < distance[cell]:
            distance[cell] = initial
            source_of[cell] = cell
            index = heap_size
            heap_size += 1
            while index > 0:
                parent = (index - 1) >> 1
                if heap_keys[parent] <= initial:
                    break
                heap_keys[index] = heap_keys[parent]
                heap_ids[index] = heap_ids[parent]
                index = parent
            heap_keys[index] = initial
            heap_ids[index] = cell

    while heap_size > 0:
        cell = int(heap_ids[0])
        current = heap_keys[0]
        count = heap_size - 1
        heap_size = count

        if count > 0:
            key = heap_keys[count]
            identifier = heap_ids[count]
            index = 0
            while True:
                left = 2 * index + 1
                if left >= count:
                    break
                right = left + 1
                child = right if right < count and heap_keys[right] < heap_keys[left] else left
                if heap_keys[child] >= key:
                    break
                heap_keys[index] = heap_keys[child]
                heap_ids[index] = heap_ids[child]
                index = child
            heap_keys[index] = key
            heap_ids[index] = identifier

        if current > distance[cell]:
            continue

        reached[reached_count] = cell
        reached_count += 1
        for edge in range(offsets[cell], offsets[cell + 1]):
            neighbour = int(adjacency[edge])
            if not mask[neighbour]:
                continue
            neighbour_type = int(crust_type[neighbour])
            if neighbour_type == craton_type:
                factor = craton_factor
            elif (int(flags[neighbour]) & weak_mask) != 0:
                factor = weak_factor
            else:
                factor = 1.0

            a = 3 * cell
            b = 3 * neighbour
            cosine = (
                float(xyz[a]) * float(xyz[b])
                + float(xyz[a + 1]) * float(xyz[b + 1])
                + float(xyz[a + 2]) * float(xyz[b + 2])
            )
            edge_distance = math.acos(max(-1.0, min(1.0, cosine))) * 6371.0
            candidate = current + edge_distance * factor
            if candidate >= distance[neighbour] or candidate > max_cost:
                continue
            distance[neighbour] = candidate
            source_of[neighbour] = int(source_of[cell])

            index = heap_size
            heap_size += 1
            while index > 0:
                parent = (index - 1) >> 1
                if heap_keys[parent] <= candidate:
                    break
                heap_keys[index] = heap_keys[parent]
                heap_ids[index] = heap_ids[parent]
                index = parent
            heap_keys[index] = candidate
            heap_ids[index] = neighbour

    return distance, source_of, reached, reached_count


def _spread_km_kernel(offsets, adjacency, edge_km, distance, max_km):
    """Full Dijkstra traversal with the same lazy-deletion binary heap as JS."""
    n = distance.size
    capacity = adjacency.size + n + 1
    heap_keys = np.empty(capacity, dtype=np.float64)
    heap_ids = np.empty(capacity, dtype=np.int32)
    heap_size = 0

    # Seed insertion order is ascending cell index, matching spreadKm.
    for cell in range(n):
        if distance[cell] < math.inf:
            key = float(distance[cell])
            index = heap_size
            heap_size += 1
            while index > 0:
                parent = (index - 1) >> 1
                if heap_keys[parent] <= key:
                    break
                heap_keys[index] = heap_keys[parent]
                heap_ids[index] = heap_ids[parent]
                index = parent
            heap_keys[index] = key
            heap_ids[index] = cell

    while heap_size > 0:
        cell = int(heap_ids[0])
        current_distance = heap_keys[0]
        count = heap_size - 1
        heap_size = count

        if count > 0:
            key = heap_keys[count]
            identifier = heap_ids[count]
            index = 0
            while True:
                left = 2 * index + 1
                if left >= count:
                    break
                right = left + 1
                child = right if right < count and heap_keys[right] < heap_keys[left] else left
                if heap_keys[child] >= key:
                    break
                heap_keys[index] = heap_keys[child]
                heap_ids[index] = heap_ids[child]
                index = child
            heap_keys[index] = key
            heap_ids[index] = identifier

        if current_distance > distance[cell]:
            continue

        for edge in range(offsets[cell], offsets[cell + 1]):
            neighbour = int(adjacency[edge])
            candidate = current_distance + edge_km[edge]
            if candidate >= distance[neighbour] or candidate > max_km:
                continue

            # Queue the stored value, not the pre-cast candidate. This is
            # important for float32 distance arrays, just as in the JS port.
            distance[neighbour] = candidate
            stored_distance = float(distance[neighbour])
            index = heap_size
            heap_size += 1
            while index > 0:
                parent = (index - 1) >> 1
                if heap_keys[parent] <= stored_distance:
                    break
                heap_keys[index] = heap_keys[parent]
                heap_ids[index] = heap_ids[parent]
                index = parent
            heap_keys[index] = stored_distance
            heap_ids[index] = neighbour

    return distance



def _pieces_of_kernel(offsets, adjacency, ordered_cells):
    """Return connected components in input-seed and CSR-neighbour order."""
    n = offsets.size - 1
    in_set = np.zeros(n, dtype=np.bool_)
    for i in range(ordered_cells.size):
        in_set[int(ordered_cells[i])] = True

    seen = np.zeros(n, dtype=np.bool_)
    queue = np.empty(n, dtype=np.int32)
    flat = np.empty(n, dtype=np.int32)
    starts = np.empty(ordered_cells.size + 1, dtype=np.int32)
    piece_count = 0
    output_count = 0

    for i in range(ordered_cells.size):
        start = int(ordered_cells[i])
        if seen[start]:
            continue

        starts[piece_count] = output_count
        piece_count += 1
        head = 0
        tail = 1
        queue[0] = start
        seen[start] = True

        while head < tail:
            cell = int(queue[head])
            head += 1
            flat[output_count] = cell
            output_count += 1

            for edge in range(offsets[cell], offsets[cell + 1]):
                neighbour = int(adjacency[edge])
                if in_set[neighbour] and not seen[neighbour]:
                    seen[neighbour] = True
                    queue[tail] = neighbour
                    tail += 1

    starts[piece_count] = output_count
    return flat[:output_count], starts[:piece_count + 1]


def _largest_piece_kernel(offsets, adjacency, ordered_cells):
    """Return the first largest component, preserving pieces_of tie rules."""
    flat, starts = _pieces_of_kernel(offsets, adjacency, ordered_cells)
    best_start = 0
    best_size = 0
    for i in range(starts.size - 1):
        start = int(starts[i])
        size = int(starts[i + 1]) - start
        if size > best_size:
            best_start = start
            best_size = size
    return flat[best_start:best_start + best_size].copy()


def _tangent_basis_components(x, y, z):
    """Return east and north tangent vectors without allocating Python lists."""
    if abs(z) > 0.9:
        ex, ey, ez = 0.0, -z, y
    else:
        ex, ey, ez = -y, x, 0.0
    length = math.sqrt(ex * ex + ey * ey + ez * ez) or 1.0
    ex, ey, ez = ex / length, ey / length, ez / length
    nx = y * ez - z * ey
    ny = z * ex - x * ez
    nz = x * ey - y * ex
    return ex, ey, ez, nx, ny, nz


def _destination_many_kernel(points, bearings, distances):
    """Batch equivalent of destination(), preserving its scalar operation order."""
    count = points.shape[0]
    out = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        ex, ey, ez, nx, ny, nz = _tangent_basis_components(x, y, z)
        dx, dy = math.cos(bearings[i]), math.sin(bearings[i])
        sine, cosine = math.sin(distances[i]), math.cos(distances[i])
        tx = ex * dx + nx * dy
        ty = ey * dx + ny * dy
        tz = ez * dx + nz * dy
        qx = x * cosine + tx * sine
        qy = y * cosine + ty * sine
        qz = z * cosine + tz * sine
        length = math.sqrt(qx * qx + qy * qy + qz * qz) or 1.0
        out[i, 0] = qx / length
        out[i, 1] = qy / length
        out[i, 2] = qz / length
    return out


def _bearing_to_many_kernel(points, targets):
    """Batch equivalent of bearing_to(), with no per-point Python allocations."""
    count = points.shape[0]
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        ex, ey, ez, nx, ny, nz = _tangent_basis_components(x, y, z)
        qx, qy, qz = targets[i, 0], targets[i, 1], targets[i, 2]
        east_dot = qx * ex + qy * ey + qz * ez
        north_dot = qx * nx + qy * ny + qz * nz
        out[i] = math.atan2(north_dot, east_dot)
    return out



def _create_cells_kernel(fields, mask, cells, crust_types, formed_at, terranes):
    """Create several cells in input order, retaining repeated-cell semantics."""
    added = 0
    for row in range(cells.size):
        cell = int(cells[row])
        if not mask[cell]:
            added += 1
        mask[cell] = 1
        for field in literal_unroll(fields):
            field[cell] = 0
        fields[0][cell] = crust_types[row]
        fields[2][cell] = formed_at[row]
        fields[3][cell] = terranes[row]
    return added


def _snapshot_cells_kernel(fields, cells):
    """Materialize a batch of snapshots in row-major CELL_FIELDS order."""
    out = np.empty((cells.size, len(fields)), dtype=np.float64)
    for row in range(cells.size):
        cell = int(cells[row])
        for column in literal_unroll((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,)):
            out[row, column] = fields[column][cell]
    return out

def _zero_cell_fields_kernel(fields, cell):
    """Reset all heterogeneous structure-of-arrays fields at one cell."""
    for field in literal_unroll(fields):
        field[cell] = 0


def _snapshot_cell_fields_kernel(fields, cell):
    """Copy a cell snapshot in upstream field order into numeric storage."""
    values = np.empty(len(fields), dtype=np.float64)
    for index in literal_unroll((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,)):
        values[index] = fields[index][cell]
    return values

if NUMBA_AVAILABLE:
    noise3d_numba = njit(cache=True, fastmath=False)(_noise3d_kernel)
    # Compile dependencies first and expose their dispatchers to the next kernel.
    _noise3d_kernel = noise3d_numba
    fbm_numba = njit(cache=True, fastmath=False)(_fbm_kernel)
    _fbm_kernel = fbm_numba
    fbm_many_numba = njit(cache=True, fastmath=False)(_fbm_many_kernel)
    hop_distance_numba = njit(cache=True, fastmath=False)(_hop_distance_kernel)
    spread_km_numba = njit(cache=True, fastmath=False)(_spread_km_kernel)
    far_field_distance_numba = njit(cache=True, fastmath=False)(_far_field_distance_kernel)
    pieces_of_numba = njit(cache=True, fastmath=False)(_pieces_of_kernel)
    _pieces_of_kernel = pieces_of_numba
    largest_piece_numba = njit(cache=True, fastmath=False)(_largest_piece_kernel)
    # Compile the helper first; Numba does not automatically infer undecorated globals.
    _tangent_basis_components = njit(cache=True, fastmath=False)(_tangent_basis_components)
    destination_many_numba = njit(cache=True, fastmath=False)(_destination_many_kernel)
    bearing_to_many_numba = njit(cache=True, fastmath=False)(_bearing_to_many_kernel)
    zero_cell_fields_numba = njit(cache=True, fastmath=False)(_zero_cell_fields_kernel)
    snapshot_cell_fields_numba = njit(cache=True, fastmath=False)(_snapshot_cell_fields_kernel)
    create_cells_numba = njit(cache=True, fastmath=False)(_create_cells_kernel)
    snapshot_cells_numba = njit(cache=True, fastmath=False)(_snapshot_cells_kernel)
else:
    noise3d_numba = fbm_numba = fbm_many_numba = hop_distance_numba = spread_km_numba = far_field_distance_numba = None
    pieces_of_numba = largest_piece_numba = None
    destination_many_numba = bearing_to_many_numba = None
    zero_cell_fields_numba = snapshot_cell_fields_numba = None
    create_cells_numba = snapshot_cells_numba = None
