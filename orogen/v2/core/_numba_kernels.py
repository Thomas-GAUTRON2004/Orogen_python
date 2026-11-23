"""Optional Numba kernels.

All kernels use float64 arithmetic, preserve the upstream operation order, and
disable fast-math. The pure-Python implementation remains the reference path.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

njit: Any
literal_unroll: Any
try:
    from numba import literal_unroll as _literal_unroll
    from numba import njit as _njit

    njit = _njit
    literal_unroll = _literal_unroll
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
    x2, y2, z2 = x0 - i2 + 2.0 * h, y0 - j2 + 2.0 * h, z0 - k2 + 2.0 * h
    x3, y3, z3 = x0 - 1.0 + 3.0 * h, y0 - 1.0 + 3.0 * h, z0 - 1.0 + 3.0 * h
    ii, jj, kk = i & 255, j & 255, k & 255
    n0 = n1 = n2 = n3 = 0.0

    a = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
    if a > 0.0:
        a *= a
        g = int(pm3[ii + int(perm[jj + int(perm[kk])])])
        n0 = a * a * (grad[g] * x0 + grad[g + 1] * y0 + grad[g + 2] * z0)

    b = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
    if b > 0.0:
        b *= b
        g = int(pm3[ii + i1 + int(perm[jj + j1 + int(perm[kk + k1])])])
        n1 = b * b * (grad[g] * x1 + grad[g + 1] * y1 + grad[g + 2] * z1)

    c = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
    if c > 0.0:
        c *= c
        g = int(pm3[ii + i2 + int(perm[jj + j2 + int(perm[kk + k2])])])
        n2 = c * c * (grad[g] * x2 + grad[g + 1] * y2 + grad[g + 2] * z2)

    d = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
    if d > 0.0:
        d *= d
        g = int(pm3[ii + 1 + int(perm[jj + 1 + int(perm[kk + 1])])])
        n3 = d * d * (grad[g] * x3 + grad[g + 1] * y3 + grad[g + 2] * z3)
    return 32.0 * (n0 + n1 + n2 + n3)


def _fbm_kernel(x, y, z, octaves, persistence, perm, pm3, grad):
    total = 0.0
    maximum = 0.0
    amplitude = 1.0
    for octave in range(octaves):
        frequency = 1 << octave
        total += amplitude * _noise3d_kernel(
            x * frequency, y * frequency, z * frequency, perm, pm3, grad
        )
        maximum += amplitude
        amplitude *= persistence
    return total / maximum


def _fbm_many_kernel(points, octaves, persistence, perm, pm3, grad):
    out = np.empty(points.shape[0], dtype=np.float64)
    for i in range(points.shape[0]):
        out[i] = _fbm_kernel(
            points[i, 0],
            points[i, 1],
            points[i, 2],
            octaves,
            persistence,
            perm,
            pm3,
            grad,
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


def _hops_from_kernel(offsets, adjacency, source_cells, max_hops):
    """Return multi-source hop distances and source ownership in queue order."""
    n = offsets.size - 1
    distance = np.full(n, -1, dtype=np.int16)
    source_of = np.full(n, -1, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    head = 0
    tail = 0

    for i in range(source_cells.size):
        cell = int(source_cells[i])
        if distance[cell] < 0:
            distance[cell] = 0
            source_of[cell] = cell
            queue[tail] = cell
            tail += 1

    while head < tail:
        cell = int(queue[head])
        head += 1
        next_depth = int(distance[cell]) + 1
        if next_depth > max_hops:
            continue
        for edge in range(offsets[cell], offsets[cell + 1]):
            neighbour = int(adjacency[edge])
            if distance[neighbour] >= 0:
                continue
            distance[neighbour] = next_depth
            source_of[neighbour] = source_of[cell]
            queue[tail] = neighbour
            tail += 1

    return distance, source_of


def _within_km_kernel(offsets, adjacency, xyz, source_mask, source_edges, min_dot):
    """Mark the union of per-source great-circle radius traversals."""
    n = source_mask.size
    reached = np.zeros(n, dtype=np.uint8)
    seen = np.full(n, -1, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)

    for source in range(n):
        if not source_mask[source]:
            continue
        reached[source] = 1
        if not source_edges[source]:
            continue

        sx = float(xyz[3 * source])
        sy = float(xyz[3 * source + 1])
        sz = float(xyz[3 * source + 2])
        head = 0
        tail = 1
        queue[0] = source
        seen[source] = source

        while head < tail:
            cell = int(queue[head])
            head += 1
            for edge in range(offsets[cell], offsets[cell + 1]):
                neighbour = int(adjacency[edge])
                if seen[neighbour] == source:
                    continue
                offset = 3 * neighbour
                cosine = (
                    float(xyz[offset]) * sx
                    + float(xyz[offset + 1]) * sy
                    + float(xyz[offset + 2]) * sz
                )
                if cosine < min_dot:
                    continue
                seen[neighbour] = source
                reached[neighbour] = 1
                queue[tail] = neighbour
                tail += 1

    return reached


def _far_field_distance_kernel(
    offsets,
    adjacency,
    xyz,
    mask,
    crust_type,
    flags,
    source_cells,
    source_works,
    strongest,
    reach_length,
    max_cost,
    craton_type,
    weak_mask,
    craton_factor,
    weak_factor,
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
                child = (
                    right
                    if right < count and heap_keys[right] < heap_keys[left]
                    else left
                )
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
                child = (
                    right
                    if right < count and heap_keys[right] < heap_keys[left]
                    else left
                )
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
    return flat[:output_count], starts[: piece_count + 1]


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
    return flat[best_start : best_start + best_size].copy()


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


def _copy_cells_kernel(
    out_fields, in_fields, mask, destinations, sources, rotation, rotate, same_raster
):
    """Copy a batch, retaining alias semantics and rotating far-field axes."""
    added = 0
    for row in range(destinations.size):
        target = int(destinations[row])
        if not mask[target]:
            added += 1
        mask[target] = 1

    if same_raster:
        # Preserve scalar copy_from ordering when reads can observe earlier writes.
        for row in range(destinations.size):
            target = int(destinations[row])
            source = int(sources[row])
            for column in literal_unroll(
                (
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                )
            ):
                out_fields[column][target] = in_fields[column][source]
            if rotate:
                x = int(out_fields[52][target])
                y = int(out_fields[53][target])
                z = int(out_fields[54][target])
                if x or y or z:
                    u = rotation[0] * x + rotation[1] * y + rotation[2] * z
                    v = rotation[3] * x + rotation[4] * y + rotation[5] * z
                    w = rotation[6] * x + rotation[7] * y + rotation[8] * z
                    length = math.sqrt(u * u + v * v + w * w) or 1.0
                    out_fields[52][target] = math.floor(127.0 * u / length + 0.5)
                    out_fields[53][target] = math.floor(127.0 * v / length + 0.5)
                    out_fields[54][target] = math.floor(127.0 * w / length + 0.5)
    else:
        # For distinct rasters, traverse each typed array contiguously.
        for column in literal_unroll(
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
            )
        ):
            destination_field = out_fields[column]
            source_field = in_fields[column]
            for row in range(destinations.size):
                destination_field[int(destinations[row])] = source_field[
                    int(sources[row])
                ]
        if rotate:
            # Only the last row for a repeated destination contains its final
            # copied axis; rotate each unique destination exactly once.
            seen = np.zeros(mask.size, dtype=np.uint8)
            for row in range(destinations.size - 1, -1, -1):
                target = int(destinations[row])
                if seen[target]:
                    continue
                seen[target] = 1
                x = int(out_fields[52][target])
                y = int(out_fields[53][target])
                z = int(out_fields[54][target])
                if x or y or z:
                    u = rotation[0] * x + rotation[1] * y + rotation[2] * z
                    v = rotation[3] * x + rotation[4] * y + rotation[5] * z
                    w = rotation[6] * x + rotation[7] * y + rotation[8] * z
                    length = math.sqrt(u * u + v * v + w * w) or 1.0
                    out_fields[52][target] = math.floor(127.0 * u / length + 0.5)
                    out_fields[53][target] = math.floor(127.0 * v / length + 0.5)
                    out_fields[54][target] = math.floor(127.0 * w / length + 0.5)
    return added


def _remove_cells_kernel(mask, cells):
    """Remove cells in order and return the number that were present."""
    removed = 0
    for row in range(cells.size):
        cell = int(cells[row])
        if mask[cell]:
            mask[cell] = 0
            removed += 1
    return removed


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


def _create_cells_blocks_kernel(
    blocks,
    mask,
    cells,
    crust_types,
    formed_at,
    terranes,
    type_field,
    formed_field,
    terrane_field,
):
    """Create several cells in input order, zeroing one dtype block at a time.

    ``blocks`` holds the per-dtype backing matrices whose rows are the crust
    fields, so clearing a cell touches a handful of matrices instead of
    unrolling over every heterogeneous field. Semantics match
    ``_create_cells_kernel``: each occurrence of a repeated cell clears it
    before the type, formation time and terrane are written.
    """
    added = 0
    for row in range(cells.size):
        cell = int(cells[row])
        if not mask[cell]:
            added += 1
        mask[cell] = 1
        for block in literal_unroll(blocks):
            for field_row in range(block.shape[0]):
                block[field_row, cell] = 0
        type_field[cell] = crust_types[row]
        formed_field[cell] = formed_at[row]
        terrane_field[cell] = terranes[row]
    return added


def _snapshot_cells_kernel(fields, cells):
    """Materialize a batch of snapshots in row-major CELL_FIELDS order."""
    out = np.empty((cells.size, len(fields)), dtype=np.float64)
    for row in range(cells.size):
        cell = int(cells[row])
        for column in literal_unroll(
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
            )
        ):
            out[row, column] = fields[column][cell]
    return out


def _zero_cell_fields_kernel(fields, cell):
    """Reset all heterogeneous structure-of-arrays fields at one cell."""
    for field in literal_unroll(fields):
        field[cell] = 0


def _snapshot_cell_fields_kernel(fields, cell):
    """Copy a cell snapshot in upstream field order into numeric storage."""
    values = np.empty(len(fields), dtype=np.float64)
    for index in literal_unroll(
        (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
        )
    ):
        values[index] = fields[index][cell]
    return values


def _record_push_kernel(
    far_work,
    far_dist,
    far_fold,
    far_at,
    far_dx,
    far_dy,
    far_dz,
    xyz,
    cell,
    felt,
    distance_km,
    source,
    time,
    fold_myr,
):
    """Numba twin of ``far_field.record_push`` for one cell.

    Storage reads go through ``np.float64``: in Numba ``float()`` of a float32
    scalar stays float32, which would round products like ``far_dist * work``
    in single precision and diverge from the Python path.
    """
    old_work = np.float64(far_work[cell])
    total_work = old_work + felt
    far_work[cell] = total_work
    far_dist[cell] = (
        np.float64(far_dist[cell]) * old_work + distance_km * felt
    ) / total_work
    elapsed = max(0.0, time - np.float64(far_at[cell]))
    previous_fold = np.float64(far_fold[cell])
    far_fold[cell] = previous_fold * math.exp(-elapsed / fold_myr) + felt
    far_at[cell] = time

    if source < 0 or source == cell:
        return
    offset = 3 * cell
    source_offset = 3 * source
    px = float(xyz[offset])
    py = float(xyz[offset + 1])
    pz = float(xyz[offset + 2])
    qx = float(xyz[source_offset])
    qy = float(xyz[source_offset + 1])
    qz = float(xyz[source_offset + 2])
    cosine = px * qx + py * qy + pz * qz
    ax = cosine * px - qx
    ay = cosine * py - qy
    az = cosine * pz - qz
    length = math.sqrt(ax * ax + ay * ay + az * az)
    if length < 1e-9:
        return
    ax = ax / length
    ay = ay / length
    az = az / length

    vx = float(far_dx[cell]) / 127.0
    vy = float(far_dy[cell]) / 127.0
    vz = float(far_dz[cell]) / 127.0
    if vx * ax + vy * ay + vz * az < 0.0:
        vx = -vx
        vy = -vy
        vz = -vz
    x = vx * old_work + ax * felt
    y = vy * old_work + ay * felt
    z = vz * old_work + az * felt
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        norm = 1.0
    far_dx[cell] = max(-127, min(127, math.floor(127.0 * x / norm + 0.5)))
    far_dy[cell] = max(-127, min(127, math.floor(127.0 * y / norm + 0.5)))
    far_dz[cell] = max(-127, min(127, math.floor(127.0 * z / norm + 0.5)))


def _emit_far_pushes_kernel(
    reached,
    distance,
    source_of,
    crust_type,
    crust_flags,
    xyz,
    far_work,
    far_dist,
    far_fold,
    far_at,
    far_dx,
    far_dy,
    far_dz,
    strongest,
    reach_length,
    time,
    fold_myr,
    craton_take,
    weak_take,
    craton_type,
    weak_mask,
    ocean_type,
    out_cells,
    out_inc,
    out_from,
    out_dist,
):
    """Apply arrivals to reached cells in order; return how many were written.

    Mirrors the Python emission loop of ``far_field.spread_push``: ocean cells
    and arrivals at or below 1e-4 are skipped, every other arrival is recorded
    and appended to the output arrays.
    """
    written = 0
    for index in range(reached.size):
        cell = reached[index]
        if crust_type[cell] == ocean_type:
            continue
        dist = distance[cell]
        arrived = strongest * math.exp(-dist / reach_length)
        if not arrived > 1e-4:
            continue
        if crust_type[cell] == craton_type:
            take = craton_take
        elif (int(crust_flags[cell]) & weak_mask) != 0:
            take = weak_take
        else:
            take = 1.0
        source = source_of[cell]
        _record_push_kernel(
            far_work,
            far_dist,
            far_fold,
            far_at,
            far_dx,
            far_dy,
            far_dz,
            xyz,
            cell,
            arrived,
            dist,
            source,
            time,
            fold_myr,
        )
        out_cells[written] = cell
        out_inc[written] = arrived * take
        out_from[written] = source
        out_dist[written] = dist
        written += 1
    return written


def _walk_kernel(xyz, offsets, adjacency, current, x, y, z):
    """Greedy walk toward a direction; numerically identical to SphereIndex.walk.

    The neighbour range of the current cell is read once per pass, as in the
    Python loop, so a move during a pass does not change which neighbours are
    scanned in that pass.
    """
    offset = 3 * current
    best = x * xyz[offset] + y * xyz[offset + 1] + z * xyz[offset + 2]
    improved = True
    while improved:
        improved = False
        low = offsets[current]
        high = offsets[current + 1]
        for k in range(low, high):
            neighbour = adjacency[k]
            offset = 3 * neighbour
            candidate = x * xyz[offset] + y * xyz[offset + 1] + z * xyz[offset + 2]
            if candidate > best:
                best = candidate
                current = neighbour
                improved = True
    return current


# Dispatcher globals are intentionally optional when Numba is not installed.
noise3d_numba: Any
fbm_numba: Any
fbm_many_numba: Any
hop_distance_numba: Any
hops_from_numba: Any
within_km_numba: Any
spread_km_numba: Any
far_field_distance_numba: Any
emit_far_pushes_numba: Any
walk_numba: Any
pieces_of_numba: Any
largest_piece_numba: Any
destination_many_numba: Any
bearing_to_many_numba: Any
zero_cell_fields_numba: Any
snapshot_cell_fields_numba: Any
create_cells_numba: Any
create_cells_blocks_numba: Any
snapshot_cells_numba: Any
copy_cells_numba: Any
remove_cells_numba: Any


# --- Simplex noise (upstream js/simplex-noise.js): 12 gradients, fBm, ridged fBm. ---
SIMPLEX_GRADIENTS = np.array(
    [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ],
    dtype=np.float64,
)


def _simplex_noise3d_kernel(x, y, z, perm, pm12, grad):
    """One simplex sample; same operation order as upstream `noise3D`."""
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
    x2, y2, z2 = x0 - i2 + 2.0 * h, y0 - j2 + 2.0 * h, z0 - k2 + 2.0 * h
    x3, y3, z3 = x0 - 1.0 + 3.0 * h, y0 - 1.0 + 3.0 * h, z0 - 1.0 + 3.0 * h
    ii, jj, kk = i & 255, j & 255, k & 255

    n0 = 0.0
    a = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
    if a > 0.0:
        a *= a
        g = pm12[ii + perm[jj + perm[kk]]]
        n0 = a * a * (grad[g, 0] * x0 + grad[g, 1] * y0 + grad[g, 2] * z0)
    n1 = 0.0
    b = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
    if b > 0.0:
        b *= b
        g = pm12[ii + i1 + perm[jj + j1 + perm[kk + k1]]]
        n1 = b * b * (grad[g, 0] * x1 + grad[g, 1] * y1 + grad[g, 2] * z1)
    n2 = 0.0
    c = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
    if c > 0.0:
        c *= c
        g = pm12[ii + i2 + perm[jj + j2 + perm[kk + k2]]]
        n2 = c * c * (grad[g, 0] * x2 + grad[g, 1] * y2 + grad[g, 2] * z2)
    n3 = 0.0
    d = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
    if d > 0.0:
        d *= d
        g = pm12[ii + 1 + perm[jj + 1 + perm[kk + 1]]]
        n3 = d * d * (grad[g, 0] * x3 + grad[g, 1] * y3 + grad[g, 2] * z3)
    return 32.0 * (n0 + n1 + n2 + n3)


def _simplex_fbm_kernel(x, y, z, octaves, persistence, perm, pm12, grad):
    total = 0.0
    peak = 0.0
    amp = 1.0
    for o in range(octaves):
        f = 1 << o
        total += amp * _simplex_noise3d_kernel(x * f, y * f, z * f, perm, pm12, grad)
        peak += amp
        amp *= persistence
    return total / peak


def _simplex_ridged_kernel(
    x, y, z, octaves, lacunarity, gain, offset, perm, pm12, grad
):
    total = 0.0
    freq = 1.0
    amp = 1.0
    prev = 1.0
    peak = 0.0
    for _ in range(octaves):
        n = _simplex_noise3d_kernel(x * freq, y * freq, z * freq, perm, pm12, grad)
        n = offset - abs(n)
        n = n * n
        total += n * amp * prev
        peak += amp
        prev = min(n, 1.0)
        freq *= lacunarity
        amp *= gain
    return total / peak


simplex_noise3d_py = _simplex_noise3d_kernel
simplex_fbm_py = _simplex_fbm_kernel
simplex_ridged_py = _simplex_ridged_kernel

if NUMBA_AVAILABLE:
    noise3d_numba = njit(cache=True, fastmath=False)(_noise3d_kernel)
    # Compile dependencies first and expose their dispatchers to the next kernel.
    _noise3d_kernel = noise3d_numba
    fbm_numba = njit(cache=True, fastmath=False)(_fbm_kernel)
    _fbm_kernel = fbm_numba
    fbm_many_numba = njit(cache=True, fastmath=False)(_fbm_many_kernel)
    hop_distance_numba = njit(cache=True, fastmath=False)(_hop_distance_kernel)
    hops_from_numba = njit(cache=True, fastmath=False)(_hops_from_kernel)
    within_km_numba = njit(cache=True, fastmath=False)(_within_km_kernel)
    spread_km_numba = njit(cache=True, fastmath=False)(_spread_km_kernel)
    far_field_distance_numba = njit(cache=True, fastmath=False)(
        _far_field_distance_kernel
    )
    # Helper dispatchers must exist before the kernel that calls them is compiled.
    _record_push_kernel = njit(cache=True, fastmath=False)(_record_push_kernel)
    emit_far_pushes_numba = njit(cache=True, fastmath=False)(_emit_far_pushes_kernel)
    walk_numba = njit(cache=True, fastmath=False)(_walk_kernel)
    pieces_of_numba = njit(cache=True, fastmath=False)(_pieces_of_kernel)
    _pieces_of_kernel = pieces_of_numba
    largest_piece_numba = njit(cache=True, fastmath=False)(_largest_piece_kernel)
    # Compile the helper first; Numba does not automatically infer undecorated globals.
    _tangent_basis_components = njit(cache=True, fastmath=False)(
        _tangent_basis_components
    )
    destination_many_numba = njit(cache=True, fastmath=False)(_destination_many_kernel)
    bearing_to_many_numba = njit(cache=True, fastmath=False)(_bearing_to_many_kernel)
    zero_cell_fields_numba = njit(cache=True, fastmath=False)(_zero_cell_fields_kernel)
    snapshot_cell_fields_numba = njit(cache=True, fastmath=False)(
        _snapshot_cell_fields_kernel
    )
    create_cells_numba = njit(cache=True, fastmath=False)(_create_cells_kernel)
    create_cells_blocks_numba = njit(cache=True, fastmath=False)(
        _create_cells_blocks_kernel
    )
    snapshot_cells_numba = njit(cache=True, fastmath=False)(_snapshot_cells_kernel)
    copy_cells_numba = njit(cache=True, fastmath=False)(_copy_cells_kernel)
    remove_cells_numba = njit(cache=True, fastmath=False)(_remove_cells_kernel)
else:
    noise3d_numba = fbm_numba = fbm_many_numba = hop_distance_numba = (
        spread_km_numba
    ) = far_field_distance_numba = walk_numba = emit_far_pushes_numba = None
    hops_from_numba = within_km_numba = None
    pieces_of_numba = largest_piece_numba = None
    destination_many_numba = bearing_to_many_numba = None
    zero_cell_fields_numba = snapshot_cell_fields_numba = None
    create_cells_numba = create_cells_blocks_numba = snapshot_cells_numba = None
    copy_cells_numba = remove_cells_numba = None

if NUMBA_AVAILABLE:
    simplex_noise3d_numba = njit(cache=True, fastmath=False)(simplex_noise3d_py)
    # fBm and ridged fBm call the sample by global name: rebind it to the compiled
    # sample so that Numba compiles the whole chain (same pattern as _noise3d_kernel).
    _simplex_noise3d_kernel = simplex_noise3d_numba
    simplex_fbm_numba = njit(cache=True, fastmath=False)(simplex_fbm_py)
    simplex_ridged_numba = njit(cache=True, fastmath=False)(simplex_ridged_py)
else:
    # Keep the optional simplex dispatchers importable when the speed extra is absent.
    simplex_noise3d_numba = simplex_fbm_numba = simplex_ridged_numba = None
