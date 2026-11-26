"""Exercise the Numba kernel source through Python dispatchers as well as JIT paths.

Numba JIT coverage reports executed source lines; py_func execution supplies
coverage.py with Python branch arcs for the same kernel bodies.
"""

import math

import numpy as np
import pytest

from orogen.v2.core import _numba_kernels as kernels

pytestmark = pytest.mark.skipif(
    not kernels.NUMBA_AVAILABLE,
    reason="install orogen[speed] to measure Numba kernel coverage",
)


def _graph():
    return (
        np.asarray([0, 1, 3, 4], dtype=np.int32),
        np.asarray([1, 0, 2, 1], dtype=np.int32),
    )


def test_numba_kernel_python_bodies_cover_noise_geometry_and_graph_branches():
    perm = np.arange(512, dtype=np.uint8)
    pm3 = (3 * (perm % 12)).astype(np.uint8)
    grad = np.asarray([0.1, 0.2, 0.3] * 12, dtype=np.float64)
    points = [
        (0.1, 0.2, 0.3),
        (0.3, 0.2, 0.1),
        (-0.4, 0.8, -0.2),
        (2.0, -3.0, 1.0),
        (0.0, 0.0, 0.0),
    ]
    for point in points:
        kernels.noise3d_numba.py_func(*point, perm, pm3, grad)
    kernels.fbm_numba.py_func(0.1, 0.2, 0.3, 3, 0.6, perm, pm3, grad)
    kernels.fbm_many_numba.py_func(
        np.asarray(points, dtype=np.float64), 3, 0.6, perm, pm3, grad
    )
    offsets, adjacency = _graph()
    sources = np.asarray([True, False, False], dtype=np.bool_)
    kernels.hop_distance_numba.py_func(offsets, adjacency, sources, 0)
    kernels.hop_distance_numba.py_func(offsets, adjacency, sources, 3)
    edge_km = np.ones(4, dtype=np.float64)
    distance = np.asarray([0.0, math.inf, math.inf], dtype=np.float64)
    kernels.spread_km_numba.py_func(offsets, adjacency, edge_km, distance.copy(), 0.5)
    kernels.spread_km_numba.py_func(offsets, adjacency, edge_km, distance.copy(), 10.0)
    selected = np.asarray([0, 1, 2], dtype=np.int32)
    flat, starts = kernels.pieces_of_numba.py_func(offsets, adjacency, selected)
    assert flat.tolist() == [0, 1, 2] and starts.tolist() == [0, 3]
    assert kernels.largest_piece_numba.py_func(
        offsets, adjacency, selected
    ).tolist() == [0, 1, 2]
    assert (
        kernels.pieces_of_numba.py_func(
            offsets, adjacency, np.asarray([], dtype=np.int32)
        )[0].size
        == 0
    )
    kernels._tangent_basis_components.py_func(1.0, 0.0, 0.95)
    kernels._tangent_basis_components.py_func(1.0, 0.0, 0.0)
    points_array = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    targets = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    kernels.destination_many_numba.py_func(
        points_array, np.asarray([0.0, 1.0]), np.asarray([0.1, 0.2])
    )
    kernels.bearing_to_many_numba.py_func(points_array, targets)


def test_numba_kernel_python_bodies_cover_crust_array_mutation_paths():
    """Coverage-only smoke test for raster mutation branches; state invariants are asserted in the raster tests."""
    fields = [np.zeros(4, dtype=np.int32) for _ in range(55)]
    mask = np.zeros(4, dtype=np.bool_)
    cells = np.asarray([0, 1, 1, 3], dtype=np.int32)
    kernels.create_cells_numba.py_func(
        fields,
        mask,
        cells,
        np.asarray([1, 2, 3, 4]),
        np.asarray([1, 2, 3, 4]),
        np.asarray([4, 3, 2, 1]),
    )
    kernels.snapshot_cells_numba.py_func(fields, cells)
    kernels.snapshot_cell_fields_numba.py_func(fields, 1)
    kernels.zero_cell_fields_numba.py_func(fields, 1)
    fields[52][0] = 127
    rotation = np.eye(3, dtype=np.float64).ravel()
    destinations = np.asarray([1, 2], dtype=np.int32)
    sources = np.asarray([0, 0], dtype=np.int32)
    kernels.copy_cells_numba.py_func(
        fields, fields, mask, destinations, sources, rotation, True, True
    )
    out_fields = [np.zeros(4, dtype=np.int32) for _ in range(55)]
    out_mask = np.zeros(4, dtype=np.bool_)
    kernels.copy_cells_numba.py_func(
        out_fields, fields, out_mask, destinations, sources, rotation, True, False
    )
    kernels.copy_cells_numba.py_func(
        out_fields, fields, out_mask, destinations, sources, rotation, False, False
    )
    kernels.remove_cells_numba.py_func(out_mask, np.asarray([1, 1, 3], dtype=np.int32))
    kernels.remove_cells_numba.py_func(out_mask, np.asarray([], dtype=np.int32))


def test_numba_weighted_distance_kernel_python_body_covers_cost_and_mask_paths():
    offsets, adjacency = _graph()
    xyz = np.asarray(
        [
            1.0,
            0.0,
            0.0,
            math.cos(0.01),
            math.sin(0.01),
            0.0,
            math.cos(0.02),
            math.sin(0.02),
            0.0,
        ],
        dtype=np.float64,
    )
    mask = np.asarray([True, True, True], dtype=np.bool_)
    crust_type = np.asarray([0, 1, 0], dtype=np.int32)
    flags = np.asarray([0, 0, 2], dtype=np.int32)
    source_cells = np.asarray([0], dtype=np.int32)
    source_works = np.asarray([100.0], dtype=np.float64)
    result = kernels.far_field_distance_numba.py_func(
        offsets,
        adjacency,
        xyz,
        mask,
        crust_type,
        flags,
        source_cells,
        source_works,
        100.0,
        1000.0,
        1000.0,
        1,
        2,
        0.3,
        1.6,
    )
    assert result[3] > 0
    blocked = mask.copy()
    blocked[1] = False
    kernels.far_field_distance_numba.py_func(
        offsets,
        adjacency,
        xyz,
        blocked,
        crust_type,
        flags,
        source_cells,
        source_works,
        100.0,
        1000.0,
        1.0,
        1,
        2,
        0.3,
        1.6,
    )


def test_numba_kernel_heap_and_copy_branches():
    """Coverage-only branch exercise; observable copy semantics are checked in the Numba parity tests."""
    offsets = np.asarray([0, 2, 4, 5, 6], dtype=np.int32)
    adjacency = np.asarray([1, 2, 0, 3, 1, 2], dtype=np.int32)
    xyz = np.asarray(
        [
            1.0,
            0.0,
            0.0,
            math.cos(0.01),
            math.sin(0.01),
            0.0,
            math.cos(0.02),
            math.sin(0.02),
            0.0,
            math.cos(0.03),
            math.sin(0.03),
            0.0,
        ],
        dtype=np.float64,
    )
    mask = np.ones(4, dtype=np.bool_)
    crust_type = np.asarray([0, 1, 0, 0], dtype=np.int32)
    flags = np.asarray([0, 0, 2, 0], dtype=np.int32)
    sources = np.asarray([0, 3], dtype=np.int32)
    works = np.asarray([100.0, 90.0], dtype=np.float64)
    kernels.far_field_distance_numba.py_func(
        offsets,
        adjacency,
        xyz,
        mask,
        crust_type,
        flags,
        sources,
        works,
        100.0,
        1000.0,
        1000.0,
        1,
        2,
        0.3,
        1.6,
    )

    distances = np.asarray([0.0, math.inf, math.inf, 0.0], dtype=np.float64)
    kernels.spread_km_numba.py_func(
        offsets, adjacency, np.ones(adjacency.size), distances, 100.0
    )
    kernels.spread_km_numba.py_func(
        offsets, adjacency, np.ones(adjacency.size), np.full(4, math.inf), 100.0
    )

    fields = [np.zeros(4, dtype=np.int32) for _ in range(55)]
    fields[52][0] = 127
    mask_a = np.zeros(4, dtype=np.bool_)
    destinations = np.asarray([1, 1, 2, 3], dtype=np.int32)
    source_cells = np.asarray([0, 0, 3, 3], dtype=np.int32)
    # Same-raster path with rotation disabled.
    kernels.copy_cells_numba.py_func(
        fields,
        fields,
        mask_a,
        np.asarray([1], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.eye(3).ravel(),
        False,
        True,
    )
    # Distinct-raster path: duplicate destination, nonzero and zero axes.
    out = [np.zeros(4, dtype=np.int32) for _ in range(55)]
    kernels.copy_cells_numba.py_func(
        out,
        fields,
        np.zeros(4, dtype=np.bool_),
        destinations,
        source_cells,
        np.eye(3).ravel(),
        True,
        False,
    )


def test_noise_kernel_all_simplex_orderings_and_corner_cutoffs():
    """Coverage-only exercise of every simplex ordering and corner cutoff."""
    from itertools import product

    perm = np.arange(512, dtype=np.uint8)
    pm3 = (3 * (perm % 12)).astype(np.uint8)
    grad = np.asarray([0.1, 0.2, 0.3] * 12, dtype=np.float64)
    kernel = kernels.noise3d_numba.py_func
    for point in product((-1.1, -0.4, -0.1, 0.2, 0.7, 1.3), repeat=3):
        kernel(*point, perm, pm3, grad)


def test_optional_numba_import_fallback(monkeypatch):
    import builtins
    import importlib

    original_import = builtins.__import__

    def without_numba(name, *args, **kwargs):
        if name == "numba":
            raise ImportError("Numba intentionally disabled for fallback test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_numba)
    try:
        importlib.reload(kernels)
        assert not kernels.NUMBA_AVAILABLE
        assert kernels.noise3d_numba is None
        assert kernels.copy_cells_numba is None
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)
        importlib.reload(kernels)


def test_numba_kernel_heap_stale_entry_and_zero_axis_branches():
    """Coverage-only exercise of stale-heap and zero-axis branches."""
    # Source insertion must bubble a lower initial Dijkstra key to the root.
    offsets = np.asarray([0, 2, 3, 4, 4], dtype=np.int32)
    adjacency = np.asarray([1, 2, 3, 1], dtype=np.int32)
    xyz = np.asarray(
        [
            1.0,
            0.0,
            0.0,
            math.cos(0.03),
            math.sin(0.03),
            0.0,
            math.cos(0.01),
            math.sin(0.01),
            0.0,
            math.cos(0.02),
            math.sin(0.02),
            0.0,
        ],
        dtype=np.float64,
    )
    mask = np.ones(4, dtype=np.bool_)
    crust = np.zeros(4, dtype=np.int32)
    flags = np.asarray([0, 2, 0, 0], dtype=np.int32)
    kernels.far_field_distance_numba.py_func(
        offsets,
        adjacency,
        xyz,
        mask,
        crust,
        flags,
        np.asarray([0, 3], dtype=np.int32),
        np.asarray([10.0, 100.0], dtype=np.float64),
        100.0,
        1000.0,
        1000.0,
        1,
        2,
        0.3,
        5.0,
    )

    # Multiple initial seeds exercise heap insertion/sift-down; a later
    # shorter route leaves a stale heap entry that must be discarded.
    spread_offsets = np.asarray([0, 2, 3, 4, 4], dtype=np.int32)
    spread_adjacency = np.asarray([1, 2, 1, 1], dtype=np.int32)
    edge_km = np.asarray([10.0, 1.0, 1.0, 1.0], dtype=np.float64)
    seeded = np.asarray([10.0, 0.0, math.inf, math.inf], dtype=np.float64)
    kernels.spread_km_numba.py_func(
        spread_offsets,
        spread_adjacency,
        edge_km,
        seeded,
        100.0,
    )
    kernels.spread_km_numba.py_func(
        spread_offsets,
        spread_adjacency,
        edge_km,
        np.asarray([0.0, math.inf, math.inf, math.inf]),
        100.0,
    )

    # Distinct raster + rotation with a zero axis takes the no-rotation branch.
    fields = [np.zeros(2, dtype=np.int32) for _ in range(55)]
    out = [np.zeros(2, dtype=np.int32) for _ in range(55)]
    kernels.copy_cells_numba.py_func(
        out,
        fields,
        np.zeros(2, dtype=np.bool_),
        np.asarray([1], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.eye(3).ravel(),
        True,
        False,
    )


def test_numba_piece_kernel_empty_and_multiple_component_size_branches():
    offsets = np.asarray([0, 1, 2, 2, 2], dtype=np.int32)
    adjacency = np.asarray([1, 0], dtype=np.int32)
    selected = np.asarray([2, 0, 1, 3], dtype=np.int32)
    pieces = kernels.pieces_of_numba.py_func(offsets, adjacency, selected)
    assert pieces[1].tolist() == [0, 1, 3, 4]
    # A later, strictly larger component replaces the first candidate.
    assert kernels.largest_piece_numba.py_func(
        offsets,
        adjacency,
        selected,
    ).tolist() == [0, 1]


def test_numba_weighted_heap_sift_down_and_stale_source_entries():
    """Coverage-only exercise of heap sift-down and stale-source branches."""
    offsets = np.zeros(5, dtype=np.int32)
    adjacency = np.asarray([], dtype=np.int32)
    xyz = np.tile(np.asarray([1.0, 0.0, 0.0]), 4)
    mask = np.ones(4, dtype=np.bool_)
    crust = np.zeros(4, dtype=np.int32)
    flags = np.zeros(4, dtype=np.int32)

    # Insert keys in a non-heap order. Popping the minimum must sift the
    # final key down through the left child.
    kernels.far_field_distance_numba.py_func(
        offsets,
        adjacency,
        xyz,
        mask,
        crust,
        flags,
        np.asarray([0, 1, 2, 3], dtype=np.int32),
        np.asarray([math.exp(-3), math.exp(-1), math.exp(-2), 1.0]),
        1.0,
        1.0,
        100.0,
        1,
        2,
        1.0,
        1.0,
    )

    # Repeated source cells can leave an obsolete larger key in the heap;
    # it must be discarded after the better source entry is processed.
    kernels.far_field_distance_numba.py_func(
        offsets,
        adjacency,
        xyz,
        mask,
        crust,
        flags,
        np.asarray([0, 0, 0], dtype=np.int32),
        np.asarray([math.exp(-3), 1.0, 1.0]),
        1.0,
        1.0,
        100.0,
        1,
        2,
        1.0,
        1.0,
    )


def test_numba_spread_heap_sifts_seeded_keys_down():
    """Coverage-only exercise of seeded-key heap reordering."""
    offsets = np.zeros(5, dtype=np.int32)
    kernels.spread_km_numba.py_func(
        offsets,
        np.asarray([], dtype=np.int32),
        np.asarray([], dtype=np.float64),
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        10.0,
    )


def test_numba_same_raster_rotation_skips_zero_axis():
    fields = [np.zeros(3, dtype=np.int32) for _ in range(55)]
    fields[52][0] = 127
    kernels.copy_cells_numba.py_func(
        fields,
        fields,
        np.zeros(3, dtype=np.bool_),
        np.asarray([1], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.eye(3).ravel(),
        True,
        True,
    )
    assert fields[52][1] == 0
    assert fields[53][1] == 0
    assert fields[54][1] == 0


def test_numba_mesh_bfs_python_bodies_cover_new_kernel_branches():
    """Exercise the new BFS bodies as Python functions for branch coverage."""
    offsets, adjacency = _graph()
    source_cells = np.asarray([0, 2, 0], dtype=np.int32)

    distance, source_of = kernels.hops_from_numba.py_func(
        offsets, adjacency, source_cells, 1
    )
    assert distance.tolist() == [0, 1, 0]
    assert source_of.tolist() == [0, 0, 2]

    cutoff_distance, cutoff_source = kernels.hops_from_numba.py_func(
        offsets, adjacency, source_cells, 0
    )
    assert cutoff_distance.tolist() == [0, -1, 0]
    assert cutoff_source.tolist() == [0, -1, 2]

    empty_distance, empty_source = kernels.hops_from_numba.py_func(
        offsets, adjacency, np.asarray([], dtype=np.int32), 3
    )
    assert np.all(empty_distance == -1)
    assert np.all(empty_source == -1)

    xyz = np.asarray(
        [
            1.0,
            0.0,
            0.0,
            math.cos(0.01),
            math.sin(0.01),
            0.0,
            math.cos(0.02),
            math.sin(0.02),
            0.0,
        ],
        dtype=np.float64,
    )
    radius_reached = kernels.within_km_numba.py_func(
        offsets,
        adjacency,
        xyz,
        np.asarray([True, False, False], dtype=np.bool_),
        np.asarray([True, False, False], dtype=np.bool_),
        math.cos(0.015),
    )
    assert radius_reached.tolist() == [1, 1, 0]

    # Sources without a non-source boundary are marked but not traversed.
    source_only = kernels.within_km_numba.py_func(
        offsets,
        adjacency,
        xyz,
        np.asarray([True, False, False], dtype=np.bool_),
        np.asarray([False, False, False], dtype=np.bool_),
        math.cos(0.015),
    )
    assert source_only.tolist() == [1, 0, 0]

    no_sources = kernels.within_km_numba.py_func(
        offsets,
        adjacency,
        xyz,
        np.zeros(3, dtype=np.bool_),
        np.zeros(3, dtype=np.bool_),
        math.cos(0.015),
    )
    assert no_sources.tolist() == [0, 0, 0]


def test_numba_record_push_python_body_covers_source_and_vector_guards():
    def record(source, source_xyz, far_axis=(0, 127, 0), felt=1.0):
        far_work = np.asarray([10.0, 0.0], dtype=np.float64)
        far_dist = np.asarray([5.0, 0.0], dtype=np.float64)
        far_fold = np.asarray([2.0, 0.0], dtype=np.float64)
        far_at = np.asarray([0.0, 0.0], dtype=np.float64)
        far_dx = np.asarray([far_axis[0], 0], dtype=np.int8)
        far_dy = np.asarray([far_axis[1], 0], dtype=np.int8)
        far_dz = np.asarray([far_axis[2], 0], dtype=np.int8)
        xyz = np.asarray([1.0, 0.0, 0.0, *source_xyz], dtype=np.float64)
        kernels._record_push_kernel.py_func(
            far_work,
            far_dist,
            far_fold,
            far_at,
            far_dx,
            far_dy,
            far_dz,
            xyz,
            0,
            felt,
            25.0,
            source,
            10.0,
            100.0,
        )
        return far_dx, far_dy, far_dz

    # Missing and self sources return after updating the scalar accumulators.
    record(-1, (0.0, 1.0, 0.0))
    record(0, (0.0, 1.0, 0.0))

    # Nearly coincident positions produce a degenerate tangent direction.
    record(1, (1.0, 1e-12, 0.0))

    # A non-aligned stored axis is flipped toward the source tangent.
    dx, dy, dz = record(1, (0.0, 1.0, 0.0))
    assert (int(dx[0]), int(dy[0]), int(dz[0])) == (0, -127, 0)

    # Zero work and a zero stored axis exercise the zero-norm fallback.
    dx, dy, dz = record(1, (0.0, 1.0, 0.0), far_axis=(0, 0, 0), felt=0.0)
    assert (int(dx[0]), int(dy[0]), int(dz[0])) == (0, 0, 0)


def test_numba_emit_far_pushes_python_body_covers_filter_and_take_paths():
    reached = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    distance = np.asarray([0.0, 0.0, 0.0, 0.0, 1000.0], dtype=np.float64)
    source_of = np.asarray([0, 0, 2, 0, 0], dtype=np.int32)
    crust_type = np.asarray([0, 1, 2, 1, 1], dtype=np.int32)
    crust_flags = np.asarray([0, 0, 0, 4, 0], dtype=np.int32)
    xyz = np.asarray(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
        ],
        dtype=np.float64,
    )
    far_work = np.zeros(5, dtype=np.float64)
    far_dist = np.zeros(5, dtype=np.float64)
    far_fold = np.zeros(5, dtype=np.float64)
    far_at = np.zeros(5, dtype=np.float64)
    far_dx = np.zeros(5, dtype=np.int8)
    far_dy = np.zeros(5, dtype=np.int8)
    far_dz = np.zeros(5, dtype=np.int8)
    out_cells = np.empty(5, dtype=np.int32)
    out_inc = np.empty(5, dtype=np.float64)
    out_from = np.empty(5, dtype=np.int32)
    out_dist = np.empty(5, dtype=np.float64)

    written = kernels.emit_far_pushes_numba.py_func(
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
        1.0,
        1.0,
        10.0,
        100.0,
        0.25,
        1.5,
        2,
        4,
        0,
        out_cells,
        out_inc,
        out_from,
        out_dist,
    )

    # Ocean and sub-threshold arrivals are filtered; normal, craton, and weak
    # cells exercise all three take-factor branches.
    assert written == 3
    assert out_cells[:written].tolist() == [1, 2, 3]
    np.testing.assert_allclose(out_inc[:written], [1.0, 0.25, 1.5])


def test_hypot3_python_body_covers_zero_and_scaled_norm():
    hypot3 = kernels._hypot3.py_func

    assert hypot3(0.0, 0.0, 0.0) == 0.0
    assert hypot3(3.0, 4.0, 12.0) == 13.0
    assert math.isclose(
        hypot3(1e200, 2e200, 2e200),
        3e200,
        rel_tol=1e-15,
    )
