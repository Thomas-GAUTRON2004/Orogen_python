"""Equivalence of the dtype-block create kernel with scalar create()."""

from __future__ import annotations

import numpy as np
import pytest

from orogen.v2.core import _numba_kernels as kernels
from orogen.v2.tectonics import crust as crust_module
from orogen.v2.tectonics.crust import CELL_FIELDS, CrustRaster, CrustType

CELLS = 16


def _dirty_raster() -> CrustRaster:
    """Return a raster whose every field holds non-zero data at every cell."""
    raster = CrustRaster(CELLS)
    for index, field in enumerate(raster.fields):
        field[:] = (index % 100) + 1
    raster.mask[:] = 1
    raster.count = CELLS
    return raster


def _assert_same(left: CrustRaster, right: CrustRaster) -> None:
    assert left.count == right.count
    np.testing.assert_array_equal(left.mask, right.mask)
    for name in CELL_FIELDS:
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def test_blocks_are_the_backing_storage_of_every_field():
    raster = CrustRaster(CELLS)
    assert sum(block.shape[0] for block in raster._numba_blocks) == len(CELL_FIELDS)
    for field in raster.fields:
        assert any(np.shares_memory(field, block) for block in raster._numba_blocks)


@pytest.mark.parametrize(
    "cells", [[3, 1, 3, 7, 1], [0, 15, -1, -16, 8], list(range(CELLS))]
)
def test_create_many_matches_scalar_create_on_dirty_raster(cells):
    batch, scalar = _dirty_raster(), _dirty_raster()
    types = np.asarray([CrustType["CRATON"]] * len(cells), dtype=np.uint8)
    times = np.linspace(-50.0, -10.0, len(cells))
    terranes = np.arange(1, len(cells) + 1, dtype=np.uint16)

    batch.create_many(cells, types, times, terranes)
    for row, cell in enumerate(cells):
        scalar.create(int(cell), int(types[row]), float(times[row]), int(terranes[row]))

    _assert_same(batch, scalar)


def test_create_many_clears_every_field_except_the_three_it_writes():
    raster = _dirty_raster()
    raster.create_many([5], CrustType["ARC"], -7.5, 4)
    for name in CELL_FIELDS:
        value = getattr(raster, name)[5]
        if name == "type":
            assert value == CrustType["ARC"]
        elif name == "formedAt":
            assert value == np.float32(-7.5)
        elif name == "terrane":
            assert value == 4
        else:
            assert value == 0, name
    assert raster.type[4] != 0 or raster.formedAt[4] != 0


def test_create_many_leaves_other_cells_untouched():
    raster, reference = _dirty_raster(), _dirty_raster()
    raster.create_many([2, 9], CrustType["OCEAN"], 0.0)
    keep = [cell for cell in range(CELLS) if cell not in (2, 9)]
    for name in CELL_FIELDS:
        np.testing.assert_array_equal(
            getattr(raster, name)[keep], getattr(reference, name)[keep]
        )


def test_create_many_counts_only_new_cells_on_a_partial_raster():
    raster = CrustRaster(CELLS)
    raster.create_many([1, 2], CrustType["CONTINENT"], 0.0)
    raster.create_many([2, 3, 3], CrustType["CONTINENT"], 0.0)
    assert raster.count == 3


@pytest.mark.skipif(
    not kernels.NUMBA_AVAILABLE,
    reason="literal_unroll is only available with Numba",
)
def test_block_kernel_python_body_matches_reference_kernel():
    """Coverage-and-parity check of the uncompiled kernel bodies."""
    cells = np.asarray([0, 1, 1, 3], dtype=np.int32)
    types = np.asarray([1, 2, 3, 4], dtype=np.uint8)
    times = np.asarray([1.0, 2.0, 3.0, 4.0])
    terranes = np.asarray([4, 3, 2, 1], dtype=np.uint16)

    reference = CrustRaster(4)
    candidate = CrustRaster(4)
    for raster in (reference, candidate):
        for index, field in enumerate(raster.fields):
            field[:] = index + 1

    added_reference = kernels._create_cells_kernel(
        reference._numba_fields, reference.mask, cells, types, times, terranes
    )
    added_candidate = kernels._create_cells_blocks_kernel(
        candidate._numba_blocks,
        candidate.mask,
        cells,
        types,
        times,
        terranes,
        candidate.type,
        candidate.formedAt,
        candidate.terrane,
    )
    assert added_reference == added_candidate == 3
    _assert_same(reference, candidate)


def test_pure_python_fallback_matches_batch_result(monkeypatch):
    compiled, fallback = _dirty_raster(), _dirty_raster()
    cells = [4, 4, 10, 2]
    compiled.create_many(cells, CrustType["CONTINENT"], -3.0, [1, 2, 3, 4])
    monkeypatch.setattr(crust_module, "NUMBA_AVAILABLE", False)
    fallback.create_many(cells, CrustType["CONTINENT"], -3.0, [1, 2, 3, 4])
    _assert_same(compiled, fallback)
