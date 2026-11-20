"""Parity of the compiled far-field emission loop with the Python reference."""

from __future__ import annotations

import math

import numpy as np
import pytest

from orogen.v2.core import _numba_kernels as kernels
from orogen.v2.tectonics import far_field
from orogen.v2.tectonics.crust import CELL_FIELDS, CrustRaster, CrustType

pytestmark = pytest.mark.skipif(
    not kernels.NUMBA_AVAILABLE, reason="emission kernel needs Numba"
)

FIELDS = ("farWork", "farDist", "farFold", "farAt", "farDx", "farDy", "farDz")


class _Mesh:
    def __init__(self, xyz: np.ndarray) -> None:
        self.xyz = xyz
        self.num_cells = xyz.size // 3


def _random_state(seed: int, count: int = 24):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * math.pi, count)
    lift = rng.uniform(-0.6, 0.6, count)
    xyz = np.empty(3 * count)
    xyz[0::3] = np.cos(angle) * np.cos(lift)
    xyz[1::3] = np.sin(angle) * np.cos(lift)
    xyz[2::3] = np.sin(lift)
    crust = CrustRaster(count)
    crust.mask[:] = (rng.random(count) < 0.85).astype(np.uint8)
    crust.type[:] = rng.integers(0, 4, count).astype(np.uint8)
    crust.flags[:] = rng.integers(0, 16, count).astype(np.uint8)
    for name in FIELDS:
        getattr(crust, name)[:] = rng.uniform(0.0, 50.0, count).astype(
            getattr(crust, name).dtype
        )
    crust.farDx[:] = rng.integers(-127, 128, count).astype(np.int8)
    crust.farDy[:] = rng.integers(-127, 128, count).astype(np.int8)
    crust.farDz[:] = rng.integers(-127, 128, count).astype(np.int8)
    return _Mesh(xyz), crust, rng


def _python_emit(mesh, crust, reached, distance, source_of, strongest, reach, time):
    config = {"farFoldMyr": 100.0, "farCratonTake": 0.15, "farWeakTake": 1.6}
    out = []
    for cell in reached:
        cell = int(cell)
        if crust.type[cell] == CrustType["OCEAN"]:
            continue
        arrived = strongest * math.exp(-float(distance[cell]) / reach)
        if not arrived > 1e-4:
            continue
        inc = arrived * far_field.far_take(
            config, int(crust.type[cell]), int(crust.flags[cell])
        )
        far_field.record_push(
            crust,
            cell,
            arrived,
            float(distance[cell]),
            mesh.xyz,
            int(source_of[cell]),
            time,
            100.0,
        )
        out.append((cell, inc, int(source_of[cell]), float(distance[cell])))
    return out


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("implementation", ["compiled", "python_body"])
def test_compiled_emission_matches_python_record_loop(seed, implementation):
    mesh, reference, rng = _random_state(seed)
    candidate = CrustRaster(mesh.num_cells)
    for name in CELL_FIELDS:
        getattr(candidate, name)[:] = getattr(reference, name)

    count = mesh.num_cells
    reached = rng.permutation(count).astype(np.int32)[: int(rng.integers(5, count))]
    distance = rng.uniform(0.0, 30_000.0, count)
    source_of = rng.integers(-1, count, count).astype(np.int32)
    strongest = float(rng.uniform(10.0, 200.0))
    reach = float(rng.uniform(5_000.0, 25_000.0))
    time = float(rng.uniform(0.0, 50.0))

    expected = _python_emit(
        mesh, reference, reached, distance, source_of, strongest, reach, time
    )

    size = reached.size
    out_cells = np.empty(size, dtype=np.int32)
    out_inc = np.empty(size, dtype=np.float32)
    out_from = np.empty(size, dtype=np.int32)
    out_dist = np.empty(size, dtype=np.float64)
    emit = (
        kernels.emit_far_pushes_numba
        if implementation == "compiled"
        else kernels.emit_far_pushes_numba.py_func
    )
    written = emit(
        reached,
        distance,
        source_of,
        candidate.type,
        candidate.flags,
        mesh.xyz,
        candidate.farWork,
        candidate.farDist,
        candidate.farFold,
        candidate.farAt,
        candidate.farDx,
        candidate.farDy,
        candidate.farDz,
        strongest,
        reach,
        time,
        100.0,
        0.15,
        1.6,
        CrustType["CRATON"],
        far_field._WEAK,
        CrustType["OCEAN"],
        out_cells,
        out_inc,
        out_from,
        out_dist,
    )

    assert written == len(expected)
    for row, (cell, inc, source, dist) in enumerate(expected):
        assert out_cells[row] == cell
        assert out_from[row] == source
        assert out_dist[row] == dist
        assert out_inc[row] == np.float32(inc)
    for name in CELL_FIELDS:
        np.testing.assert_array_equal(
            getattr(candidate, name), getattr(reference, name), err_msg=name
        )


def test_emission_skips_ocean_and_negligible_arrivals():
    mesh, crust, _ = _random_state(3, count=4)
    crust.type[:] = CrustType["OCEAN"]
    crust.type[1] = CrustType["CONTINENT"]
    distance = np.array([0.0, 0.0, 0.0, 1e9])
    source_of = np.full(4, -1, dtype=np.int32)
    reached = np.array([0, 1, 3], dtype=np.int32)
    out = [np.empty(3, dtype=t) for t in (np.int32, np.float32, np.int32, np.float64)]
    written = kernels.emit_far_pushes_numba(
        reached,
        distance,
        source_of,
        crust.type,
        crust.flags,
        mesh.xyz,
        crust.farWork,
        crust.farDist,
        crust.farFold,
        crust.farAt,
        crust.farDx,
        crust.farDy,
        crust.farDz,
        10.0,
        1000.0,
        0.0,
        100.0,
        0.15,
        1.6,
        CrustType["CRATON"],
        far_field._WEAK,
        CrustType["OCEAN"],
        *out,
    )
    assert written == 1
    assert out[0][0] == 1


@pytest.mark.parametrize("seed", range(4))
def test_record_kernel_python_body_matches_record_push(seed):
    """Run the uncompiled record body so its lines are traced by coverage."""
    mesh, reference, rng = _random_state(100 + seed)
    candidate = CrustRaster(mesh.num_cells)
    for name in CELL_FIELDS:
        getattr(candidate, name)[:] = getattr(reference, name)
    kernel = kernels._record_push_kernel.py_func
    for _ in range(40):
        cell = int(rng.integers(0, mesh.num_cells))
        source = int(rng.integers(-1, mesh.num_cells))
        felt = float(rng.uniform(0.0, 80.0))
        distance = float(rng.uniform(0.0, 20_000.0))
        time = float(rng.uniform(0.0, 60.0))
        far_field.record_push(
            reference, cell, felt, distance, mesh.xyz, source, time, 100.0
        )
        kernel(
            candidate.farWork,
            candidate.farDist,
            candidate.farFold,
            candidate.farAt,
            candidate.farDx,
            candidate.farDy,
            candidate.farDz,
            mesh.xyz,
            cell,
            felt,
            distance,
            source,
            time,
            100.0,
        )
    for name in FIELDS:
        np.testing.assert_array_equal(
            getattr(candidate, name), getattr(reference, name), err_msg=name
        )


def test_python_emission_loop_skips_ocean_and_negligible_arrivals(monkeypatch):
    """Without Numba, spread_push keeps land cells with a visible arrival only.

    A 12-cell equatorial ring at 30 degree spacing puts cells 1-2 steps from the
    source within reach and cells 3 steps away (about 10,000 km, arrival near
    4.5e-5 with a 1,000 km reach) below the 1e-4 threshold.
    """
    from types import SimpleNamespace

    from orogen.v2.tectonics import far_field as module

    count = 12
    angle = np.arange(count) * (2.0 * math.pi / count)
    xyz = np.empty(3 * count)
    xyz[0::3] = np.cos(angle)
    xyz[1::3] = np.sin(angle)
    xyz[2::3] = 0.0
    mesh = _Mesh(xyz)
    mesh.adj_offset = np.arange(count + 1, dtype=np.int32) * 2
    mesh.adj_list = np.array(
        [c for cell in range(count) for c in ((cell + 1) % count, (cell - 1) % count)],
        dtype=np.int32,
    )
    crust = CrustRaster(count)
    crust.mask[:] = 1
    crust.type[:] = CrustType["CONTINENT"]
    crust.type[1] = CrustType["OCEAN"]
    monkeypatch.setattr(module, "NUMBA_AVAILABLE", False)
    config = {
        "farReachKm": 1_000.0,
        "farFrontRefKm": 1_000.0,
        "farMaxKm": 1_000_000.0,
        "farCratonCost": 0.3,
        "farWeakCost": 1.6,
        "farCratonTake": 0.15,
        "farWeakTake": 1.6,
        "farFoldMyr": 100.0,
    }
    world = SimpleNamespace(mesh=mesh, config=config, time=0.0)
    plate = SimpleNamespace(crust=crust)
    result = module.spread_push(world, plate, {0: 1.0}, 1000.0, 0.0)
    assert 1 not in result["cells"]  # ocean is crossed but never recorded
    assert 0 in result["cells"]
    assert 3 not in result["cells"]  # arrival below the 1e-4 threshold
    assert 2 in result["cells"]


def test_record_kernel_guards_antipodal_sources_and_zero_push():
    """Cover the degenerate-geometry and zero-norm guards of the record body."""
    kernel = kernels._record_push_kernel.py_func
    xyz = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    work = np.array([2.0, 0.0], dtype=np.float32)
    dist = np.array([10.0, 0.0], dtype=np.float32)
    fold = np.zeros(2, dtype=np.float32)
    at = np.zeros(2, dtype=np.float32)
    dx = np.array([50, 0], dtype=np.int8)
    dy = np.zeros(2, dtype=np.int8)
    dz = np.zeros(2, dtype=np.int8)

    # Antipodal source: the tangent direction vanishes, so the axis is kept.
    kernel(work, dist, fold, at, dx, dy, dz, xyz, 0, 1.0, 5.0, 1, 1.0, 100.0)
    assert dx[0] == 50 and dy[0] == 0 and dz[0] == 0
    assert work[0] == pytest.approx(3.0)

    # Zero felt with a zero axis: the norm guard falls back to one.
    work[:] = [2.0, 0.0]
    kernel(work, dist, fold, at, dx, dy, dz, xyz, 0, 0.0, 5.0, -1, 2.0, 100.0)
    assert work[0] == pytest.approx(2.0)
