"""Reproducible paired JavaScript/Python benchmark worker.

Python workloads run with every loaded Orogen V2 Numba dispatch gate disabled
and enabled. step_world is a Python-only tectonic comparison because the
repository does not yet contain a complete JavaScript tectonic orchestrator.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

import orogen.v2.tectonics.far_field as far_field_module
from orogen.v2.core._numba_kernels import NUMBA_AVAILABLE
from orogen.v2.core.fast_noise import FastNoise
from orogen.v2.core.mesh_bfs import (
    hop_distance,
    hops_from,
    largest_piece_of,
    pieces_of,
    within_km,
)
from orogen.v2.core.min_heap import spread_km
from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.core.sphere_geometry import bearing_to_many, destination_many
from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType
from orogen.v2.tectonics.stepper import step_world
from orogen.v2.tectonics.world import TectonicWorld

N_NOISE = 40_000
N_GEOM = 80_000
N_CELLS = 40_000
N_STEP_CELLS = int(os.environ.get("BENCH_STEP_CELLS", "512"))
ROUNDS = max(3, int(os.environ.get("BENCH_ROUNDS", "10")))
WARMUPS = max(1, int(os.environ.get("BENCH_WARMUPS", "2")))
_sink = 0.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples: list[float]) -> dict[str, object]:
    """Return descriptive statistics and raw timing samples."""
    return {
        "median_ms": round(statistics.median(samples), 6),
        "mean_ms": round(statistics.fmean(samples), 6),
        "stddev_ms": round(statistics.stdev(samples), 6) if len(samples) > 1 else 0.0,
        "p95_ms": round(_percentile(samples, 0.95), 6),
        "min_ms": round(min(samples), 6),
        "max_ms": round(max(samples), 6),
        "samples_ms": [round(value, 6) for value in samples],
    }


def measure(function) -> dict[str, object]:
    """Warm the callable, then summarize its measured execution times."""
    for _ in range(WARMUPS):
        function()
    samples = []
    for _ in range(ROUNDS):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1_000.0)
    return summarize(samples)


@contextmanager
def numba_mode(enabled: bool):
    """Temporarily switch every loaded Orogen V2 Numba dispatch gate."""
    previous = {}
    active = bool(enabled and NUMBA_AVAILABLE)
    for name, module in tuple(sys.modules.items()):
        if name.startswith("orogen.v2.") and hasattr(module, "NUMBA_AVAILABLE"):
            previous[name] = module.NUMBA_AVAILABLE
            module.NUMBA_AVAILABLE = active
    try:
        yield active
    finally:
        for name, value in previous.items():
            module = sys.modules.get(name)
            if module is not None:
                module.NUMBA_AVAILABLE = value


class MeshFixture:
    """Weak-referenceable mesh container for synthetic graph workloads."""


def make_core_inputs():
    """Build identical deterministic arrays and a synthetic CSR graph."""
    points = np.asarray(
        [
            (
                math.sin(index * 0.013) * 17.0,
                math.cos(index * 0.007) * 11.0,
                math.sin(index * 0.003 + 0.4) * 23.0,
            )
            for index in range(N_NOISE)
        ],
        dtype=np.float64,
    )
    vectors = np.empty((N_GEOM, 3), dtype=np.float64)
    for index in range(N_GEOM):
        z = 1.0 - 2.0 * (index + 0.5) / N_GEOM
        angle = index * 2.399963229728653
        radius = math.sqrt(1.0 - z * z)
        vectors[index] = (radius * math.cos(angle), radius * math.sin(angle), z)

    xyz = np.empty(N_CELLS * 3, dtype=np.float32)
    for index in range(N_CELLS):
        z = 1.0 - 2.0 * (index + 0.5) / N_CELLS
        angle = index * 2.399963229728653
        radius = math.sqrt(1.0 - z * z)
        xyz[3 * index : 3 * index + 3] = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            z,
        )

    offsets = np.arange(N_CELLS + 1, dtype=np.int32) * 6
    adjacency = np.empty(N_CELLS * 6, dtype=np.int32)
    for index in range(N_CELLS):
        adjacency[index * 6 : index * 6 + 3] = [
            (index + step) % N_CELLS for step in (1, 2, 3)
        ]
        adjacency[index * 6 + 3 : index * 6 + 6] = [
            (index - step) % N_CELLS for step in (1, 2, 3)
        ]

    mesh = MeshFixture()
    mesh.num_cells = N_CELLS
    mesh.xyz = xyz
    mesh.adj_offset = offsets
    mesh.adj_list = adjacency
    return points, vectors, mesh


def make_far_field_case():
    """Prepare the far-field diffusion fixture outside timed regions."""
    angle = np.arange(N_CELLS, dtype=np.float64) * (2.0 * math.pi / N_CELLS)
    xyz = np.empty(N_CELLS * 3, dtype=np.float64)
    xyz[0::3] = np.cos(angle)
    xyz[1::3] = np.sin(angle)
    xyz[2::3] = 0.0

    offsets = np.arange(N_CELLS + 1, dtype=np.int32) * 6
    adjacency = np.empty(N_CELLS * 6, dtype=np.int32)
    for cell in range(N_CELLS):
        adjacency[cell * 6 : cell * 6 + 3] = [
            (cell + step) % N_CELLS for step in (1, 2, 3)
        ]
        adjacency[cell * 6 + 3 : cell * 6 + 6] = [
            (cell - step) % N_CELLS for step in (1, 2, 3)
        ]

    mesh = MeshFixture()
    mesh.num_cells = N_CELLS
    mesh.xyz = xyz
    mesh.adj_offset = offsets
    mesh.adj_list = adjacency
    crust = CrustRaster(N_CELLS)
    crust.mask.fill(1)
    crust.count = N_CELLS
    crust.type.fill(CrustType["CONTINENT"])
    crust.type[::211] = CrustType["CRATON"]
    crust.flags[::97] |= CrustFlag["SUTURE"]
    config = {
        "farReachKm": 21_000.0,
        "farFrontRefKm": 1_000.0,
        "farMaxKm": 21_000.0,
        "farCratonCost": 0.3,
        "farWeakCost": 1.6,
        "farCratonTake": 0.15,
        "farWeakTake": 1.6,
        "farFoldMyr": 100.0,
    }
    world = SimpleNamespace(mesh=mesh, config=config, time=10.0)
    plate = SimpleNamespace(crust=crust)
    sources = {0: 100.0, N_CELLS // 2: 80.0}
    return world, plate, sources


def benchmark_far_field(case):
    """Measure spread_push, including its Python-side history writes."""
    world, plate, sources = case
    history_fields = (
        "farWork",
        "farDist",
        "farFold",
        "farAt",
        "farDx",
        "farDy",
        "farDz",
    )

    def run():
        global _sink
        for name in history_fields:
            getattr(plate.crust, name).fill(0)
        result = far_field_module.spread_push(
            world, plate, sources, 1_000.0, world.time
        )
        _sink = len(result["cells"])

    return measure(run)


def make_step_world_pool(mesh, count: int):
    """Prepare independent four-plate worlds before timing individual steps."""
    worlds = []
    points = mesh.xyz.reshape(-1, 3)
    longitude = np.arctan2(points[:, 1], points[:, 0])
    labels = (np.floor((longitude + math.pi) / (math.pi / 2.0)).astype(int)) % 4

    for _ in range(count):
        world = TectonicWorld(mesh, 20_261_119)
        for plate_id in range(4):
            plate = world.add_plate(f"benchmark-{plate_id}")
            cells = np.flatnonzero(labels == plate_id).astype(np.int32)
            types = np.where(
                cells % 9 == 0,
                CrustType["OCEAN"],
                CrustType["CONTINENT"],
            ).astype(np.uint8)
            plate.crust.create_many(
                cells,
                types,
                -100.0,
                np.full(cells.size, plate_id + 1, dtype=np.uint16),
            )
            if cells.size:
                stride = max(1, cells.size // 16)
                plate.crust.flags[cells[::stride]] |= CrustFlag["RIFT_MARGIN"]
            plate.omega = [
                0.0010 * (plate_id + 1),
                0.00045 * (-1 if plate_id % 2 else 1),
                0.0007 * (plate_id - 1.5),
            ]
            plate.refresh_extent(mesh.xyz)
        worlds.append(world)
    return worlds


def benchmark_step_world(mesh):
    """Measure a step on a fresh four-plate world for every sample."""
    worlds = make_step_world_pool(mesh, WARMUPS + ROUNDS)
    cursor = 0

    def run():
        nonlocal cursor
        global _sink
        state = step_world(worlds[cursor], 0.5)
        cursor += 1
        _sink = (
            float(state["classifiedBoundaries"])
            + float(state["resolve"].get("created", 0))
            + float(state["resolve"].get("consumed", 0))
            + float(state["collisions"]["tracked"])
        )

    timing = measure(run)
    timing["fixture"] = {
        "mesh_cells": int(mesh.num_cells),
        "plates": 4,
        "dt_myr": 0.5,
        "warmups": WARMUPS,
        "measured_steps": ROUNDS,
        "world_setup_included": False,
        "javascript_equivalent_available": False,
    }
    return timing


def run_mode(points, vectors, mesh, far_field_case, step_mesh, enabled: bool):
    """Run common and tectonic workloads under one Numba dispatch mode."""
    with numba_mode(enabled):
        noise = FastNoise(9876)
        selected_cells = np.asarray(
            [
                cell
                for start in range(0, N_CELLS, 68)
                for cell in range(start, min(start + 64, N_CELLS))
            ],
            dtype=np.int32,
        )
        copy_source = CrustRaster(N_CELLS)
        copy_cells = np.arange(0, N_CELLS, 2, dtype=np.int32)
        copy_source.create_many(
            copy_cells, CrustType["CONTINENT"], -100.0, copy_cells + 1
        )

        def bench_noise():
            global _sink
            _sink = float(np.sum(noise.fbm_many(points, 5, 2.0 / 3.0)))

        def bench_geometry():
            global _sink
            targets = destination_many(vectors, 0.71, 0.013)
            bearings = bearing_to_many(vectors, targets)
            _sink = float(np.sum(bearings + np.sum(vectors * targets, axis=1)))

        def bench_bfs():
            global _sink
            distance = hop_distance(mesh, lambda cell: cell == 0, N_CELLS)
            _sink = int(distance[-1])

        def bench_components():
            global _sink
            components = pieces_of(mesh, selected_cells)
            _sink = sum(len(component) for component in components)

        def bench_largest_piece():
            global _sink
            largest = largest_piece_of(mesh, selected_cells)
            _sink = len(largest)

        def bench_within_km():
            global _sink
            reached = within_km(mesh, lambda cell: cell == 0, 5_000.0)
            _sink = int(np.count_nonzero(reached))

        def bench_multi_source_bfs():
            global _sink
            distance, source = hops_from(mesh, (0, N_CELLS // 2))
            _sink = int(distance[-1]) + int(source[-1])

        def bench_dijkstra():
            global _sink
            distance = np.full(N_CELLS, np.inf, dtype=np.float64)
            distance[0] = 0.0
            spread_km(mesh, distance, max_km=20_000.0)
            _sink = float(distance[-1])

        def bench_crust_create_snapshot():
            global _sink
            raster = CrustRaster(N_CELLS)
            cells = np.arange(0, N_CELLS, 2, dtype=np.int32)
            raster.create_many(
                cells, CrustType["CONTINENT"], -100.0, cells + 1
            )
            snapshots = raster.snapshot_many(cells)
            _sink = float(snapshots.size + snapshots[-1, -1])

        def bench_crust_copy():
            global _sink
            target = CrustRaster(N_CELLS)
            target.copy_many(copy_cells, copy_source, copy_cells)
            _sink = target.count

        timings = {
            "noise_fbm_ms": measure(bench_noise),
            "spherical_geometry_ms": measure(bench_geometry),
            "mesh_bfs_ms": measure(bench_bfs),
            "mesh_components_ms": measure(bench_components),
            "mesh_largest_piece_ms": measure(bench_largest_piece),
            "mesh_within_km_ms": measure(bench_within_km),
            "mesh_multi_source_bfs_ms": measure(bench_multi_source_bfs),
            "mesh_dijkstra_ms": measure(bench_dijkstra),
            "crust_create_snapshot_ms": measure(bench_crust_create_snapshot),
            "crust_copy_ms": measure(bench_crust_copy),
        }
        far_field_timing = benchmark_far_field(far_field_case)
        step_world_timing = benchmark_step_world(step_mesh)

    return {
        "timings": timings,
        "far_field_diffusion": far_field_timing,
        "step_world": step_world_timing,
    }


def main():
    """Run the pure-Python and optional Numba benchmark suites."""
    points, vectors, mesh = make_core_inputs()
    far_field_case = make_far_field_case()
    step_mesh = build_sim_mesh(N_STEP_CELLS, 20_261_119)

    pure = run_mode(points, vectors, mesh, far_field_case, step_mesh, False)
    accelerated = (
        run_mode(points, vectors, mesh, far_field_case, step_mesh, True)
        if NUMBA_AVAILABLE
        else None
    )

    payload = {
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "numba_available": NUMBA_AVAILABLE,
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
        },
        "configuration": {
            "rounds": ROUNDS,
            "warmups": WARMUPS,
            "noise_points": N_NOISE,
            "geometry_points": N_GEOM,
            "graph_cells": N_CELLS,
            "step_world_requested_cells": N_STEP_CELLS,
        },
        "python_pure": pure,
        "python_numba": accelerated,
    }
    print(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
