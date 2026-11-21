"""Measure initial-world construction on reusable spherical meshes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.initial_world import build_start_world


def _validate_partition(world: Any, cell_count: int) -> None:
    """Ensure the initial plates partition every mesh cell exactly once."""
    coverage = np.zeros(cell_count, dtype=np.uint8)
    for plate in world.plates:
        mask = np.asarray(plate.crust.mask, dtype=np.uint8)
        if mask.shape != (cell_count,):
            raise RuntimeError(
                "plate crust mask does not match the actual simulation mesh size: "
                f"expected {cell_count}, got {mask.size}"
            )
        coverage += mask
    invalid = int(np.count_nonzero(coverage != 1))
    if invalid:
        raise RuntimeError(
            f"initial-world plate partition is invalid in {invalid} cells"
        )


def _benchmark_size(
    cell_count: int, seed: int, warmups: int, repeats: int
) -> dict[str, Any]:
    mesh_started = time.perf_counter()
    mesh = build_sim_mesh(cell_count, seed)
    mesh_seconds = time.perf_counter() - mesh_started
    actual_cell_count = int(mesh.num_cells)

    timings: list[float] = []
    attempts: list[int] = []
    for run_index in range(warmups + repeats):
        started = time.perf_counter()
        built = build_start_world(mesh, seed)
        elapsed = time.perf_counter() - started
        _validate_partition(built["world"], actual_cell_count)
        if not built["start"].get("fractionMet", False):
            raise RuntimeError("initial-world builder did not meet its land fraction")
        if run_index >= warmups:
            timings.append(elapsed)
            attempts.append(int(built["start"].get("attempts", 0)))
        del built

    return {
        "requested_cells": cell_count,
        "num_cells": actual_cell_count,
        "mesh_build_seconds": mesh_seconds,
        "warmups": warmups,
        "repeats": repeats,
        "build_seconds": timings,
        "median_build_seconds": statistics.median(timings),
        "layout_attempts": attempts,
        "partition_valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 4096, 40000],
        help="mesh cell counts to benchmark",
    )
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--label", default="unknown")
    args = parser.parse_args()
    if any(size < 4 for size in args.sizes):
        parser.error("each mesh size must be at least four cells")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be non-negative and repeats must be positive")

    results = [
        _benchmark_size(size, args.seed, args.warmups, args.repeats)
        for size in args.sizes
    ]
    print(
        json.dumps(
            {
                "label": args.label,
                "seed": args.seed,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
