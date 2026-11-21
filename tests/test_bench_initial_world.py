from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bench_initial_world


def test_validate_partition_rejects_mask_with_wrong_mesh_size():
    world = SimpleNamespace(
        plates=[SimpleNamespace(crust=SimpleNamespace(mask=np.ones(5, dtype=np.uint8)))]
    )

    with pytest.raises(
        RuntimeError, match="does not match the actual simulation mesh size"
    ):
        bench_initial_world._validate_partition(world, 4)


def test_benchmark_uses_actual_mesh_cell_count(monkeypatch):
    mesh = SimpleNamespace(num_cells=5)
    world = SimpleNamespace(
        plates=[SimpleNamespace(crust=SimpleNamespace(mask=np.ones(5, dtype=np.uint8)))]
    )
    built = {"world": world, "start": {"fractionMet": True, "attempts": 1}}
    monkeypatch.setattr(bench_initial_world, "build_sim_mesh", lambda *_: mesh)
    monkeypatch.setattr(bench_initial_world, "build_start_world", lambda *_: built)

    result = bench_initial_world._benchmark_size(4, 941, warmups=0, repeats=1)

    assert result["requested_cells"] == 4
    assert result["num_cells"] == 5
    assert result["partition_valid"] is True
