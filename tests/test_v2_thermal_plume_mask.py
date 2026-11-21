"""Parity of the vectorised plume mask with the original selection."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from orogen.v2.tectonics import thermal


def _pick_setup(seed: int, count: int = 160):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    mesh = types.SimpleNamespace(num_cells=count, xyz=points.reshape(-1))
    hotspots = []
    for _ in range(int(rng.integers(0, 4))):
        v = rng.normal(size=3)
        hotspots.append({"xyz": v / np.linalg.norm(v), "alive": True})
    config = {
        "plumeOceanWeight": float(rng.uniform(0.0, 1.0)),
        "plumeSpacingKm": float(rng.uniform(800.0, 4000.0)),
    }
    world = types.SimpleNamespace(
        mesh=mesh,
        config=config,
        mantleHeat=rng.random(count),
        hotspots=hotspots,
    )
    return world, rng


def _old_pick(world, rng_draw):
    """Original plume selection, reproduced verbatim for the reference."""
    mesh, config, mantle_heat = world.mesh, world.config, world.mantleHeat
    count = int(mesh.num_cells)
    xyz = np.asarray(mesh.xyz, dtype=np.float64).reshape(-1, 3)
    ocean_weight = float(config["plumeOceanWeight"])
    near = math.cos(float(config["plumeSpacingKm"]) / 6371.0)
    live = [
        np.asarray(h["xyz"], dtype=np.float64) for h in world.hotspots if h["alive"]
    ]
    weights = np.zeros(count, dtype=np.float32)
    total = 0.0
    for cell in range(count):
        position = xyz[cell]
        if any(float(np.dot(position, plume)) > near for plume in live):
            continue
        weight = ocean_weight + (1.0 - ocean_weight) * float(mantle_heat[cell])
        weights[cell] = weight
        total += float(weight)
    if total <= 0.0:
        return -1
    draw = float(rng_draw()) * total
    for cell in range(count):
        draw -= float(weights[cell])
        if draw <= 0.0 and weights[cell] > 0:
            return cell
    return -1


@pytest.mark.parametrize("seed", range(40))
def test_plume_site_matches_original_selection(seed):
    world, rng = _pick_setup(seed)
    draw_seed = int(rng.integers(0, 10**6))
    expected = _old_pick(world, np.random.default_rng(draw_seed).random)
    actual = thermal.pick_plume_site(world, np.random.default_rng(draw_seed).random)
    assert actual == expected


def test_plume_mask_decides_cells_exactly_at_the_spacing_threshold():
    """A cell sitting on the spacing boundary must match the scalar comparison."""
    rng = np.random.default_rng(11)
    count = 64
    plume = np.array([0.0, 0.0, 1.0])
    near = math.cos(1500.0 / 6371.0)
    angles = np.linspace(0.0, 2 * math.pi, count, endpoint=False)
    # Points whose dot with the plume is within ulps of the threshold.
    xyz = np.empty((count, 3))
    xyz[:, 0] = np.sin(np.arccos(near)) * np.cos(angles)
    xyz[:, 1] = np.sin(np.arccos(near)) * np.sin(angles)
    xyz[:, 2] = near + rng.normal(scale=1e-13, size=count)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    exact = np.array(
        [float(np.dot(xyz[c], plume)) > near for c in range(count)], dtype=bool
    )
    assert np.array_equal(exact, thermal._cells_near_plumes(xyz, [plume], near))


def test_plume_mask_handles_no_live_plumes():
    xyz = np.eye(3)
    assert not thermal._cells_near_plumes(xyz, [], 0.5).any()
