import math
from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.ops import (
    names_after_split,
    plume_junctions,
    pull_axis_of,
    try_rift,
)
from orogen.v2.tectonics.ops import rift as rift_module

from test_v2_tectonic_rift_operations import make_world, square_junction_fixture


def test_pull_axis_requires_enough_trench_and_finds_principal_axis():
    world, plate = make_world()
    plate.capCenter = [1.0, 0.0, 0.0]
    assert pull_axis_of(plate, world.mesh) is None

    plate.crust.flags[0] |= CrustFlag["TRENCH"]
    assert pull_axis_of(plate, world.mesh) is not None  # 1000 km of trench

    plate.crust.flags[3] |= CrustFlag["TRENCH"]
    axis = pull_axis_of(plate, world.mesh)
    assert axis is not None
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert abs(axis[1]) == pytest.approx(1.0, abs=1e-5)


def test_pull_axis_rejects_a_degenerate_initial_power_vector():
    world, plate = make_world()
    plate.capCenter = [0.0, 0.0, 1.0]
    plate.crust.flags[4] |= CrustFlag["TRENCH"]
    assert pull_axis_of(plate, world.mesh) is None


def test_names_after_split_use_craton_letters_when_distinct():
    world, plate = make_world()
    fresh = world.add_plate("new")
    world.terraneLetters = {10: "A", 30: "B"}
    assert names_after_split(world, plate, fresh, "old") == ("A", "B")


def test_names_after_split_allocate_unused_numbered_names():
    world, plate = make_world()
    plate.name = "Rift·1"
    world.add_plate("Rift·2")
    world.add_plate("Rift·4")
    left, right = names_after_split(world, plate, world.plates[-1], "Rift·1")
    assert (left, right) == ("Rift·3", "Rift·5")


def test_try_rift_splits_plate_and_logs_event(monkeypatch):
    world, plate = make_world()
    plate.crust.flags[1] |= CrustFlag["SUTURE"]
    found = {
        "path": [1, 2],
        "failed_arm": [3],
        "across": [0.0, 0.0, 1.0],
        "value": 0.25,
    }
    calls = []

    monkeypatch.setattr(rift_module, "pull_axis_of", lambda *_args: None)
    monkeypatch.setattr(rift_module, "rift_context", lambda *_args: {"ctx": True})
    monkeypatch.setattr(
        rift_module,
        "find_rift",
        lambda _world, _plate, junction, *_args: (
            calls.append(junction) or found if junction == 4 else None
        ),
    )
    monkeypatch.setattr(
        rift_module, "spread_belt", lambda *_args, **_kwargs: None
    )

    result = try_rift(
        world,
        plate,
        [-1, 3, 4],
        {"text": "the test plume", "kind": "plume", "hotspot": 7},
        need_cratons_both_sides=False,
    )

    assert result is True
    assert calls == [3, 4]
    assert world.lastRiftAt == world.time
    event = world.events[-1]
    assert event["kind"] == "rift"
    assert event["data"] == {
        "plate": 0,
        "fresh": 1,
        "cause": "plume",
        "hotspot": 7,
        "lengthKm": 2000.0,
        "weakKm": 1000.0,
        "failedArmKm": 1000.0,
        "ageMyr": world.time,
    }
    assert "the test plume" in event["text"]
    assert "failed arm" in event["text"]


def test_try_rift_returns_false_when_no_junction_yields_a_cut(monkeypatch):
    world, plate = make_world()
    monkeypatch.setattr(rift_module, "pull_axis_of", lambda *_args: None)
    monkeypatch.setattr(rift_module, "rift_context", lambda *_args: {})
    monkeypatch.setattr(rift_module, "find_rift", lambda *_args: None)
    assert try_rift(
        world,
        plate,
        [-1, 0],
        {"text": "no cut", "kind": "plume", "hotspot": -1},
    ) is False
    assert len(world.plates) == 1
    assert world.events == []


def test_plume_junctions_samples_evenly_spaced_land_cells():
    world, plate = square_junction_fixture()
    world.mesh.index = SimpleNamespace(nearest=lambda *_args: 24)
    world.config["plumeJunctionKm"] = 500.0
    candidates = plume_junctions(world, plate, [1.0, 0.0, 0.0])
    assert len(candidates) == 12
    assert candidates[0] == 24
    assert len(set(candidates)) == 12
    assert all(plate.crust.mask[cell] for cell in candidates)
    assert all(
        plate.crust.type[cell] != CrustType["CRATON"]
        for cell in candidates
    )


def test_plume_junctions_falls_back_from_a_craton_and_returns_small_set():
    world, plate = square_junction_fixture()
    world.mesh.index = SimpleNamespace(nearest=lambda *_args: 24)
    world.config["plumeJunctionKm"] = 100.0
    plate.crust.type[24] = CrustType["CRATON"]
    candidates = plume_junctions(world, plate, [1.0, 0.0, 0.0])
    assert 24 not in candidates
    assert candidates
    assert len(candidates) <= 12
