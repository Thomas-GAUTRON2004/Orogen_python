import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.world import (
    BoundaryKind, EventKind, Provisional, TectonicWorld,
)


def test_world_allocates_state_and_independent_seed_streams():
    mesh = build_sim_mesh(24, 91)
    config = {"simCells": 24, "nested": {"step": 2}}
    world = TectonicWorld(mesh, 1234, config)
    other = TectonicWorld(mesh, 1234, config)
    assert world.owner.shape == (mesh.num_cells,)
    assert world.owner.dtype == np.int16
    assert np.all(world.owner == -1)
    assert world.boundaryKind.dtype == np.uint8
    assert world.mantleHeat.dtype == np.float32
    assert world.rng() == other.rng()
    assert world.rngSubduction() == other.rngSubduction()
    assert world.rng() != world.rngSubduction()
    world.config["nested"]["step"] = 99
    assert config["nested"]["step"] == 2


def test_world_plate_ids_terrane_ids_names_and_events():
    mesh = build_sim_mesh(24, 91)
    world = TectonicWorld(mesh, 1, {"simCells": 24})
    a = world.add_plate("A")
    b = world.add_plate("B")
    assert (a.id, b.id, a.born) == (0, 1, 0.0)
    a.crust.create(0, 2, -1000, 7)
    assert world.next_terrane_id() == 8
    world.terraneLetters = {7: "L"}
    assert world.name_by_cratons(a, "plate") == "L"
    world.log(EventKind["START"], "start", {"plate": 0})
    assert world.events[0] == {
        "time": 0.0, "kind": "start", "text": "start", "data": {"plate": 0}
    }
    with pytest.raises(ValueError):
        world.log("not-an-event", "invalid")
    assert BoundaryKind["COLLISION"] == 5
    assert Provisional["LIKE_NEIGHBOUR"] == 2


def test_world_requires_upstream_configuration_until_defaults_are_ported():
    with pytest.raises(ValueError):
        TectonicWorld(build_sim_mesh(12, 1), 1, None)
