"""Tests for adaptive tectonic stepping."""
import math
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.simulation import fastest_speed, closing_rate, choose_step
from orogen.v2.tectonics.world import TectonicWorld


def _world():
    mesh = build_sim_mesh(48, 41)
    config = {"maxStepMyr": 4.0, "maxCellsPerStep": 1.5, "maxSpeedOceanic": 100.0}
    return mesh, TectonicWorld(mesh, 41, config)


def test_fastest_speed_zero_and_equator_crossing():
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.capCos = math.cos(0.2)
    assert fastest_speed([0, 0, 0], plate, [1, 0, 0]) == 0.0
    # A pole orthogonal to the cap centre reaches the equator within the cap.
    assert fastest_speed([0, 0, 0.01], plate, [1, 0, 0]) == 0.01


def test_fastest_speed_cap_does_not_reach_equator():
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.capCos = math.cos(0.1)
    speed = fastest_speed([0, 0, 0.01], plate, [0, 0, 1])
    assert speed == pytest.approx(0.01 * math.sin(0.1))


def test_closing_rate_uses_two_fastest_living_nonempty_plates():
    mesh, world = _world()
    a, b, c = world.add_plate("A"), world.add_plate("B"), world.add_plate("C")
    for plate in (a, b, c):
        plate.crust.create(0, 0, 0, 1)
        plate.refresh_extent(mesh.xyz)
        plate.capCos = -1.0
    a.omega = [0.0, 0.0, 0.03]
    b.omega = [0.0, 0.0, 0.02]
    c.omega = [0.0, 0.0, 0.01]
    assert closing_rate(world) == 0.05
    c.alive = False
    assert closing_rate(world) == 0.05


def test_choose_step_stationary_and_rate_limited():
    mesh, world = _world()
    plate = world.add_plate("P")
    assert choose_step(world) == 4.0
    plate.crust.create(0, 0, 0, 1)
    plate.refresh_extent(mesh.xyz)
    plate.capCos = -1.0
    plate.omega = [0.0, 0.0, 1.0]
    result = choose_step(world)
    assert result <= 4.0
    assert result > 0.0
