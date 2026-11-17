from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.thermal import (
    insulate_at_start, plume_heat_factor, update_mantle_heat,
)


class Mesh:
    num_cells = 5


def make_world():
    plate = Plate(0, 5)
    plate.crust.create(0, CrustType["CONTINENT"], 0.0)
    plate.crust.create(1, CrustType["CRATON"], 0.0)
    plate.crust.create(2, CrustType["OCEAN"], 0.0)
    plate.crust.create(3, CrustType["ARC"], 0.0)
    return SimpleNamespace(
        mesh=Mesh(), plates=[plate],
        owner=np.array([0, 0, 0, 0, -1], dtype=np.int16),
        ownerCell=np.arange(5, dtype=np.int32),
        mantleHeat=np.zeros(5, dtype=np.float32),
        hotspotsReady=False,
        config={
            "mantleInsulateMyr": 100.0,
            "mantleCoolMyr": 200.0,
            "riftHeatGain": 5.0,
            "riftHeatFloor": 1.0,
            "riftHeatPower": 2.0,
        },
    )


def test_insulate_at_start_marks_only_continent_and_craton():
    world = make_world()
    insulate_at_start(world)
    assert world.mantleHeat.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_update_mantle_heat_initializes_then_warms_and_cools():
    world = make_world()
    heat = update_mantle_heat(world, 0.0)
    assert world.hotspotsReady is True
    assert heat.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]

    heat[0] = 0.0
    heat[2] = 1.0
    update_mantle_heat(world, 100.0)
    assert heat[0] == pytest.approx(1.0 - np.exp(-1.0))
    assert heat[1] == pytest.approx(1.0)
    assert heat[2] == pytest.approx(np.exp(-0.5))
    assert heat[3] == 0.0
    assert heat[4] == 0.0


def test_update_mantle_heat_rejects_invalid_state_and_time():
    world = make_world()
    with pytest.raises(ValueError):
        update_mantle_heat(world, -1.0)
    with pytest.raises(ValueError):
        update_mantle_heat(world, float("nan"))
    world.mantleHeat = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError):
        update_mantle_heat(world, 1.0)
    world = make_world()
    world.hotspotsReady = True
    world.config["mantleInsulateMyr"] = 0.0
    with pytest.raises(ValueError):
        update_mantle_heat(world, 1.0)


def test_plume_heat_factor_uses_continental_mean_and_disabled_fallback():
    world = make_world()
    world.hotspotsReady = True
    world.mantleHeat[:] = [1.0, 0.5, 1.0, 1.0, 1.0]
    factor = plume_heat_factor(world, world.plates[0])
    mean = 0.75
    assert factor == pytest.approx((1.0 + 5.0 * mean ** 2) / 6.0)
    assert plume_heat_factor(
        SimpleNamespace(config={"riftHeatGain": 0.0}), world.plates[0]
    ) == 1.0
    world.owner[:] = -1
    assert plume_heat_factor(world, world.plates[0]) == pytest.approx(1 / 6)
