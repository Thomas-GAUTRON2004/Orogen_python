from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.thermal import (
    insulate_at_start,
    pick_plume_site,
    plume_heat_factor,
    update_mantle_heat,
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
        mesh=Mesh(),
        plates=[plate],
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
    assert factor == pytest.approx((1.0 + 5.0 * mean**2) / 6.0)
    assert (
        plume_heat_factor(
            SimpleNamespace(config={"riftHeatGain": 0.0}), world.plates[0]
        )
        == 1.0
    )
    world.owner[:] = -1
    assert plume_heat_factor(world, world.plates[0]) == pytest.approx(1 / 6)


def test_pick_plume_site_weights_heat_and_keeps_ocean_probability():
    world = make_world()
    world.mesh.xyz = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )
    world.config.update({"plumeOceanWeight": 0.0, "plumeSpacingKm": 0.0})
    world.mantleHeat[:] = [0.0, 1.0, 0.0, 0.0, 0.0]
    assert pick_plume_site(world, lambda: 0.0) == 1
    world.config["plumeOceanWeight"] = 0.2
    world.mantleHeat[:] = 0.0
    assert pick_plume_site(world, lambda: 0.0) == 0


def test_pick_plume_site_excludes_live_plume_spacing_and_returns_none_when_empty():
    world = make_world()
    world.mesh.xyz = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.999,
            0.0447,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )
    world.hotspots = [{"xyz": [1.0, 0.0, 0.0], "alive": True}]
    world.config.update({"plumeOceanWeight": 0.0, "plumeSpacingKm": 500.0})
    world.mantleHeat[:] = [1.0, 1.0, 0.0, 0.0, 0.0]
    assert pick_plume_site(world, lambda: 0.0) == -1
    world.hotspots[0]["alive"] = False
    assert pick_plume_site(world, lambda: 0.0) == 0


def test_pick_plume_site_handles_all_cells_excluded_and_validates_inputs():
    world = make_world()
    world.mesh.xyz = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )
    world.hotspots = [
        {"xyz": [1.0, 0.0, 0.0], "alive": True},
        {"xyz": [-1.0, 0.0, 0.0], "alive": True},
    ]
    world.config.update({"plumeOceanWeight": 0.0, "plumeSpacingKm": 15000.0})
    world.mantleHeat[:] = 1.0
    assert pick_plume_site(world, lambda: 0.5) == -1
    world.config["plumeOceanWeight"] = 1.1
    with pytest.raises(ValueError):
        pick_plume_site(world, lambda: 0.5)


def test_insulate_at_start_rejects_mismatched_raster():
    world = make_world()
    world.mantleHeat = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError):
        insulate_at_start(world)


def test_plume_heat_factor_handles_degenerate_normalization():
    world = make_world()
    world.config.update({"riftHeatGain": 1.0, "riftHeatFloor": -1.0})
    assert plume_heat_factor(world, world.plates[0]) == 1.0


def test_pick_plume_site_rejects_coordinate_size_mismatch_and_bad_rng_fallback():
    world = make_world()
    world.mesh.xyz = np.zeros(3, dtype=np.float64)
    with pytest.raises(ValueError):
        pick_plume_site(world, lambda: 0.5)

    world = make_world()
    world.mesh.xyz = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )
    world.config.update({"plumeOceanWeight": 1.0, "plumeSpacingKm": 0.0})
    assert pick_plume_site(world, lambda: 1.1) == -1
