from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.plumes import birth_plume, update_plumes


class Mesh:
    num_cells = 4
    cell_area_km2 = 1000.0
    xyz = np.array(
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
        ],
        dtype=np.float64,
    )
    adj_offset = np.array([0, 3, 6, 9, 12], dtype=np.int32)
    adj_list = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], dtype=np.int32)

    class Index:
        def nearest(self, x, y, z):
            return int(np.argmax(Mesh.xyz.reshape(-1, 3) @ np.array([x, y, z])))

    index = Index()


def make_world():
    mesh = Mesh()
    plate = Plate(0, 4)
    plate.name = "Pangaea"
    plate.crust.create(0, CrustType["CONTINENT"], 0.0)
    plate.crust.create(1, CrustType["CRATON"], 0.0)
    plate.refresh_extent(mesh.xyz)
    events = []
    return SimpleNamespace(
        mesh=mesh,
        plates=[plate],
        time=0.0,
        owner=np.zeros(4, dtype=np.int16),
        ownerCell=np.arange(4, dtype=np.int32),
        provisional=np.zeros(4, dtype=np.uint8),
        mantleHeat=np.zeros(4, dtype=np.float32),
        hotspotsReady=True,
        hotspots=[],
        plumes=[],
        pendingPlumeRifts=[],
        config={
            "mantleInsulateMyr": 100.0,
            "mantleCoolMyr": 150.0,
            "plumeOceanWeight": 0.3,
            "plumeSpacingKm": 0.0,
            "plumeLifeMinMyr": 80.0,
            "plumeLifeMaxMyr": 80.0,
            "plumeRiftChance": 1.0,
            "riftMinContinentKm2": 1000.0,
            "riftHeatGain": 0.0,
            "riftHeatFloor": 1.0,
            "riftHeatPower": 1.0,
            "hotspotTrackKm": 250.0,
            "hotspotEdificeKm": 50.0,
            "hotspotBuildKmPerMyr": 2.0,
            "hotspotMaxKm": 10.0,
            "hotspotQuietMinMyr": 1.5,
            "hotspotQuietMaxMyr": 5.0,
            "hotspotBurstMinMyr": 0.6,
            "hotspotBurstMaxMyr": 2.0,
        },
        rngHotspot=lambda: 0.0,
        rngVolcano=lambda: 0.5,
        log=lambda kind, text, data=None: events.append(
            {
                "time": 0.0,
                "kind": kind,
                "text": text,
                "data": data or {},
            }
        ),
        events=events,
    )


def test_birth_without_world_collections_uses_coordinate_fallback_and_unique_names():
    world = make_world()
    del world.hotspots
    del world.pendingPlumeRifts
    world.mesh = SimpleNamespace(
        num_cells=4,
        cell_area_km2=1000.0,
        xyz=np.array(
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
            ]
        ),
    )

    first = birth_plume(
        world,
        [1.0, 0.0, 0.0],
        head=False,
        rift=False,
        life_myr=20.0,
        strength=1.0,
    )
    second = birth_plume(
        world,
        [0.0, 1.0, 0.0],
        head=False,
        rift=False,
        life_myr=20.0,
        strength=1.0,
    )

    assert first["name"] == "Kakaka"
    assert second["name"] == "Plume 2"
    assert len(world.hotspots) == 2
    assert world.pendingPlumeRifts == []


def test_nearest_cell_fallback_rejects_mesh_coordinate_mismatch():
    world = make_world()
    world.mesh = SimpleNamespace(num_cells=4, xyz=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="mesh coordinates"):
        birth_plume(
            world,
            [1.0, 0.0, 0.0],
            head=False,
            rift=False,
            name="Invalid mesh",
            life_myr=10.0,
            strength=1.0,
        )


def test_attribute_schedule_and_invalid_rift_target():
    world = make_world()
    item = SimpleNamespace(
        time=0.0,
        xyz=[0.0, 1.0, 0.0],
        done=False,
        head=False,
        rift=False,
        name="Object plume",
        lifeMyr=20.0,
        strength=0.8,
    )
    world.plumes = [item]

    update_plumes(world, 0.0)

    assert item.done is True
    assert world.hotspots[0]["name"] == "Object plume"
    assert world.events[0]["data"]["head"] is False

    plume = birth_plume(
        world,
        [1.0, 0.0, 0.0],
        head=False,
        rift=True,
        rift_plate=99,
        name="No target",
        life_myr=10.0,
        strength=1.0,
    )
    assert plume["name"] == "No target"
    assert world.pendingPlumeRifts == []
