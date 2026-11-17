from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.plumes import _erupted_during, _stamp_track, birth_plume, update_plumes
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.world import EventKind


class Mesh:
    num_cells = 4
    cell_area_km2 = 1000.0
    xyz = np.array([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
    ], dtype=np.float64)
    adj_offset = np.array([0, 3, 6, 9, 12], dtype=np.int32)
    adj_list = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], dtype=np.int32)

    class Index:
        def nearest(self, x, y, z):
            points = Mesh.xyz.reshape(-1, 3)
            return int(np.argmax(points @ np.array([x, y, z])))

    index = Index()


def make_world():
    mesh = Mesh()
    plate = Plate(0, mesh.num_cells)
    plate.name = "Pangaea"
    plate.crust.create(0, CrustType["CONTINENT"], 0.0)
    plate.crust.create(1, CrustType["CRATON"], 0.0)
    plate.crust.create(2, CrustType["OCEAN"], 0.0)
    plate.crust.create(3, CrustType["ARC"], 0.0)
    plate.refresh_extent(mesh.xyz)
    events = []
    return SimpleNamespace(
        mesh=mesh,
        plates=[plate],
        time=0.0,
        owner=np.array([0, 0, 0, 0], dtype=np.int16),
        ownerCell=np.arange(4, dtype=np.int32),
        provisional=np.zeros(4, dtype=np.uint8),
        mantleHeat=np.zeros(4, dtype=np.float32),
        hotspotsReady=False,
        hotspots=[],
        plumes=[],
        pendingPlumeRifts=[],
        events=events,
        config={
            "mantleInsulateMyr": 100.0,
            "mantleCoolMyr": 150.0,
            "hotspotInitialCount": 0,
            "plumeRatePerMyr": 0.0,
            "plumeOceanWeight": 0.3,
            "plumeSpacingKm": 0.0,
            "plumeLifeMinMyr": 80.0,
            "plumeLifeMaxMyr": 80.0,
            "plumeRiftChance": 1.0,
            "riftMinContinentKm2": 1000.0,
            "riftHeatGain": 0.0,
            "riftHeatFloor": 1.0,
            "riftHeatPower": 1.0,
            "lipAreaMinMkm2": 0.15,
            "lipAreaMaxMkm2": 4.0,
            "lipThicknessKm": 2.0,
            "lipLobes": 0.5,
            "lipEdgeNoise": 0.3,
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
        log=lambda kind, text, data=None: events.append({
            "time": 0.0, "kind": kind, "text": text, "data": data or {},
        }),
    )


def test_first_update_seeds_initial_plumes_and_mantle_memory():
    world = make_world()
    world.config["hotspotInitialCount"] = 1
    world.time = 5.0

    state = update_plumes(world, 10.0)

    assert world.hotspotsReady is True
    assert len(world.hotspots) == 1
    plume = world.hotspots[0]
    assert plume["alive"] is True
    assert plume["lip"] is None
    assert plume["bornAt"] <= world.time
    assert plume["dieAt"] > world.time
    assert world.events[0]["kind"] == EventKind["PLUME_RISE"]
    assert world.mantleHeat[0] == pytest.approx(1.0)
    assert state == {"born": 1, "expired": 0, "alive": 1}


def test_scheduled_mantle_frame_head_queues_forced_rift():
    world = make_world()
    world.hotspotsReady = True
    world.time = 12.0
    world.plumes.append({
        "time": 10.0, "xyz": [0.0, 1.0, 0.0], "head": True,
        "rift": True, "name": "Tavora", "lifeMyr": 40.0, "strength": 1.2,
    })

    state = update_plumes(world, 1.0)

    assert world.plumes[0]["done"] is True
    assert world.hotspots[0]["xyz"] == pytest.approx([0.0, 1.0, 0.0])
    assert world.hotspots[0]["name"] == "Tavora"
    assert world.pendingPlumeRifts[0]["plate"] == 0
    assert world.pendingPlumeRifts[0]["cell"] == -1
    assert world.pendingPlumeRifts[0]["cause"]["hotspot"] == 0
    assert state["alive"] == 1


def test_scheduled_plate_cell_head_and_invalid_schedule():
    world = make_world()
    world.hotspotsReady = True
    world.time = 2.0
    world.plumes.extend([
        {"time": 1.0, "plate": 0, "cell": 1},
        {"time": 1.0, "plate": 9, "cell": 1},
        {"time": 4.0, "xyz": [1.0, 0.0, 0.0]},
    ])

    update_plumes(world, 0.5)

    assert world.plumes[0]["done"] is True
    assert world.plumes[1]["done"] is True
    assert world.plumes[2].get("done") is None
    assert len(world.hotspots) == 1
    assert world.pendingPlumeRifts[0]["cell"] == 1


def test_probability_birth_and_no_birth_when_site_is_excluded():
    world = make_world()
    world.hotspotsReady = True
    world.config["plumeRatePerMyr"] = 1.0

    update_plumes(world, 1.0)
    assert len(world.hotspots) == 1
    assert world.hotspots[0]["alive"] is True

    world = make_world()
    world.hotspotsReady = True
    world.config.update({"plumeRatePerMyr": 1.0, "plumeOceanWeight": 0.0, "plumeSpacingKm": 20000.0})
    world.hotspots.append({
        "id": 0, "name": "Existing", "xyz": [1.0, 0.0, 0.0],
        "bornAt": -10.0, "dieAt": 100.0, "alive": True,
    })
    update_plumes(world, 1.0)
    assert len(world.hotspots) == 1


def test_plume_death_event_and_dead_plumes_are_not_reprocessed():
    world = make_world()
    world.hotspotsReady = True
    world.time = 20.0
    world.hotspots.append({
        "id": 7, "name": "Old", "xyz": [1.0, 0.0, 0.0],
        "bornAt": -5.0, "dieAt": 10.0, "alive": True,
    })
    world.hotspots.append({
        "id": 8, "name": "Already dead", "xyz": [0.0, 1.0, 0.0],
        "bornAt": -20.0, "dieAt": -1.0, "alive": False,
    })

    state = update_plumes(world, 0.0)

    assert world.hotspots[0]["alive"] is False
    assert world.events[-1]["kind"] == EventKind["PLUME_DEATH"]
    assert world.events[-1]["data"]["lifeMyr"] == pytest.approx(25.0)
    assert state == {"born": 2, "expired": 1, "alive": 0}


def test_birth_plume_rejects_invalid_position_and_skips_ocean_rift():
    world = make_world()
    with pytest.raises(ValueError):
        birth_plume(world, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        birth_plume(world, [1.0, float("nan"), 0.0])

    world.owner[:] = -1
    world.hotspotsReady = True
    plume = birth_plume(
        world, [1.0, 0.0, 0.0], rift=True,
        name="Open", life_myr=50.0, strength=1.0,
    )
    assert plume["name"] == "Open"
    assert world.events[-1]["data"]["plate"] == -1
    assert world.pendingPlumeRifts == []


def test_chance_rift_respects_continental_area_and_rng_draw():
    world = make_world()
    world.hotspotsReady = True
    world.config["riftMinContinentKm2"] = 1e9
    birth_plume(
        world, [1.0, 0.0, 0.0], rift="chance",
        name="Small", life_myr=10.0, strength=1.0,
    )
    assert world.pendingPlumeRifts == []

    world.config["riftMinContinentKm2"] = 1000.0
    world.config["plumeRiftChance"] = 0.5
    world.rngHotspot = lambda: 0.99
    birth_plume(
        world, [1.0, 0.0, 0.0], rift="chance",
        name="NoRift", life_myr=10.0, strength=1.0,
    )
    assert world.pendingPlumeRifts == []


def test_update_plumes_rejects_bad_dt_and_marks_non_living_plate_schedule_done():
    world = make_world()
    with pytest.raises(ValueError):
        update_plumes(world, -1.0)

    world.hotspotsReady = True
    world.plates[0].alive = False
    world.plumes.append({"time": 0.0, "plate": 0, "cell": 0})
    update_plumes(world, 0.0)
    assert world.plumes[0]["done"] is True


def test_initial_plume_seed_stops_when_all_candidate_cells_are_live():
    world = make_world()
    world.config.update({
        "hotspotInitialCount": 1,
        "plumeOceanWeight": 0.0,
        "plumeSpacingKm": 20000.0,
    })
    points = world.mesh.xyz.reshape(-1, 3)
    world.hotspots = [
        {
            "id": i, "name": f"Active{i}", "xyz": point.tolist(),
            "bornAt": -10.0, "dieAt": 100.0, "alive": True,
        }
        for i, point in enumerate(points)
    ]

    state = update_plumes(world, 1.0)

    assert world.hotspotsReady is True
    assert len(world.hotspots) == 4
    assert state == {"born": 4, "expired": 0, "alive": 4}


def test_head_eruption_stamps_a_connected_lip_and_records_its_area():
    world = make_world()
    world.mesh.xyz = np.array([
        1.0, 0.0, 0.0,
        np.cos(0.01), np.sin(0.01), 0.0,
        np.cos(0.02), np.sin(0.02), 0.0,
        0.0, 1.0, 0.0,
    ], dtype=np.float64)
    world.mesh.adj_offset = np.array([0, 1, 3, 5, 6], dtype=np.int32)
    world.mesh.adj_list = np.array([1, 0, 2, 1, 3, 2], dtype=np.int32)
    world.mesh.index = SimpleNamespace(nearest=lambda x, y, z: 0)
    world.config.update({
        "lipAreaMinMkm2": 4.0, "lipAreaMaxMkm2": 4.0,
        "lipLobes": 0.0, "lipEdgeNoise": 0.0,
    })
    rv_values = iter([0.5] * 11 + [42 / 1e9])
    world.rngVolcano = lambda: next(rv_values)

    plume = birth_plume(
        world, [1.0, 0.0, 0.0], head=True, rift=False,
        name="Test plume", life_myr=50.0, strength=1.0,
    )

    lip = plume["lip"]
    assert lip["name"] == "Test plume Traps"
    assert lip["onLand"] is True
    assert lip["flows"] == 2
    assert lip["cells"] == 3
    assert world.events[0]["kind"] == EventKind["PLUME_RISE"]
    assert world.events[1]["kind"] == EventKind["LIP"]
    assert world.events[1]["data"]["areaMkm2"] == pytest.approx(0.003)
    for cell in range(3):
        assert world.plates[0].crust.volcType[cell] == 2
        assert world.plates[0].crust.volcSource[cell] == 1
        assert world.plates[0].crust.volcKm[cell] > 0.0
        assert world.plates[0].crust.flags[cell] & 16


def test_head_eruption_does_not_stamp_provisional_or_unowned_cells():
    world = make_world()
    world.mesh.xyz = np.array([
        1.0, 0.0, 0.0,
        np.cos(0.01), np.sin(0.01), 0.0,
        np.cos(0.02), np.sin(0.02), 0.0,
        0.0, 1.0, 0.0,
    ], dtype=np.float64)
    world.mesh.adj_offset = np.array([0, 1, 3, 5, 6], dtype=np.int32)
    world.mesh.adj_list = np.array([1, 0, 2, 1, 3, 2], dtype=np.int32)
    world.mesh.index = SimpleNamespace(nearest=lambda x, y, z: 0)
    world.provisional[1] = 1
    world.owner[2] = -1
    world.config.update({
        "lipAreaMinMkm2": 4.0, "lipAreaMaxMkm2": 4.0,
        "lipLobes": 0.0, "lipEdgeNoise": 0.0,
    })

    plume = birth_plume(
        world, [1.0, 0.0, 0.0], head=True, rift=False,
        name="Filtered", life_myr=50.0, strength=1.0,
    )

    assert plume["lip"]["cells"] == 1
    assert world.plates[0].crust.volcType[0] == 2
    assert world.plates[0].crust.volcType[1] == 0
    assert world.plates[0].crust.volcType[2] == 0


def test_head_eruption_rejects_invalid_area_configuration():
    world = make_world()
    world.config["lipAreaMinMkm2"] = 0.0
    with pytest.raises(ValueError, match="LIP area bounds"):
        birth_plume(
            world, [1.0, 0.0, 0.0], head=True, rift=False,
            name="Bad area", life_myr=10.0, strength=1.0,
        )


def test_tail_track_preserves_lip_and_builds_a_separate_volcano():
    world = make_world()
    world.time = 10.0
    world.hotspotsReady = True
    world.mesh.xyz = np.array([
        1.0, 0.0, 0.0,
        np.cos(0.01), np.sin(0.01), 0.0,
        np.cos(0.02), np.sin(0.02), 0.0,
        0.0, 1.0, 0.0,
    ], dtype=np.float64)
    world.mesh.adj_offset = np.array([0, 1, 3, 5, 6], dtype=np.int32)
    world.mesh.adj_list = np.array([1, 0, 2, 1, 3, 2], dtype=np.int32)
    world.mesh.index = SimpleNamespace(nearest=lambda x, y, z: 0)
    crust = world.plates[0].crust
    crust.volcType[0] = 2
    crust.volcKm[0] = 2.0
    plume = {
        "id": 0, "name": "Track", "xyz": [1.0, 0.0, 0.0],
        "bornAt": 0.0, "dieAt": 100.0, "alive": True, "strength": 0.5,
        "pulseUntil": 100.0, "bursting": True, "pulseIntensity": 1.2,
    }
    world.hotspots = [plume]

    update_plumes(world, 1.0)

    assert crust.volcType[0] == 2
    assert crust.volcKm[0] == pytest.approx(3.2)
    assert crust.volcType[1] == 1
    assert crust.volcType[2] == 1
    assert crust.volcKm[1] == 0.0
    assert crust.volcKm[2] == 0.0
    assert crust.volcAt[0] == pytest.approx(10.0)
    assert crust.volcSource[2] == 1


def test_erupted_during_advances_quiet_and_burst_intervals():
    world = make_world()
    rng_values = iter([0.0, 0.0, 0.0, 0.5])
    world.rngVolcano = lambda: next(rng_values)
    plume = {
        "pulseUntil": None, "bursting": False, "pulseIntensity": 0.0,
    }

    erupted = _erupted_during(world, plume, 0.0, 2.0)

    assert erupted == pytest.approx(0.3)
    assert plume["bursting"] is False
    assert plume["pulseIntensity"] == pytest.approx(0.5)
    assert plume["pulseUntil"] == pytest.approx(3.85)


def test_track_skips_unowned_provisional_and_empty_crust_sites():
    world = make_world()
    world.time = 10.0
    world.mesh.index = SimpleNamespace(nearest=lambda x, y, z: 0)
    plume = {
        "id": 0, "xyz": [1.0, 0.0, 0.0], "strength": 1.0,
        "pulseUntil": None, "bursting": False, "pulseIntensity": 0.0,
    }

    world.owner[0] = -1
    _stamp_track(world, plume, 1.0)
    assert plume["pulseUntil"] is None

    world.owner[0] = 0
    world.provisional[0] = 1
    _stamp_track(world, plume, 1.0)
    assert plume["pulseUntil"] is None

    world.provisional[0] = 0
    world.plates[0].crust.mask[0] = 0
    _stamp_track(world, plume, 1.0)
    assert plume["pulseUntil"] is None
