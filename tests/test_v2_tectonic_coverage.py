import copy

import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.core.rotation import quat_from_axis_angle
from orogen.v2.tectonics.crust import (
    BeltSide,
    CrustFlag,
    CrustRaster,
    CrustType,
    OrogenyType,
    VolcanicType,
)
from orogen.v2.tectonics.plate import Plate, frame_rotation
from orogen.v2.tectonics.world import EventKind, TectonicWorld


def test_crust_create_recreate_count_remove_and_snapshot_restore():
    crust = CrustRaster(4)
    with pytest.raises(ValueError):
        CrustRaster(0)
    crust.create(1, CrustType["CONTINENT"], -12.5, 7)
    crust.thickKm[1] = 31.5
    crust.flags[1] = CrustFlag["SUTURE"]
    assert crust.count == 1
    snapshot = crust.snapshot(1)
    crust.create(1, CrustType["OCEAN"], -4, 2)
    assert crust.count == 1
    assert crust.thickKm[1] == 0
    crust.restore(2, snapshot)
    assert crust.count == 2
    assert crust.type[2] == CrustType["CONTINENT"]
    assert crust.terrane[2] == 7
    assert crust.flags[2] == CrustFlag["SUTURE"]
    with pytest.raises(ValueError):
        crust.restore(3, [1])
    crust.remove(1)
    crust.remove(1)
    assert crust.count == 1
    assert crust.mask.tolist() == [0, 0, 1, 0]


def test_crust_copy_rotates_far_field_direction_and_preserves_fields():
    source = CrustRaster(3)
    target = CrustRaster(3)
    source.create(0, CrustType["OCEAN"], -30, 12)
    source.thickKm[0] = 8.5
    source.farDx[0], source.farDy[0], source.farDz[0] = 127, 0, 0
    target.copy_from(2, source, 0, [0, -1, 0, 1, 0, 0, 0, 0, 1])
    assert target.count == 1
    assert target.thickKm[2] == pytest.approx(8.5)
    assert target.farDx[2] == 0
    assert target.farDy[2] == 127
    target.copy_from(2, source, 0)
    assert target.count == 1
    assert target.farDx[2] == 127


def test_crust_trench_subduction_and_orogeny_lifecycle():
    crust = CrustRaster(2)
    crust.create(0, CrustType["CONTINENT"], 0, 1)
    crust.arm_trench(0, 40)
    crust.arm_trench(0, 25)
    assert crust.flags[0] & CrustFlag["TRENCH"]
    assert crust.slabKm[0] == pytest.approx(40)
    crust.record_subduction(0, 12.5)
    assert crust.slabKm[0] == pytest.approx(52.5)

    crust.record_orogeny(0, OrogenyType["INTRAPLATE"], 3, 10, 5)
    assert crust.oroType[0] == OrogenyType["INTRAPLATE"]
    assert crust.oroStart[0] == pytest.approx(10)
    assert crust.oroWork[0] == pytest.approx(3)
    crust.record_orogeny(0, OrogenyType["ANDEAN"], 2, 12, 5)
    assert crust.oroType[0] == OrogenyType["ANDEAN"]
    assert crust.oroWork[0] == pytest.approx(5)
    crust.record_orogeny(0, OrogenyType["ARC"], 1, 13, 5)
    assert crust.oroType[0] == OrogenyType["ANDEAN"]
    crust.record_orogeny(0, OrogenyType["URAL"], 4, 30, 5)
    assert crust.oroType[0] == OrogenyType["URAL"]
    assert crust.oroStart[0] == pytest.approx(30)
    assert crust.oroWork[0] == pytest.approx(4)


def test_plate_frames_extent_and_reshaped_flags():
    mesh = build_sim_mesh(64, 31)
    plate = Plate(4, mesh.num_cells)
    assert np.allclose(plate.to_world(1, 0, 0), [1, 0, 0])
    assert np.allclose(plate.to_plate_frame(0, 1, 0), [0, 1, 0])
    plate.set_quat(quat_from_axis_angle(0, 0, 1, np.pi / 2))
    assert np.allclose(plate.to_world(1, 0, 0), [0, 1, 0], atol=1e-7)
    assert np.allclose(plate.to_plate_frame(0, 1, 0), [1, 0, 0], atol=1e-7)
    assert np.allclose(frame_rotation(plate, plate), np.eye(3), atol=1e-7)
    plate.crust.create(0, CrustType["CRATON"], 0, 1)
    plate.crust.create(1, CrustType["OCEAN"], 0, 2)
    plate.mark_reshaped()
    assert plate.dirty and plate.reshaped
    plate.refresh_extent(mesh.xyz)
    assert plate.stats == {"cells": 2, "continentalCells": 1}
    assert plate.crust.count == 2
    assert not plate.dirty
    cx, cy, cz, cap_cos = plate.cap_in_world(mesh)
    assert np.isclose(np.linalg.norm([cx, cy, cz]), 1)
    assert -1 <= cap_cos <= 1


def test_plate_empty_and_antipodal_extent_fallbacks():
    mesh = build_sim_mesh(24, 13)
    plate = Plate(0, mesh.num_cells)
    plate.refresh_extent(mesh.xyz)
    assert plate.stats["cells"] == 0
    assert plate.capCenter == [0.0, 0.0, 1.0]
    assert plate.capCos == -1.0
    plate.crust.create(0, CrustType["OCEAN"], 0)
    plate.crust.create(1, CrustType["OCEAN"], 0)
    points = np.zeros((mesh.num_cells, 3))
    points[0] = [1, 0, 0]
    points[1] = [-1, 0, 0]
    plate.refresh_extent(points)
    assert plate.capCenter == [0.0, 0.0, 1.0]
    assert plate.capCos == -1.0


def test_world_config_copy_rng_terrane_naming_and_event_validation():
    mesh = build_sim_mesh(24, 13)
    config = {"nested": {"value": 1}}
    world = TectonicWorld(mesh, 77, config, {"extra": True})
    config["nested"]["value"] = 9
    assert world.config["nested"]["value"] == 1
    assert world.config["extra"] is True
    same = TectonicWorld(mesh, 77, {"nested": {"value": 1}, "extra": True})
    assert [world.rng() for _ in range(5)] == [same.rng() for _ in range(5)]
    with pytest.raises(ValueError):
        TectonicWorld(mesh, 1, None)

    plate = world.add_plate("old")
    plate.crust.create(0, CrustType["CRATON"], 0, 8)
    assert world.next_terrane_id() == 9
    assert world.next_terrane_id() == 10
    world.terraneLetters = {8: "B", 9: "A"}
    assert world.name_by_cratons(plate, "fallback") == "B"
    plate.crust.type[0] = CrustType["OCEAN"]
    assert world.name_by_cratons(plate, "fallback") == "fallback"
    world.log(EventKind["START"], "started", {"seed": 77})
    assert world.events[-1] == {
        "time": 0.0,
        "kind": "start",
        "text": "started",
        "data": {"seed": 77},
    }
    world.log(EventKind["RIFT"], "rift")
    assert world.events[-1]["data"] == {}
    with pytest.raises(ValueError):
        world.log("not-an-event", "invalid")


def test_world_terrane_overflow():
    mesh = build_sim_mesh(12, 7)
    world = TectonicWorld(mesh, 1, {})
    world.lastTerraneId = 0xFFFF
    with pytest.raises(OverflowError):
        world.next_terrane_id()


def test_rotated_plate_gather_maps_world_cells_to_plate_frame():
    from orogen.v2.core.rotation import quat_from_axis_angle
    from orogen.v2.tectonics.crust import CrustType
    from orogen.v2.tectonics.gather import for_each_claim
    from orogen.v2.tectonics.world import TectonicWorld

    mesh = build_sim_mesh(48, 17)
    world = TectonicWorld(mesh, 19, {"simCells": 48})
    plate = world.add_plate("rotated")
    plate.set_quat(quat_from_axis_angle(0, 0, 1, np.pi / 2))
    world_cell = 5
    x, y, z = mesh.xyz.reshape(-1, 3)[world_cell]
    local = plate.to_plate_frame(float(x), float(y), float(z))
    raster_cell = int(mesh.index.nearest(*map(float, local)))
    plate.crust.create(raster_cell, CrustType["OCEAN"], 0, 4)
    plate.refresh_extent(mesh.xyz)

    claims = []
    for_each_claim(world, lambda w, p, c: claims.append((w, p.id, c)))
    assert (world_cell, plate.id, raster_cell) in claims


def test_crust_restore_over_existing_cell_does_not_increment_count():
    crust = CrustRaster(2)
    crust.create(0, CrustType["CONTINENT"], 0, 1)
    values = crust.snapshot(0)
    crust.restore(0, values)
    assert crust.count == 1
    assert crust.mask.tolist() == [1, 0]
