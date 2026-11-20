import numpy as np

from orogen.v2.tectonics.crust import (
    CELL_FIELDS,
    CrustFlag,
    CrustRaster,
    CrustType,
    OrogenyType,
)
from orogen.v2.tectonics.plate import Plate, frame_rotation


def test_crust_raster_create_snapshot_restore_and_remove():
    raster = CrustRaster(8)
    raster.create(3, CrustType["CONTINENT"], 12.5, 9)
    raster.arm_trench(3, 40.0)
    raster.record_subduction(3, 2.5)
    raster.record_orogeny(3, OrogenyType["ANDEAN"], 5.0, 12.5, 10.0)
    assert raster.count == 1
    assert raster.mask[3] == 1
    assert raster.flags[3] & CrustFlag["TRENCH"]
    assert np.isclose(raster.slabKm[3], 42.5)
    assert np.isclose(raster.oroWork[3], 5.0)
    snap = raster.snapshot(3)
    raster.create(3, CrustType["OCEAN"], 20.0)
    raster.restore(4, snap)
    assert raster.type[4] == CrustType["CONTINENT"]
    assert raster.terrane[4] == 9
    raster.remove(3)
    assert raster.count == 1
    raster.remove(4)
    assert raster.count == 0
    assert len(CELL_FIELDS) == len(set(CELL_FIELDS)) == len(raster.fields)


def test_orogeny_escalation_and_quiet_gap_restart():
    raster = CrustRaster(2)
    raster.create(0, CrustType["CONTINENT"], 0)
    raster.record_orogeny(0, OrogenyType["ANDEAN"], 10, 10, 20)
    raster.record_orogeny(0, OrogenyType["HIMALAYAN"], 5, 15, 20)
    assert raster.oroType[0] == OrogenyType["HIMALAYAN"]
    assert raster.oroWork[0] == 15
    raster.record_orogeny(0, OrogenyType["ARC"], 3, 50, 20)
    assert raster.oroType[0] == OrogenyType["ARC"]
    assert raster.oroWork[0] == 3
    assert raster.oroStart[0] == 50


def test_plate_frame_transforms_and_cap_extent():
    plate = Plate(4, 5)
    plate.crust.create(0, CrustType["CRATON"], 0)
    plate.crust.create(1, CrustType["OCEAN"], 0)
    plate.refresh_extent(
        np.array([1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1], dtype=np.float32)
    )
    assert plate.stats == {"cells": 2, "continentalCells": 1}
    world = plate.to_world(0, 0, 1)
    local = plate.to_plate_frame(*world)
    assert np.allclose(local, [0, 0, 1])
    assert (
        plate.cell_at(type("Index", (), {"nearest": lambda self, x, y, z: 2})(), *world)
        == 2
    )
    assert np.allclose(frame_rotation(plate, plate), np.eye(3))
    plate.mark_reshaped()
    assert plate.reshaped and plate.dirty
