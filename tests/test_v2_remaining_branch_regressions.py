"""Regression tests for remaining tectonic decision branches."""

from types import SimpleNamespace

import numpy as np

from orogen.v2.tectonics import backarc, kinematics, resolve
from orogen.v2.tectonics.crust import CrustRaster, CrustType
from orogen.v2.tectonics.ops.merge_plates import _fill_inverse_warp_holes
from orogen.v2.tectonics.ops.plate_accretion import landmass_of
from orogen.v2.tectonics.spread_belt import _update_belt_front


def test_zero_rotation_returns_the_original_pose_without_normalizing():
    pose = [2.0, 0.0, 0.0, 0.0]
    assert kinematics.pose_at_constant_omega(pose, [0.0, 0.0, 0.0], 50.0) == pose


def test_spread_front_keeps_an_equal_or_worse_existing_front():
    crust = CrustRaster(1)
    crust.create(0, CrustType["CONTINENT"], 0.0)
    crust.oroType[0] = 7
    crust.beltReach[0] = 100.0
    crust.beltAt[0] = 10.0
    crust.beltU[0] = 0.5
    crust.beltShear[0] = 12
    _update_belt_front(crust, 0, {"keepShear": False, "shear": 30}, 0, 7, 1, 0.5, 200.0, 10.0)
    assert crust.beltReach[0] == 100.0
    assert crust.beltShear[0] == 12


def test_backarc_band_does_not_cross_unowned_crust_cells():
    points = np.asarray(
        [[1.0, 0.0, 0.0], [0.999, 0.04, 0.0], [0.996, 0.08, 0.0]],
        dtype=np.float32,
    )
    mesh = SimpleNamespace(
        num_cells=3,
        xyz=points.reshape(-1),
        adj_offset=np.asarray([0, 1, 3, 4], dtype=np.int32),
        adj_list=np.asarray([1, 0, 2, 1], dtype=np.int32),
    )
    crust = CrustRaster(3)
    crust.create(0, CrustType["CONTINENT"], 0.0)
    crust.create(2, CrustType["CONTINENT"], 0.0)
    plate = SimpleNamespace(crust=crust)
    world = SimpleNamespace(mesh=mesh, ownerCell=np.arange(3, dtype=np.int32))
    strip = backarc._band_behind(world, plate, [0], 1000.0)
    assert strip.tolist() == [1, 0, 0]


def test_inverse_warp_hole_fill_uses_land_majority_and_ignores_sparse_gaps():
    output = CrustRaster(6)
    for cell in (1, 2, 3):
        output.create(cell, CrustType["CONTINENT"], -10.0, cell)
    output.create(4, CrustType["OCEAN"], -20.0, 4)
    offsets = np.asarray([0, 4, 4, 4, 4, 4, 4], dtype=np.int32)
    adjacency = np.asarray([1, 2, 3, 4], dtype=np.int32)
    _fill_inverse_warp_holes(output, np.zeros(6, dtype=np.int8), offsets, adjacency, 6)
    assert output.mask[0]
    assert output.type[0] == CrustType["CONTINENT"]
    assert not output.mask[5]


def test_landmass_walk_ignores_ocean_empty_and_duplicate_seeds():
    crust = CrustRaster(4)
    crust.create(0, CrustType["CONTINENT"], 0.0)
    crust.create(1, CrustType["OCEAN"], 0.0)
    crust.create(2, CrustType["CRATON"], 0.0)
    mesh = SimpleNamespace(
        adj_offset=np.asarray([0, 2, 3, 4, 4], dtype=np.int32),
        adj_list=np.asarray([1, 2, 0, 0], dtype=np.int32),
    )
    plate = SimpleNamespace(crust=crust)
    assert landmass_of(plate, mesh, [1, 3, 0, 0]) == [0, 2]


def test_fracture_coordinate_uses_zero_relative_rotation_without_a_partner():
    points = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mesh = SimpleNamespace(
        xyz=points.reshape(-1),
        index=SimpleNamespace(nearest=lambda *_args: 0),
        adj_offset=np.asarray([0, 1, 2], dtype=np.int32),
        adj_list=np.asarray([1, 0], dtype=np.int32),
    )
    grower = SimpleNamespace(
        id=0,
        omega=[0.0, 0.0, 0.0],
        to_plate_frame=lambda x, y, z: [x, y, z],
    )
    world = SimpleNamespace(mesh=mesh, owner=np.asarray([0, 0]), plates=[grower])
    value = resolve._fracture_at(world, grower, 1, [1.0, 0.0, 0.0])
    assert np.isfinite(value)
