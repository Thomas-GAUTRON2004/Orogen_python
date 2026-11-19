"""Regression tests for remaining tectonic decision branches."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics import backarc, kinematics
from orogen.v2.tectonics.crust import CrustRaster, CrustType
from orogen.v2.tectonics.ops import merge_plates as merge_ops
from orogen.v2.tectonics.ops.merge_plates import _fill_inverse_warp_holes
from orogen.v2.tectonics.ops.plate_accretion import landmass_of
from orogen.v2.tectonics.spread_belt import _update_belt_front

resolve = import_module("orogen.v2.tectonics.resolve")



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
    _update_belt_front(
        crust, 0, {"keepShear": False, "shear": 30}, 0, 7, 1, 0.5, 200.0, 10.0
    )
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



def test_backarc_bridge_fills_between_disconnected_crust_pieces():
    mesh = SimpleNamespace(
        num_cells=5,
        edge_km=100.0,
        adj_offset=np.asarray([0, 1, 3, 5, 7, 8], dtype=np.int32),
        adj_list=np.asarray([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int32),
    )
    crust = CrustRaster(5)
    crust.create(0, CrustType["CONTINENT"], -20.0, 10)
    crust.create(4, CrustType["OCEAN"], -5.0, 20)
    plate = SimpleNamespace(crust=crust, mark_reshaped=lambda: None)

    filled = backarc._bridge_pieces(plate, mesh, 300.0)

    assert filled == 3
    assert crust.mask[1] and crust.mask[2] and crust.mask[3]
    assert crust.type[1] == CrustType["OCEAN"]
    assert crust.type[2] == CrustType["OCEAN"]
    assert crust.type[3] == CrustType["OCEAN"]
    assert crust.formedAt[1] == -20.0
    assert crust.formedAt[2] == -5.0
    assert crust.formedAt[3] == -5.0



def test_craton_picker_falls_back_when_requested_count_exceeds_core():
    points = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    picked = __import__(
        "orogen.v2.tectonics.initial_world", fromlist=["_pick_craton_centres"]
    )._pick_craton_centres(
        points,
        np.asarray([0, 1], dtype=np.int32),
        points[0],
        0.0,
        3,
        1.0e9,
        100.0,
        lambda: 0.5,
    )
    assert picked.tolist() == [0, 1]


def test_surface_velocity_reuses_a_caller_owned_output_buffer():
    output = [9.0, 9.0, 9.0]
    result = kinematics.surface_velocity([1.0, 2.0, 3.0], 4.0, 5.0, 6.0, output)
    assert result is output
    assert output == [-3.0, 6.0, -3.0]



def test_backarc_band_skips_a_duplicate_world_to_plate_seed():
    mesh = SimpleNamespace(
        num_cells=2,
        xyz=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32).reshape(
            -1
        ),
        adj_offset=np.asarray([0, 1, 2], dtype=np.int32),
        adj_list=np.asarray([1, 0], dtype=np.int32),
    )
    crust = CrustRaster(2)
    crust.create(0, CrustType["CONTINENT"], -10.0)
    world = SimpleNamespace(mesh=mesh, ownerCell=np.asarray([0, 0], dtype=np.int32))

    strip = backarc._band_behind(world, SimpleNamespace(crust=crust), [0, 1], 1000.0)

    assert strip.tolist() == [1, 0]



def test_backarc_bridge_skips_interior_cells_without_crust_neighbours_and_noop():
    def line_mesh(count):
        rows = []
        for cell in range(count):
            row = []
            if cell:
                row.append(cell - 1)
            if cell + 1 < count:
                row.append(cell + 1)
            rows.append(row)
        offsets, adjacency = [0], []
        for row in rows:
            adjacency.extend(row)
            offsets.append(len(adjacency))
        return SimpleNamespace(
            num_cells=count,
            edge_km=100.0,
            adj_offset=np.asarray(offsets, dtype=np.int32),
            adj_list=np.asarray(adjacency, dtype=np.int32),
        )

    mesh = line_mesh(7)
    crust = CrustRaster(7)
    crust.create(0, CrustType["CONTINENT"], -20.0)
    crust.create(6, CrustType["OCEAN"], -5.0)
    changed = []
    plate = SimpleNamespace(crust=crust, mark_reshaped=lambda: changed.append(True))

    assert backarc._bridge_pieces(plate, mesh, 700.0) == 2
    assert crust.mask[1] and crust.mask[5]
    assert not crust.mask[2] and not crust.mask[3] and not crust.mask[4]
    assert changed == [True]

    far_crust = CrustRaster(7)
    far_crust.create(0, CrustType["CONTINENT"], -20.0)
    far_crust.create(6, CrustType["OCEAN"], -5.0)
    unchanged = []
    far_plate = SimpleNamespace(
        crust=far_crust, mark_reshaped=lambda: unchanged.append(True)
    )
    assert backarc._bridge_pieces(far_plate, mesh, 100.0) == 0
    assert unchanged == []



def test_merged_crust_handles_zero_length_warp_and_empty_filtered_sources(monkeypatch):
    mesh = SimpleNamespace(
        num_cells=1,
        xyz=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        index=SimpleNamespace(
            nearest=lambda *_args: pytest.fail("zero vector must not locate")
        ),
        adj_offset=np.asarray([0, 0], dtype=np.int32),
        adj_list=np.asarray([], dtype=np.int32),
    )
    monkeypatch.setattr(merge_ops, "_stamp_cratons_rigidly", lambda *_args: None)

    upper = CrustRaster(1)
    upper.create(0, CrustType["CONTINENT"], -10.0)
    empty_lower = CrustRaster(1)
    merged, _origin = merge_ops._build_merged_crust(
        upper,
        empty_lower,
        np.asarray([-1], dtype=np.int32),
        np.asarray([[6371.0, 0.0, 0.0]], dtype=np.float64),
        mesh,
        np.eye(3),
    )
    assert merged.mask[0]

    empty_upper = CrustRaster(1)
    empty, _origin = merge_ops._build_merged_crust(
        empty_upper,
        empty_lower,
        np.asarray([-1], dtype=np.int32),
        np.zeros((1, 3), dtype=np.float64),
        mesh,
        np.eye(3),
    )
    assert empty.count == 0

    craton_upper = CrustRaster(1)
    craton_upper.create(0, CrustType["CRATON"], -100.0)
    ocean_lower = CrustRaster(1)
    ocean_lower.create(0, CrustType["OCEAN"], -5.0)
    preserved, _origin = merge_ops._build_merged_crust(
        craton_upper,
        ocean_lower,
        np.asarray([0], dtype=np.int32),
        np.zeros((1, 3), dtype=np.float64),
        mesh,
        np.eye(3),
    )
    assert preserved.type[0] == CrustType["CRATON"]
