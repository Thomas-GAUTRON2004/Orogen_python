"""Tests for the collision-time plate welding operation."""

import numpy as np

from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType
from orogen.v2.tectonics.ops.merge_plates import _stamp_cratons_rigidly, merge_plates
from orogen.v2.tectonics.plate import Plate


def merge_config():
    return {
        "indentFalloffKm": 50.0,
        "indentGain": 0.01,
        "escapeShare": 0.7,
        "sutureHalfWidthKm": 200.0,
        "sealStraitKm": 100.0,
        "inlandSeaMaxKm2": 0.0,
        "apronKm": 100.0,
        "cratonRimKm": 200.0,
        "cratonRimCost": 2.0,
        "cratonRimShare": 0.25,
        "weakCellCost": 0.5,
        "orogenyQuietMyr": 10.0,
        "beltWorkShare": 0.5,
        "thickeningPerWorkKm": 0.1,
        "belts": {
            "himalayan": {"baseKm": 100.0, "perKm": 0.5, "maxKm": 500.0},
            "foreland": {"baseKm": 80.0, "perKm": 0.2, "maxKm": 300.0},
            "ural": {"baseKm": 50.0, "perKm": 0.2, "maxKm": 200.0},
        },
        "beltSpeedFactorMin": 0.6,
        "beltSpeedFactorMax": 1.6,
        "beltSpeedRefCmYr": 5.0,
        "beltSpeedExponent": 0.5,
    }


class Index:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

    def nearest(self, x, y, z):
        point = np.asarray([x, y, z], dtype=float)
        return int(np.argmin(np.sum((self.points - point) ** 2, axis=1)))


class Mesh:
    def __init__(self):
        angles = np.arange(7, dtype=float) * 0.02
        points = np.column_stack((
            np.cos(angles), np.sin(angles), np.zeros(7)
        ))
        self.xyz = points.astype(np.float32).ravel()
        self.num_cells = 7
        self.edge_km = 100.0
        self.edge_rad = self.edge_km / 6371.0
        self.cell_area_km2 = 100.0
        self.index = Index(points)
        self.adj_offset = np.asarray([0, 4, 8, 11, 14, 16, 17, 17], dtype=np.int32)
        self.adj_list = np.asarray([
            4, 1, 2, 3,
            0, 2, 3, 5,
            0, 1, 3,
            0, 2, 4,
            0, 3,
            1,
        ], dtype=np.int32)


def merge_fixture(gain=0.01):
    # A five-cell star gives the missing centre four crust neighbours so the
    # inverse-warp pinhole repair is exercised as well.
    mesh = Mesh()
    upper, lower = Plate(0, 7), Plate(1, 7)
    upper.name, lower.name = "Upper", "Lower"
    upper.capCenter = [0.0, 0.0, 1.0]
    lower.capCenter = [0.0, 0.0, -1.0]
    upper.crust.create(1, CrustType["CONTINENT"], -10.0, 10)
    upper.crust.create(2, CrustType["CONTINENT"], -15.0, 15)
    upper.crust.create(4, CrustType["CRATON"], -100.0, 20)
    upper.crust.create(5, CrustType["CONTINENT"], -25.0, 25)
    lower.crust.create(1, CrustType["CONTINENT"], -20.0, 30)
    lower.crust.create(2, CrustType["CONTINENT"], -25.0, 35)
    lower.crust.create(3, CrustType["CONTINENT"], -30.0, 50)
    lower.crust.create(4, CrustType["CONTINENT"], -40.0, 60)
    lower.crust.create(5, CrustType["OCEAN"], -5.0, 40)
    upper.refresh_extent(mesh.xyz)
    lower.refresh_extent(mesh.xyz)
    upper.hue, lower.hue = 11, 22
    world = type("World", (), {})()
    world.mesh = mesh
    world.time = 25.0
    world.config = merge_config()
    world.config["indentGain"] = gain
    world.name_by_cratons = lambda _plate, fallback: fallback
    return world, upper, lower


def test_merge_plates_redraws_upper_and_retires_lower_with_a_suture():
    world, upper, lower = merge_fixture()
    result = merge_plates(
        world,
        upper,
        lower,
        [1],
        {"shorteningKm": 500.0, "meanRateCmYr": 5.0},
    )

    assert result["overlapCells"] == 1
    assert result["maxIndentKm"] > 0.0
    assert result["meanIndentKm"] > 0.0
    assert result["sealedCells"] == 0
    assert upper.crust.mask[0] == 1  # the four-neighbour pinhole was filled
    assert upper.crust.type[4] == CrustType["CRATON"]\n    assert upper.crust.type[0] == CrustType["CONTINENT"]
    assert upper.crust.flags[1] & CrustFlag["SUTURE"]
    assert not (upper.crust.flags[4] & CrustFlag["SUTURE"])
    assert upper.crust.count == int(upper.stats["cells"])
    assert upper.stats["continentalCells"] == 6
    assert upper.born == world.time
    assert upper.bornOfRift is False
    assert upper.wings == {}
    assert upper.hue == 22
    assert lower.alive is False
    assert lower.crust.count == 0
    assert lower.omega == [0.0, 0.0, 0.0]
    assert lower.dynamics is None
    assert lower.stats["cells"] == 0


def test_merge_plates_handles_a_world_overlap_fallback_without_overlap_raster():
    world, upper, lower = merge_fixture(gain=0.0)
    upper.crust.remove(1)
    lower.crust.remove(1)
    upper.refresh_extent(world.mesh.xyz)
    lower.refresh_extent(world.mesh.xyz)

    result = merge_plates(
        world,
        upper,
        lower,
        [1],
        {"shorteningKm": 200.0, "meanRateCmYr": 2.0},
    )

    assert result["overlapCells"] == 1
    assert result["meanIndentKm"] > 0.0
    assert lower.alive is False
    assert upper.crust.count > 0


def test_merge_plates_with_zero_indent_keeps_raster_locations_stable():
    world, upper, lower = merge_fixture(gain=0.0)
    result = merge_plates(
        world,
        upper,
        lower,
        [1],
        {"shorteningKm": 50.0, "meanRateCmYr": 1.0},
    )
    assert result["overlapCells"] == 1
    assert upper.crust.mask[1] == 1
    assert upper.crust.mask[3] == 1
    assert upper.crust.mask[4] == 1


def test_merge_hole_repair_copies_open_sea_when_sea_neighbours_dominate():
    world, upper, lower = merge_fixture(gain=0.0)
    upper.crust.remove(2)
    upper.crust.remove(4)
    lower.crust.type[2] = CrustType["OCEAN"]
    lower.crust.type[3] = CrustType["OCEAN"]
    lower.crust.type[4] = CrustType["OCEAN"]
    result = merge_plates(
        world,
        upper,
        lower,
        [1],
        {"shorteningKm": 100.0, "meanRateCmYr": 2.0},
    )
    assert result["overlapCells"] == 1
    assert upper.crust.mask[0] == 1
    assert upper.crust.type[0] == CrustType["OCEAN"]


def test_stamp_cratons_skips_a_zero_length_stamping_position():
    world, upper, _lower = merge_fixture(gain=0.0)
    world.mesh.xyz[12:15] = 0.0
    crust = CrustRaster(world.mesh.num_cells)
    crust.create(4, CrustType["CRATON"], -100.0, 20)
    output = CrustRaster(world.mesh.num_cells)
    _stamp_cratons_rigidly(
        output,
        crust,
        np.zeros((world.mesh.num_cells, 3), dtype=np.float32),
        world.mesh,
    )
    assert output.count == 0
