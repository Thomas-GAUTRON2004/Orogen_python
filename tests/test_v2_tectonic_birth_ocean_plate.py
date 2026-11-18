import importlib

import numpy as np

from orogen.v2.tectonics.ops import birth_ocean_plate, ocean_plate_name

birth_module = importlib.import_module(
    "orogen.v2.tectonics.ops.birth_ocean_plate"
)
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.world import TectonicWorld


class Index:
    def __init__(self, points):
        self.mapping = {
            tuple(round(float(v), 5) for v in point): i
            for i, point in enumerate(points)
        }

    def nearest(self, x, y, z):
        key = (round(float(x), 5), round(float(y), 5), round(float(z), 5))
        return self.mapping[key]


class Mesh:
    def __init__(self):
        points = [
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
        self.num_cells = len(points)
        self.xyz = np.asarray(points, dtype=np.float32).reshape(-1)
        self.index = Index(points)
        self.edge_km = 1000.0
        self.cell_area_km2 = 100.0
        self.adj_offset = np.asarray([0, 1, 3, 5, 7, 8], dtype=np.int32)
        self.adj_list = np.asarray([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int32)


def make_world(minimum_area=150.0):
    mesh = Mesh()
    world = TectonicWorld(
        mesh,
        seed=42,
        config={
            "oceanPlateMinAreaKm2": minimum_area,
        },
    )
    world.time = 18.0
    background = world.add_plate("Panthalassa")
    background.isBackground = True
    for cell in (1, 2, 3):
        background.crust.create(cell, CrustType["OCEAN"], -100.0, 5)
    world.owner[1:4] = background.id
    world.ownerCell[:] = np.arange(mesh.num_cells, dtype=np.int32)
    return world, background


def test_birth_ocean_plate_transfers_reachable_background_crust_and_logs():
    world, background = make_world()
    fresh = birth_ocean_plate(
        world, background, seed_cells=[0], reach_km=1000.0, name="Thalassa"
    )

    assert fresh is not None
    assert fresh.id == 1
    assert fresh.name == "Thalassa"
    assert fresh.isOceanBorn is True
    assert fresh.born == world.time
    assert fresh.omega == [0.0, 0.0, 0.0]
    assert fresh.crust.mask.tolist() == [0, 1, 1, 0, 0]
    assert fresh.crust.count == 2
    assert background.crust.count == 1
    assert background.crust.mask[3] == 1

    event = world.events[-1]
    assert event["kind"] == "plate-born"
    assert event["data"] == {"plate": 1, "areaMkm2": 0.0002}
    assert "0.0 Mkm²" in event["text"]


def test_birth_ocean_plate_respects_minimum_area_and_does_not_add_plate():
    world, background = make_world(minimum_area=250.0)
    fresh = birth_ocean_plate(
        world, background, seed_cells=[0], reach_km=1000.0, name="small"
    )
    assert fresh is None
    assert len(world.plates) == 1
    assert background.crust.count == 3
    assert world.events == []


def test_birth_ocean_plate_returns_none_when_no_background_is_reachable():
    world, background = make_world()
    world.owner[:] = -1
    fresh = birth_ocean_plate(
        world, background, seed_cells=[4], reach_km=1000.0, name="empty"
    )
    assert fresh is None
    assert len(world.plates) == 1


def test_birth_ocean_plate_marks_empty_transfer_dead(monkeypatch):
    world, background = make_world()
    monkeypatch.setattr(birth_module, "transfer_cells", lambda *_args: 0)
    fresh = birth_ocean_plate(
        world, background, seed_cells=[0], reach_km=1000.0, name="empty"
    )
    assert fresh is None
    assert len(world.plates) == 2
    assert world.plates[-1].alive is False
    assert world.events == []


def test_ocean_plate_names_increment_by_existing_ocean_born_plates():
    world, background = make_world()
    assert ocean_plate_name(world) == "Thalassa"
    a = world.add_plate("Thalassa")
    a.isOceanBorn = True
    b = world.add_plate("another")
    b.isOceanBorn = True
    assert ocean_plate_name(world) == "Thalassa 3"

def test_birth_ocean_plate_deduplicates_raster_cells_and_skips_empty_rasters():
    world, background = make_world(minimum_area=0.0)
    # World cells 1 and 2 map to the same raster cell; cell 3 maps to an
    # unoccupied raster cell. Only one unique, occupied raster cell transfers.
    world.ownerCell[2] = 1
    world.ownerCell[3] = 0
    fresh = birth_ocean_plate(
        world, background, seed_cells=[0, 3], reach_km=1000.0, name="dedup"
    )
    assert fresh is not None
    assert fresh.crust.count == 1
    assert fresh.crust.mask[1]
    assert not fresh.crust.mask[0]

