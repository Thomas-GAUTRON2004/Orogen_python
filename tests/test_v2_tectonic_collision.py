import math
from types import SimpleNamespace

import numpy as np

from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.collision import (
    are_young_rift_partners,
    choose_sides,
    collect_contacts,
    forget_stale_collisions,
    underthrust,
)


class Index:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

    def nearest(self, x, y, z):
        target = np.asarray([x, y, z], dtype=float)
        return int(np.argmin(np.sum((self.points - target) ** 2, axis=1)))


class Mesh:
    def __init__(self, points, neighbours, cell_area=100.0):
        self.num_cells = len(points)
        self.xyz = np.asarray(points, dtype=np.float32).reshape(-1)
        self.index = Index(points)
        self.cell_area_km2 = float(cell_area)
        self.edge_km = 100.0
        self.edge_rad = self.edge_km / 6371.0
        offsets = [0]
        adjacency = []
        for row in neighbours:
            adjacency.extend(row)
            offsets.append(len(adjacency))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)


def contact_world():
    points = [[1, 0, 0], [0, 1, 0], [-1, 0, 0]]
    mesh = Mesh(points, [[1], [0, 2], [1]])
    a = Plate(0, 3)
    b = Plate(1, 3)
    a.crust.create(0, CrustType["CONTINENT"], -100.0, 10)
    a.crust.create(2, CrustType["CONTINENT"], -100.0, 12)
    b.crust.create(0, CrustType["CONTINENT"], -90.0, 20)
    b.crust.create(1, CrustType["CONTINENT"], -90.0, 21)
    world = SimpleNamespace(
        mesh=mesh,
        owner=np.asarray([0, 1, 0], dtype=np.int16),
        ownerCell=np.asarray([0, 0, 2], dtype=np.int32),
        landClaims=np.asarray([1, 1, 2], dtype=np.uint8),
        collider=np.asarray([-1, -1, 1], dtype=np.int16),
        plates=[a, b],
    )
    return world, a, b


def underthrust_world(*, lower_kind=1, share=1.0, upper_present=True):
    points = [[1, 0, 0], [math.sqrt(0.5), math.sqrt(0.5), 0], [0, 1, 0], [-1, 0, 0]]
    mesh = Mesh(points, [[1], [0, 2], [1, 3], [2]])
    upper = Plate(0, 4)
    lower = Plate(1, 4)
    upper.crust.create(2, CrustType["CONTINENT"], -50.0, 12)
    lower.crust.create(0, CrustType["CONTINENT"], -100.0, 10)
    lower.crust.create(2, lower_kind, -100.0, 22)
    if not upper_present:
        upper.crust.remove(2)
    world = SimpleNamespace(
        mesh=mesh,
        owner=np.asarray([1, 0, 0, -1], dtype=np.int16),
        ownerCell=np.asarray([0, 1, 2, 0], dtype=np.int32),
        landClaims=np.asarray([1, 2, 2, 0], dtype=np.uint8),
        budget={"underthrustCells": 0, "created": 0, "consumed": 0,
                "erodedCells": 0, "underthrustCells": 0, "recycledCells": 0},
        config={
            "underthrustShare": share,
            "overlapCapKm": 1.0,
            "underthrustThickenKm": 10.0,
            "underthrustMaxThickKm": 50.0,
        },
    )
    return world, upper, lower


def test_collect_contacts_separates_overlap_and_touching_pairs():
    world, _a, _b = contact_world()
    result = collect_contacts(world)
    assert result["overlap"] == {"0:1": [2]}
    assert result["touching"] == {"0:1": [0, 1]}


def test_collect_contacts_ignores_ocean_background_and_rift_margins():
    world, a, b = contact_world()
    a.crust.flags[0] |= CrustFlag["RIFT_MARGIN"]
    b.crust.flags[0] |= CrustFlag["RIFT_MARGIN"]
    assert collect_contacts(world)["touching"] == {}

    a.crust.flags[0] = 0
    b.isBackground = True
    assert collect_contacts(world)["touching"] == {}


def test_collect_contacts_skips_unowned_and_same_plate_overlap_sentinels():
    world, _a, _b = contact_world()
    world.owner[0] = -1
    world.collider[2] = -1
    assert collect_contacts(world) == {"overlap": {}, "touching": {"0:1": [2]}}


def test_forget_stale_collisions_removes_only_expired_unsutured_records():
    world = SimpleNamespace(
        time=50.0,
        config={"collisionForgetMyr": 20.0},
        collisions={
            "0:1": {"lastActive": 10.0, "sutured": False},
            "0:2": {"lastActive": 40.0, "sutured": False},
            "1:2": {"lastActive": 1.0, "sutured": True},
        },
    )
    assert forget_stale_collisions(world) == 1
    assert set(world.collisions) == {"0:2", "1:2"}


def test_are_young_rift_partners_checks_mutuality_and_time_window():
    a = Plate(0, 2, born=10.0)
    b = Plate(1, 2, born=12.0)
    a.riftPartner = 1
    b.riftPartner = 0
    cfg = {"riftPushMyr": 20.0}
    assert are_young_rift_partners(a, b, 31.0, cfg)
    assert not are_young_rift_partners(a, b, 32.0, cfg)
    b.riftPartner = -1
    assert not are_young_rift_partners(a, b, 31.0, cfg)


def test_choose_sides_prefers_trench_flag_count_then_continental_share():
    world, a, b = contact_world()
    a.crust.create(1, CrustType["CONTINENT"], -100.0, 11)
    b.crust.create(2, CrustType["CONTINENT"], -90.0, 22)
    a.crust.flags[1] |= CrustFlag["TRENCH"]
    assert choose_sides(world, a, b, [0]) == {"upper": a, "lower": b}

    a.crust.flags[1] = 0
    a.stats["continentalCells"] = 2
    b.stats["continentalCells"] = 3
    assert choose_sides(world, a, b, [0]) == {"upper": b, "lower": a}

    b.stats["continentalCells"] = 2
    assert choose_sides(world, a, b, [0]) == {"upper": a, "lower": b}


def test_underthrust_consumes_deepest_lower_continent_and_thickens_upper():
    world, upper, lower = underthrust_world()
    upper.crust.create(2, CrustType["CONTINENT"], -50.0, 12)
    record = {"frontKm": 100.0}
    consumed = underthrust(world, upper, lower, [1, 2], record, 1.0)
    assert consumed == 1
    assert lower.crust.mask[2] == 0
    assert upper.crust.thickKm[2] == 10.0
    assert upper.crust.rootKm[2] == 10.0
    assert record["underthrustDue"] == 0.0
    assert record["underthrustCells"] == 1
    assert world.budget["underthrustCells"] == 1
    assert lower.reshaped is True
    assert upper.dirty is True


def test_underthrust_skips_ocean_and_craton_and_does_not_change_budget():
    world, upper, lower = underthrust_world(lower_kind=CrustType["CRATON"])
    record = {"frontKm": 100.0}
    assert underthrust(world, upper, lower, [1, 2], record, 1.0) == 0
    assert lower.crust.mask[2] == 1
    assert world.budget["underthrustCells"] == 0


def test_underthrust_returns_early_when_disabled_or_no_shortening():
    world, upper, lower = underthrust_world()
    record = {"frontKm": 100.0}
    world.config["underthrustShare"] = 0.0
    assert underthrust(world, upper, lower, [1, 2], record, 1.0) == 0
    world.config["underthrustShare"] = 1.0
    assert underthrust(world, upper, lower, [1, 2], record, 0.0) == 0


def test_underthrust_returns_when_no_lower_land_touches_overlap():
    world, upper, lower = underthrust_world()
    world.owner[0] = 0
    record = {"frontKm": 100.0}
    assert underthrust(world, upper, lower, [1, 2], record, 1.0) == 0
    assert lower.crust.mask[2] == 1


def test_underthrust_can_consume_without_upper_land_above():
    world, upper, lower = underthrust_world(upper_present=False)
    record = {"frontKm": 100.0}
    assert underthrust(world, upper, lower, [1, 2], record, 1.0) == 1
    assert lower.crust.mask[2] == 0
    assert upper.dirty is True


def test_choose_sides_deduplicates_neighbour_cells_shared_by_contact_cells():
    world, a, b = contact_world()
    world.mesh.adj_offset = np.asarray([0, 2, 4, 4], dtype=np.int32)
    world.mesh.adj_list = np.asarray([1, 2, 0, 2], dtype=np.int32)
    a.crust.create(1, CrustType["CONTINENT"], -100.0, 11)
    b.crust.create(1, CrustType["CONTINENT"], -90.0, 21)
    a.crust.flags[2] |= CrustFlag["TRENCH"]
    result = choose_sides(world, a, b, [0, 1])
    assert result == {"upper": a, "lower": b}


def test_underthrust_stops_after_target_number_of_cells():
    world, upper, lower = underthrust_world()
    lower.crust.create(3, CrustType["CONTINENT"], -100.0, 23)
    upper.crust.create(3, CrustType["CONTINENT"], -50.0, 13)
    record = {"frontKm": 100.0}
    consumed = underthrust(world, upper, lower, [1, 2, 3], record, 1.0)
    assert consumed == 1
    assert lower.crust.mask[3] == 0
    assert lower.crust.mask[2] == 1
    assert world.budget["underthrustCells"] == 1
