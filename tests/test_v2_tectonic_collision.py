import math
from types import SimpleNamespace

import numpy as np

from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType
from orogen.v2.tectonics.plate import Plate
from orogen.v2.tectonics.collision import (
    accrete_arcs,
    are_young_rift_partners,
    choose_sides,
    collect_contacts,
    dock_if_terrane,
    enclosed_sea_between_km2,
    forget_stale_collisions,
    is_rift_jitter,
    open_sea_between,
    recycle_small_arc,
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
    world.config["overlapCapKm"] = 2.0
    lower.crust.create(3, CrustType["CONTINENT"], -100.0, 23)
    upper.crust.create(3, CrustType["CONTINENT"], -50.0, 13)
    record = {"frontKm": 100.0}
    consumed = underthrust(world, upper, lower, [1, 2, 3], record, 1.0)
    assert consumed == 1
    assert lower.crust.mask[3] == 0
    assert lower.crust.mask[2] == 1
    assert world.budget["underthrustCells"] == 1


def sea_world():
    points = [
        [math.cos(i * 0.1), math.sin(i * 0.1), 0.0]
        for i in range(7)
    ]
    mesh = Mesh(
        points,
        [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5]],
        cell_area=100.0,
    )
    upper = Plate(0, 7)
    lower = Plate(1, 7)
    upper.crust.create(0, CrustType["CONTINENT"], -100.0, 10)
    lower.crust.create(3, CrustType["CONTINENT"], -100.0, 20)
    owner = np.asarray([0, -1, -1, 1, -1, -1, -1], dtype=np.int16)
    owner_cell = np.arange(7, dtype=np.int32)
    land_claims = np.asarray([1, 0, 0, 1, 0, 0, 0], dtype=np.uint8)
    world = SimpleNamespace(
        mesh=mesh,
        owner=owner,
        ownerCell=owner_cell,
        landClaims=land_claims,
        plates=[upper, lower],
        config={"sutureGapKm": 500.0},
    )
    return world, upper, lower


def test_enclosed_sea_between_counts_only_small_component_bordered_by_both_plates():
    world, upper, lower = sea_world()
    area = enclosed_sea_between_km2(world, upper, lower)
    assert area == 200.0


def test_enclosed_sea_between_returns_zero_when_no_sea_component_or_joint_border():
    world, upper, lower = sea_world()
    world.owner[1] = upper.id
    world.ownerCell[1] = 1
    upper.crust.create(1, CrustType["CONTINENT"], -20.0, 11)
    world.owner[3] = -1  # removes the lower-plate border; remaining sea joins the world ocean
    assert enclosed_sea_between_km2(world, upper, lower) == 0.0


def test_open_sea_between_measures_facing_coasts_and_width():
    world, upper, lower = sea_world()
    result = open_sea_between(world, upper, lower)
    assert result == {"km2": 200.0, "widestKm": 200.0}


def test_open_sea_between_returns_zero_when_no_opposed_sea_is_within_gap():
    world, upper, lower = sea_world()
    world.config["sutureGapKm"] = 50.0
    assert open_sea_between(world, upper, lower) == {
        "km2": 0.0, "widestKm": 0.0
    }


def test_is_rift_jitter_requires_small_overlap_and_both_rift_margins():
    world, a, b = contact_world()
    a.crust.create(1, CrustType["CONTINENT"], -100.0, 11)
    b.crust.create(1, CrustType["CONTINENT"], -90.0, 21)
    a.crust.flags[1] |= CrustFlag["RIFT_MARGIN"]
    b.crust.flags[1] |= CrustFlag["RIFT_MARGIN"]
    world.config = {"riftJitterFrontKm": 100.0}
    assert is_rift_jitter(world, a, b, [1])

    b.crust.flags[1] = 0
    assert not is_rift_jitter(world, a, b, [1])
    b.crust.flags[1] |= CrustFlag["RIFT_MARGIN"]
    world.config["riftJitterFrontKm"] = 1.0
    assert not is_rift_jitter(world, a, b, [0, 1, 2])


def test_recycle_small_arc_recycles_short_components_and_preserves_large_arcs():
    world, a, _b = contact_world()
    a.crust.create(0, CrustType["ARC"], -100.0, 30)
    a.crust.create(1, CrustType["ARC"], -100.0, 31)
    a.crust.remove(2)
    world.config = {"arcAccreteMinKm2": 500.0, "terraneMinAreaKm2": 50.0}
    world.time = 10.0
    world.budget = {"recycledCells": 0}
    world.events = []
    world.log = lambda kind, text, data=None: world.events.append({"kind": kind, "text": text, "data": {} if data is None else data})
    assert recycle_small_arc(world, a, 0)
    assert a.crust.mask[0] == 0
    assert a.crust.mask[1] == 0
    assert world.budget["recycledCells"] == 2
    assert a.alive is False
    assert world.events[-1]["kind"] == "arc-recycled"

    world, a, _b = contact_world()
    a.crust.create(0, CrustType["ARC"], -100.0, 30)
    world.config = {"arcAccreteMinKm2": 50.0, "terraneMinAreaKm2": 50.0}
    world.time = 10.0
    world.budget = {"recycledCells": 0}
    world.events = []
    world.log = lambda kind, text, data=None: world.events.append({"kind": kind, "text": text, "data": {} if data is None else data})
    assert not recycle_small_arc(world, a, 0)
    assert a.crust.mask[0] == 1


def test_recycle_small_arc_rejects_non_arc_cells():
    world, a, _b = contact_world()
    world.config = {"arcAccreteMinKm2": 500.0, "terraneMinAreaKm2": 50.0}
    world.budget = {"recycledCells": 0}
    world.events = []
    assert not recycle_small_arc(world, a, 0)
    assert world.events == []


def test_accrete_arcs_scrapes_arc_to_non_arc_host_and_logs_threshold():
    world, donor, host = contact_world()
    donor.crust.type[0] = CrustType["ARC"]
    donor.crust.terrane[0] = 30
    donor.stats["continentalCells"] = 0
    host.stats["continentalCells"] = 3
    world.time = 15.0
    world.arcScrapes = {}
    world.budget = {"recycledCells": 0}
    world.events = []
    world.log = lambda kind, text, data=None: world.events.append({"kind": kind, "text": text, "data": {} if data is None else data})
    host.crust.remove(0)
    world.config = {
        "arcAccreteMinKm2": 50.0,
        "terraneMinAreaKm2": 100.0,
        "arcScrapeLogKm2": 50.0,
    }

    result = accrete_arcs(world, donor, host, [0])

    assert result == {"core": [], "accreted": True}
    assert donor.crust.mask[0] == 0
    assert host.crust.type[0] == CrustType["ARC"]
    assert world.arcScrapes["0:1"]["cells"] == 1
    assert world.arcScrapes["0:1"]["logged"] is True
    assert world.events[-1]["kind"] == "arc-scrape"


def test_accrete_arcs_returns_core_cells_when_no_arcs_overlap():
    world, a, b = contact_world()
    result = accrete_arcs(world, a, b, [0, 2])
    assert result == {"core": [0, 2], "accreted": False}


def test_dock_if_terrane_returns_false_when_both_landmasses_are_large(monkeypatch):
    world, a, b = contact_world()
    world.config = {"terraneMaxAreaKm2": 50.0, "terraneForeArcKm": 100.0}
    monkeypatch.setattr(
        "orogen.v2.tectonics.collision.landmass_of",
        lambda plate, mesh, seeds: list(seeds),
    )
    assert not dock_if_terrane(world, a, b, [0])
