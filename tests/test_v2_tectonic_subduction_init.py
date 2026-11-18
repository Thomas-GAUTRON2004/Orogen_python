import numpy as np
import pytest

from orogen.v2.tectonics import subduction_init
from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.subduction_init import (
    _carve_intraplate_ocean_cells,
    _collect_subduction_contacts,
    _found_subduction_pair,
    _founder_run_probability,
    _old_ocean_margin_runs,
    _plate_can_founder_margin,
    _start_intraplate_ocean_founder,
    _subduction_cell_is_stale,
    _subduction_founder_metrics,
    arm_margin,
    choose_foundering,
    founder_within_plates,
    initiate_subduction,
    is_on_segment,
)
from orogen.v2.tectonics.world import TectonicWorld


class Index:
    def __init__(self, points):
        self.mapping = {
            tuple(round(float(v), 5) for v in point): i
            for i, point in enumerate(points)
        }

    def nearest(self, x, y, z):
        return self.mapping[
            (round(float(x), 5), round(float(y), 5), round(float(z), 5))
        ]


class Mesh:
    def __init__(self, edge_km=100.0, cell_area=100.0):
        self.points = [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
            [-0.8, 0.6, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
        self.num_cells = len(self.points)
        self.xyz = np.asarray(self.points, dtype=np.float32).reshape(-1)
        self.index = Index(self.points)
        self.edge_km = float(edge_km)
        self.edge_rad = self.edge_km / 6371.0
        self.cell_area_km2 = float(cell_area)
        self.adj_offset = np.asarray([0, 1, 3, 5, 7, 9, 10], dtype=np.int32)
        self.adj_list = np.asarray(
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
            dtype=np.int32,
        )


def config(**overrides):
    result = {
        "subductionInitRatePerMyr": 1.0,
        "subductionInitMaxOpeningCmYr": 5.0,
        "subductionInitMinLengthKm": 100.0,
        "subductionInitAgeMyr": 20.0,
        "subductionInitAgeCap": 3.0,
        "subductionInitSlabKm": 80.0,
        "oceanPlateReachKm": 200.0,
        "oceanPlateMinAreaKm2": 100.0,
        "disperseFounderFactor": 0.25,
        "marginFounderRatePerMyr": 1.0,
        "marginFounderMinShare": 0.5,
    }
    result.update(overrides)
    return result


def make_boundary_world(*, background=False, **overrides):
    mesh = Mesh()
    world = TectonicWorld(mesh, seed=24, config=config(**overrides))
    world.time = 0.0
    if background:
        passive = world.add_plate("Panthalassa")
        passive.isBackground = True
        for cell in (1, 2, 3, 4):
            passive.crust.create(cell, CrustType["OCEAN"], -200.0, 0)
        active = world.add_plate("Pacific")
        for cell in (0, 1):
            active.crust.create(cell, CrustType["OCEAN"], -100.0, 10 + cell)
        world.owner[0] = active.id
        world.owner[1:5] = passive.id
    else:
        active = world.add_plate("Ocean A")
        upper = world.add_plate("Continent B")
        active.crust.create(0, CrustType["OCEAN"], -100.0, 10)
        active.crust.create(1, CrustType["OCEAN"], -100.0, 11)
        upper.crust.create(0, CrustType["CONTINENT"], -80.0, 20)
        upper.crust.create(1, CrustType["OCEAN"], -80.0, 21)
        world.owner[0] = active.id
        world.owner[1] = upper.id
    world.ownerCell[:] = np.arange(mesh.num_cells, dtype=np.int32)
    world.convergence[0] = 1.0
    world.convergence[1] = -10.0
    world.rngSubduction = lambda: 0.0
    world.rngMarginFounder = lambda: 0.0
    return world


def make_intraplate_world(**overrides):
    mesh = Mesh(edge_km=100.0, cell_area=100.0)
    world = TectonicWorld(mesh, seed=56, config=config(**overrides))
    world.time = 100.0
    plate = world.add_plate("Pangaea")
    plate.crust.create(0, CrustType["CONTINENT"], -1000.0, 1)
    for cell in range(1, 5):
        plate.crust.create(cell, CrustType["OCEAN"], -100.0, 2)
    plate.wings = {1: [0.0, 1.0, 0.0], 2: [1.0, 0.0, 0.0]}
    plate.refresh_extent(mesh.xyz)
    world.rngMarginFounder = lambda: 0.0
    return world, plate


def test_initiate_subduction_arms_old_ocean_margin_and_logs_foundering():
    world = make_boundary_world()
    assert initiate_subduction(world, 1.0) == 1

    active, upper = world.plates
    assert upper.crust.flags[0] & CrustFlag["TRENCH"]
    assert upper.crust.flags[1] & CrustFlag["TRENCH"]
    assert upper.crust.slabKm[0] == pytest.approx(80.0)
    assert upper.dirty is True

    event = world.events[-1]
    assert event["kind"] == "subduction-init"
    assert event["data"] == {
        "upper": upper.id,
        "lower": active.id,
        "lengthKm": 100.0,
        "ageMyr": 100.0,
    }
    assert "100 km" in event["text"]


def test_initiate_subduction_skips_when_rate_disabled_or_margin_diverges():
    world = make_boundary_world(subductionInitRatePerMyr=0.0)
    assert initiate_subduction(world, 1.0) == 0
    assert world.events == []

    world = make_boundary_world()
    world.convergence[0] = -6.0
    assert initiate_subduction(world, 1.0) == 0
    assert world.events == []


def test_initiate_subduction_skips_existing_trenches_and_stale_plates():
    world = make_boundary_world()
    world.plates[0].crust.flags[0] |= CrustFlag["TRENCH"]
    assert initiate_subduction(world, 1.0) == 0

    world = make_boundary_world()
    world.plates[1].reshaped = True
    assert initiate_subduction(world, 1.0) == 0


def test_initiate_subduction_obeys_age_and_length_thresholds():
    world = make_boundary_world(subductionInitMinLengthKm=101.0)
    assert initiate_subduction(world, 1.0) == 0

    world = make_boundary_world(subductionInitAgeMyr=101.0)
    assert initiate_subduction(world, 1.0) == 0


def test_initiate_subduction_uses_seeded_probability_and_dispersal_factor():
    world = make_boundary_world(subductionInitRatePerMyr=0.0001)
    world.rngSubduction = lambda: 0.5
    assert initiate_subduction(world, 1.0) == 0

    world = make_boundary_world()
    world.cycle = {"phase": "dispersing", "breakupAt": -100.0}
    world.config["disperseFounderFactor"] = 0.0
    assert initiate_subduction(world, 1.0) == 0


def test_initiate_subduction_carves_a_real_plate_from_passive_background():
    world = make_boundary_world(background=True)
    assert initiate_subduction(world, 1.0) == 1

    background, active, ocean = world.plates
    assert ocean.isOceanBorn is True
    assert ocean.name == "Thalassa"
    assert ocean.crust.count == 3
    assert background.crust.count == 1
    assert active.crust.flags[0] & CrustFlag["TRENCH"]
    assert len(world.events) == 2
    assert world.events[0]["kind"] == "plate-born"
    assert world.events[1]["kind"] == "subduction-init"
    assert world.events[1]["data"]["lower"] == ocean.id


def test_initiate_subduction_skips_when_passive_ocean_cannot_be_carved():
    world = make_boundary_world(
        background=True,
        oceanPlateMinAreaKm2=1000.0,
    )
    assert initiate_subduction(world, 1.0) == 0
    assert len(world.plates) == 2
    assert world.events == []


def test_choose_foundering_prefers_land_as_upper_plate():
    world = make_boundary_world()
    ocean, continent = world.plates
    assert choose_foundering(world, ocean, continent, [0]) == (continent, ocean)


def test_choose_foundering_breaks_ocean_ocean_ties_by_formation_age():
    world = make_boundary_world()
    plate_a, plate_b = world.plates
    plate_b.crust.type[0] = CrustType["OCEAN"]
    plate_a.crust.formedAt[0] = -120.0
    plate_b.crust.formedAt[0] = -80.0
    assert choose_foundering(world, plate_a, plate_b, [0]) == (plate_a, plate_b)

    plate_a.crust.formedAt[0] = -80.0
    plate_b.crust.formedAt[0] = -120.0
    assert choose_foundering(world, plate_a, plate_b, [0]) == (plate_b, plate_a)


def test_is_on_segment_checks_direct_lower_cell_then_neighbours():
    world = make_boundary_world()
    upper, lower = world.plates
    assert is_on_segment(world, lower, 0, upper)

    lower.crust.remove(0)
    lower.crust.remove(1)
    lower.crust.create(2, CrustType["OCEAN"], -50.0, 4)
    assert is_on_segment(world, lower, 1, upper)

    lower.crust.remove(2)
    assert not is_on_segment(world, lower, 1, upper)


def test_arm_margin_sets_dirty_even_when_upper_cell_is_missing():
    world = make_boundary_world()
    upper, lower = world.plates
    upper.crust.remove(0)
    upper.crust.remove(1)
    arm_margin(world, upper, lower, [0])
    assert upper.dirty is True


def test_founder_within_plates_cuts_old_outer_ocean_and_arms_margin():
    world, plate = make_intraplate_world(oceanPlateReachKm=500.0)
    assert founder_within_plates(world, 1.0) == 1

    fresh = world.plates[1]
    assert fresh.isOceanBorn is True
    assert fresh.name == "Thalassa"
    assert fresh.crust.mask.tolist() == [0, 0, 1, 1, 1, 0]
    assert fresh.wings == {2: [1.0, 0.0, 0.0]}
    assert plate.wings == {1: [0.0, 1.0, 0.0]}
    assert plate.crust.flags[1] & CrustFlag["TRENCH"]
    assert plate.crust.slabKm[1] == pytest.approx(80.0)
    assert plate.dirty is True
    assert world.events[0]["kind"] == "plate-born"
    assert world.events[1]["kind"] == "subduction-init"


def test_founder_within_plates_respects_disabled_rate_minimum_share_and_coast_age():
    world, _plate = make_intraplate_world(marginFounderRatePerMyr=0.0)
    assert founder_within_plates(world, 1.0) == 0

    world, _plate = make_intraplate_world(marginFounderMinShare=2.0)
    assert founder_within_plates(world, 1.0) == 0

    world, plate = make_intraplate_world()
    plate.crust.flags[1] |= CrustFlag["TRENCH"]
    assert founder_within_plates(world, 1.0) == 0


def test_founder_within_plates_rejects_short_or_unripe_margins():
    world, _plate = make_intraplate_world(subductionInitMinLengthKm=101.0)
    assert founder_within_plates(world, 1.0) == 0

    world, plate = make_intraplate_world(subductionInitAgeMyr=201.0)
    assert founder_within_plates(world, 1.0) == 0


def test_founder_within_plates_rejects_weak_chance_and_small_ocean_take():
    world, _plate = make_intraplate_world(marginFounderRatePerMyr=0.0001)
    world.rngMarginFounder = lambda: 0.5
    assert founder_within_plates(world, 1.0) == 0

    world, _plate = make_intraplate_world(oceanPlateReachKm=100.0)
    world.config["oceanPlateMinAreaKm2"] = 400.0
    assert founder_within_plates(world, 1.0) == 0


def test_founder_within_plates_restores_plate_after_failed_transfer(monkeypatch):
    world, plate = make_intraplate_world()
    monkeypatch.setattr(subduction_init, "transfer_cells", lambda *_args: 0)
    assert founder_within_plates(world, 1.0) == 0
    assert world.plates[-1].alive is False
    assert world.events == []


def test_founder_within_plates_respects_dispersal_cycle_delay():
    world, _plate = make_intraplate_world(disperseFounderFactor=0.0)
    world.cycle = {"phase": "dispersing", "breakupAt": -200.0}
    assert founder_within_plates(world, 1.0) == 0


def test_initiate_subduction_skips_dead_plate_after_contact_gathering():
    world = make_boundary_world()
    world.plates[1].alive = False
    assert initiate_subduction(world, 1.0) == 0
    assert world.events == []


def test_initiate_subduction_rejects_a_fragmented_contact_segment(monkeypatch):
    world = make_boundary_world(subductionInitMinLengthKm=150.0)
    active, upper = world.plates
    active.crust.create(2, CrustType["OCEAN"], -100.0, 12)
    upper.crust.create(3, CrustType["OCEAN"], -80.0, 23)
    world.owner[2] = active.id
    world.owner[3] = upper.id
    world.ownerCell[2] = 2
    world.ownerCell[3] = 3
    world.convergence[2] = 1.0
    monkeypatch.setattr(
        subduction_init,
        "largest_piece_of",
        lambda _mesh, cells: [int(cells[0])],
    )
    assert initiate_subduction(world, 1.0) == 0
    assert world.events == []


def test_initiate_subduction_handles_background_as_second_plate():
    mesh = Mesh()
    world = TectonicWorld(mesh, seed=24, config=config())
    world.time = 0.0
    active = world.add_plate("Pacific")
    background = world.add_plate("Panthalassa")
    background.isBackground = True
    active.crust.create(0, CrustType["OCEAN"], -100.0, 10)
    for cell in (1, 2, 3, 4):
        background.crust.create(cell, CrustType["OCEAN"], -200.0, 0)
    world.owner[0] = active.id
    world.owner[1:5] = background.id
    world.ownerCell[:] = np.arange(mesh.num_cells, dtype=np.int32)
    world.convergence[0] = 1.0
    world.rngSubduction = lambda: 0.0

    assert initiate_subduction(world, 1.0) == 1
    fresh = world.plates[-1]
    assert fresh.isOceanBorn is True
    assert active.crust.flags[0] & CrustFlag["TRENCH"]
    assert world.events[-1]["data"]["lower"] == fresh.id


def test_choose_foundering_allows_left_plate_to_be_the_continental_upper():
    world = make_boundary_world()
    plate_a, plate_b = world.plates
    plate_a.crust.type[0] = CrustType["CONTINENT"]
    plate_b.crust.type[0] = CrustType["OCEAN"]
    assert choose_foundering(world, plate_a, plate_b, [0]) == (plate_a, plate_b)


def test_founder_within_plates_skips_dead_and_empty_land_plates():
    world, plate = make_intraplate_world()
    plate.alive = False
    assert founder_within_plates(world, 1.0) == 0

    world, plate = make_intraplate_world()
    plate.stats["continentalCells"] = 0
    assert founder_within_plates(world, 1.0) == 0


def test_founder_within_plates_groups_contiguous_old_coast(monkeypatch):
    world, plate = make_intraplate_world(oceanPlateReachKm=500.0)
    plate.crust.type[3] = CrustType["CONTINENT"]
    plate.refresh_extent(world.mesh.xyz)
    monkeypatch.setattr(
        subduction_init,
        "exterior_sea",
        lambda _mesh, _crust: np.zeros(world.mesh.num_cells, dtype=np.uint8),
    )

    # The two old coast cells are connected; the later seed is already assigned
    # to that same run and must not start a second component.
    assert founder_within_plates(world, 1.0) == 0
    assert len(world.plates) == 1


def test_founder_within_plates_rejects_runs_not_on_exterior_sea(monkeypatch):
    world, _plate = make_intraplate_world()
    monkeypatch.setattr(
        subduction_init,
        "exterior_sea",
        lambda _mesh, _crust: np.zeros(world.mesh.num_cells, dtype=np.uint8),
    )
    assert founder_within_plates(world, 1.0) == 0
    assert len(world.plates) == 1
    assert world.events == []


def test_intraplate_founder_helpers_preserve_margin_selection_and_transfer():
    world, plate = make_intraplate_world(oceanPlateReachKm=500.0)
    num_cells = world.mesh.num_cells
    minimum_cells = 100.0 / world.mesh.cell_area_km2
    assert _plate_can_founder_margin(plate, minimum_cells, num_cells, 0.0)
    assert not _plate_can_founder_margin(plate, minimum_cells, num_cells, 2.0)

    run = [1, 2]
    length_age_chance = _founder_run_probability(
        world,
        run,
        plate.crust.formedAt,
        float(world.time),
        1.0,
        float(world.mesh.edge_km),
        100.0,
        100.0,
        2.0,
        0.01,
    )
    assert length_age_chance == pytest.approx((200.0, 200.0, 0.02))
    world.cycle = {"phase": "dispersing", "breakupAt": -200.0}
    world.config["disperseFounderFactor"] = 0.25
    dispersed = _founder_run_probability(
        world,
        run,
        plate.crust.formedAt,
        float(world.time),
        1.0,
        float(world.mesh.edge_km),
        100.0,
        100.0,
        2.0,
        0.01,
    )
    assert dispersed == pytest.approx((200.0, 200.0, 0.005))
    assert (
        _founder_run_probability(
            world,
            [1],
            plate.crust.formedAt,
            float(world.time),
            1.0,
            float(world.mesh.edge_km),
            100.0,
            101.0,
            2.0,
            0.01,
        )
        is None
    )

    coast, run_of, runs = _old_ocean_margin_runs(
        world.mesh, plate.crust, float(world.time), 100.0
    )
    assert runs
    selected = runs[0]
    labels, take = _carve_intraplate_ocean_cells(
        world.mesh, plate.crust, coast, run_of, selected, 5
    )
    assert take
    assert all(labels[cell] == 1 for cell in take)
    assert all(plate.crust.type[cell] == 0 for cell in take)

    started = _start_intraplate_ocean_founder(
        world,
        plate,
        selected,
        take,
        labels,
        coast,
        len(selected) * float(world.mesh.edge_km),
        200.0,
        float(world.time),
        float(world.mesh.cell_area_km2),
        world.config,
    )
    assert started is True
    fresh = world.plates[-1]
    assert fresh.isOceanBorn
    assert fresh.crust.count == len(take)
    assert all(plate.crust.flags[cell] & CrustFlag["TRENCH"] for cell in selected)

def test_subduction_contact_helpers_group_and_measure_the_same_founder_pair():
    world = make_boundary_world()
    contacts = _collect_subduction_contacts(world)

    assert list(contacts) == [(0, 1)]
    entry = contacts[(0, 1)]
    assert entry["cells"] == [0]
    assert entry["ages"] == pytest.approx(100.0)
    assert _subduction_cell_is_stale(world.plates[0], 0) is False

    metrics = _subduction_founder_metrics(world, entry, 1.0, 100.0)
    assert metrics is not None
    length_km, age_myr, chance = metrics
    assert length_km == pytest.approx(100.0)
    assert age_myr == pytest.approx(100.0)
    assert chance > 0.0

    world.config["subductionInitMinLengthKm"] = 101.0
    assert _subduction_founder_metrics(world, entry, 1.0, 100.0) is None


def test_found_subduction_pair_helper_creates_the_same_trench_event():
    world = make_boundary_world()
    contacts = _collect_subduction_contacts(world)

    assert _found_subduction_pair(
        world,
        (0, 1),
        contacts[(0, 1)],
        1.0,
        float(world.mesh.edge_km),
    )
    assert world.plates[1].crust.flags[0] & CrustFlag["TRENCH"]
    assert world.events[-1]["kind"] == "subduction-init"
    assert world.events[-1]["data"]["upper"] == 1
    assert world.events[-1]["data"]["lower"] == 0
