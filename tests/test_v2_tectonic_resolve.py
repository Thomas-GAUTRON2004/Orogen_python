"""Tests for the tectonic convergence/divergence resolver."""

import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.resolve import (
    _is_advancing_on,
    _is_behind,
    _nearest_crust,
    decide_birth,
    resolve,
)
from orogen.v2.tectonics.world import TectonicWorld


def _world():
    mesh = build_sim_mesh(48, 23)
    world = TectonicWorld(mesh, 5, {"simCells": 48, "closingCmYr": 0.1})
    return mesh, world


def test_resolve_empty_world_and_refreshes_plates():
    mesh, world = _world()
    plate = world.add_plate("empty")
    result = resolve(world)
    assert result == {"created": 0, "consumed": 0, "collisionCells": 0}
    assert np.all(world.owner == -1)
    assert plate.reshaped is False


def test_resolve_preserves_land_collision_and_reports_it():
    mesh, world = _world()
    a, b = world.add_plate("A"), world.add_plate("B")
    a.crust.create(0, CrustType["CONTINENT"], -20, 1)
    b.crust.create(0, CrustType["CRATON"], -10, 2)
    a.refresh_extent(mesh.xyz)
    b.refresh_extent(mesh.xyz)
    result = resolve(world)
    assert result["collisionCells"] == 1
    assert result["consumed"] == 0
    assert a.crust.mask[0] and b.crust.mask[0]
    assert world.collider[0] == b.id
    assert world.claimCount[0] == 2


def test_resolve_subducts_older_ocean_and_updates_budget():
    mesh, world = _world()
    old, young = world.add_plate("old"), world.add_plate("young")
    old.crust.create(0, CrustType["OCEAN"], -40, 1)
    young.crust.create(0, CrustType["OCEAN"], -10, 2)
    old.refresh_extent(mesh.xyz)
    young.refresh_extent(mesh.xyz)
    result = resolve(world)
    assert result["consumed"] == 1
    assert not old.crust.mask[0]
    assert young.crust.mask[0]
    assert world.owner[0] == young.id
    assert world.consumedAt[0] == 1
    assert world.budget["consumed"] == 1


def test_resolve_continent_overrides_ocean():
    mesh, world = _world()
    ocean, continent = world.add_plate("ocean"), world.add_plate("continent")
    ocean.crust.create(0, CrustType["OCEAN"], -10, 1)
    continent.crust.create(0, CrustType["CONTINENT"], -10, 2)
    ocean.refresh_extent(mesh.xyz)
    continent.refresh_extent(mesh.xyz)
    result = resolve(world)
    assert result["consumed"] == 1
    assert not ocean.crust.mask[0]
    assert world.owner[0] == continent.id


def test_decide_birth_rejects_grower_without_nearby_crust():
    mesh, world = _world()
    plate = world.add_plate("empty")
    result = decide_birth(world, plate, 0, [(plate, plate.cap_in_world(mesh))])
    assert result["born"] is False
    assert result["why"] == "not attached to this plate's crust"


def test_nearest_crust_and_behind_helpers():
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["OCEAN"], 0, 1)
    point = plate.to_world(*map(float, mesh.xyz[:3]))
    hit = _nearest_crust(plate, mesh, point)
    assert hit["cell"] == 0
    assert hit["dot"] > 0.999
    assert _is_behind(plate, 0, 0, mesh) is False
    assert _is_advancing_on(plate, plate, 0, mesh, point, 0.0) is False


def test_candidate_cells_empty_when_no_gap_bins_marked():
    from orogen.v2.tectonics.resolve import candidate_cells

    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["OCEAN"], 0, 1)
    plate.refresh_extent(mesh.xyz)
    world.occupancyStamp = 1
    assert candidate_cells(world, plate, [mesh.num_cells - 1]) == []


def test_resolve_incoming_older_ocean_is_subducted():
    mesh, world = _world()
    young_holder, old_arrival = world.add_plate("young"), world.add_plate("old")
    young_holder.crust.create(0, CrustType["OCEAN"], -10, 1)
    old_arrival.crust.create(0, CrustType["OCEAN"], -40, 2)
    young_holder.refresh_extent(mesh.xyz)
    old_arrival.refresh_extent(mesh.xyz)
    result = resolve(world)
    assert result["consumed"] == 1
    assert young_holder.crust.mask[0]
    assert not old_arrival.crust.mask[0]


def test_resolve_clears_stale_owner_after_claim_pass(monkeypatch):
    import importlib

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["CONTINENT"], 0, 1)
    plate.refresh_extent(mesh.xyz)

    def claim_then_remove(target, callback):
        callback(0, plate, 0)
        plate.crust.remove(0)

    monkeypatch.setattr(module, "for_each_claim", claim_then_remove)
    module._gather_and_subduct(world)
    assert world.owner[0] == -1
    assert world.claimCount[0] == 0
    assert world.landClaims[0] == 0


def test_growth_creates_trench_apron(monkeypatch):
    import importlib

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    from orogen.v2.tectonics.crust import CrustFlag

    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["OCEAN"], 0, 1)
    plate.crust.flags[0] |= CrustFlag["TRENCH"]
    plate.crust.slabKm[0] = 80
    plate.refresh_extent(mesh.xyz)
    monkeypatch.setattr(module, "candidate_cells", lambda w, p, gaps: [1])
    monkeypatch.setattr(
        module,
        "decide_birth",
        lambda w, p, c, everyone: {"born": True, "trenchParent": 0},
    )
    monkeypatch.setattr(module, "_fracture_at", lambda *args: 0.25)
    assert module._grow_crust_into_gaps(world, [1]) == 1
    assert plate.crust.mask[1]
    assert plate.crust.flags[1] & CrustFlag["TRENCH"]
    assert plate.crust.flags[1] & CrustFlag["APRON"]
    assert plate.crust.slabKm[1] == 80


def test_is_advancing_on_handles_coincident_and_forward_points():
    from types import SimpleNamespace

    mesh, world = _world()
    x = SimpleNamespace(omega=[0.0, 0.0, 1.0])
    y = SimpleNamespace(
        omega=[0.0, 0.0, 0.0], to_world=lambda a, b, c: np.array([1.0, 0.0, 0.0])
    )
    point = np.array([1.0, 0.0, 0.0])
    assert _is_advancing_on(x, y, 0, mesh, point, 0.0) is True
    y.to_world = lambda a, b, c: np.array([0.0, 1.0, 0.0])
    assert _is_advancing_on(x, y, 0, mesh, point, 0.0) is True


def test_is_advancing_on_rejects_non_closing_and_slow_motion():
    from types import SimpleNamespace

    mesh, world = _world()
    point = np.array([1.0, 0.0, 0.0])
    stationary = SimpleNamespace(
        omega=[0.0, 0.0, 0.0], to_world=lambda a, b, c: np.array([a, b, c])
    )
    moving = SimpleNamespace(omega=[0.0, 0.0, 1.0])
    assert _is_advancing_on(stationary, stationary, 0, mesh, point, 0.0) is False
    assert _is_advancing_on(moving, stationary, 0, mesh, point, 1000.0) is False


def test_is_behind_skips_crust_neighbours_and_handles_fully_occupied_edge():
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["OCEAN"], 0, 1)
    # Neighbours already occupied are skipped; the edge still has no open side.
    for nb in mesh.adj_list[mesh.adj_offset[0] : mesh.adj_offset[1]]:
        plate.crust.create(int(nb), CrustType["OCEAN"], 0, 1)
    assert _is_behind(plate, 0, 0, mesh) is False


def test_decide_birth_rejects_existing_foreign_crust(monkeypatch):
    import importlib

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    mesh, world = _world()
    grower, rival = world.add_plate("grower"), world.add_plate("rival")
    grower.crust.create(0, CrustType["OCEAN"], 0, 1)
    rival.crust.create(0, CrustType["OCEAN"], 0, 2)
    grower.refresh_extent(mesh.xyz)
    rival.refresh_extent(mesh.xyz)
    rival.cell_at = lambda index, x, y, z: 0
    monkeypatch.setattr(
        module, "_nearest_crust", lambda *args, **kwargs: {"cell": 0, "dot": 1.0}
    )
    monkeypatch.setattr(module, "may_have_crust_near", lambda *args: True)
    result = module.decide_birth(world, grower, 1, [(rival, rival.cap_in_world(mesh))])
    assert result["born"] is False
    assert result["why"] == "not a gap: another plate has crust here"


def test_decide_birth_skips_rival_without_nearby_crust(monkeypatch):
    import importlib

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    mesh, world = _world()
    grower, rival = world.add_plate("grower"), world.add_plate("rival")
    grower.crust.create(0, CrustType["OCEAN"], 0, 1)
    grower.refresh_extent(mesh.xyz)
    rival.refresh_extent(mesh.xyz)
    monkeypatch.setattr(
        module,
        "_nearest_crust",
        lambda plate, mesh, point, under=-1: (
            {"cell": 0, "dot": 1.0} if plate is grower else {"cell": -1, "dot": -2.0}
        ),
    )
    monkeypatch.setattr(module, "may_have_crust_near", lambda *args: True)
    result = module.decide_birth(world, grower, 1, [(rival, rival.cap_in_world(mesh))])
    assert result["born"] is True


def test_decide_birth_closing_boundary_rejected(monkeypatch):
    import importlib

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    mesh, world = _world()
    grower, rival = world.add_plate("grower"), world.add_plate("rival")
    grower.crust.create(0, CrustType["OCEAN"], 0, 1)
    rival.crust.create(1, CrustType["OCEAN"], 0, 2)
    grower.refresh_extent(mesh.xyz)
    rival.refresh_extent(mesh.xyz)
    monkeypatch.setattr(
        module,
        "_nearest_crust",
        lambda plate, *args, **kwargs: (
            {"cell": 0, "dot": 0.99} if plate is grower else {"cell": 1, "dot": 0.0}
        ),
    )
    monkeypatch.setattr(module, "may_have_crust_near", lambda *args: True)
    monkeypatch.setattr(module, "_is_advancing_on", lambda *args: True)
    result = module.decide_birth(world, grower, 2, [(rival, rival.cap_in_world(mesh))])
    assert result["born"] is False
    assert result["why"] == "closing boundary"


def test_decide_birth_trench_parent_respects_wing_direction(monkeypatch):
    import importlib

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    from orogen.v2.tectonics.crust import CrustFlag

    mesh, world = _world()
    grower = world.add_plate("grower")
    grower.crust.create(0, CrustType["OCEAN"], 0, 1)
    grower.crust.flags[0] |= CrustFlag["TRENCH"]
    grower.crust.wingKm[0] = 5
    grower.refresh_extent(mesh.xyz)
    rival = world.add_plate("rival")
    rival.cap_in_world = lambda mesh: [0.0, 0.0, 1.0, -2.0]
    monkeypatch.setattr(
        module, "_nearest_crust", lambda *args, **kwargs: {"cell": 0, "dot": 1.0}
    )
    monkeypatch.setattr(module, "may_have_crust_near", lambda *args: False)
    monkeypatch.setattr(module, "_is_behind", lambda *args: True)
    result = module.decide_birth(
        world,
        grower,
        1,
        [(grower, grower.cap_in_world(mesh)), (rival, rival.cap_in_world(mesh))],
    )
    assert result == {"born": True, "trenchParent": -1}


def test_is_behind_open_edge_returns_direction_test():
    mesh, world = _world()
    plate = world.add_plate("P")
    plate.crust.create(0, CrustType["OCEAN"], 0, 1)
    assert isinstance(_is_behind(plate, 0, 1, mesh), bool)


def test_resolve_skips_gap_assignment_when_the_world_view_has_no_gaps(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    world = SimpleNamespace(
        occupancyStamp=0,
        owner=np.asarray([0, 0], dtype=np.int32),
        budget={"created": 0, "consumed": 0},
        landClaims=np.zeros(2, dtype=np.int32),
        plates=[],
        mesh=SimpleNamespace(xyz=np.zeros(6, dtype=np.float32)),
    )
    monkeypatch.setattr(module, "_gather_and_subduct", lambda _world: 0)
    monkeypatch.setattr(
        module, "_assign_gap_owners", lambda *_args: pytest.fail("no gaps to assign")
    )

    assert module.resolve(world) == {"created": 0, "consumed": 0, "collisionCells": 0}


def test_fracture_walk_runs_to_its_depth_limit_without_a_partner(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    count = 10
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
    mesh = SimpleNamespace(
        num_cells=count,
        xyz=np.tile(np.asarray([1.0, 0.0, 0.0]), count),
        index=SimpleNamespace(nearest=lambda *_args: 0),
        adj_offset=np.asarray(offsets, dtype=np.int32),
        adj_list=np.asarray(adjacency, dtype=np.int32),
    )
    grower = SimpleNamespace(
        id=0,
        omega=[0.0, 0.0, 0.0],
        to_plate_frame=lambda *values: list(values),
    )
    world = SimpleNamespace(
        mesh=mesh,
        owner=np.full(count, -1, dtype=np.int32),
        plates=[grower],
    )
    monkeypatch.setattr(module, "fracture_coordinate", lambda *_args: 0.25)

    assert module._fracture_at(world, grower, 0, [1.0, 0.0, 0.0]) == 0.25


def test_fracture_walk_keeps_the_leading_partner_when_a_later_owner_lags(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    mesh = SimpleNamespace(
        num_cells=4,
        xyz=np.tile(np.asarray([1.0, 0.0, 0.0]), 4),
        index=SimpleNamespace(nearest=lambda *_args: 0),
        adj_offset=np.asarray([0, 3, 3, 3, 3], dtype=np.int32),
        adj_list=np.asarray([1, 2, 3], dtype=np.int32),
    )
    grower = SimpleNamespace(
        id=0,
        omega=[0.0, 0.0, 0.0],
        to_plate_frame=lambda *values: list(values),
    )
    first = SimpleNamespace(isBackground=False, omega=[0.0, 0.0, 0.0])
    second = SimpleNamespace(isBackground=False, omega=[0.0, 0.0, 0.0])
    world = SimpleNamespace(
        mesh=mesh,
        owner=np.asarray([-1, 1, 1, 2], dtype=np.int32),
        plates=[grower, first, second],
    )
    monkeypatch.setattr(module, "fracture_coordinate", lambda *_args: 0.5)

    assert module._fracture_at(world, grower, 0, [1.0, 0.0, 0.0]) == 0.5


def test_candidate_cells_deduplicates_gaps_that_map_to_one_plate_cell(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    mesh, world = _world()
    grower = world.add_plate("grower")
    grower.crust.create(0, CrustType["OCEAN"], 0.0, 1)
    grower.cap_in_world = lambda _mesh: [1.0, 0.0, 0.0, -1.0]
    grower.cell_at = lambda *_args: 0
    monkeypatch.setattr(
        module,
        "bin_index",
        lambda _mesh: SimpleNamespace(
            bin_of=np.zeros(mesh.num_cells, dtype=np.int32), count=1
        ),
    )
    monkeypatch.setattr(module, "may_have_crust_near", lambda *_args: True)

    candidates = module.candidate_cells(world, grower, [1, 2])

    assert len(candidates) == len(set(candidates))


def test_gap_owner_assignment_skips_foreign_hits_without_a_crust_cell(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("orogen.v2.tectonics.resolve")
    mesh = SimpleNamespace(
        num_cells=1,
        xyz=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        index=SimpleNamespace(),
    )

    class Candidate:
        def __init__(self, identifier):
            self.id = identifier
            self.alive = True
            self.isBackground = False
            self.crust = __import__(
                "orogen.v2.tectonics.crust", fromlist=["CrustRaster"]
            ).CrustRaster(1)
            self.crust.create(0, CrustType["CONTINENT"], -10.0)
            self.cell_at = lambda *_args: 0

    best, foreign = Candidate(0), Candidate(1)
    world = SimpleNamespace(
        mesh=mesh,
        owner=np.asarray([-1], dtype=np.int32),
        ownerCell=np.asarray([-1], dtype=np.int32),
        plates=[best, foreign],
        occupancyStamp=1,
        claimCount=np.zeros(1, dtype=np.int32),
        provisional=np.zeros(1, dtype=np.int8),
        provisionalLike=np.full(1, -1, dtype=np.int32),
        config={},
    )
    monkeypatch.setattr(
        module,
        "bin_index",
        lambda _mesh: SimpleNamespace(bin_of=np.zeros(1, dtype=np.int32)),
    )
    monkeypatch.setattr(module, "may_have_crust_near", lambda *_args: True)
    monkeypatch.setattr(
        module,
        "_nearest_crust",
        lambda plate, *_args: (
            {"cell": 0, "dot": 0.9} if plate is best else {"cell": -1, "dot": -2.0}
        ),
    )

    module._assign_gap_owners(world, [0])

    assert world.owner.tolist() == [0]
    assert world.provisional.tolist() == [1]
