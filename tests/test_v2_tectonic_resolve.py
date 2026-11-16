"""Tests for the tectonic convergence/divergence resolver."""
import numpy as np

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.resolve import resolve, decide_birth, _nearest_crust, _is_behind, _is_advancing_on
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
