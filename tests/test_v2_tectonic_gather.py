import numpy as np

from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.crust import CrustType
from orogen.v2.tectonics.gather import clear_world_view, for_each_claim
from orogen.v2.tectonics.world import TectonicWorld


def test_gather_identity_plate_claims_and_clears_world_view():
    mesh = build_sim_mesh(48, 17)
    world = TectonicWorld(mesh, 3, {"simCells": 48})
    plate = world.add_plate("background")
    plate.isBackground = True
    plate.crust.create(0, CrustType["CRATON"], -10, 1)
    plate.crust.create(7, CrustType["OCEAN"], -5, 2)
    plate.refresh_extent(mesh.xyz)

    world.owner.fill(4)
    world.ownerCell.fill(9)
    world.claimCount.fill(3)
    world.landClaims.fill(2)
    world.collider.fill(1)
    world.consumedAt.fill(1)
    world.provisional.fill(1)
    clear_world_view(world)
    assert np.all(world.owner == -1)
    assert np.all(world.ownerCell == 9)
    assert np.all(world.claimCount == 0)
    assert np.all(world.landClaims == 0)
    assert np.all(world.collider == -1)
    assert np.all(world.consumedAt == 0)
    assert np.all(world.provisional == 0)

    claims = []
    for_each_claim(world, lambda w, p, c: claims.append((w, p.id, c)))
    assert claims == [(0, plate.id, 0), (7, plate.id, 7)]


def test_gather_skips_dead_and_empty_plates():
    mesh = build_sim_mesh(24, 9)
    world = TectonicWorld(mesh, 2, {"simCells": 24})
    dead = world.add_plate("dead")
    dead.crust.create(0, CrustType["OCEAN"], 0, 1)
    dead.refresh_extent(mesh.xyz)
    dead.alive = False
    world.add_plate("empty")
    claims = []
    for_each_claim(world, lambda w, p, c: claims.append((w, p.id, c)))
    assert claims == []
