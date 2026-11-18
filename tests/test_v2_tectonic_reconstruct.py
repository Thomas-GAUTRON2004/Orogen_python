from orogen.v2.core.sim_mesh import build_sim_mesh
from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.polarity import overrides
from orogen.v2.tectonics.reconstruct import reconstruct
from orogen.v2.tectonics.world import TectonicWorld


def _pair():
    mesh = build_sim_mesh(48, 23)
    world = TectonicWorld(mesh, 5, {"simCells": 48})
    a = world.add_plate("A")
    b = world.add_plate("B")
    return mesh, world, a, b


def test_reconstruct_counts_overlaps_and_prefers_land():
    mesh, world, ocean, land = _pair()
    ocean.crust.create(0, CrustType["OCEAN"], -20, 1)
    land.crust.create(0, CrustType["CRATON"], -10, 2)
    ocean.refresh_extent(mesh.xyz)
    land.refresh_extent(mesh.xyz)
    reconstruct(world)
    assert world.owner[0] == land.id
    assert world.ownerCell[0] == 0
    assert world.claimCount[0] == 2
    assert world.landClaims[0] == 1
    assert ocean.crust.mask[0] and land.crust.mask[0]


def test_reconstruct_first_claim_wins_when_all_ocean_and_saturates_counters():
    mesh = build_sim_mesh(48, 23)
    world = TectonicWorld(mesh, 5, {"simCells": 48})
    plates = [world.add_plate(str(i)) for i in range(260)]
    for plate in plates:
        plate.crust.create(0, CrustType["OCEAN"], 0, plate.id)
        plate.refresh_extent(mesh.xyz)
    reconstruct(world)
    assert world.owner[0] == plates[0].id
    assert world.claimCount[0] == 255
    assert world.landClaims[0] == 0


def test_polarity_land_ocean_trench_background_age_and_collision():
    _, _, a, b = _pair()
    a.crust.create(0, CrustType["CONTINENT"], 10, 1)
    b.crust.create(0, CrustType["OCEAN"], 0, 2)
    assert overrides(a, 0, b, 0) is True
    assert overrides(b, 0, a, 0) is False

    a.crust.type[0] = CrustType["OCEAN"]
    b.crust.type[0] = CrustType["OCEAN"]
    a.crust.formedAt[0] = -10
    b.crust.formedAt[0] = -30
    assert overrides(a, 0, b, 0) is True
    assert overrides(b, 0, a, 0) is False
    assert overrides(a, 0, b, 0, decided=b.id) is False
    assert overrides(a, 0, b, 0, decided=999) is True

    a.crust.flags[0] |= CrustFlag["TRENCH"]
    assert overrides(a, 0, b, 0) is True
    assert overrides(b, 0, a, 0) is False
    a.crust.flags[0] = 0
    a.isBackground = True
    assert overrides(a, 0, b, 0) is False
    assert overrides(b, 0, a, 0) is True
    a.isBackground = False
    b.isBackground = False
    a.crust.formedAt[0] = b.crust.formedAt[0] = 0
    assert overrides(a, 0, b, 0) is (a.id < b.id)
    assert overrides(a, 1, b, 0) is (a.id > b.id)
    a.crust.type[0] = CrustType["CRATON"]
    b.crust.type[0] = CrustType["CONTINENT"]
    assert overrides(a, 0, b, 0) is None
