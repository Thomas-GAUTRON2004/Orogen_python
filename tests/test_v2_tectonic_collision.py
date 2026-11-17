function content(value) {
    const iterate =
      typeof value === "string" ? undefined : value?.[iteratorKey];
    if (typeof iterate !== "function") {
      return emitBlock(value);
    }

    // Reuse the method from a potentially stateful getter, with its original
    // receiver. The for-of loop closes the iterator if the host rejects a block.
    const blocks = { [iteratorKey]: () => apply(iterate, value, []) };
    for (const block of blocks) {
      emitBlock(block);
    }
  }

def test_enclosed_sea_between_returns_zero_when_all_cells_are_land():
    world, upper, _lower = sea_world()
    for cell in range(1, world.mesh.num_cells):
        world.owner[cell] = upper.id
        world.ownerCell[cell] = cell
        upper.crust.create(cell, CrustType["CONTINENT"], -10.0, cell + 10)
    assert enclosed_sea_between_km2(world, upper, world.plates[1]) == 0.0


def test_open_sea_between_counts_facing_but_not_directly_opposed_cells(monkeypatch):
    from orogen.v2.tectonics import collision

    world, upper, lower = sea_world()
    calls = {"count": 0}

    def shallow_angle(*_args):
        calls["count"] += 1
        return [1.0, 0.0, 0.0] if calls["count"] % 2 else [
            -0.7, math.sqrt(0.51), 0.0
        ]

    monkeypatch.setattr(collision, "tangent_toward", shallow_angle)
    result = open_sea_between(world, upper, lower)
    assert result["km2"] > 0.0
    assert result["widestKm"] == 0.0


def test_recycle_skips_arc_recycled_event_for_tiny_fragment():
    from orogen.v2.tectonics.collision import recycle

    world, donor, _host = contact_world()
    donor.crust.create(0, CrustType["ARC"], -20.0, 30)
    world.budget = {"recycledCells": 0}
    world.events = []
    world.log = lambda kind, text, data=None: world.events.append({"kind": kind})
    world.mesh.cell_area_km2 = 100.0
    world.config = {"terraneMinAreaKm2": 1000.0}

    assert recycle(world, donor, [0]) == 1
    assert world.budget["recycledCells"] == 1
    assert world.events == []


def test_accrete_arcs_selects_one_donor_when_both_plates_have_arc_cells():
    world, a, b = contact_world()
    a.crust.type[0] = CrustType["ARC"]
    b.crust.type[0] = CrustType["ARC"]
    a.crust.remove(0)
    world.time = 15.0
    world.arcScrapes = {}
    world.budget = {"recycledCells": 0}
    world.events = []
    world.log = lambda kind, text, data=None: world.events.append({"kind": kind, "text": text, "data": {} if data is None else data})
    world.config = {
        "arcAccreteMinKm2": 50.0,
        "terraneMinAreaKm2": 100.0,
        "arcScrapeLogKm2": 1000.0,
    }

    result = accrete_arcs(world, a, b, [0])

    assert result == {"core": [], "accreted": True}
    assert b.crust.mask[0] == 0
    assert a.crust.type[0] == CrustType["ARC"]
    assert world.arcScrapes["1:0"]["cells"] == 1
    assert world.arcScrapes["1:0"]["logged"] is False
    assert world.events == []


def test_accrete_arcs_skips_transfer_when_donor_is_dead():
    world, donor, host = contact_world()
    donor.crust.type[0] = CrustType["ARC"]
    donor.alive = False
    world.time = 10.0
    world.arcScrapes = {}
    world.budget = {"recycledCells": 0}
    world.events = []
    world.config = {"arcAccreteMinKm2": 100.0, "terraneMinAreaKm2": 100.0}
    assert accrete_arcs(world, donor, host, [0]) == {
        "core": [], "accreted": False
    }
    assert world.arcScrapes == {}


def test_dock_if_terrane_docks_using_accretion_and_logs_event(monkeypatch):
    import importlib

    collision_module = importlib.import_module(
        "orogen.v2.tectonics.collision"
    )
    accretion_module = importlib.import_module(
        "orogen.v2.tectonics.ops.plate_accretion"
    )
    world, a, b = contact_world()
    world.time = 25.0
    world.log = lambda kind, text, data=None: world.events.append(
        {"kind": kind, "text": text, "data": {} if data is None else data}
    )
    world.events = []
    world.config = {
        "terraneMaxAreaKm2": 200.0,
        "arcAccreteMinKm2": 50.0,
        "terraneMinAreaKm2": 50.0,
    }
    a.stats["continentalCells"] = 3
    b.stats["continentalCells"] = 1
    monkeypatch.setattr(
        accretion_module,
        "landmass_of",
        lambda plate, _mesh, _seeds: [0] if plate.id == 0 else [1],
    )
    calls = []
    monkeypatch.setattr(
        accretion_module,
        "accrete_terrane",
        lambda world, host, donor, mass, rate: (
            calls.append((host.id, donor.id, list(mass), rate))
            or {"cells": 1, "sealed": 2}
        ),
    )

    assert collision_module.dock_if_terrane(world, a, b, [0])
    assert calls and calls[0][:3] == (0, 1, [1])
    assert world.events[-1]["kind"] == "dock"


def test_dock_if_terrane_returns_after_scrap_without_dock_event(monkeypatch):
    import importlib

    collision_module = importlib.import_module(
        "orogen.v2.tectonics.collision"
    )
    accretion_module = importlib.import_module(
        "orogen.v2.tectonics.ops.plate_accretion"
    )
    world, a, b = contact_world()
    world.time = 25.0
    world.events = []
    world.config = {
        "terraneMaxAreaKm2": 200.0,
        "arcAccreteMinKm2": 50.0,
        "terraneMinAreaKm2": 50.0,
    }
    monkeypatch.setattr(
        accretion_module,
        "landmass_of",
        lambda plate, _mesh, _seeds: [0] if plate.id == 0 else [1],
    )
    monkeypatch.setattr(
        accretion_module,
        "accrete_terrane",
        lambda *_args: {"scrap": True},
    )
    assert collision_module.dock_if_terrane(world, a, b, [0])
    assert world.events == []


def test_dock_if_terrane_recycles_tiny_arc_piece_instead_of_accreting(monkeypatch):
    import importlib

    collision_module = importlib.import_module(
        "orogen.v2.tectonics.collision"
    )
    accretion_module = importlib.import_module(
        "orogen.v2.tectonics.ops.plate_accretion"
    )
    world, a, b = contact_world()
    b.crust.type[0] = CrustType["ARC"]
    world.time = 25.0
    world.events = []
    world.budget = {"recycledCells": 0}
    world.log = lambda kind, text, data=None: world.events.append(
        {"kind": kind, "text": text, "data": {} if data is None else data}
    )
    world.config = {
        "terraneMaxAreaKm2": 200.0,
        "arcAccreteMinKm2": 200.0,
        "terraneMinAreaKm2": 1000.0,
    }
    monkeypatch.setattr(
        accretion_module,
        "landmass_of",
        lambda plate, _mesh, _seeds: [0] if plate.id == 0 else [0],
    )
    monkeypatch.setattr(
        accretion_module,
        "accrete_terrane",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must recycle arc")),
    )
    assert collision_module.dock_if_terrane(world, a, b, [0])
    assert world.budget["recycledCells"] == 1
    assert world.events == []


def test_dock_if_terrane_returns_false_when_one_mass_is_missing(monkeypatch):
    import importlib

    collision_module = importlib.import_module(
        "orogen.v2.tectonics.collision"
    )
    accretion_module = importlib.import_module(
        "orogen.v2.tectonics.ops.plate_accretion"
    )
    world, a, b = contact_world()
    world.config = {"terraneMaxAreaKm2": 200.0}
    monkeypatch.setattr(
        accretion_module,
        "landmass_of",
        lambda plate, _mesh, _seeds: [] if plate.id == 0 else [0],
    )
    assert not collision_module.dock_if_terrane(world, a, b, [0])
