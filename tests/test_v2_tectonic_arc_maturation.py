from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.arc_maturation import mature_arcs
from orogen.v2.tectonics.crust import CrustFlag, CrustRaster, CrustType


class Mesh:
    def __init__(self, num_cells, neighbours, edge_km=10.0):
        self.num_cells = num_cells
        self.edge_km = edge_km
        offsets = [0]
        adjacency = []
        for row in neighbours:
            adjacency.extend(row)
            offsets.append(len(adjacency))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)


def make_world():
    mesh = Mesh(
        6,
        [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]],
    )
    crust = CrustRaster(6)
    crust.create(0, CrustType["CONTINENT"], 0.0, 10)
    crust.create(1, CrustType["ARC"], 0.0, 11)
    crust.create(2, CrustType["ARC"], 48.0, 12)
    crust.create(3, CrustType["CRATON"], -100.0, 13)
    crust.create(4, CrustType["ARC"], 0.0, 14)
    crust.create(5, CrustType["ARC"], 0.0, 15)
    crust.flags[1] |= CrustFlag["LIP"]
    crust.thickKm[1] = 42.0
    crust.volcType[1] = 2
    crust.volcKm[1] = 8.5
    crust.coverKm[1] = 3.0
    crust.flags[4] |= CrustFlag["MARGIN_ISLE"]

    plate = SimpleNamespace(
        id=0,
        alive=True,
        isBackground=False,
        crust=crust,
        dirty=False,
    )
    return SimpleNamespace(
        mesh=mesh,
        plates=[plate],
        config={
            "arcContinentKmPerMyr": 1.0,
            "arcContinentAgeMyr": 20.0,
        },
        time=50.0,
        budget={},
    ), plate


def test_arc_maturation_advances_one_cell_ring_and_preserves_crust_history():
    world, plate = make_world()
    crust = plate.crust

    assert mature_arcs(world, 5.0) == 0
    assert crust.type[1] == CrustType["ARC"]
    assert plate.arcMatureKm == pytest.approx(5.0)
    assert plate.dirty is False

    assert mature_arcs(world, 5.0) == 1
    assert crust.type[1] == CrustType["CONTINENT"]
    assert crust.type[2] == CrustType["ARC"]  # too young
    assert crust.type[4] == CrustType["ARC"]  # margin isle
    assert crust.type[5] == CrustType["ARC"]  # does not touch continental crust

    assert crust.terrane[1] == 11
    assert crust.formedAt[1] == pytest.approx(0.0)
    assert crust.flags[1] & CrustFlag["LIP"]
    assert crust.thickKm[1] == pytest.approx(42.0)
    assert crust.volcType[1] == 2
    assert crust.volcKm[1] == pytest.approx(8.5)
    assert crust.coverKm[1] == pytest.approx(3.0)
    assert plate.dirty is True
    assert world.budget["arcMaturedCells"] == 1
    assert plate.arcMatureKm == pytest.approx(10.0 - 10.0 * np.sqrt(3.0) / 2.0)


def test_arc_maturation_advances_one_ring_per_call_and_updates_budget():
    world, plate = make_world()
    crust = plate.crust
    assert mature_arcs(world, 10.0) == 1
    assert crust.type[1] == CrustType["CONTINENT"]
    assert crust.type[2] == CrustType["ARC"]

    world.time = 100.0
    # Accumulated remainder is less than one cell spacing; add enough for one ring.
    assert mature_arcs(world, 8.0) == 1
    assert crust.type[2] == CrustType["CONTINENT"]
    assert crust.type[4] == CrustType["ARC"]
    assert world.budget["arcMaturedCells"] == 2


def test_arc_maturation_does_not_change_empty_or_unqualified_plates():
    world, plate = make_world()
    empty = SimpleNamespace(
        alive=True, isBackground=False, crust=CrustRaster(6), dirty=False
    )
    background = SimpleNamespace(
        alive=True, isBackground=True, crust=plate.crust, dirty=False
    )
    dead = SimpleNamespace(
        alive=False, isBackground=False, crust=plate.crust, dirty=False
    )
    world.plates = [empty, background, dead]
    assert mature_arcs(world, 10.0) == 0
    assert not hasattr(empty, "arcMatureKm")
    assert not hasattr(background, "arcMatureKm")
    assert not hasattr(dead, "arcMatureKm")
    assert world.budget == {}


def test_arc_maturation_returns_early_when_rate_or_dt_is_nonpositive():
    world, plate = make_world()
    world.config["arcContinentKmPerMyr"] = 0.0
    assert mature_arcs(world, 10.0) == 0
    assert not hasattr(plate, "arcMatureKm")

    world.config["arcContinentKmPerMyr"] = 1.0
    assert mature_arcs(world, 0.0) == 0
    assert mature_arcs(world, -1.0) == 0
    assert not hasattr(plate, "arcMatureKm")


def test_arc_maturation_consumes_a_front_step_even_when_no_cell_is_due():
    world, plate = make_world()
    # No old arc touches continental or cratonic crust.
    plate.crust.remove(1)
    plate.crust.remove(2)
    plate.crust.remove(4)
    plate.crust.remove(5)

    assert mature_arcs(world, 10.0) == 0
    assert plate.arcMatureKm == pytest.approx(
        10.0 - 10.0 * np.sqrt(3.0) / 2.0
    )
    assert plate.dirty is False
    assert world.budget == {}
