import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustFlag, CrustType
from orogen.v2.tectonics.ops.scrape_arc import scrape_arc
from orogen.v2.tectonics.plate import Plate


class Index:
    def __init__(self):
        self.cells = {
            (1.0, 0.0, 0.0): 0,
            (0.0, 1.0, 0.0): 1,
            (-1.0, 0.0, 0.0): 2,
            (0.0, -1.0, 0.0): 3,
        }

    def nearest(self, x, y, z):
        return self.cells[(round(x, 5), round(y, 5), round(z, 5))]


class Mesh:
    def __init__(self):
        self.xyz = np.asarray(
            [1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0],
            dtype=np.float32,
        )
        self.num_cells = 4
        self.index = Index()


def make_plates():
    mesh = Mesh()
    host = Plate(0, 4)
    donor = Plate(1, 4)
    host.crust.create(0, CrustType["CONTINENT"], -100.0, 10)
    host.crust.create(2, CrustType["OCEAN"], -10.0, 20)
    host.crust.flags[0] = CrustFlag["SUTURE"]
    host.crust.flags[2] = CrustFlag["TRENCH"]

    for cell in range(4):
        donor.crust.create(cell, CrustType["ARC"], -50.0, 30 + cell)
    donor.crust.flags[1] = CrustFlag["TRENCH"] | CrustFlag["LIP"]
    donor.crust.farDx[1] = 127
    donor.crust.volcKm[1] = 14.0
    donor.crust.thickKm[1] = 23.0
    donor.omega = [0.1, 0.2, 0.3]
    donor.dynamics = {"tension": 0.5}
    world = type("World", (), {"mesh": mesh})()
    return world, host, donor


def test_scrape_arc_absorbs_host_land_and_moves_cells_to_empty_or_ocean():
    world, host, donor = make_plates()
    result = scrape_arc(world, host, donor, [0, 1, 2])

    assert result == {"moved": 2, "absorbed": 1}
    assert donor.crust.mask.tolist() == [0, 0, 0, 1]
    assert donor.crust.count == 1
    assert donor.alive is True
    assert donor.omega == [0.1, 0.2, 0.3]
    assert donor.dynamics == {"tension": 0.5}

    assert host.crust.type[0] == CrustType["CONTINENT"]
    assert host.crust.type[1] == CrustType["ARC"]
    assert host.crust.type[2] == CrustType["ARC"]
    assert host.crust.terrane[1] == 31
    assert host.crust.farDx[1] == 127
    assert host.crust.volcKm[1] == pytest.approx(14.0)
    assert host.crust.thickKm[1] == pytest.approx(23.0)
    assert host.crust.flags[1] & CrustFlag["SUTURE"]
    assert host.crust.flags[1] & CrustFlag["LIP"]
    assert not (host.crust.flags[1] & CrustFlag["TRENCH"])
    assert host.crust.flags[2] & CrustFlag["SUTURE"]
    assert not (host.crust.flags[2] & CrustFlag["TRENCH"])

    assert host.reshaped is True
    assert donor.reshaped is True
    assert host.crust.count == 3
    assert host.stats["cells"] == 3
    assert donor.stats["cells"] == 1


def test_scrape_arc_kills_empty_donor_and_zeros_its_motion():
    world, host, donor = make_plates()
    result = scrape_arc(world, host, donor, [0, 1, 2, 3])

    assert result == {"moved": 3, "absorbed": 1}
    assert donor.crust.count == 0
    assert donor.alive is False
    assert donor.omega == [0.0, 0.0, 0.0]
    assert donor.dynamics is None
    assert donor.stats["cells"] == 0


def test_scrape_arc_skips_donor_cells_that_are_not_occupied():
    world, host, donor = make_plates()
    donor.crust.remove(1)
    result = scrape_arc(world, host, donor, [1])

    assert result == {"moved": 0, "absorbed": 0}
    assert donor.crust.count == 3
    assert host.crust.count == 2
    assert donor.alive is True
