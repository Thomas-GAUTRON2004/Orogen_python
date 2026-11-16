"""Per-cell crust history raster ported from Orogen V2 model/crust.js."""
from __future__ import annotations

import math
import numpy as np

CrustType = {"OCEAN": 0, "CONTINENT": 1, "CRATON": 2, "ARC": 3}
OrogenyType = {
    "NONE": 0, "ANDEAN": 1, "LARAMIDE": 2, "URAL": 3, "HIMALAYAN": 4,
    "ARC": 5, "RIFT": 6, "TRANSPRESSION": 7, "INTRAPLATE": 8,
}
VolcanicType = {"NONE": 0, "TRACK": 1, "LIP": 2}
BeltSide = {"NONE": 0, "UPPER": 1, "LOWER": 2}
CrustFlag = {
    "TRENCH": 1, "SUTURE": 2, "FAILED_RIFT": 4, "RIFT_MARGIN": 8,
    "LIP": 16, "APRON": 32, "MARGIN_ISLE": 64, "EROSIVE": 128,
}

CELL_FIELDS = (
    "type flags formedAt terrane slabKm wingKm oroType oroWork oroStart oroLast "
    "thickKm beltU beltReach beltAt beltSide beltShear prevType prevLast prevU "
    "prevReach prevSide prevShear rootKm upliftKm baseKm rugged prevRugged "
    "volcWornKm riverIncKm volcType volcAt volcKm volcSource fzU geoFlags "
    "coverKm coverMaxKm coverMaxAt coverCarb coverSalt coverCoal coverOrganic "
    "exhumKm unroofKm floodMyr floodLast arcMyr riftAt farWork farDist farFold "
    "farAt farDx farDy farDz"
).split()

_DTYPES = {
    "type": np.uint8, "flags": np.uint8, "formedAt": np.float32,
    "terrane": np.uint16, "slabKm": np.float32, "wingKm": np.float32,
    "oroType": np.uint8, "oroWork": np.float32, "oroStart": np.float32,
    "oroLast": np.float32, "thickKm": np.float32, "beltU": np.float32,
    "beltReach": np.float32, "beltAt": np.float32, "beltSide": np.uint8,
    "beltShear": np.int8, "prevType": np.uint8, "prevLast": np.float32,
    "prevU": np.float32, "prevReach": np.float32, "prevSide": np.uint8,
    "prevShear": np.int8, "rootKm": np.float32, "upliftKm": np.float32,
    "baseKm": np.float32, "rugged": np.float32, "prevRugged": np.float32,
    "volcWornKm": np.float32, "riverIncKm": np.float32, "volcType": np.uint8,
    "volcAt": np.float32, "volcKm": np.float32, "volcSource": np.uint16,
    "fzU": np.float32, "geoFlags": np.uint8, "coverKm": np.float32,
    "coverMaxKm": np.float32, "coverMaxAt": np.float32, "coverCarb": np.float32,
    "coverSalt": np.float32, "coverCoal": np.float32, "coverOrganic": np.float32,
    "exhumKm": np.float32, "unroofKm": np.float32, "floodMyr": np.float32,
    "floodLast": np.float32, "arcMyr": np.float32, "riftAt": np.float32,
    "farWork": np.float32, "farDist": np.float32, "farFold": np.float32,
    "farAt": np.float32, "farDx": np.int8, "farDy": np.int8, "farDz": np.int8,
}


class CrustRaster:
    """Structure-of-arrays record for crust in one plate's local frame."""

    def __init__(self, num_cells: int):
        if num_cells < 1:
            raise ValueError("num_cells must be positive")
        self.numCells = int(num_cells)
        self.mask = np.zeros(num_cells, dtype=np.uint8)
        for name in CELL_FIELDS:
            setattr(self, name, np.zeros(num_cells, dtype=_DTYPES[name]))
        self.count = 0
        self.fields = [getattr(self, name) for name in CELL_FIELDS]

    def create(self, cell: int, crust_type: int, formed_at: float, terrane: int = 0) -> None:
        if not self.mask[cell]:
            self.count += 1
        self.mask[cell] = 1
        for field in self.fields:
            field[cell] = 0
        self.type[cell] = crust_type
        self.formedAt[cell] = formed_at
        self.terrane[cell] = terrane

    def copy_from(self, cell: int, other: "CrustRaster", source: int, rot=None) -> None:
        if not self.mask[cell]:
            self.count += 1
        self.mask[cell] = 1
        for mine, theirs in zip(self.fields, other.fields):
            mine[cell] = theirs[source]
        if rot is not None and (self.farDx[cell] or self.farDy[cell] or self.farDz[cell]):
            x, y, z = int(self.farDx[cell]), int(self.farDy[cell]), int(self.farDz[cell])
            u = rot[0] * x + rot[1] * y + rot[2] * z
            v = rot[3] * x + rot[4] * y + rot[5] * z
            w = rot[6] * x + rot[7] * y + rot[8] * z
            length = math.hypot(u, v, w) or 1.0
            self.farDx[cell] = math.floor(127 * u / length + 0.5)
            self.farDy[cell] = math.floor(127 * v / length + 0.5)
            self.farDz[cell] = math.floor(127 * w / length + 0.5)

    def snapshot(self, cell: int) -> list:
        return [field[cell].item() for field in self.fields]

    def restore(self, cell: int, values) -> None:
        if len(values) != len(self.fields):
            raise ValueError("snapshot field count does not match CELL_FIELDS")
        if not self.mask[cell]:
            self.count += 1
        self.mask[cell] = 1
        for field, value in zip(self.fields, values):
            field[cell] = value

    def remove(self, cell: int) -> None:
        if self.mask[cell]:
            self.mask[cell] = 0
            self.count -= 1

    def arm_trench(self, cell: int, slab_km: float) -> None:
        self.flags[cell] |= CrustFlag["TRENCH"]
        self.slabKm[cell] = max(float(self.slabKm[cell]), slab_km)

    def record_subduction(self, cell: int, km: float) -> None:
        self.slabKm[cell] += km

    def record_orogeny(
        self, cell: int, oro_type: int, work_km: float, time_myr: float,
        quiet_myr: float,
    ) -> None:
        if self.oroType[cell] == OrogenyType["NONE"] or time_myr - self.oroLast[cell] > quiet_myr:
            self.oroStart[cell] = time_myr
            self.oroWork[cell] = 0
            self.oroType[cell] = oro_type
        elif (
            (oro_type > self.oroType[cell] or self.oroType[cell] == OrogenyType["INTRAPLATE"])
            and oro_type not in (
                OrogenyType["ARC"], OrogenyType["TRANSPRESSION"],
                OrogenyType["INTRAPLATE"],
            )
        ):
            self.oroType[cell] = oro_type
        self.oroWork[cell] += work_km
        self.oroLast[cell] = time_myr
