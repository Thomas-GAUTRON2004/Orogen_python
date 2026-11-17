"""Per-cell crust history raster ported from Orogen V2 model/crust.js."""
from __future__ import annotations

import math
import numpy as np

from ..core._numba_kernels import (
    NUMBA_AVAILABLE, zero_cell_fields_numba, snapshot_cell_fields_numba,
    create_cells_numba, snapshot_cells_numba, copy_cells_numba, remove_cells_numba,
)

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

        # Allocate one backing matrix per dtype instead of one allocation per
        # field. Each field remains a contiguous, independently writable 1-D
        # view with exactly the same dtype and values as before.
        fields_by_dtype = {}
        for name in CELL_FIELDS:
            dtype = np.dtype(_DTYPES[name])
            fields_by_dtype.setdefault(dtype, []).append(name)
        for dtype, names in fields_by_dtype.items():
            storage = np.zeros((len(names), num_cells), dtype=dtype)
            for row, name in enumerate(names):
                setattr(self, name, storage[row])

        self.count = 0
        self.fields = [getattr(self, name) for name in CELL_FIELDS]
        self._numba_fields = tuple(self.fields)


    def _cell_indices(self, cells) -> np.ndarray:
        raw = np.asarray(cells)
        if raw.ndim != 1:
            raise ValueError("cells must be a one-dimensional integer sequence")
        if raw.size == 0:
            return np.empty(0, dtype=np.int32)
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("cells must be a one-dimensional integer sequence")
        indices = np.asarray(raw, dtype=np.int64)
        indices = np.where(indices < 0, indices + self.numCells, indices)
        if np.any(indices < 0) or np.any(indices >= self.numCells):
            raise IndexError("cell index out of range")
        return np.ascontiguousarray(indices, dtype=np.int32)

    @staticmethod
    def _batch_values(values, count: int, name: str, dtype) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        if array.ndim == 0:
            return np.full(count, array.item(), dtype=dtype)
        if array.ndim != 1 or array.size != count:
            raise ValueError(f"{name} must be a scalar or have one value per cell")
        return np.ascontiguousarray(array)

    def create_many(
        self, cells, crust_type, formed_at, terrane=0,
    ) -> None:
        """Create multiple cells in order using one compiled call when available.

        Scalar metadata is broadcast; one-dimensional metadata arrays may supply
        a distinct value for each cell. Repeated cell indices retain sequential
        create() semantics: each occurrence clears the cell before setting it.
        """
        indices = self._cell_indices(cells)
        count = len(indices)
        types = self._batch_values(crust_type, count, "crust_type", np.uint8)
        times = self._batch_values(formed_at, count, "formed_at", np.float64)
        terranes = self._batch_values(terrane, count, "terrane", np.uint16)
        if NUMBA_AVAILABLE:
            self.count += int(create_cells_numba(
                self._numba_fields, self.mask, indices, types, times, terranes,
            ))
        else:
            for row, cell in enumerate(indices):
                self.create(
                    int(cell), int(types[row]), float(times[row]), int(terranes[row]),
                )

    def snapshot_many(self, cells) -> list[list]:
        """Return snapshots for multiple cells in input order."""
        indices = self._cell_indices(cells)
        if NUMBA_AVAILABLE:
            return snapshot_cells_numba(self._numba_fields, indices).tolist()
        return [self.snapshot(int(cell)) for cell in indices]

    def copy_many(self, cells, other: "CrustRaster", sources=None, rot=None) -> None:
        """Copy a sequence of source cells in one compiled call when available.

        Destination and source indices are processed in order, so repeated
        destinations and same-raster overlapping copies retain scalar semantics.
        """
        destinations = self._cell_indices(cells)
        source_indices = other._cell_indices(cells if sources is None else sources)
        if destinations.size != source_indices.size:
            raise ValueError("sources must have one index per destination cell")
        if not destinations.size:
            return
        rotate = rot is not None
        rotation = (
            np.ascontiguousarray(np.asarray(rot, dtype=np.float64).reshape(9))
            if rotate else np.zeros(9, dtype=np.float64)
        )
        if NUMBA_AVAILABLE:
            self.count += int(copy_cells_numba(
                self._numba_fields, other._numba_fields, self.mask,
                destinations, source_indices, rotation, rotate, self is other,
            ))
        else:
            for target, source in zip(destinations, source_indices):
                self.copy_from(int(target), other, int(source), rotation if rotate else None)

    def remove_many(self, cells) -> None:
        """Remove cells in order, updating count once for each present cell."""
        indices = self._cell_indices(cells)
        if not indices.size:
            return
        if NUMBA_AVAILABLE:
            self.count -= int(remove_cells_numba(self.mask, indices))
        else:
            for cell in indices:
                self.remove(int(cell))

    def create(self, cell: int, crust_type: int, formed_at: float, terrane: int = 0) -> None:
        if not self.mask[cell]:
            self.count += 1
        self.mask[cell] = 1
        if NUMBA_AVAILABLE:
            zero_cell_fields_numba(self._numba_fields, cell)
        else:
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
            matrix = np.asarray(rot, dtype=np.float64).reshape(9)
            u = matrix[0] * x + matrix[1] * y + matrix[2] * z
            v = matrix[3] * x + matrix[4] * y + matrix[5] * z
            w = matrix[6] * x + matrix[7] * y + matrix[8] * z
            length = math.hypot(u, v, w) or 1.0
            self.farDx[cell] = math.floor(127 * u / length + 0.5)
            self.farDy[cell] = math.floor(127 * v / length + 0.5)
            self.farDz[cell] = math.floor(127 * w / length + 0.5)

    def snapshot(self, cell: int) -> list:
        if NUMBA_AVAILABLE:
            # float64 represents every Float32 and current integer field value exactly;
            # tolist() boxes the snapshot in one C-level conversion, matching JS Numbers.
            return snapshot_cell_fields_numba(self._numba_fields, cell).tolist()
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
