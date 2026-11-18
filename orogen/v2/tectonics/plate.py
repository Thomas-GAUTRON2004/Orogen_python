"""Rigid tectonic plate model ported from Orogen V2 model/plate.js."""

from __future__ import annotations

import math

import numpy as np

from ..core.rotation import quat_identity, quat_to_mat3
from .crust import CrustRaster, CrustType


class Plate:
    """A rigid orientation, angular velocity and plate-frame crust raster."""

    def __init__(self, plate_id: int, num_cells: int, born: float = 0.0):
        self.id = int(plate_id)
        self.name = str(plate_id)
        self.alive = True
        self.born = float(born)
        self.bornOfRift = False
        self.hue = int(plate_id)
        self.isBackground = False
        self.quat = quat_identity()
        self.mat = np.asarray(quat_to_mat3(self.quat), dtype=np.float64).reshape(3, 3)
        self.omega = [0.0, 0.0, 0.0]
        self.crust = CrustRaster(num_cells)
        self.dirty = False
        self.reshaped = False
        self.capCenter = [0.0, 0.0, 1.0]
        self.capCos = -1.0
        self.stats = {"cells": 0, "continentalCells": 0}
        self.bins = None
        self.dynamics = None
        self.wingBudgetKm = 0.0
        self.wings: dict[object, object] = {}
        self.arcTerraneId = None
        self.marginIsleTerrane = None
        self.riftPartner = -1
        self.oceanSplitWith = -1
        self.splitAt = -math.inf
        self.riftPushDir = None
        self.flatSlab = False
        self.flatSlabKm = 0.0
        self.lastFedAt = -math.inf
        self.backArcOf = -1
        self.backArcPushDir = None
        self.rollbackCmYr = 0.0
        self.lastBackArcAt = -math.inf
        self.isOceanBorn = False
        self.scene: dict[str, object] = {}

    def set_quat(self, quaternion) -> None:
        self.quat = list(quaternion)
        self.mat = np.asarray(quat_to_mat3(self.quat), dtype=np.float64).reshape(3, 3)

    def to_world(self, x: float, y: float, z: float) -> np.ndarray:
        return self.mat @ np.asarray([x, y, z], dtype=np.float64)

    def to_plate_frame(self, x: float, y: float, z: float) -> np.ndarray:
        return self.mat.T @ np.asarray([x, y, z], dtype=np.float64)

    def cell_at(self, index, x: float, y: float, z: float) -> int:
        local = self.to_plate_frame(x, y, z)
        return int(index.nearest(*map(float, local)))

    def cap_in_world(self, mesh) -> list[float]:
        centre = self.to_world(*self.capCenter)
        radius = math.acos(max(-1.0, min(1.0, self.capCos))) + 2 * mesh.edge_rad
        cap_cos = -2.0 if radius >= math.pi else math.cos(radius)
        return [*map(float, centre), cap_cos]

    def mark_reshaped(self) -> None:
        self.reshaped = True
        self.dirty = True

    def refresh_extent(self, xyz) -> None:
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        mask = self.crust.mask.astype(bool)
        cells = np.flatnonzero(mask)
        count = int(cells.size)
        continental = int(np.count_nonzero(self.crust.type[mask] != CrustType["OCEAN"]))
        self.stats["cells"] = count
        self.stats["continentalCells"] = continental
        self.crust.count = count
        self.dirty = False
        if count == 0:
            self.capCenter, self.capCos = [0.0, 0.0, 1.0], -1.0
            return
        total = points[cells].sum(axis=0)
        length = float(np.linalg.norm(total))
        if length < 1e-6 * count:
            self.capCenter, self.capCos = [0.0, 0.0, 1.0], -1.0
            return
        centre = total / length
        min_dot = float(np.min(points[cells] @ centre))
        self.capCenter = centre.tolist()
        self.capCos = min_dot


def frame_rotation(source: Plate, target: Plate) -> np.ndarray:
    """Rotation from source plate coordinates into target plate coordinates."""
    return target.mat.T @ source.mat
