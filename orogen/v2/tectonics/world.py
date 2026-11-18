"""Mutable tectonic-world state container ported from Orogen V2 model/world.js."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..core.seed import make_rng
from .crust import CrustType
from .plate import Plate

EventKind = {
    "START": "start",
    "PLATE_BORN": "plate-born",
    "COLLISION": "collision",
    "SUTURE": "suture",
    "DOCK": "dock",
    "ARC_SCRAPE": "arc-scrape",
    "ARC_COLLAPSE": "arc-collapse",
    "ISLE_MERGE": "isle-merge",
    "ARC_RECYCLED": "arc-recycled",
    "FRAGMENT": "fragment",
    "RIFT": "rift",
    "SUBDUCTION_INIT": "subduction-init",
    "OCEAN_SPLIT": "ocean-split",
    "MICROPLATE": "microplate",
    "SLAB_FLAT": "slab-flat",
    "SLAB_STEEP": "slab-steep",
    "BACKARC_OPEN": "backarc-open",
    "BACKARC_CLOSE": "backarc-close",
    "PLUME_RISE": "plume-rise",
    "PLUME_DEATH": "plume-death",
    "LIP": "lip",
    "SUPERCONTINENT": "supercontinent",
    "DISPERSED": "dispersed",
}
_EVENT_KINDS = frozenset(EventKind.values())

BoundaryKind = {
    "NONE": 0,
    "RIDGE": 1,
    "TRANSFORM": 2,
    "SUBDUCTING": 3,
    "OVERRIDING": 4,
    "COLLISION": 5,
}
Provisional = {"NO": 0, "NEWBORN": 1, "LIKE_NEIGHBOUR": 2}


class TectonicWorld:
    """World-frame ownership and per-plate crust state at one simulation instant.

    The upstream world deep-copies the global TECTONICS configuration. Until that
    full config module is ported, callers must pass the matching config mapping.
    """

    def __init__(
        self, sim_mesh, seed: int, config: dict, overrides: dict | None = None
    ):
        if config is None:
            raise ValueError(
                "config must be supplied; upstream TECTONICS defaults are not yet ported"
            )
        self.mesh = sim_mesh
        self.seed = int(seed)
        self.config = deepcopy(config)
        if overrides:
            self.config.update(overrides)
        # Independent deterministic streams: adding draws to one mechanism must not
        # perturb the random sequence of any other mechanism.
        self.rng = make_rng((self.seed ^ 0x5EED) & 0xFFFFFFFF)
        self.rngSubduction = make_rng((self.seed ^ 0x51AB) & 0xFFFFFFFF)
        self.rngMarginFounder = make_rng((self.seed ^ 0xF0D2) & 0xFFFFFFFF)
        self.rngOceanSplit = make_rng((self.seed ^ 0x0C5A) & 0xFFFFFFFF)
        self.rngHotspot = make_rng((self.seed ^ 0x7075) & 0xFFFFFFFF)
        self.rngVolcano = make_rng((self.seed ^ 0x7A1C) & 0xFFFFFFFF)
        self.rngBackArc = make_rng((self.seed ^ 0xBAC4) & 0xFFFFFFFF)
        self.rngMarginIsle = make_rng((self.seed ^ 0x15E1) & 0xFFFFFFFF)
        self.rngCycle = make_rng((self.seed ^ 0xC1C1) & 0xFFFFFFFF)

        n = int(sim_mesh.num_cells)
        self.hotspots = []
        self.mantleHeat = np.zeros(n, dtype=np.float32)
        self.hotspotsReady = False
        self.cycle = None
        self.pendingPlumeRifts = []
        self.lastRiftAt = None
        self.plumes = []
        self.terraneLetters = None
        self.time = 0.0
        self.step = 0
        self.occupancyStamp = 0
        self.lastDt = 0.0
        self.slowMyr = 0.0
        self.plates = []
        self.goneCrust = None

        self.owner = np.full(n, -1, dtype=np.int16)
        self.ownerCell = np.zeros(n, dtype=np.int32)
        self.claimCount = np.zeros(n, dtype=np.uint8)
        self.landClaims = np.zeros(n, dtype=np.uint8)
        self.collider = np.full(n, -1, dtype=np.int16)
        self.oceanOverrider = np.full(n, -1, dtype=np.int16)
        self.consumedAt = np.zeros(n, dtype=np.uint8)
        self.provisional = np.zeros(n, dtype=np.uint8)
        self.provisionalLike = np.zeros(n, dtype=np.int32)
        self.boundaryKind = np.zeros(n, dtype=np.uint8)
        self.convergence = np.zeros(n, dtype=np.float32)

        self.budget = {
            "created": 0,
            "consumed": 0,
            "erodedCells": 0,
            "underthrustCells": 0,
            "recycledCells": 0,
        }
        self.events = []
        self.collisions = {}
        self.arcScrapes = {}
        self.lastTerraneId = 0
        self.erosionNoise = None
        self.erosionThreshold = 0.0

    def next_terrane_id(self) -> int:
        if self.lastTerraneId == 0:
            for plate in self.plates:
                occupied = plate.crust.mask.astype(bool)
                if np.any(occupied):
                    self.lastTerraneId = max(
                        self.lastTerraneId, int(plate.crust.terrane[occupied].max())
                    )
        if self.lastTerraneId >= 0xFFFF:
            raise OverflowError("terrane ids exhausted (Uint16)")
        self.lastTerraneId += 1
        return self.lastTerraneId

    def add_plate(self, name: str) -> Plate:
        plate = Plate(len(self.plates), self.mesh.num_cells, self.time)
        plate.name = name
        self.plates.append(plate)
        return plate

    def name_by_cratons(self, plate: Plate, fallback: str) -> str:
        if self.terraneLetters is None:
            return fallback
        held = set()
        crust = plate.crust
        for cell in range(crust.numCells):
            terrane = int(crust.terrane[cell])
            if crust.mask[cell] and crust.type[cell] == CrustType["CRATON"]:
                letter = self.terraneLetters.get(terrane)
                if letter:
                    held.add(letter)
        return "".join(sorted(held)) if held else fallback

    def log(self, kind: str, text: str, data: dict | None = None) -> None:
        if kind not in _EVENT_KINDS:
            raise ValueError(f"unknown event kind: {kind}")
        self.events.append(
            {
                "time": self.time,
                "kind": kind,
                "text": text,
                "data": {} if data is None else data,
            }
        )
