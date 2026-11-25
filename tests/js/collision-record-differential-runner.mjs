import { readFileSync } from "node:fs";

import { trackCollisions } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/collision.js";

const input = JSON.parse(readFileSync(0, "utf8"));

const RECORD_FIELDS = [
  "upper", "lower", "startedAt", "lastActive", "shorteningKm", "frontKm",
  "closingCmYr", "meanRateCmYr", "stalledMyr", "contactMyr", "cells",
  "sutured", "underthrustDue", "underthrustCells", "frontMaxKm",
  "gapKm2", "gapWidestKm", "checkedAt", "gapBestKm2", "gapShrankAt",
  "ringKm2", "weldCandidate", "weldBlockedByRing", "mergeEligible", "ringLogged",
];

function run(spec) {
  const numCells = spec.owners.length;
  const mesh = {
    numCells,
    xyz: new Float64Array(spec.xyz),
    adjOffset: new Uint32Array(spec.adjOffset),
    adjList: new Int32Array(spec.adjList),
    edgeRad: spec.edgeRad ?? 0.02,
    edgeKm: spec.edgeKm,
    cellAreaKm2: spec.cellAreaKm2,
    index: { nearest: () => 0 },
  };
  const plates = spec.plates.map((source, id) => ({
    id,
    name: source.name,
    alive: true,
    isBackground: false,
    born: 0,
    riftPartner: -1,
    stats: { continentalCells: source.continentalCells ?? 1 },
    omega: source.omega ?? [0, 0, 0],
    capCenter: source.capCenter,
    crust: {
      mask: new Uint8Array(source.mask),
      type: new Uint8Array(source.type),
      flags: new Uint16Array(source.flags ?? source.mask.map(() => 0)),
      terrane: new Int32Array(source.terrane ?? source.mask.map(() => id + 1)),
      formedAt: new Float64Array(source.formedAt ?? source.mask.map(() => 0)),
      slabKm: new Float32Array(source.mask.map(() => 0)),
      coverKm: new Float32Array(source.mask.map(() => 0)),
      count: source.mask.reduce((sum, value) => sum + value, 0),
    },
    cellAt() { return 0; },
    toWorld(x, y, z) { return [x, y, z]; },
    toPlateFrame(x, y, z) { return [x, y, z]; },
  }));
  const events = [];
  const collisions = new Map();
  const world = {
    mesh,
    owner: new Int16Array(spec.owners),
    ownerCell: new Int32Array(spec.ownerCell),
    landClaims: new Uint8Array(spec.landClaims),
    collider: new Int16Array(spec.collider),
    plates,
    collisions,
    arcScrapes: new Map(),
    budget: { recycledCells: 0 },
    config: spec.config,
    time: spec.time ?? 12,
    log(kind, text, data) { events.push({ kind, text, data }); },
  };
  const changed = trackCollisions(world, spec.dt ?? 1);
  const record = collisions.get("0:1") ?? null;
  return {
    result: {
      started: record ? 1 : 0,
      tracked: record ? 1 : 0,
      underthrustCells: record?.underthrustCells ?? 0,
      changed,
    },
    record: record === null
      ? null
      : Object.fromEntries(RECORD_FIELDS.map((field) => {
          const value = record[field];
          return [field, Number.isFinite(value) ? value : value ?? null];
        })),
    events,
  };
}

console.log(JSON.stringify(input.cases.map(run)));
