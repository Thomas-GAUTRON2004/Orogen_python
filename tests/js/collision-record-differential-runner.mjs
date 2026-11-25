import { readFileSync } from "node:fs";

import { trackCollisions } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/collision.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function run(spec) {
  const mesh = {
    numCells: 1,
    xyz: new Float64Array([1, 0, 0]),
    adjOffset: new Uint32Array([0, 0]),
    adjList: new Int32Array(0),
    edgeRad: 0.02,
    edgeKm: 100,
    cellAreaKm2: 10000,
    index: { nearest: () => 0 },
  };
  const plates = spec.plates.map((source, id) => ({
    id,
    name: source.name,
    alive: true,
    isBackground: false,
    born: 0,
    riftPartner: -1,
    stats: { continentalCells: 1 },
    omega: [0, 0, 0],
    capCenter: source.capCenter,
    crust: {
      mask: new Uint8Array([1]),
      type: new Uint8Array([source.type]),
      flags: new Uint16Array([0]),
      terrane: new Int32Array([id + 1]),
      formedAt: new Float64Array([0]),
      slabKm: new Float32Array([0]),
      coverKm: new Float32Array([0]),
      count: 1,
    },
    cellAt() { return 0; },
    toWorld(x, y, z) { return [x, y, z]; },
    toPlateFrame(x, y, z) { return [x, y, z]; },
  }));
  const events = [];
  const collisions = new Map();
  const world = {
    mesh,
    owner: new Int16Array([0]),
    ownerCell: new Int32Array([0]),
    landClaims: new Uint8Array([2]),
    collider: new Int16Array([1]),
    plates,
    collisions,
    config: spec.config,
    time: 12,
    log(kind, text, data) { events.push({ kind, text, data }); },
  };
  const changed = trackCollisions(world, 1);
  const record = collisions.get("0:1") ?? null;
  const fields = [
    "upper", "lower", "startedAt", "lastActive", "shorteningKm", "frontKm",
    "closingCmYr", "meanRateCmYr", "stalledMyr", "contactMyr", "cells",
    "sutured", "underthrustDue", "underthrustCells", "frontMaxKm",
  ];
  return {
    result: {
      started: record ? 1 : 0,
      tracked: record ? 1 : 0,
      underthrustCells: record?.underthrustCells ?? 0,
      changed,
    },
    record: record === null
      ? null
      : Object.fromEntries(fields.map((field) => [field, record[field]])),
    events,
  };
}

console.log(JSON.stringify(input.cases.map(run)));
