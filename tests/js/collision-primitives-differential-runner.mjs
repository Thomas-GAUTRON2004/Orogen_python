import { readFileSync } from "node:fs";

import {
  areYoungRiftPartners,
  trackCollisions,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/step/collision.js";

const input = JSON.parse(readFileSync(0, "utf8"));

const young = input.youngCases.map((x) =>
  areYoungRiftPartners(
    { id: x.a.id, riftPartner: x.a.partner, born: x.a.born },
    { id: x.b.id, riftPartner: x.b.partner, born: x.b.born },
    x.time,
    { riftPushMyr: x.window }
  )
);

const spec = input.staleCase;
const collisions = new Map(
  Object.entries(spec.records).map(([key, record]) => [key, { ...record }])
);
const world = {
  mesh: {
    numCells: spec.owners.length,
    xyz: new Float64Array(3 * spec.owners.length),
    adjOffset: new Int32Array(spec.owners.length + 1),
    adjList: new Int32Array(0),
    index: null,
    edgeKm: 100,
    cellAreaKm2: 10000,
  },
  owner: new Int16Array(spec.owners),
  ownerCell: new Int32Array(spec.owners.length),
  landClaims: new Uint8Array(spec.owners.length),
  collider: new Int16Array(spec.owners.length).fill(-1),
  plates: [],
  collisions,
  config: { collisionForgetMyr: spec.forgetAfter },
  time: spec.time,
};
const merged = trackCollisions(world, 1);
const remaining = Object.fromEntries(
  [...collisions.entries()].sort(([a], [b]) => a.localeCompare(b))
);
console.log(JSON.stringify({ young, stale: { merged, remaining } }));
