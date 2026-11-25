import { readFileSync } from "node:fs";

import {
  landmasses,
  trackCycle,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/step/supercontinent.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function makeWorld(spec) {
  let drawIndex = 0;
  return {
    config: spec.config,
    time: spec.time,
    events: spec.events,
    cycle: spec.cycle,
    mesh: {
      numCells: spec.owners.length,
      adjOffset: new Uint32Array(spec.adjOffset),
      adjList: new Int32Array(spec.adjList),
      cellAreaKm2: spec.cellAreaKm2,
      xyz: new Float64Array(spec.xyz),
    },
    owner: new Int16Array(spec.owners),
    ownerCell: new Int32Array(spec.ownerCells),
    plates: spec.plates.map((plate) => ({
      isBackground: false,
      crust: { type: new Uint8Array(plate.types) },
    })),
    mantleHeat: new Float32Array(spec.mantleHeat),
    rngCycle: () => spec.rng[drawIndex++] ?? 0,
    log: () => {},
  };
}

function snapshotCycle(cycle) {
  if (cycle === null || cycle === undefined) return null;
  const snapshot = { ...cycle };
  snapshot.near = cycle.near === null || cycle.near === undefined
    ? null
    : Array.from(cycle.near);
  if (!Number.isFinite(snapshot.gatherAt)) snapshot.gatherAt = null;
  return snapshot;
}

const masses = input.massCases.map((spec) => landmasses(makeWorld(spec)));
const cycles = input.cycleCases.map((spec) => {
  const world = makeWorld(spec);
  trackCycle(world);
  return snapshotCycle(world.cycle);
});

console.log(JSON.stringify({ masses, cycles }));
