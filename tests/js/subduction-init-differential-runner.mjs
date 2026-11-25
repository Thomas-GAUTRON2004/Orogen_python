import { readFileSync } from "node:fs";

import { CrustFlag } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";
import { EventKind } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/world.js";
import { initiateSubduction } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/subduction-init.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function makeWorld(spec) {
  const mesh = {
    numCells: spec.numCells,
    adjOffset: new Int32Array(spec.adjOffset),
    adjList: new Int32Array(spec.adjList),
    xyz: new Float64Array(spec.xyz),
    edgeKm: spec.edgeKm,
    index: {},
  };
  const plates = spec.plates.map((source, id) => {
    const flags = new Uint16Array(source.flags);
    const slabKm = new Float32Array(source.slabKm);
    const crust = {
      mask: new Uint8Array(source.mask),
      type: new Uint8Array(source.type),
      flags,
      formedAt: new Float64Array(source.formedAt),
      slabKm,
      armTrench(cell, depth) {
        this.flags[cell] |= CrustFlag.TRENCH;
        this.slabKm[cell] = Math.max(this.slabKm[cell], depth);
      },
    };
    return {
      id,
      name: source.name,
      alive: source.alive,
      isBackground: false,
      reshaped: source.reshaped,
      dirty: false,
      crust,
      cellAt(_index, x) {
        return Math.max(0, Math.min(spec.numCells - 1, Math.round(x) + source.offset));
      },
      toWorld(x, y, z) {
        return [x, y - source.worldYOffset, z];
      },
    };
  });
  const events = [];
  let drawIndex = 0;
  return {
    mesh,
    owner: new Int16Array(spec.owner),
    ownerCell: new Int32Array(spec.ownerCell),
    convergence: new Float64Array(spec.convergence),
    plates,
    config: spec.config,
    time: spec.time,
    cycle: spec.cycle,
    rngSubduction: () => spec.draws[drawIndex++] ?? 0,
    log(kind, message, data) {
      events.push({ kind, message, data });
    },
    events,
  };
}

const results = input.cases.map((spec) => {
  const world = makeWorld(spec);
  const started = initiateSubduction(world, spec.dt);
  return {
    started,
    flags: world.plates.map((plate) => Array.from(plate.crust.flags)),
    slabKm: world.plates.map((plate) => Array.from(plate.crust.slabKm)),
    dirty: world.plates.map((plate) => plate.dirty),
    events: world.events,
  };
});
console.log(JSON.stringify(results));
