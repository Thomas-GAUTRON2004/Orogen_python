import { readFileSync } from "node:fs";

import { Plate } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/plate.js";
import { resolve } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/resolve.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function run(spec) {
  const numCells = spec.xyz.length / 3;
  const mesh = {
    numCells,
    xyz: new Float64Array(spec.xyz),
    adjOffset: new Int32Array(spec.adjOffset),
    adjList: new Int32Array(spec.adjList),
    edgeRad: 0.02,
    edgeKm: 100,
    cellAreaKm2: 10000,
    index: { nearest: () => 0 },
    _bins: {
      binOf: new Uint16Array(numCells),
      neighbours: [new Uint16Array([0])],
      count: 1,
      at: () => 0,
    },
  };
  const plates = spec.plates.map((source, id) => {
    const plate = new Plate(id, numCells);
    plate.name = source.name;
    plate.alive = source.alive ?? true;
    plate.isBackground = source.isBackground ?? false;
    for (let cell = 0; cell < numCells; cell++) {
      if (source.mask[cell]) {
        plate.crust.create(
          cell,
          source.type[cell],
          source.formedAt?.[cell] ?? 0
        );
      }
    }
    return plate;
  });
  const world = {
    mesh,
    plates,
    owner: new Int16Array(numCells).fill(-1),
    ownerCell: new Int32Array(numCells).fill(-1),
    claimCount: new Uint16Array(numCells),
    landClaims: new Uint8Array(numCells),
    collider: new Int16Array(numCells).fill(-1),
    consumedAt: new Uint8Array(numCells),
    provisional: new Uint8Array(numCells),
    oceanOverrider: new Int16Array(numCells).fill(-1),
    budget: { created: 0, consumed: 0 },
    time: 12,
    occupancyStamp: 0,
    config: { closingCmYr: 0 },
    log() {},
  };
  const result = resolve(world);
  return {
    result,
    owner: Array.from(world.owner),
    ownerCell: Array.from(world.ownerCell),
    claimCount: Array.from(world.claimCount),
    landClaims: Array.from(world.landClaims),
    collider: Array.from(world.collider),
    consumedAt: Array.from(world.consumedAt),
    provisional: Array.from(world.provisional),
    budget: { ...world.budget },
    plates: plates.map((plate) => ({
      alive: plate.alive,
      isBackground: plate.isBackground,
      mask: Array.from(plate.crust.mask),
      type: Array.from(plate.crust.type),
      count: plate.crust.count,
      dirty: plate.dirty,
      reshaped: plate.reshaped,
      bins: plate.bins === null ? null : Array.from(plate.bins),
      fzU: Array.from(plate.crust.fzU),
    })),
  };
}

console.log(JSON.stringify(input.cases.map(run)));
