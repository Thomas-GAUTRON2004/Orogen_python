import { readFileSync } from "node:fs";

import { chooseJunction } from "../../external/planet_heightmap_generation/v2/js/tectonics/ops/rift.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const results = input.cases.map(spec => {
  const mesh = {
    adjOffset: Uint32Array.from(spec.adjOffset),
    adjList: Uint32Array.from(spec.adjList),
    numCells: spec.mask.length,
    edgeKm: spec.edgeKm,
  };
  const crust = {
    mask: Uint8Array.from(spec.mask),
    type: Uint8Array.from(spec.type),
    flags: Uint8Array.from(spec.flags),
  };
  const world = {
    mesh,
    config: {
      riftMinArmKm: spec.riftMinArmKm,
      riftWeaknessCost: spec.riftWeaknessCost,
    },
  };
  return chooseJunction(world, { crust }, () => spec.rng);
});
console.log(JSON.stringify(results));
