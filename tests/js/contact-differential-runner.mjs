import { readFileSync } from "node:fs";

import { contactAt } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/contact.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function run(spec) {
  const mesh = {
    numCells: spec.numCells,
    xyz: new Float64Array(spec.xyz),
    adjOffset: new Uint32Array(spec.adjOffset),
    adjList: new Int32Array(spec.adjList),
  };
  const world = {
    mesh,
    owner: new Int16Array(spec.owner),
    occupancyStamp: spec.occupancyStamp ?? 0,
  };
  const results = [];
  for (const query of spec.queries) {
    if (query.owner !== undefined) world.owner.set(query.owner);
    if (query.occupancyStamp !== undefined) {
      world.occupancyStamp = query.occupancyStamp;
    }
    const out = [9, 9, 9, 9];
    const across = contactAt(world, query.cell, out);
    results.push({ across, normal: out });
  }
  return results;
}

console.log(JSON.stringify(input.cases.map(run)));
