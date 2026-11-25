import { readFileSync } from "node:fs";

import {
  maskedCells,
  sidesOf,
  partition,
  exteriorSea,
  elongationOf,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/ops/rift-partition.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function evaluate(spec) {
  const mesh = {
    numCells: spec.numCells,
    xyz: new Float64Array(spec.xyz),
    adjOffset: new Int32Array(spec.adjOffset),
    adjList: new Int32Array(spec.adjList),
    edgeKm: spec.edgeKm,
  };
  const crust = {
    count: spec.mask.reduce((sum, value) => sum + value, 0),
    mask: new Uint8Array(spec.mask),
    type: new Uint8Array(spec.type),
    terrane: new Int32Array(spec.terrane),
  };
  const cells = spec.cells === null ? maskedCells(crust) : new Int32Array(spec.cells);
  const part = partition(mesh, crust, spec.path, cells);
  return {
    masked: Array.from(maskedCells(crust)),
    partition: {
      side: Array.from(part.side),
      land: Array.from(part.land),
      cratons: Array.from(part.cratons),
    },
    summary: sidesOf(mesh, crust, spec.path, cells),
    exterior: Array.from(exteriorSea(mesh, crust)),
    elongation: [
      elongationOf(mesh, part.side, 1, cells),
      elongationOf(mesh, part.side, 2, cells),
    ],
  };
}

console.log(JSON.stringify(input.cases.map(evaluate)));
