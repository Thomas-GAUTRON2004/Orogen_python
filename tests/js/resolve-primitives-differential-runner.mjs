import { readFileSync } from "node:fs";

import {
  candidateCells,
  decideBirth,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/step/resolve.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function makeMesh(spec) {
  return {
    numCells: spec.numCells,
    xyz: new Float64Array(spec.xyz),
    adjOffset: new Int32Array(spec.adjOffset),
    adjList: new Int32Array(spec.adjList),
    edgeRad: spec.edgeRad,
    index: { nearest: () => 0 },
    _bins: {
      binOf: new Uint16Array(spec.numCells),
      neighbours: [new Uint16Array([0])],
      count: 1,
      at: () => 0,
    },
  };
}

function makePlate(spec) {
  return {
    id: spec.id,
    isBackground: false,
    omega: spec.omega ?? [0, 0, 0],
    bins: new Uint32Array([7]),
    crust: {
      mask: new Uint8Array(spec.mask),
      type: new Uint8Array(spec.mask.length),
      flags: new Uint16Array(spec.flags ?? spec.mask.length),
      wingKm: new Float32Array(spec.wingKm ?? spec.mask.length),
    },
    capInWorld() { return [0, 0, 1, spec.capFloor]; },
    cellAt() { return spec.underCell; },
    toWorld(x, y, z) { return [x, y - (spec.worldYOffset ?? 0), z]; },
    toPlateFrame(x, y, z) { return [x, y, z]; },
  };
}

const candidateSpec = input.candidate;
const candidateMesh = makeMesh(candidateSpec);
const candidatePlate = makePlate(candidateSpec.plate);
const candidates = candidateCells(
  { mesh: candidateMesh, occupancyStamp: 7 },
  candidatePlate,
  candidateSpec.gaps
);

const decisions = input.decisions.map((spec) => {
  const mesh = makeMesh(spec);
  const grower = makePlate(spec.grower);
  const rival = spec.rival ? makePlate(spec.rival) : null;
  const everyone = [{ plate: grower, cap: [0, 0, 1, -1] }];
  if (rival) everyone.push({ plate: rival, cap: [0, 0, 1, -1] });
  const result = decideBirth(
    { mesh, occupancyStamp: 7, config: { closingCmYr: 0 } },
    grower,
    spec.cell,
    everyone
  );
  let why = result.why;
  if (why?.startsWith("not attached")) why = "not-attached";
  return { born: result.born, why: why ?? null, trenchParent: result.trenchParent ?? null };
});

console.log(JSON.stringify({ candidates, decisions }));
