import { readFileSync } from "node:fs";

import { trackCollisions } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/collision.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function run(spec) {
  const mesh = {
    numCells: spec.owners.length,
    xyz: new Float64Array(3 * spec.owners.length),
    adjOffset: new Int32Array(spec.adjOffset),
    adjList: new Int32Array(spec.adjList),
    index: null,
    edgeKm: 100,
    cellAreaKm2: 10000,
  };
  const plates = spec.plates.map((plate, id) => ({
    id,
    name: `plate-${id}`,
    alive: false,
    isBackground: plate.isBackground,
    crust: {
      mask: new Uint8Array(plate.mask),
      type: new Uint8Array(plate.type),
      flags: new Uint16Array(plate.flags),
    },
  }));
  const collisions = new Map();
  const world = {
    mesh,
    owner: new Int16Array(spec.owners),
    ownerCell: new Int32Array(spec.ownerCell),
    landClaims: new Uint8Array(spec.landClaims),
    collider: new Int16Array(spec.collider),
    plates,
    collisions,
    config: { collisionForgetMyr: 50 },
    time: 0,
  };

  const captured = [];
  const originalSet = Map.prototype.set;
  Map.prototype.set = function (key, value) {
    if (this !== collisions && !captured.includes(this)) captured.push(this);
    return originalSet.call(this, key, value);
  };
  try {
    trackCollisions(world, 1);
  } finally {
    Map.prototype.set = originalSet;
  }

  const asObject = (map) => Object.fromEntries(
    [...map.entries()].map(([key, cells]) => [key, Array.from(cells)])
  );
  return {
    overlap: captured[0] ? asObject(captured[0]) : {},
    touching: captured[1] ? asObject(captured[1]) : {},
  };
}

console.log(JSON.stringify(input.cases.map(run)));
