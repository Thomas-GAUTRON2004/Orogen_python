import { readFileSync } from "node:fs";

import { trackCollisions } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/collision.js";
import { CrustType } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const spec = input.scenario;
const mesh = {
  numCells: 1,
  xyz: new Float64Array([1, 0, 0]),
  adjOffset: new Int32Array([0, 0]),
  adjList: new Int32Array(0),
  index: {},
  edgeKm: 100,
  cellAreaKm2: spec.cellAreaKm2,
};
function makePlate(id, type, continentalCells) {
  const crust = {
    mask: new Uint8Array([1]),
    type: new Uint8Array([type]),
    count: 1,
    remove(cell) {
      this.mask[cell] = 0;
      this.count--;
    },
  };
  return {
    id,
    name: id === 0 ? "Arc donor" : "Continental host",
    alive: true,
    isBackground: false,
    omega: [0.001, 0.002, 0.003],
    stats: { continentalCells },
    crust,
    cellAt() { return 0; },
    markReshaped() { this.reshaped = true; },
  };
}
const donor = makePlate(0, CrustType.ARC, 0);
const host = makePlate(1, CrustType.CONTINENT, 1);
const world = {
  mesh,
  owner: new Int16Array([0]),
  ownerCell: new Int32Array([0]),
  landClaims: new Uint8Array([2]),
  collider: new Int16Array([1]),
  plates: [donor, host],
  collisions: new Map(),
  arcScrapes: new Map(),
  budget: { recycledCells: 0 },
  config: {
    collisionForgetMyr: 50,
    arcAccreteMinKm2: spec.arcAccreteMinKm2,
    terraneMinAreaKm2: 100000,
  },
  time: 5,
  log() {},
};
const merged = trackCollisions(world, 1);
console.log(JSON.stringify({
  merged,
  donorAlive: donor.alive,
  donorMask: Array.from(donor.crust.mask),
  donorCount: donor.crust.count,
  donorReshaped: Boolean(donor.reshaped),
  donorOmega: donor.omega,
  recycledCells: world.budget.recycledCells,
  hostAlive: host.alive,
}));
