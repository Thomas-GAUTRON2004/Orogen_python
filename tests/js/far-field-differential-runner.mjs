import { readFileSync } from "node:fs";
import { TectonicWorld } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/world.js";
import { CrustType, CrustFlag, CELL_FIELDS } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";
import {
  feedFarField, evolveFarField, spreadPush, recordPush,
  lineStrike, farTake, recentPush,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/step/far-field.js";

const payload = JSON.parse(readFileSync(0, "utf8"));
const mesh = {
  numCells: payload.mesh.numCells,
  xyz: new Float32Array(payload.mesh.xyz),
  adjOffset: new Int32Array(payload.mesh.adjOffset),
  adjList: new Int32Array(payload.mesh.adjList),
};
const world = new TectonicWorld(mesh, payload.seed, payload.config);
const plate = world.addPlate("Reference");
const types = [CrustType.CONTINENT, CrustType.CRATON, CrustType.CONTINENT, CrustType.CONTINENT, CrustType.OCEAN];
for (let cell = 0; cell < types.length; cell++) plate.crust.create(cell, types[cell], -20, cell + 1);
for (const cell of [1, 2, 3]) plate.crust.flags[cell] |= CrustFlag.SUTURE;
plate.crust.flags[4] |= CrustFlag.FAILED_RIFT;
world.time = 10;

const list = value => Array.from(value ?? []);
const direct = spreadPush(world, plate, new Map([[0, 100]]), 1000, world.time);
recordPush(plate.crust, 3, 4.5, 200, mesh.xyz, 0, 15, 100);
const strike = lineStrike(plate.crust, mesh, 2);
const noStrike = lineStrike(plate.crust, mesh, 0);
const factors = {
  craton: farTake(world.config, CrustType.CRATON, 0),
  weak: farTake(world.config, CrustType.CONTINENT, CrustFlag.SUTURE),
  normal: farTake(world.config, CrustType.CONTINENT, 0),
  recent: recentPush(10, 5, 15, 100),
  future: recentPush(10, 15, 5, 100),
  zero: recentPush(0, 0, 100, 100),
};
const dead = world.addPlate("Dead");
dead.alive = false;
feedFarField(world, plate, "front", [0, 1], [10, 20], 500, 0.5);
feedFarField(world, plate, "front", [1, 2], [2, 4], 700, 0.25);
feedFarField(world, plate, "ignored-zero", [0], 10, 900, 0);
feedFarField(world, plate, "ignored-empty", [], 10, 900, 1);
feedFarField(world, dead, "dead-front", [1], 30, 800, 1);
const pending = [...world.farPending.entries()].map(([key, value]) => ({
  key,
  plate: value.plate.id,
  cells: [...value.cells.entries()].sort((a, b) => a[0] - b[0]),
  frontKm: value.frontKm,
}));
const early = evolveFarField(world, 1);
const due = evolveFarField(world, 1);
const crust = Object.fromEntries(CELL_FIELDS.map(field => [field, list(plate.crust[field])]));
crust.mask = list(plate.crust.mask);
const final = {
  direct: { cells: list(direct.cells), inc: list(direct.inc), from: list(direct.from), dist: list(direct.dist) },
  strike,
  noStrike,
  factors,
  pending,
  early,
  due,
  farClock: world.farClock,
  farLastWindow: world.farLastWindow,
  pendingAfter: [...world.farPending.keys()],
  crust,
};
process.stdout.write(JSON.stringify(final));
