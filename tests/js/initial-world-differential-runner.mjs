import { readFileSync } from "node:fs";
import { meshFromArrays } from "../../external/planet_heightmap_generation/v2/js/core/sim-mesh.js";
import { buildStartWorld } from "../../external/planet_heightmap_generation/v2/js/tectonics/init/start-world.js";
import { CELL_FIELDS } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";

const payload = JSON.parse(readFileSync(0, "utf8"));
const source = payload.mesh;
const mesh = meshFromArrays({
  numCells: source.numCells,
  xyz: new Float32Array(source.xyz),
  adjOffset: new Int32Array(source.adjOffset),
  adjList: new Int32Array(source.adjList),
});
const { world, start } = buildStartWorld(mesh, payload.seed, payload.config);
const list = value => Array.from(value ?? []);
const crustSnapshot = crust => ({
  mask: list(crust.mask),
  ...Object.fromEntries(CELL_FIELDS.map(field => [field, list(crust[field])])),
});
const plates = world.plates.map(plate => ({
  id: plate.id,
  name: plate.name,
  born: plate.born,
  alive: plate.alive,
  isBackground: Boolean(plate.isBackground),
  omega: list(plate.omega),
  quat: list(plate.quat),
  stats: plate.stats,
  crust: crustSnapshot(plate.crust),
}));
const worldState = {
  budget: world.budget,
  events: world.events,
  owner: list(world.owner),
  ownerCell: list(world.ownerCell),
  claimCount: list(world.claimCount),
  landClaims: list(world.landClaims),
  collider: list(world.collider),
  oceanOverrider: list(world.oceanOverrider),
  consumedAt: list(world.consumedAt),
  provisional: list(world.provisional),
  provisionalLike: list(world.provisionalLike),
  boundaryKind: list(world.boundaryKind),
  convergence: list(world.convergence),
  mantleHeat: list(world.mantleHeat),
  startSeamKm: {
    major: list(world.startSeamKm.major),
    majorRange: list(world.startSeamKm.majorRange),
    ring: list(world.startSeamKm.ring),
  },
};
const startState = {
  cratons: start.cratons,
  centre: start.centre,
  rimCentre: start.rimCentre,
  reference: start.reference,
  elongation: start.elongation,
  seams: list(start.seams),
  owner: list(start.owner),
  skirtKm: start.skirtKm,
  rim: list(start.rim),
  passiveFrom: start.passiveFrom,
  passiveTo: start.passiveTo,
  attempts: start.attempts,
  cratonAreaScale: start.cratonAreaScale,
  fractionMet: start.fractionMet,
  continentId: start.continentId,
  oceanId: start.oceanId,
  landCells: start.landCells,
  continentalFraction: start.continentalFraction,
  apronCells: start.apronCells,
  trench: list(start.trench),
};
process.stdout.write(JSON.stringify({ start: startState, world: worldState, plates }));
