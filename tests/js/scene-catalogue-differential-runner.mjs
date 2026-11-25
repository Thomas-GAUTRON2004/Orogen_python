import { readFileSync } from "node:fs";
import { meshFromArrays } from "../../external/planet_heightmap_generation/v2/js/core/sim-mesh.js";
import { CELL_FIELDS } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";
import { BUDGET_SCENES } from "../../external/planet_heightmap_generation/v2/js/tectonics/scenarios/budget-scenes.js";
import { COLLISION_SCENES } from "../../external/planet_heightmap_generation/v2/js/tectonics/scenarios/collision-scenes.js";
import { EVENT_SCENES } from "../../external/planet_heightmap_generation/v2/js/tectonics/scenarios/event-scenes.js";
import { FORCE_SCENES } from "../../external/planet_heightmap_generation/v2/js/tectonics/scenarios/force-scenes.js";
import { HOTSPOT_SCENES } from "../../external/planet_heightmap_generation/v2/js/tectonics/scenarios/hotspot-scenes.js";
import { MARGIN_SCENES } from "../../external/planet_heightmap_generation/v2/js/tectonics/scenarios/margin-scenes.js";
import { RIFT_SCENES } from "../../external/planet_heightmap_generation/v2/js/tectonics/scenarios/rift-scenes.js";
import { TERRAIN_SCENES } from "../../external/planet_heightmap_generation/v2/js/tectonics/scenarios/terrain-scenes.js";

const payload = JSON.parse(readFileSync(0, "utf8"));
const source = payload.mesh;
const mesh = meshFromArrays({
  numCells: source.numCells,
  xyz: new Float32Array(source.xyz),
  adjOffset: new Int32Array(source.adjOffset),
  adjList: new Int32Array(source.adjList),
});
const catalogues = {
  budget: BUDGET_SCENES,
  collision: COLLISION_SCENES,
  event: EVENT_SCENES,
  force: FORCE_SCENES,
  hotspot: HOTSPOT_SCENES,
  margin: MARGIN_SCENES,
  rift: RIFT_SCENES,
  terrain: TERRAIN_SCENES,
};
const list = value => Array.from(value ?? []);
function snapshot(world) {
  return {
    time: world.time,
    step: world.step,
    config: world.config,
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
    plumes: world.plumes ?? [],
    hotspots: world.hotspots ?? [],
    plates: world.plates.map(plate => ({
      id: plate.id,
      name: plate.name,
      alive: plate.alive,
      born: plate.born,
      isBackground: Boolean(plate.isBackground),
      omega: list(plate.omega),
      quat: list(plate.quat),
      stats: plate.stats,
      crust: {
        mask: list(plate.crust.mask),
        ...Object.fromEntries(CELL_FIELDS.map(field => [field, list(plate.crust[field])])),
      },
    })),
  };
}
const results = [];
for (const item of payload.cases) {
  const catalogue = catalogues[item.catalogue];
  if (!catalogue) throw new Error(`Unknown scene catalogue: ${item.catalogue}`);
  const scene = catalogue.find(candidate => candidate.id === item.id);
  if (!scene || typeof scene.build !== "function") {
    throw new Error(`Missing build function for ${item.catalogue}/${item.id}`);
  }
  const world = scene.build(mesh, payload.seed);
  if (item.drive && typeof scene.drive === "function") scene.drive(world);
  results.push({ catalogue: item.catalogue, id: item.id, state: snapshot(world) });
}
process.stdout.write(JSON.stringify(results));
