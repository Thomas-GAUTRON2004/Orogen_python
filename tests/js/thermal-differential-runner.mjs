import { readFileSync } from "node:fs";

import {
  updateHotspots,
  insulateAtStart,
  pickPlumeSite,
  isContinental,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/step/hotspots.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function makeWorld(spec) {
  return {
    config: spec.config,
    mesh: {
      numCells: spec.owners.length,
      xyz: new Float64Array(spec.xyz),
    },
    owner: new Int16Array(spec.owners),
    ownerCell: new Int32Array(spec.ownerCells),
    plates: spec.plates.map((plate) => ({
      crust: { type: new Uint8Array(plate.types) },
    })),
    mantleHeat: new Float32Array(spec.heat),
    hotspots: spec.hotspots ?? [],
    hotspotsReady: true,
    plumes: [],
    time: spec.time ?? 0,
    rngHotspot: () => 0,
    log: () => {},
  };
}

const insulationWorld = makeWorld(input.insulation);
const continental = input.insulation.owners.map((_, cell) =>
  isContinental(insulationWorld, cell)
);
insulateAtStart(insulationWorld);

const warmWorld = makeWorld(input.warm);
updateHotspots(warmWorld, input.warm.dt);

const picks = input.pickCases.map((scenario) => {
  const world = makeWorld(scenario);
  return scenario.draws.map((draw) =>
    pickPlumeSite(world, () => draw)
  );
});

console.log(JSON.stringify({
  insulation: {
    continental,
    heat: Array.from(insulationWorld.mantleHeat),
  },
  warmed: Array.from(warmWorld.mantleHeat),
  picks,
}));
