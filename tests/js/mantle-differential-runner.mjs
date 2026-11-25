import { readFileSync } from "node:fs";

import {
  flowOfCycle,
  mantleFlowAt,
  dispersalField,
  repelAt,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/step/supercontinent.js";
import { CrustType } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const { flowConfig, flowCases, dispersal } = input.mantle;

const flows = flowCases.map((scenario) => {
  const [x, y, z] = scenario.point;
  const flow = flowOfCycle(scenario.cycle, flowConfig, x, y, z, [10, 10, 10]);
  const wrapper = mantleFlowAt(
    { cycle: scenario.cycle, config: flowConfig },
    x, y, z, [10, 10, 10]
  );
  return { flow, wrapper };
});

const world = {
  config: dispersal.config,
  cycle: dispersal.cycle,
  mesh: {
    numCells: dispersal.owners.length,
    xyz: new Float64Array(dispersal.xyz),
    cellAreaKm2: dispersal.cellAreaKm2,
  },
  owner: new Int16Array(dispersal.owners),
  ownerCell: new Int32Array(dispersal.ownerCells),
  landClaims: new Uint8Array(dispersal.landClaims),
  collider: new Int16Array(dispersal.collider),
  plates: dispersal.plates.map((plate) => ({
    isBackground: plate.isBackground,
    crust: { type: new Uint8Array(plate.types) },
  })),
};

const field = dispersalField(world);
const fieldSnapshot = field === null ? null : {
  centres: field.centres,
  touching: [...field.touching].sort((a, b) => a - b),
  k: field.k,
  reach: field.reach,
};
const pushes = dispersal.repelQueries.map((query) =>
  repelAt(field, query.plateId, ...query.point, [0, 0, 0])
);
const disabledByConfig = dispersalField({
  ...world,
  config: { ...world.config, dispersalRepulsion: 0 },
}) === null;
const disabledByPhase = dispersalField({
  ...world,
  cycle: { ...world.cycle, phase: "assembled" },
}) === null;

console.log(JSON.stringify({
  flows,
  field: fieldSnapshot,
  pushes,
  disabledByConfig,
  disabledByPhase,
}));
