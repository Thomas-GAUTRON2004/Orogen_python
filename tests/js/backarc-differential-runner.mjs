import { readFileSync } from "node:fs";

import {
  rollbackCmYr,
  steerBackArcStrips,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/step/backarc.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const config = {
  rollbackMinAgeMyr: input.rollback.config.rollbackMinAgeMyr,
  rollbackFullAgeMyr: input.rollback.config.rollbackFullAgeMyr,
  rollbackMaxCmYr: input.rollback.config.rollbackMaxCmYr,
};
const rollback = input.rollback.ages.map(age => rollbackCmYr(age, config));

const plates = input.steering.plates.map(spec => ({
  alive: spec.alive,
  backArcOf: spec.backArcOf,
  backArcPushDir: spec.backArcPushDir,
  born: spec.born,
  rollbackCmYr: spec.rollbackCmYr,
  capCenter: spec.capCenter,
  omega: [...spec.omega],
  toWorld(x, y, z) {
    return [x, y, z];
  },
}));
steerBackArcStrips({
  plates,
  config: { backArcActiveMyr: input.steering.activeMyr },
  time: input.steering.time,
});

console.log(JSON.stringify({
  rollback,
  steering: plates.map(plate => plate.omega),
}));
