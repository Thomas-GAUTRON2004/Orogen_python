import { readFileSync } from "node:fs";

import { isWordSeed, seedOf } from "../../external/planet_heightmap_generation/v2/js/core/seed.js";
import { makeRng } from "../../external/planet_heightmap_generation/js/rng.js";
import {
  quatFromAxisAngle,
  quatIntegrate,
  quatMul,
  quatNormalize,
  quatToEulerPole,
  quatToMat3,
} from "../../external/planet_heightmap_generation/v2/js/core/rotation.js";
import { fractureCoordinate } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/segments.js";
import { fastestSpeed, solve3 } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/torque-balance.js";

const request = JSON.parse(readFileSync(0, "utf8"));

const functions = {
  "tectonics.fastestSpeed": (input) =>
    fastestSpeed(input.omega, { capCos: input.capCos }, input.centre),
  "tectonics.solve3": (input) => solve3(input.matrix, input.vector),
  "tectonics.fractureCoordinate": (input) =>
    fractureCoordinate(...input.point, input.omega),
  "rotation.quaternionOperations": (input) => {
    const q1 = quatFromAxisAngle(...input.q1Axis, input.q1Angle);
    const q2 = quatFromAxisAngle(...input.q2Axis, input.q2Angle);
    const quaternion = quatNormalize(quatMul(q1, q2));
    const integrated = quatIntegrate(quaternion, input.omega, input.dt);
    return {
      quaternion: Array.from(quaternion),
      integrated: Array.from(integrated),
      matrix: Array.from(quatToMat3(integrated, new Float64Array(9))),
    };
  },
  "rotation.eulerPole": (input) => {
    const q1 = quatFromAxisAngle(...input.q1Axis, input.q1Angle);
    const q2 = quatFromAxisAngle(...input.q2Axis, input.q2Angle);
    const quaternion = quatNormalize(quatMul(q1, q2));
    const integrated = quatIntegrate(quaternion, input.omega, input.dt);
    return quatToEulerPole(integrated);
  },
  "seed.seedAndRng": (input) => {
    const rng = makeRng(input.rngSeed);
    return {
      seeds: input.texts.map((text) => ({
        text,
        word: isWordSeed(text),
        seed: seedOf(text),
      })),
      random: Array.from(
        { length: input.randomCount },
        () => rng(),
      ),
    };
  },
};

const fn = functions[request.function];
if (typeof fn !== "function") {
  throw new Error(`Unknown differential function: ${request.function}`);
}

const result = await fn(request.input);
process.stdout.write(JSON.stringify(result));
