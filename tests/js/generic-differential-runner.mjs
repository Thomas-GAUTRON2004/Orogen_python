import { readFileSync } from "node:fs";\nimport { CrustRaster, CrustFlag, CELL_FIELDS } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";

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
import {
  encodeCoverKm, decodeCoverKm,
  encodeExhumKm, decodeExhumKm,
  encodeUnroofKm, decodeUnroofKm,
  encodeFloodMyr, decodeFloodMyr,
  encodeArcMyr, decodeArcMyr,
  encodeAge, decodeAge,
  encodeFloodAge, decodeFloodAge,
  encodeFacies, decodeFacies,
  GEO_FRAME_FIELDS, GEOLOGY_READS,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/model/geology.js";
import { thickenCover } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/geology.js";
import { beltReliefKm, buildHeights } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/heights.js";
import { broadProfile } from "../../external/planet_heightmap_generation/v2/js/terrain/profiles.js";\nimport { chooseStep } from "../../external/planet_heightmap_generation/v2/js/tectonics/simulate.js";\nimport { binIndex, markBin, mayHaveCrustNear } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/occupancy.js";
import {
  latLonToXyz, xyzToLatLon, omegaFromEulerPole, surfaceVelocity,
  speedCmYrAt, advancePlates, poseAtConstantOmega,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/step/kinematics.js";

const request = JSON.parse(readFileSync(0, "utf8"));

const functions = {\n  "tectonics.crustSnapshot": (input) => {\n    const crust = new CrustRaster(input.size);\n    crust.create(input.cell, input.type, input.longitude, input.terrane);\n    crust.armTrench(input.cell, input.trenchDepth);\n    crust.recordSubduction(input.cell, input.subduction);\n    for (const [type, work, start, last] of input.orogeny) crust.recordOrogeny(input.cell, type, work, start, last);\n    return {\n      fields: CELL_FIELDS, count: crust.count, mask: Array.from(crust.mask),\n      snapshot: crust.snapshot(input.cell), flags: crust.flags[input.cell],\n      slabKm: crust.slabKm[input.cell], oroType: crust.oroType[input.cell],\n      oroWork: crust.oroWork[input.cell], oroStart: crust.oroStart[input.cell],\n      oroLast: crust.oroLast[input.cell], flagTrench: CrustFlag.TRENCH,\n    };\n  },\n  "tectonics.crustBranches": (input) => {\n    const crust = new CrustRaster(input.size);\n    for (let i = 0; i < input.size; i++) crust.create(i, input.baseType, input.longitude, input.terraneStart + i);\n    for (const [cell, type, work, start, last] of input.operations) crust.recordOrogeny(cell, type, work, start, last);\n    return Array.from({ length: input.size }, (_, i) => crust.snapshot(i));\n  },\n  "tectonics.plateKinematics": (input) => {\n    const origin = latLonToXyz(...input.origin);\n    const velocityPoint = latLonToXyz(...input.velocityPoint);\n    const omega = omegaFromEulerPole(origin, input.eulerSpeed);\n    const velocity = surfaceVelocity(omega, ...velocityPoint);\n    const q0 = quatFromAxisAngle(...input.poseAxis, input.poseAngle);\n    const pose = poseAtConstantOmega(q0, omega, input.poseDt);\n    const world = {\n      plates: [\n        { alive: true, quat: q0, omega, setQuat(q) { this.quat = q; } },\n        { alive: false, quat: q0, omega: [1, 2, 3], setQuat(q) { this.quat = q; } },\n      ],\n      time: input.worldTime, step: input.worldStep,\n    };\n    advancePlates(world, input.advanceDt);\n    return {\n      xyz: origin, latlon: xyzToLatLon(...velocityPoint), omega, velocity,\n      speed: speedCmYrAt(omega, ...velocityPoint), pose,\n      advanced: world.plates[0].quat, time: world.time, step: world.step,\n      dead: world.plates[1].quat,\n    };\n  },\n  "tectonics.occupancyBins": (input) => {\n    const mesh = { numCells: input.numCells, xyz: new Float32Array(input.xyz), edgeKm: input.edgeKm };\n    const bins = binIndex(mesh);\n    const plate = {};\n    markBin(plate, bins.binOf[0], input.terrane, bins.count);\n    return {\n      count: bins.count,\n      binOf: Array.from(bins.binOf),\n      counts: bins.neighbours.map((values) => values.length),\n      near: Array.from({ length: input.numCells }, (_, index) => mayHaveCrustNear(plate, index, input.terrane, bins)),\n      absent: mayHaveCrustNear({}, 0, input.terrane, bins),\n    };\n  },\n  "tectonics.chooseStep": (input) => {\n    const world = {\n      config: input.config,\n      mesh: { edgeRad: input.edgeRad },\n      plates: input.plates.map((plate) => ({\n        alive: plate.alive, omega: plate.omega, capCos: plate.capCos,\n        capCenter: plate.capCenter, crust: { count: plate.count },\n        toWorld(a, b, c) { return [a, b, c]; },\n      })),\n    };\n    return chooseStep(world);\n  },\n  "tectonics.geologyCodecs": (input) => {\n    const pairs = (values, encode, decode) => values.map((value) => { const encoded = encode(value); return [encoded, decode(encoded)]; });\n    const floodAge = input.floodAge.map(([last, time, ever]) => { const encoded = encodeFloodAge(last, time, ever); const decoded = decodeFloodAge(encoded); return [encoded, Number.isFinite(decoded) ? decoded : null]; });\n    const facies = input.facies.map((values) => { const encoded = encodeFacies(...values); return [encoded, decodeFacies(encoded)]; });\n    return {\n      cover: pairs(input.cover, encodeCoverKm, decodeCoverKm),\n      exhum: pairs(input.exhum, encodeExhumKm, decodeExhumKm),\n      unroof: pairs(input.unroof, encodeUnroofKm, decodeUnroofKm),\n      flood: pairs(input.flood, encodeFloodMyr, decodeFloodMyr),\n      arc: pairs(input.arc, encodeArcMyr, decodeArcMyr),\n      age: pairs(input.age, encodeAge, decodeAge),\n      floodAge, facies,\n      decodeFacies: input.decodeFacies.map((value) => decodeFacies(value)),\n      fields: GEO_FRAME_FIELDS, reads: GEOLOGY_READS,\n    };\n  },\n  "tectonics.buildHeights": (input) => {\n    const crust = {\n      type: Uint8Array.of(input.crustType),\n      rootKm: Float32Array.of(input.rootKm),\n      upliftKm: Float32Array.of(input.upliftKm),\n      rugged: Float32Array.of(input.rugged),\n    };\n    buildHeights(crust, 0, input.type, input.side, input.u, input.reach, input.work, input.thickened, { heights: input.heights }, input.amp);\n    return { rootKm: crust.rootKm[0], upliftKm: crust.upliftKm[0], rugged: crust.rugged[0] };\n  },
  "tectonics.broadProfile": (input) =>
    broadProfile(input.type, input.side, input.distance, input.reach, input.jitter, input.wear),
  "tectonics.beltReliefKm": (input) =>
    beltReliefKm(input.type, input.reach, input.heights),
  "tectonics.thickenCover": (input) => {
    const crust = {
      coverKm: new Float32Array(input.coverKm),
      coverMaxKm: new Float32Array(input.coverMaxKm),
      coverMaxAt: new Float32Array(input.coverMaxAt),
      unroofKm: new Float32Array(input.unroofKm),
    };
    thickenCover(crust, 0, input.thickenedKm, input.time, input.geology);
    return {
      coverKm: crust.coverKm[0],
      coverMaxKm: crust.coverMaxKm[0],
      coverMaxAt: crust.coverMaxAt[0],
      unroofKm: crust.unroofKm[0],
    };
  },
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
