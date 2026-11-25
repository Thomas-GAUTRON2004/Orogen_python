import { readFileSync } from "node:fs";
import { CrustRaster, CrustFlag, CELL_FIELDS } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";
import { Plate, frameRotation } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/plate.js";

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
import { broadProfile } from "../../external/planet_heightmap_generation/v2/js/terrain/profiles.js";
import { chooseStep } from "../../external/planet_heightmap_generation/v2/js/tectonics/simulate.js";
import { binIndex, markBin, mayHaveCrustNear } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/occupancy.js";
import {
  latLonToXyz, xyzToLatLon, omegaFromEulerPole, surfaceVelocity,
  speedCmYrAt, advancePlates, poseAtConstantOmega,
} from "../../external/planet_heightmap_generation/v2/js/tectonics/step/kinematics.js";

const request = JSON.parse(readFileSync(0, "utf8"));

const functions = {
  "tectonics.plateFrameTransform": (input) => {
    const q1 = quatFromAxisAngle(...input.q1Axis, input.q1Angle);
    const q2 = quatFromAxisAngle(...input.q2Axis, input.q2Angle);
    const integrated = quatIntegrate(quatNormalize(quatMul(q1, q2)), input.omega, input.dt);
    const p = input.plate;
    const plate = new Plate(p.id, p.numCells, p.radius);
    plate.setQuat(integrated);
    const world = Array.from(plate.toWorld(...p.worldPoint, [0, 0, 0]));
    const local = Array.from(plate.toPlateFrame(...world, [0, 0, 0]));
    for (const [cell, type, longitude, terrane] of p.crustCells) plate.crust.create(cell, type, longitude, terrane);
    plate.refreshExtent(new Float32Array(p.xyz));
    const other = new Plate(p.otherId, p.otherNumCells);
    other.setQuat(q1);
    const rotation = frameRotation(plate, other);
    return { world, local, matrix: Array.from(plate.mat), stats: plate.stats, capCenter: plate.capCenter, capCos: plate.capCos, cap: plate.capInWorld({ edgeRad: p.capEdgeRad }), frameRotation: Array.from(rotation) };
  },
  "tectonics.plateExtents": (input) => {
    const q = quatFromAxisAngle(...input.qAxis, input.qAngle);
    const results = {};
    for (const scenario of input.cases) {
      const plate = new Plate(input.id, scenario.numCells, input.radius);
      plate.setQuat(q);
      for (const cell of scenario.occupied) plate.crust.create(cell, scenario.types[cell], scenario.longitude, scenario.terraneStart + cell);
      plate.dirty = true;
      plate.refreshExtent(new Float32Array(scenario.xyz));
      results[scenario.name] = { stats: plate.stats, count: plate.crust.count, dirty: plate.dirty, capCenter: plate.capCenter, capCos: plate.capCos, cap: plate.capInWorld({ edgeRad: input.edgeRad }) };
    }
    return results;
  },
  "tectonics.crustSnapshot": (input) => {
    const crust = new CrustRaster(input.size);
    crust.create(input.cell, input.type, input.longitude, input.terrane);
    crust.armTrench(input.cell, input.trenchDepth);
    crust.recordSubduction(input.cell, input.subduction);
    for (const [type, work, start, last] of input.orogeny) crust.recordOrogeny(input.cell, type, work, start, last);
    return {
      fields: CELL_FIELDS, count: crust.count, mask: Array.from(crust.mask),
      snapshot: crust.snapshot(input.cell), flags: crust.flags[input.cell],
      slabKm: crust.slabKm[input.cell], oroType: crust.oroType[input.cell],
      oroWork: crust.oroWork[input.cell], oroStart: crust.oroStart[input.cell],
      oroLast: crust.oroLast[input.cell], flagTrench: CrustFlag.TRENCH,
    };
  },
  "tectonics.crustBranches": (input) => {
    const crust = new CrustRaster(input.size);
    for (let i = 0; i < input.size; i++) crust.create(i, input.baseType, input.longitude, input.terraneStart + i);
    for (const [cell, type, work, start, last] of input.operations) crust.recordOrogeny(cell, type, work, start, last);
    return Array.from({ length: input.size }, (_, i) => crust.snapshot(i));
  },
  "tectonics.plateKinematics": (input) => {
    const origin = latLonToXyz(...input.origin);
    const velocityPoint = latLonToXyz(...input.velocityPoint);
    const omega = omegaFromEulerPole(origin, input.eulerSpeed);
    const velocity = surfaceVelocity(omega, ...velocityPoint);
    const q0 = quatFromAxisAngle(...input.poseAxis, input.poseAngle);
    const pose = poseAtConstantOmega(q0, omega, input.poseDt);
    const world = {
      plates: [
        { alive: true, quat: q0, omega, setQuat(q) { this.quat = q; } },
        { alive: false, quat: q0, omega: [1, 2, 3], setQuat(q) { this.quat = q; } },
      ],
      time: input.worldTime, step: input.worldStep,
    };
    advancePlates(world, input.advanceDt);
    return {
      xyz: origin, latlon: xyzToLatLon(...velocityPoint), omega, velocity,
      speed: speedCmYrAt(omega, ...velocityPoint), pose,
      advanced: world.plates[0].quat, time: world.time, step: world.step,
      dead: world.plates[1].quat,
    };
  },
  "tectonics.occupancyBins": (input) => {
    const mesh = { numCells: input.numCells, xyz: new Float32Array(input.xyz), edgeKm: input.edgeKm };
    const bins = binIndex(mesh);
    const plate = {};
    markBin(plate, bins.binOf[0], input.terrane, bins.count);
    return {
      count: bins.count,
      binOf: Array.from(bins.binOf),
      counts: bins.neighbours.map((values) => values.length),
      near: Array.from({ length: input.numCells }, (_, index) => mayHaveCrustNear(plate, index, input.terrane, bins)),
      absent: mayHaveCrustNear({}, 0, input.terrane, bins),
    };
  },
  "tectonics.chooseStep": (input) => {
    const world = {
      config: input.config,
      mesh: { edgeRad: input.edgeRad },
      plates: input.plates.map((plate) => ({
        alive: plate.alive, omega: plate.omega, capCos: plate.capCos,
        capCenter: plate.capCenter, crust: { count: plate.count },
        toWorld(a, b, c) { return [a, b, c]; },
      })),
    };
    return chooseStep(world);
  },
  "tectonics.geologyCodecs": (input) => {
    const pairs = (values, encode, decode) => values.map((value) => { const encoded = encode(value); return [encoded, decode(encoded)]; });
    const floodAge = input.floodAge.map(([last, time, ever]) => { const encoded = encodeFloodAge(last, time, ever); const decoded = decodeFloodAge(encoded); return [encoded, Number.isFinite(decoded) ? decoded : null]; });
    const facies = input.facies.map((values) => { const encoded = encodeFacies(...values); return [encoded, decodeFacies(encoded)]; });
    return {
      cover: pairs(input.cover, encodeCoverKm, decodeCoverKm),
      exhum: pairs(input.exhum, encodeExhumKm, decodeExhumKm),
      unroof: pairs(input.unroof, encodeUnroofKm, decodeUnroofKm),
      flood: pairs(input.flood, encodeFloodMyr, decodeFloodMyr),
      arc: pairs(input.arc, encodeArcMyr, decodeArcMyr),
      age: pairs(input.age, encodeAge, decodeAge),
      floodAge, facies,
      decodeFacies: input.decodeFacies.map((value) => decodeFacies(value)),
      fields: GEO_FRAME_FIELDS, reads: GEOLOGY_READS,
    };
  },
  "tectonics.buildHeights": (input) => {
    const crust = {
      type: Uint8Array.of(input.crustType),
      rootKm: Float32Array.of(input.rootKm),
      upliftKm: Float32Array.of(input.upliftKm),
      rugged: Float32Array.of(input.rugged),
    };
    buildHeights(crust, 0, input.type, input.side, input.u, input.reach, input.work, input.thickened, { heights: input.heights }, input.amp);
    return { rootKm: crust.rootKm[0], upliftKm: crust.upliftKm[0], rugged: crust.rugged[0] };
  },
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
