import { readFileSync } from "node:fs";

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
import { fractureCoordinate } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/segments.js";
import { thickenCover } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/geology.js";
import { beltReliefKm, buildHeights } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/heights.js";
import { broadProfile } from "../../external/planet_heightmap_generation/v2/js/terrain/profiles.js";
import { CrustType } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";
import { fastestSpeed, solve3 } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/torque-balance.js";
import { chooseStep } from "../../external/planet_heightmap_generation/v2/js/tectonics/simulate.js";

const input = JSON.parse(readFileSync(0, "utf8"));

const codecs = {
  cover: input.codecs.cover.map((x) => {
    const encoded = encodeCoverKm(x);
    return [encoded, decodeCoverKm(encoded)];
  }),
  exhum: input.codecs.exhum.map((x) => {
    const encoded = encodeExhumKm(x);
    return [encoded, decodeExhumKm(encoded)];
  }),
  unroof: input.codecs.unroof.map((x) => {
    const encoded = encodeUnroofKm(x);
    return [encoded, decodeUnroofKm(encoded)];
  }),
  flood: input.codecs.flood.map((x) => {
    const encoded = encodeFloodMyr(x);
    return [encoded, decodeFloodMyr(encoded)];
  }),
  arc: input.codecs.arc.map((x) => {
    const encoded = encodeArcMyr(x);
    return [encoded, decodeArcMyr(encoded)];
  }),
  age: input.codecs.age.map((x) => {
    const encoded = encodeAge(x);
    return [encoded, decodeAge(encoded)];
  }),
  floodAge: input.codecs.floodAge.map(([last, time, ever]) => {
    const encoded = encodeFloodAge(last, time, ever);
    const decoded = decodeFloodAge(encoded);
    return [encoded, Number.isFinite(decoded) ? decoded : null];
  }),
  facies: input.codecs.facies.map((values) => {
    const encoded = encodeFacies(...values);
    return [encoded, decodeFacies(encoded)];
  }),
  decodeFacies: input.codecs.decodeFacies.map((encoded) => decodeFacies(encoded)),
  fields: GEO_FRAME_FIELDS,
  reads: GEOLOGY_READS,
};

const covers = input.covers.map((scenario) => {
  const crust = {
    coverKm: new Float32Array(scenario.coverKm),
    coverMaxKm: new Float32Array(scenario.coverMaxKm),
    coverMaxAt: new Float32Array(scenario.coverMaxAt),
    unroofKm: new Float32Array(scenario.unroofKm),
  };
  thickenCover(crust, 0, scenario.thickenedKm, scenario.time, scenario.geology);
  return {
    coverKm: crust.coverKm[0],
    coverMaxKm: crust.coverMaxKm[0],
    coverMaxAt: crust.coverMaxAt[0],
    unroofKm: crust.unroofKm[0],
  };
});

const segments = input.segments.map((x) =>
  fractureCoordinate(x.point[0], x.point[1], x.point[2], x.omega)
);

const profiles = input.profiles.map((x) =>
  broadProfile(x.type, x.side, x.distance, x.reach, x.jitter, x.wear)
);

const reliefs = input.reliefs.map((x) => beltReliefKm(x.type, x.reach, input.heights));

const heightStates = input.heightCases.map((x) => {
  const crust = {
    type: Uint8Array.of(x.crustType),
    rootKm: Float32Array.of(x.rootKm),
    upliftKm: Float32Array.of(x.upliftKm),
    rugged: Float32Array.of(x.rugged),
  };
  buildHeights(
    crust, 0, x.type, x.side, x.u, x.reach, x.work, x.thickened,
    { heights: input.heights }, x.amp
  );
  return [crust.rootKm[0], crust.upliftKm[0], crust.rugged[0]];
});

const systems = input.linearSystems.map((x) => solve3(x.matrix, x.vector));
const speeds = input.speedCases.map((x) =>
  fastestSpeed(x.omega, { capCos: x.capCos }, x.centre)
);
const world = {
  config: input.simulation.config,
  mesh: { edgeRad: input.simulation.edgeRad },
  plates: input.simulation.plates.map((x) => ({
    alive: x.alive,
    omega: x.omega,
    capCos: x.capCos,
    capCenter: x.capCenter,
    crust: { count: x.count },
    toWorld(a, b, c) { return [a, b, c]; },
  })),
};

console.log(JSON.stringify({
  codecs, covers, segments, profiles, reliefs, heightStates, systems, speeds,
  step: chooseStep(world),
}));
