// Differential harness for the three prescribed-motion scenes.
// The scene constructors and kinematics are imported from the pinned upstream submodule.
// Input is a JSON payload containing the mesh arrays and case parameters.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { meshFromArrays } from "../core/sim-mesh.mjs";

const UP = process.env.UPSTREAM ?? path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../../external/planet_heightmap_generation",
);
const url = (relative) => new URL(path.join(UP, relative), "file:///");
const upstream = (relative) => import(url(relative).href);

const [
  { buildRigidDriftWorld, setRigidDriftTime },
  { buildWilsonWorld, driveWilson },
  { buildNudgeScene },
] = await Promise.all([
  upstream("v2/js/tectonics/scenarios/rigid-drift.js"),
  upstream("v2/js/tectonics/scenarios/wilson-cycle.js"),
  upstream("v2/js/tectonics/scenarios/start-world-nudge.js"),
]);

const payload = JSON.parse(readFileSync(0, "utf8"));
const source = payload.mesh;
const mesh = meshFromArrays({
  numCells: source.numCells,
  xyz: new Float32Array(source.xyz),
  adjOffset: new Int32Array(source.adjOffset),
  adjList: new Int32Array(source.adjList),
});

const array = (value) => Array.from(value ?? []);
const crustSnapshot = (plate) => {
  const crust = plate.crust;
  return {
    mask: array(crust.mask),
    type: array(crust.type),
    flags: array(crust.flags),
    formedAt: array(crust.formedAt),
    terrane: array(crust.terrane),
  };
};

function plateSnapshot(plate) {
  const scene = plate.scene ?? {};
  const result = {
    id: plate.id,
    name: plate.name,
    isBackground: Boolean(plate.isBackground),
    omega: array(plate.omega),
    quat: array(plate.quat),
    crust: crustSnapshot(plate),
  };
  if (scene.q0) result.q0 = array(scene.q0);
  if (scene.centre) result.centre = array(scene.centre);
  if (scene.pole) result.pole = array(scene.pole);
  if (scene.def) {
    result.def = {
      name: scene.def.name,
      centre: array(scene.def.centre),
      poleOffsetDeg: scene.def.poleOffsetDeg,
      speedCmYr: scene.def.speedCmYr,
    };
  }
  if (scene.motion) {
    result.motion = {
      pole: array(scene.motion.pole),
      speedCmYr: scene.motion.speedCmYr,
    };
  }
  return result;
}

function rigidDrift() {
  const testCase = payload.cases.rigidDrift;
  const world = buildRigidDriftWorld(mesh, testCase.seed);
  const plates = world.plates.map(plateSnapshot);
  const times = testCase.timesMyr.map((time) => {
    setRigidDriftTime(world, time);
    return {
      time: world.time,
      quats: world.plates.map((plate) => array(plate.quat)),
    };
  });
  return { plates, times };
}

function wilson() {
  const testCase = payload.cases.wilson;
  const world = buildWilsonWorld(mesh, testCase.seed);
  const plates = world.plates.map(plateSnapshot);
  const times = testCase.timesMyr.map((time) => {
    world.time = time;
    driveWilson(world);
    return {
      time: world.time,
      omegas: world.plates.map((plate) => array(plate.omega)),
    };
  });
  return { plates, times };
}

function nudge() {
  const testCase = payload.cases.nudge;
  const result = buildNudgeScene(mesh, testCase.seed);
  const { world, start } = result;
  result.drive(world);
  const cratons = start.cratons.map((craton) => ({
    id: craton.id,
    letter: craton.letter,
    cluster: craton.cluster,
    centre: array(craton.centre),
    radius: craton.radius,
    reach: craton.reach,
    inner: craton.inner,
    grain: array(craton.grain),
  }));
  return {
    start: {
      continentId: start.continentId,
      centre: array(start.centre),
      reference: array(start.reference),
      passiveFrom: start.passiveFrom,
      passiveTo: start.passiveTo,
      landCells: start.landCells,
      apronCells: start.apronCells,
      cratons,
    },
    heading: array(result.heading),
    pole: array(result.pole),
    drivenOmega: array(world.plates[start.continentId].omega),
    plates: world.plates.map(plateSnapshot),
  };
}

console.log(JSON.stringify({
  rigidDrift: rigidDrift(),
  wilson: wilson(),
  nudge: nudge(),
}));
