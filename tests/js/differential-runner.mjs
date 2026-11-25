import { readFileSync } from "node:fs";
const input = JSON.parse(readFileSync(0, "utf8"));

import { seedOf, isWordSeed } from "../../external/planet_heightmap_generation/v2/js/core/seed.js";
import { makeRng } from "../../external/planet_heightmap_generation/js/rng.js";
import { quatFromAxisAngle, quatMul, quatNormalize, quatToMat3, quatIntegrate, quatToEulerPole } from "../../external/planet_heightmap_generation/v2/js/core/rotation.js";
import { CrustRaster, CrustType, OrogenyType, CrustFlag, CELL_FIELDS } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";
import { Plate, frameRotation } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/plate.js";
import { latLonToXyz, xyzToLatLon, omegaFromEulerPole, surfaceVelocity, speedCmYrAt, advancePlates, poseAtConstantOmega } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/kinematics.js";

const seedTexts = input.seedTexts;
const seeds = seedTexts.map(text => ({text, word:isWordSeed(text), seed:seedOf(text)}));
const rng = makeRng(input.random.seed);
const random = Array.from({length:input.random.count}, () => rng());

const q1 = quatFromAxisAngle(...input.quaternion.q1Axis, input.quaternion.q1Angle);
const q2 = quatFromAxisAngle(...input.quaternion.q2Axis, input.quaternion.q2Angle);
const q = quatNormalize(quatMul(q1, q2));
const omega = input.quaternion.omega;
const integrated = quatIntegrate(q, omega, input.quaternion.dt);
const matrix = Array.from(quatToMat3(integrated, new Float64Array(9)));
const euler = quatToEulerPole(integrated);

const crustInput = input.crust;
const crust = new CrustRaster(crustInput.size);
crust.create(crustInput.cell, crustInput.type, crustInput.longitude, crustInput.terrane);
crust.armTrench(crustInput.cell, crustInput.trenchDepth);
crust.recordSubduction(crustInput.cell, crustInput.subduction);
for (const [type, work, start, last] of crustInput.orogeny) {
  crust.recordOrogeny(crustInput.cell, type, work, start, last);
}
const snap = crust.snapshot(2);
const crustOut = {
  fields: CELL_FIELDS,
  count: crust.count,
  mask: Array.from(crust.mask),
  snapshot: snap,
  flags: crust.flags[2],
  slabKm: crust.slabKm[2],
  oroType: crust.oroType[2],
  oroWork: crust.oroWork[2],
  oroStart: crust.oroStart[2],
  oroLast: crust.oroLast[2],
  flagTrench: CrustFlag.TRENCH,
};

const plateInput = input.plate;
const plate = new Plate(plateInput.id, plateInput.numCells, plateInput.radius);
plate.setQuat(integrated);
const world = Array.from(plate.toWorld(...plateInput.worldPoint, [0,0,0]));
const local = Array.from(plate.toPlateFrame(...world, [0,0,0]));
const xyz = new Float32Array(plateInput.xyz);
for (const [cell, type, longitude, terrane] of plateInput.crustCells) {
  plate.crust.create(cell, type, longitude, terrane);
}
plate.refreshExtent(xyz);
const other = new Plate(plateInput.otherId, plateInput.otherNumCells);
other.setQuat(q1);
const rotation = Array.from(frameRotation(plate, other));
const cap = plate.capInWorld({edgeRad:plateInput.capEdgeRad});

const branchInput = input.crustBranches;
const branchCrust = new CrustRaster(branchInput.size);
for (let i=0;i<branchInput.size;i++) branchCrust.create(i, CrustType.CONTINENT, 0, i+1);
for (const [cell, type, work, start, last] of branchInput.operations) {
  branchCrust.recordOrogeny(cell, type, work, start, last);
}
const crustBranches = Array.from({length:5},(_,i)=>branchCrust.snapshot(i));

function extentCase({occupied, xyz:coords, types, numCells}) {
  const p = new Plate(8, numCells, 2);
  p.setQuat(q1);
  for (const i of occupied) p.crust.create(i, types[i], -1, i+1);
  p.dirty = true;
  p.refreshExtent(new Float32Array(coords));
  return {stats:p.stats,count:p.crust.count,dirty:p.dirty,capCenter:p.capCenter,capCos:p.capCos,cap:p.capInWorld({edgeRad:.015})};
}
const extents = Object.fromEntries(input.extentCases.map(({name, ...scenario}) => [name, extentCase(scenario)]));


const kinInput = input.kinematics;
const kinOrigin = latLonToXyz(...kinInput.origin);
const kinVelocityPoint = latLonToXyz(...kinInput.velocityPoint);
const kinOmega = omegaFromEulerPole(kinOrigin, kinInput.eulerSpeed);
const kinVelocity = surfaceVelocity(kinOmega, ...kinVelocityPoint);
const kinQ0 = quatFromAxisAngle(...kinInput.poseAxis, kinInput.poseAngle);
const kinPose = poseAtConstantOmega(kinQ0, kinOmega, kinInput.poseDt);
const kinWorld = {plates:[{alive:true,quat:kinQ0,omega:kinOmega,setQuat(q){this.quat=q;}},{alive:false,quat:kinQ0,omega:[1,2,3],setQuat(q){this.quat=q;}}],time:kinInput.worldTime,step:kinInput.worldStep};
advancePlates(kinWorld, kinInput.advanceDt);
const kinematics = {xyz:kinOrigin,latlon:xyzToLatLon(...kinVelocityPoint),omega:kinOmega,velocity:kinVelocity,speed:speedCmYrAt(kinOmega,...kinVelocityPoint),pose:kinPose,advanced:kinWorld.plates[0].quat,time:kinWorld.time,step:kinWorld.step,dead:kinWorld.plates[1].quat};


import { binIndex, markBin, mayHaveCrustNear } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/occupancy.js";
const occInput = input.occupancy;
const occMesh = {numCells:occInput.numCells, xyz:new Float32Array(occInput.xyz), edgeKm:occInput.edgeKm};
const occBins = binIndex(occMesh);
const occPlate = {};
markBin(occPlate, occBins.binOf[0], occInput.terrane, occBins.count);
const occupancy = {binOf:Array.from(occBins.binOf), counts:occBins.neighbours.map(x=>x.length), near:Array.from({length:occInput.numCells},(_,i)=>mayHaveCrustNear(occPlate,i,occInput.terrane,occBins)), absent:mayHaveCrustNear({},0,19,occBins), count:occBins.count};

console.log(JSON.stringify({
  extents, seeds, random, quaternion:q, integrated, matrix, euler, crust:crustOut, crustBranches, extents, kinematics, occupancy,
  plate:{world, local, matrix:Array.from(plate.mat), stats:plate.stats,
         capCenter:plate.capCenter, capCos:plate.capCos, cap, frameRotation:rotation}
}));
