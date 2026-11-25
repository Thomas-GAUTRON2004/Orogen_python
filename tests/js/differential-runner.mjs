import { readFileSync } from "node:fs";
const input = JSON.parse(readFileSync(0, "utf8"));

import { seedOf, isWordSeed } from "../../external/planet_heightmap_generation/v2/js/core/seed.js";
import { makeRng } from "../../external/planet_heightmap_generation/js/rng.js";
import { quatFromAxisAngle, quatMul, quatNormalize, quatToMat3, quatIntegrate, quatToEulerPole } from "../../external/planet_heightmap_generation/v2/js/core/rotation.js";
import { CrustRaster, CrustType, OrogenyType, CrustFlag, CELL_FIELDS } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";
import { Plate, frameRotation } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/plate.js";
import { latLonToXyz, xyzToLatLon, omegaFromEulerPole, surfaceVelocity, speedCmYrAt, advancePlates, poseAtConstantOmega } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/kinematics.js";

const seedTexts = ["", "0", "42", "-17", "3.14159", "0x10", "hello", "Orogen 3.0.1", "é"];
const seeds = seedTexts.map(text => ({text, word:isWordSeed(text), seed:seedOf(text)}));
const rng = makeRng(123456);
const random = Array.from({length:12}, () => rng());

const q1 = quatFromAxisAngle(0, 0, 1, Math.PI / 3);
const q2 = quatFromAxisAngle(1, 0, 0, -Math.PI / 7);
const q = quatNormalize(quatMul(q1, q2));
const omega = [0.001, -0.002, 0.003];
const integrated = quatIntegrate(q, omega, 12.5);
const matrix = Array.from(quatToMat3(integrated, new Float64Array(9)));
const euler = quatToEulerPole(integrated);

const crust = new CrustRaster(6);
crust.create(2, CrustType.CONTINENT, -120.25, 17);
crust.armTrench(2, 35.5);
crust.recordSubduction(2, 2.75);
crust.recordOrogeny(2, OrogenyType.ANDEAN, 8.125, 20, 15);
crust.recordOrogeny(2, OrogenyType.HIMALAYAN, 1.5, 25, 15);
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

const plate = new Plate(3, 6, 4.5);
plate.setQuat(integrated);
const world = Array.from(plate.toWorld(0.25, -0.5, 0.75, [0,0,0]));
const local = Array.from(plate.toPlateFrame(...world, [0,0,0]));
const xyz = new Float32Array([1,0,0, 0,1,0, 0,0,1, -1,0,0, 0,-1,0, 0,0,-1]);
plate.crust.create(0, CrustType.CRATON, -10, 1);
plate.crust.create(1, CrustType.OCEAN, -5, 2);
plate.refreshExtent(xyz);
const other = new Plate(4, 6);
other.setQuat(q1);
const rotation = Array.from(frameRotation(plate, other));
const cap = plate.capInWorld({edgeRad:0.02});

const branchCrust = new CrustRaster(5);
for (let i=0;i<5;i++) branchCrust.create(i, CrustType.CONTINENT, 0, i+1);
branchCrust.recordOrogeny(0, OrogenyType.INTRAPLATE, 2, 10, 5);
branchCrust.recordOrogeny(0, OrogenyType.ANDEAN, 3, 12, 5);
branchCrust.recordOrogeny(1, OrogenyType.HIMALAYAN, 4, 10, 5);
branchCrust.recordOrogeny(1, OrogenyType.ARC, 1, 11, 5);
branchCrust.recordOrogeny(2, OrogenyType.ANDEAN, 4, 10, 5);
branchCrust.recordOrogeny(2, OrogenyType.RIFT, 1, 30, 5);
branchCrust.recordOrogeny(3, OrogenyType.TRANSPRESSION, 4, 10, 5);
branchCrust.recordOrogeny(3, OrogenyType.URAL, 1, 11, 5);
branchCrust.recordOrogeny(4, OrogenyType.ANDEAN, 4, 10, 5);
branchCrust.recordOrogeny(4, OrogenyType.HIMALAYAN, 1, 11, 5);
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


const kinOmega = omegaFromEulerPole(latLonToXyz(35,-120), 4.5);
const kinVelocity = surfaceVelocity(kinOmega, ...latLonToXyz(-12,77));
const kinQ0 = quatFromAxisAngle(0,1,0,0.2);
const kinPose = poseAtConstantOmega(kinQ0, kinOmega, 3.25);
const kinWorld = {plates:[{alive:true,quat:kinQ0,omega:kinOmega,setQuat(q){this.quat=q;}},{alive:false,quat:kinQ0,omega:[1,2,3],setQuat(q){this.quat=q;}}],time:7,step:2};
advancePlates(kinWorld, 1.75);
const kinematics = {xyz:latLonToXyz(35,-120),latlon:xyzToLatLon(...latLonToXyz(-12,77)),omega:omegaFromEulerPole(latLonToXyz(35,-120),4.5),velocity:kinVelocity,speed:speedCmYrAt(kinOmega,...latLonToXyz(-12,77)),pose:kinPose,advanced:kinWorld.plates[0].quat,time:kinWorld.time,step:kinWorld.step,dead:kinWorld.plates[1].quat};


import { binIndex, markBin, mayHaveCrustNear } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/occupancy.js";
const occMesh = {numCells:4, xyz:new Float32Array([1,0,0, 0,1,0, 0,0,1, -1,0,0]), edgeKm:500};
const occBins = binIndex(occMesh);
const occPlate = {};
markBin(occPlate, occBins.binOf[0], 19, occBins.count);
const occupancy = {binOf:Array.from(occBins.binOf), counts:occBins.neighbours.map(x=>x.length), near:[0,1,2,3].map(i=>mayHaveCrustNear(occPlate,i,19,occBins)), absent:mayHaveCrustNear({},0,19,occBins), count:occBins.count};

console.log(JSON.stringify({
  extents, seeds, random, quaternion:q, integrated, matrix, euler, crust:crustOut, crustBranches, extents, kinematics, occupancy,
  plate:{world, local, matrix:Array.from(plate.mat), stats:plate.stats,
         capCenter:plate.capCenter, capCos:plate.capCos, cap, frameRotation:rotation}
}));
