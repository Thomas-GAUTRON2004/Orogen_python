import { seedOf, isWordSeed } from "./core/seed.mjs";
import { makeRng } from "./js/rng.mjs";
import { quatFromAxisAngle, quatMul, quatNormalize, quatToMat3, quatIntegrate, quatToEulerPole } from "./core/rotation.mjs";
import { CrustRaster, CrustType, OrogenyType, CrustFlag, CELL_FIELDS } from "./tectonics/crust.mjs";
import { Plate, frameRotation } from "./tectonics/plate.mjs";

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
console.log(JSON.stringify({
  seeds, random, quaternion:q, integrated, matrix, euler, crust:crustOut,
  plate:{world, local, matrix:Array.from(plate.mat), stats:plate.stats,
         capCenter:plate.capCenter, capCos:plate.capCos, cap, frameRotation:rotation}
}));
