// Thin driver for the upstream start world (planet_heightmap_generation @ ed466f9).
//
// It calls the upstream buildStartWorld and prints what the Python port is compared
// with. No upstream step is reimplemented here: the layout search, landmass, rim,
// trench, inherited ranges and far-field pushes all come from init/start-world.js and
// the modules it imports, through the pointers in this directory.
import { readFileSync } from "node:fs";
import { meshFromArrays } from "./core/sim-mesh.mjs";
import { buildStartWorld } from "./tectonics/init/start-world.mjs";
import { CrustFlag } from "./tectonics/crust.mjs";

const payload = JSON.parse(readFileSync(0, "utf8"));
const source = payload.mesh;
const mesh = meshFromArrays({
    numCells: source.numCells,
    xyz: new Float32Array(source.xyz),
    adjOffset: new Int32Array(source.adjOffset),
    adjList: new Int32Array(source.adjList),
});

// The config payload holds the keys the Python test compares; upstream merges them
// over its own TECTONICS defaults, exactly as `new TectonicWorld(mesh, seed, overrides)`.
const { world, start } = buildStartWorld(mesh, payload.seed, payload.config);
const [continent, ocean] = world.plates;
const crust = continent.crust;
const cells = mesh.numCells;
const list = (array) => Array.from(array);

const trench = [];
for (let cell = 0; cell < cells; cell++) if (crust.flags[cell] & CrustFlag.TRENCH) trench.push(cell);

const initialRecords = {};
for (const field of [
    "oroType", "oroWork", "oroStart", "oroLast", "beltU", "beltReach", "beltAt", "beltSide",
    "beltShear", "prevType", "prevLast", "prevU", "prevReach", "prevSide", "prevShear",
    "prevRugged", "thickKm", "rootKm", "upliftKm", "rugged", "farWork", "farDist", "farFold",
    "farAt", "farDx", "farDy", "farDz",
]) initialRecords[field] = list(crust[field]);

console.log(JSON.stringify({
    attempts: start.attempts,
    cratonAreaScale: start.cratonAreaScale,
    fractionMet: start.fractionMet,
    elongation: start.elongation,
    centre: start.centre,
    rimCentre: start.rimCentre,
    reference: start.reference,
    skirtKm: start.skirtKm,
    cratons: start.cratons,
    seams: list(start.seams),
    owner: list(start.owner),
    rim: start.rim,
    passiveFrom: start.passiveFrom,
    passiveTo: start.passiveTo,
    continentName: continent.name,
    born: continent.born,
    continentMask: list(crust.mask),
    oceanMask: list(ocean.crust.mask),
    oceanFormedAt: list(ocean.crust.formedAt),
    type: list(crust.type),
    flags: list(crust.flags),
    formedAt: list(crust.formedAt),
    terrane: list(crust.terrane),
    slabKm: list(crust.slabKm),
    trench,
    startSeamKm: { major: list(world.startSeamKm.major), majorRange: list(world.startSeamKm.majorRange), ring: list(world.startSeamKm.ring) },
    initialRecords,
}));
