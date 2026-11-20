import { readFileSync } from "node:fs";
import { makeRng } from "./js/rng.mjs";
import { FastNoise } from "./core/fast-noise.mjs";
import { meshFromArrays } from "./core/sim-mesh.mjs";
import { placeCratons } from "./tectonics/init/cratons.mjs";
import { shapeLandmass } from "./tectonics/init/landmass.mjs";
import { plateOutlineWithApron, chooseActiveRim } from "./tectonics/init/rim-trench.mjs";
import { tangentBasis, cellDistanceKm } from "./core/sphere-geom.mjs";

const payload = JSON.parse(readFileSync(0, "utf8"));
const { seed, config } = payload;
const source = payload.mesh;
const mesh = meshFromArrays({
    numCells: source.numCells,
    xyz: new Float32Array(source.xyz),
    adjOffset: new Int32Array(source.adjOffset),
    adjList: new Int32Array(source.adjList),
});

function shrinkSteps(floor) {
    const lowest = Math.max(0.05, Math.min(1.0, Number(floor)));
    const steps = [];
    let value = 1.0;
    while (value > lowest) {
        steps.push(value);
        value *= 0.85;
    }
    steps.push(lowest);
    return steps;
}

function centroid(type) {
    const total = [0, 0, 0];
    let count = 0;
    for (let cell = 0; cell < mesh.numCells; cell++) {
        if (type[cell] === 0) continue;
        total[0] += mesh.xyz[3 * cell];
        total[1] += mesh.xyz[3 * cell + 1];
        total[2] += mesh.xyz[3 * cell + 2];
        count++;
    }
    if (!count) return [0, 0, 0];
    const length = Math.hypot(...total) || 1;
    return total.map(value => value / length);
}

function halfSpan(type, centre) {
    let farthest = 1.0;
    let found = false;
    for (let cell = 0; cell < mesh.numCells; cell++) {
        if (type[cell] === 0) continue;
        found = true;
        const dot = mesh.xyz[3 * cell] * centre[0]
            + mesh.xyz[3 * cell + 1] * centre[1]
            + mesh.xyz[3 * cell + 2] * centre[2];
        farthest = Math.min(farthest, Math.max(-1.0, Math.min(1.0, dot)));
    }
    return found ? Math.acos(farthest) * 180 / Math.PI : 0.0;
}

function summarise(candidate) {
    if (!candidate) return null;
    const layout = candidate.layout;
    const land = candidate.land;
    return {
        areaScale: candidate.areaScale,
        excess: candidate.excess,
        attempts: candidate.attempts,
        middle: candidate.middle,
        fractionMet: candidate.fractionMet,
        layout: {
            elongation: layout.elongation,
            clusters: layout.clusters,
            cratons: layout.cratons.map(item => ({
                id: item.id,
                letter: item.letter,
                cluster: item.cluster,
                centre: item.centre,
                radius: item.radius,
                reach: item.reach,
                inner: item.inner,
                grain: item.grain,
            })),
        },
        land: {
            type: Array.from(land.type),
            terrane: Array.from(land.terrane),
            suture: Array.from(land.suture),
            owner: Array.from(land.owner),
            pair: Array.from(land.pair),
            pairRange: Array.from(land.pairRange),
            skirtKm: land.skirtKm,
            rimCentre: land.rimCentre,
            passive: land.passive,
        },
        world: assembleWorldSummary(layout, land),
    };
}

function compareHeapItems(left, right) {
    return left[0] < right[0] || (left[0] === right[0] && left[1] < right[1]);
}

function heapPush(heap, item) {
    let index = heap.length;
    heap.push(item);
    while (index > 0) {
        const parent = Math.floor((index - 1) / 2);
        if (!compareHeapItems(item, heap[parent])) break;
        heap[index] = heap[parent];
        index = parent;
    }
    heap[index] = item;
}

function heapPop(heap) {
    const first = heap[0];
    const last = heap.pop();
    if (heap.length && last) {
        let index = 0;
        while (true) {
            const left = index * 2 + 1;
            if (left >= heap.length) break;
            const right = left + 1;
            const child = right < heap.length && compareHeapItems(heap[right], heap[left])
                ? right : left;
            if (!compareHeapItems(heap[child], last)) break;
            heap[index] = heap[child];
            index = child;
        }
        heap[index] = last;
    }
    return first;
}

function seamDistanceSummary(land, seamKind, keepStrength) {
    const distance = new Float64Array(mesh.numCells).fill(Infinity);
    const strength = new Float32Array(mesh.numCells);
    const heap = [];
    for (let cell = 0; cell < mesh.numCells; cell++) {
        if (land.suture[cell] !== seamKind) continue;
        distance[cell] = 0;
        if (keepStrength) strength[cell] = land.pairRange[land.pair[cell]];
        heapPush(heap, [0, cell]);
    }
    while (heap.length) {
        const [current, cell] = heapPop(heap);
        if (current > distance[cell] + 1e-9 || current > 3000) continue;
        for (let edge = mesh.adjOffset[cell]; edge < mesh.adjOffset[cell + 1]; edge++) {
            const neighbour = mesh.adjList[edge];
            if (land.type[neighbour] === 0) continue;
            const candidate = current + cellDistanceKm(mesh.xyz, cell, neighbour);
            if (candidate >= distance[neighbour] || candidate > 3000) continue;
            distance[neighbour] = candidate;
            strength[neighbour] = strength[cell];
            heapPush(heap, [candidate, neighbour]);
        }
    }
    return {
        distance: Array.from(Float32Array.from(distance)),
        strength: Array.from(strength),
    };
}

function assembleWorldSummary(layout, land) {
    const isLand = Uint8Array.from(land.type, type => type === 0 ? 0 : 1);
    const { inPlate, rim } = plateOutlineWithApron(mesh, isLand, config);
    const reference = tangentBasis(land.rimCentre)[0];
    const trench = chooseActiveRim(mesh, rim, land.rimCentre, reference, land.passive);
    const numCells = mesh.numCells;
    const type = new Uint8Array(numCells);
    const flags = new Uint8Array(numCells);
    const formedAt = new Float32Array(numCells);
    const terrane = new Uint16Array(numCells);
    const slabKm = new Float32Array(numCells);
    const continentMask = Array.from(inPlate);
    const oceanMask = Array.from(inPlate, value => value ? 0 : 1);
    const oceanFormedAt = Array.from(oceanMask, value => value ? -Number(config.initialOceanAgeMyr) : 0);
    for (let cell = 0; cell < numCells; cell++) {
        if (!inPlate[cell]) continue;
        const landType = land.type[cell];
        type[cell] = landType;
        if (landType === 0) {
            flags[cell] |= 32; // CrustFlag.APRON
            formedAt[cell] = -Number(config.initialOceanAgeMyr);
            if (trench.active[cell]) {
                flags[cell] |= 1; // CrustFlag.TRENCH
                slabKm[cell] = Number(config.slabMatureKm);
            }
        } else {
            formedAt[cell] = -Number(config.initialLandAgeMyr);
            terrane[cell] = land.terrane[cell];
            if (land.suture[cell] === 1 || land.suture[cell] === 3) {
                flags[cell] |= 2; // CrustFlag.SUTURE
            }
        }
    }
    const majorSeams = seamDistanceSummary(land, 3, true);
    const ringSeams = seamDistanceSummary(land, 2, false);
    return {
        continentName: layout.cratons.map(craton => craton.letter).join(""),
        continentMask,
        oceanMask,
        type: Array.from(type),
        flags: Array.from(flags),
        formedAt: Array.from(formedAt),
        terrane: Array.from(terrane),
        slabKm: Array.from(slabKm),
        oceanFormedAt,
        born: -Number(config.riftTenureMyr),
        rim,
        trench: Array.from(trench.active, (value, cell) => value && land.type[cell] === 0 ? cell : -1).filter(cell => cell >= 0),
        passiveFrom: trench.passiveFrom,
        passiveTo: trench.passiveTo,
        reference,
        startSeamKm: {
            major: majorSeams.distance,
            majorRange: majorSeams.strength,
            ring: ringSeams.distance,
        },
    };
}

const maxAttempts = 24;
const relaxSpanAfter = 18;
const fractionTolerance = 0.004;
let closest = null;
let chosen = null;
for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const stretch = 1.0 - attempt / maxAttempts;
    const tried = [];
    let fit = null;
    for (const areaScale of shrinkSteps(config.cratonMinAreaScale)) {
        const layoutRng = makeRng(seed * 31 + attempt * 7919 + 17);
        const layout = placeCratons(layoutRng, config, stretch, areaScale);
        if (!layout) continue;
        const passiveRng = makeRng(seed * 31 + attempt * 7919 + 29);
        const passive = {
            from: passiveRng() * 360 - 180,
            span: 360 * (1 - Math.min(1, Math.max(0, config.rimTrenchCoverage))),
        };
        const noise = new FastNoise(seed + attempt * 101);
        const shapeRng = makeRng(seed * 31 + attempt * 7919 + 41);
        const land = shapeLandmass(mesh, layout, noise, shapeRng, config, stretch, passive);
        if (!land) continue;
        let landCount = 0;
        for (let cell = 0; cell < land.type.length; cell++) {
            if (land.type[cell] !== 0) landCount++;
        }
        const excess = landCount / mesh.numCells - config.continentalFraction;
        const candidate = { layout, land, areaScale, excess, attempts: attempt + 1 };
        tried.push(candidate);
        if (Math.abs(excess) <= fractionTolerance) {
            fit = candidate;
            break;
        }
    }
    for (const candidate of tried) {
        if (!closest || Math.abs(candidate.excess) < Math.abs(closest.excess)) {
            closest = candidate;
        }
    }
    chosen = fit;
    if (!chosen && attempt === maxAttempts - 1) chosen = closest;
    if (!chosen) continue;
    const middle = centroid(chosen.land.type);
    const tooWide = halfSpan(chosen.land.type, middle)
        > Math.acos(1 - 2 * config.continentalFraction) * 180 / Math.PI
            + config.spanAllowanceDeg;
    if (fit && tooWide && attempt < relaxSpanAfter) continue;
    chosen = { ...chosen, middle, fractionMet: Boolean(fit) };
    break;
}
console.log(JSON.stringify(summarise(chosen)));
