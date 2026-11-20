import { readFileSync } from "node:fs";
import { makeRng } from "./js/rng.mjs";
import { FastNoise } from "./core/fast-noise.mjs";
import { meshFromArrays } from "./core/sim-mesh.mjs";
import { placeCratons } from "./tectonics/init/cratons.mjs";
import { shapeLandmass } from "./tectonics/init/landmass.mjs";
import { plateOutlineWithApron, chooseActiveRim } from "./tectonics/init/rim-trench.mjs";
import { tangentBasis, cellDistanceKm } from "./core/sphere-geom.mjs";
import { hopDistance } from "./core/mesh-bfs.mjs";

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

// These two record stages follow the pinned upstream
// v2/js/tectonics/init/start-world.js, step/orogeny.js and step/far-field.js
// at ed466f9db71292e912d8d0160e689a67a994f2d9.
function smoothRange(a, b, value) {
    const t = Math.max(0, Math.min(1, (value - a) / (b - a)));
    return t * t * (3 - 2 * t);
}

function spreadInitialRange(crust, sources, widths, works, time, config) {
    const count = mesh.numCells;
    const u = new Float32Array(count).fill(Infinity);
    const sourceOf = new Int32Array(count).fill(-1);
    const seaKm = new Float32Array(count);
    const rimKm = new Float32Array(count);
    const queue = [];
    const rimMax = Number(config.cratonRimKm ?? 0);
    const seaLimit = Number(config.apronKm) * 1.5 + mesh.edgeKm;

    for (let i = 0; i < sources.length; i++) {
        const cell = sources[i];
        if (widths[i] <= 0 || works[i] <= 0) continue;
        if (sourceOf[cell] < 0) queue.push(cell);
        else if (widths[i] <= widths[sourceOf[cell]]) continue;
        u[cell] = 0;
        sourceOf[cell] = i;
    }

    for (let head = 0; head < queue.length; head++) {
        const cell = queue[head];
        const here = u[cell];
        const atSea = crust.type[cell] === 0;
        const reach = widths[sourceOf[cell]];
        for (let edge = mesh.adjOffset[cell]; edge < mesh.adjOffset[cell + 1]; edge++) {
            const neighbour = mesh.adjList[edge];
            if (!crust.mask[neighbour]) continue;
            if (crust.type[neighbour] === 0) {
                if (!atSea) continue;
                const sea = seaKm[cell] + cellDistanceKm(mesh.xyz, cell, neighbour);
                if (sea > seaLimit || u[neighbour] < Infinity) continue;
                u[neighbour] = 0;
                sourceOf[neighbour] = sourceOf[cell];
                seaKm[neighbour] = sea;
                queue.push(neighbour);
                continue;
            }
            const distance = cellDistanceKm(mesh.xyz, cell, neighbour);
            let hop = distance;
            if (crust.flags[neighbour] & (2 | 4 | 8)) {
                hop *= Number(config.weakCellCost);
            }
            let rim = 0;
            if (crust.type[neighbour] === 2) {
                rim = crust.type[cell] === 2 ? rimKm[cell] + distance : 0.5 * distance;
                if (rim > rimMax) continue;
                hop *= Number(config.cratonRimCost);
            }
            const next = atSea ? 0 : here + hop / reach;
            if (next >= u[neighbour] || next > 1) continue;
            u[neighbour] = next;
            sourceOf[neighbour] = sourceOf[cell];
            rimKm[neighbour] = rim;
            queue.push(neighbour);
        }
    }

    const reached = [...new Set(queue)].sort((a, b) => a - b);
    for (const cell of reached) {
        if (u[cell] === Infinity || crust.type[cell] === 0) continue;
        const rimCell = crust.type[cell] === 2;
        if (rimCell && !(rimMax > 0)) continue;
        let work = works[sourceOf[cell]];
        if (rimCell) work *= Number(config.cratonRimShare);
        const width = widths[sourceOf[cell]];
        const recordWork = u[cell] === 0
            ? work
            : work * Number(config.beltWorkShare) * (1 - u[cell]);

        const wasType = crust.oroType[cell];
        const wasStart = crust.oroStart[cell];
        const wasLast = crust.oroLast[cell];
        if (wasType === 0 || time - wasLast > Number(config.orogenyQuietMyr)) {
            crust.oroStart[cell] = time;
            crust.oroWork[cell] = 0;
            crust.oroType[cell] = 3; // OrogenyType.URAL
        } else if ((3 > wasType || wasType === 8) && ![5, 7, 8].includes(3)) {
            crust.oroType[cell] = 3;
        }
        crust.oroWork[cell] += recordWork;
        crust.oroLast[cell] = time;

        if (wasType !== 0 && crust.beltReach[cell] > 0
            && (crust.oroType[cell] !== wasType || crust.oroStart[cell] !== wasStart)) {
            crust.prevType[cell] = wasType;
            crust.prevLast[cell] = wasLast;
            crust.prevU[cell] = crust.beltU[cell];
            crust.prevReach[cell] = crust.beltReach[cell];
            crust.prevSide[cell] = crust.beltSide[cell];
            crust.prevShear[cell] = crust.beltShear[cell];
            crust.prevRugged[cell] = crust.rugged[cell];
            crust.rugged[cell] = 0;
            crust.beltReach[cell] = 0;
        }
        if (crust.oroType[cell] === 3
            && (crust.beltReach[cell] === 0 || crust.beltAt[cell] < time || u[cell] < crust.beltU[cell])) {
            crust.beltU[cell] = u[cell];
            crust.beltReach[cell] = width;
            crust.beltAt[cell] = time;
            crust.beltSide[cell] = 0;
            crust.beltShear[cell] = 0;
        }

        const thickened = rimCell ? 0
            : work * Number(config.beltWorkShare) * (1 - u[cell] ** 2)
                * Number(config.thickeningPerWorkKm);
        crust.thickKm[cell] += thickened;
        const heights = config.heights ?? {};
        // Match upstream buildHeights: the crustal root grows with the belt's thickening.
        crust.rootKm[cell] += thickened * Number(heights.rootGain?.[3] ?? 1);
        const relief = Number(heights.reliefKm?.[3] ?? 1.7)
            * (Number(heights.youngShare ?? 0.55)
                + (1 - Number(heights.youngShare ?? 0.55))
                    * Math.min(1, width / Number(heights.fullReachKm?.[3] ?? 250)));
        const amplitude = (rimCell ? Number(config.cratonRimShare) : 1) * relief;
        const share = 1 - Math.exp(-work / Number(heights.upliftBuildKm?.[3] ?? 30));
        const W = Math.max(40, width);
        const d = u[cell] * width;
        const fade = 1 - smoothRange(0.7, 1.05, d / W);
        const target = amplitude * fade * 0.6
            * Math.exp(-((d / Math.max(110, 0.35 * W)) ** 2));
        if (target >= 0 ? target > crust.upliftKm[cell] : target < crust.upliftKm[cell]) {
            crust.upliftKm[cell] += (target - crust.upliftKm[cell]) * share;
        }
        if (amplitude > crust.rugged[cell]) {
            crust.rugged[cell] += (amplitude - crust.rugged[cell]) * share;
        }
    }
}

function recordInitialFarPush(crust, cell, felt, distance, source, time, config) {
    const oldWork = crust.farWork[cell];
    const total = oldWork + felt;
    crust.farWork[cell] = total;
    crust.farDist[cell] = (crust.farDist[cell] * oldWork + distance * felt) / total;
    crust.farFold[cell] = crust.farFold[cell]
        * Math.exp(-Math.max(0, time - crust.farAt[cell]) / Number(config.farFoldMyr))
        + felt;
    crust.farAt[cell] = time;
    if (source < 0 || source === cell) return;
    const offset = 3 * cell;
    const sourceOffset = 3 * source;
    const px = mesh.xyz[offset], py = mesh.xyz[offset + 1], pz = mesh.xyz[offset + 2];
    const qx = mesh.xyz[sourceOffset], qy = mesh.xyz[sourceOffset + 1], qz = mesh.xyz[sourceOffset + 2];
    const dot = px * qx + py * qy + pz * qz;
    let ax = dot * px - qx, ay = dot * py - qy, az = dot * pz - qz;
    const length = Math.hypot(ax, ay, az);
    if (length < 1e-9) return;
    ax /= length; ay /= length; az /= length;
    let vx = crust.farDx[cell] / 127, vy = crust.farDy[cell] / 127, vz = crust.farDz[cell] / 127;
    if (vx * ax + vy * ay + vz * az < 0) { vx = -vx; vy = -vy; vz = -vz; }
    const x = vx * oldWork + ax * felt;
    const y = vy * oldWork + ay * felt;
    const z = vz * oldWork + az * felt;
    const norm = Math.hypot(x, y, z) || 1;
    crust.farDx[cell] = Math.round(127 * x / norm);
    crust.farDy[cell] = Math.round(127 * y / norm);
    crust.farDz[cell] = Math.round(127 * z / norm);
}

function spreadInitialFarPush(crust, sources, frontKm, time, config) {
    const length = Number(config.farReachKm)
        * Math.min(1.6, Math.max(0.5, Math.sqrt(frontKm / Number(config.farFrontRefKm))));
    const strongest = Math.max(...sources.map(item => item[1]));
    if (!(strongest > 0)) return;
    const distance = new Float64Array(mesh.numCells).fill(Infinity);
    const sourceOf = new Int32Array(mesh.numCells).fill(-1);
    const heap = [];
    for (const [cell, work] of sources) {
        const initial = -length * Math.log(work / strongest);
        if (initial < distance[cell]) {
            distance[cell] = initial;
            sourceOf[cell] = cell;
            heapPush(heap, [initial, cell]);
        }
    }
    const reached = [];
    while (heap.length) {
        const [current, cell] = heapPop(heap);
        if (current > distance[cell]) continue;
        reached.push(cell);
        for (let edge = mesh.adjOffset[cell]; edge < mesh.adjOffset[cell + 1]; edge++) {
            const neighbour = mesh.adjList[edge];
            if (!crust.mask[neighbour]) continue;
            let factor = 1;
            if (crust.type[neighbour] === 2) factor = Number(config.farCratonCost);
            else if (crust.flags[neighbour] & (2 | 4 | 8)) factor = Number(config.farWeakCost);
            const candidate = current + cellDistanceKm(mesh.xyz, cell, neighbour) * factor;
            if (candidate >= distance[neighbour] || candidate > Number(config.farMaxKm)) continue;
            distance[neighbour] = candidate;
            sourceOf[neighbour] = sourceOf[cell];
            heapPush(heap, [candidate, neighbour]);
        }
    }
    for (const cell of reached) {
        if (crust.type[cell] === 0) continue;
        const felt = strongest * Math.exp(-distance[cell] / length);
        if (!(felt > 1e-4)) continue;
        recordInitialFarPush(crust, cell, felt, distance[cell], sourceOf[cell], time, config);
    }
}

function buildInitialInheritedRecords(land, crust, config, seed) {
    const noise = new FastNoise((seed ^ 0x5a7e) >>> 0);
    const frequency = 6371 / Number(config.startRangeVaryKm);
    for (const [name, kind] of [["inner", 1], ["ring", 2], ["major", 3]]) {
        const [widthKm, workKm, ageMyr, keep] = config.startRanges[name];
        if (!(workKm > 0)) continue;
        const sources = [];
        const strengths = [];
        for (let cell = 0; cell < mesh.numCells; cell++) {
            if (!crust.mask[cell] || land.suture[cell] !== kind || crust.type[cell] === 0) continue;
            const x = mesh.xyz[3 * cell], y = mesh.xyz[3 * cell + 1], z = mesh.xyz[3 * cell + 2];
            const value = noise.noise3D(x * frequency + 7.1 * kind, y * frequency, z * frequency)
                + 0.4 * noise.noise3D(
                    x * 2.7 * frequency + 3.3,
                    y * 2.7 * frequency + 8.1 * kind,
                    z * 2.7 * frequency + 1.9,
                );
            const variation = Number(config.startRangeVary);
            let strength = 1 - variation * Math.max(0, Math.min(1, 0.5 + 0.6 * value));
            if (kind === 3) strength *= land.pairRange[land.pair[cell]];
            sources.push(cell);
            strengths.push(strength);
        }
        if (!sources.length) continue;
        const before = {
            root: crust.rootKm.slice(),
            uplift: crust.upliftKm.slice(),
            rugged: crust.rugged.slice(),
        };
        spreadInitialRange(
            crust,
            sources,
            strengths.map(strength => widthKm * (0.7 + 0.3 * strength)),
            strengths.map(strength => workKm * strength),
            -ageMyr,
            config,
        );
        const nearest = new Float64Array(mesh.numCells).fill(Infinity);
        const nearestStrength = new Float64Array(mesh.numCells).fill(1);
        const heap = [];
        for (let i = 0; i < sources.length; i++) {
            nearest[sources[i]] = 0;
            nearestStrength[sources[i]] = strengths[i];
            heapPush(heap, [0, sources[i]]);
        }
        while (heap.length) {
            const [current, cell] = heapPop(heap);
            if (current > nearest[cell] + 1e-9 || current > 2 * widthKm) continue;
            for (let edge = mesh.adjOffset[cell]; edge < mesh.adjOffset[cell + 1]; edge++) {
                const neighbour = mesh.adjList[edge];
                if (!crust.mask[neighbour]) continue;
                const candidate = current + cellDistanceKm(mesh.xyz, cell, neighbour);
                if (candidate >= nearest[neighbour] || candidate > 2 * widthKm) continue;
                nearest[neighbour] = candidate;
                nearestStrength[neighbour] = nearestStrength[cell];
                heapPush(heap, [candidate, neighbour]);
            }
        }
        for (let cell = 0; cell < mesh.numCells; cell++) {
            const factor = keep * (Number.isFinite(nearest[cell]) ? nearestStrength[cell] : 1);
            crust.rootKm[cell] = before.root[cell] + (crust.rootKm[cell] - before.root[cell]) * factor;
            crust.upliftKm[cell] = before.uplift[cell] + (crust.upliftKm[cell] - before.uplift[cell]) * factor;
            crust.rugged[cell] = before.rugged[cell] + (crust.rugged[cell] - before.rugged[cell]) * factor;
        }
    }

    if (!(Number(config.farStartWorkKm) > 0)) return;
    const pairs = new Map();
    for (let cell = 0; cell < mesh.numCells; cell++) {
        if (land.suture[cell] !== 3 || land.type[cell] === 0) continue;
        const pair = land.pair[cell];
        if (!pairs.has(pair)) pairs.set(pair, []);
        pairs.get(pair).push(cell);
    }
    for (const pair of [...pairs.keys()].sort((a, b) => a - b)) {
        const cells = pairs.get(pair);
        const work = Number(config.farStartWorkKm) * land.pairRange[pair];
        spreadInitialFarPush(
            crust,
            cells.map(cell => [cell, work]),
            cells.length * mesh.edgeKm,
            -Number(config.startRanges.major[2]),
            config,
        );
    }
}

function initialRecordArrays(crust) {
    return Object.fromEntries([
        "oroType", "oroWork", "oroStart", "oroLast",
        "beltU", "beltReach", "beltAt", "beltSide", "beltShear",
        "prevType", "prevLast", "prevU", "prevReach", "prevSide", "prevShear",
        "prevRugged", "thickKm", "rootKm", "upliftKm", "rugged",
        "farWork", "farDist", "farFold", "farAt", "farDx", "farDy", "farDz",
    ].map(name => [name, Array.from(crust[name])]));
}

function assembleWorldSummary(layout, land) {
    const isLand = Uint8Array.from(land.type, type => type === 0 ? 0 : 1);
    const { inPlate, rim } = plateOutlineWithApron(mesh, isLand, config);
    const apronHops = Math.max(1, Math.round(config.apronKm / mesh.edgeKm));
    const smoothHops = Math.max(1, Math.round(config.apronSmoothKm / mesh.edgeKm));
    const fromLand = hopDistance(mesh, cell => isLand[cell] === 1, apronHops + smoothHops);
    const grown = Uint8Array.from(fromLand, distance => distance >= 0 ? 1 : 0);
    const fromOutside = hopDistance(mesh, cell => grown[cell] === 0, smoothHops);
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
    const crust = {
        mask: Uint8Array.from(inPlate),
        type,
        flags,
        oroType: new Uint8Array(numCells),
        oroWork: new Float32Array(numCells),
        oroStart: new Float32Array(numCells),
        oroLast: new Float32Array(numCells),
        beltU: new Float32Array(numCells),
        beltReach: new Float32Array(numCells),
        beltAt: new Float32Array(numCells),
        beltSide: new Uint8Array(numCells),
        beltShear: new Int8Array(numCells),
        prevType: new Uint8Array(numCells),
        prevLast: new Float32Array(numCells),
        prevU: new Float32Array(numCells),
        prevReach: new Float32Array(numCells),
        prevSide: new Uint8Array(numCells),
        prevShear: new Int8Array(numCells),
        prevRugged: new Float32Array(numCells),
        thickKm: new Float32Array(numCells),
        rootKm: new Float32Array(numCells),
        upliftKm: new Float32Array(numCells),
        rugged: new Float32Array(numCells),
        farWork: new Float32Array(numCells),
        farDist: new Float32Array(numCells),
        farFold: new Float32Array(numCells),
        farAt: new Float32Array(numCells),
        farDx: new Int8Array(numCells),
        farDy: new Int8Array(numCells),
        farDz: new Int8Array(numCells),
    };
    buildInitialInheritedRecords(land, crust, config, seed);
    const initialRecords = initialRecordArrays(crust);
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
        apronDiagnostics: {
            edgeKm: mesh.edgeKm,
            apronHops,
            smoothHops,
            landCells: Array.from(isLand).reduce((sum, value) => sum + value, 0),
            grownCells: Array.from(grown).reduce((sum, value) => sum + value, 0),
            inPlateCells: Array.from(inPlate).reduce((sum, value) => sum + value, 0),
            outsideReached: Array.from(fromOutside).filter(distance => distance >= 0).length,
        },
        startSeamKm: {
            major: majorSeams.distance,
            majorRange: majorSeams.strength,
            ring: ringSeams.distance,
        },
        initialRecords,
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
