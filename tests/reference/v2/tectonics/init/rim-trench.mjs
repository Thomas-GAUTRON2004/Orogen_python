// Upstream V2 start-world rim and trench construction, vendored from
// raguilar011095/planet_heightmap_generation at ed466f9db71292e912d8d0160e689a67a994f2d9.
import { kmToHops } from "../../core/sim-mesh.mjs";
import { hopDistance, touches } from "../../core/mesh-bfs.mjs";
import { polarFrame } from "../../core/sphere-geom.mjs";

export function plateOutlineWithApron(simMesh, isLand, cfg) {
    const { numCells } = simMesh;
    const apron = kmToHops(simMesh, cfg.apronKm);
    const smooth = kmToHops(simMesh, cfg.apronSmoothKm);
    const fromLand = hopDistance(simMesh, cell => isLand[cell] === 1, apron + smooth);
    const grown = new Uint8Array(numCells);
    for (let cell = 0; cell < numCells; cell++) {
        if (fromLand[cell] >= 0) grown[cell] = 1;
    }
    const fromOutside = hopDistance(simMesh, cell => grown[cell] === 0, smooth);
    const inPlate = new Uint8Array(numCells);
    for (let cell = 0; cell < numCells; cell++) {
        if (grown[cell] && fromOutside[cell] < 0) inPlate[cell] = 1;
    }
    const rim = [];
    for (let cell = 0; cell < numCells; cell++) {
        if (inPlate[cell] && touches(simMesh, cell, neighbour => !inPlate[neighbour])) {
            rim.push(cell);
        }
    }
    return { inPlate, rim };
}

export function chooseActiveRim(simMesh, rim, centre, reference, passive) {
    const { xyz, numCells } = simMesh;
    const toPolar = polarFrame(centre, reference);
    const active = new Uint8Array(numCells);
    for (const cell of rim) {
        const azimuth = toPolar(
            xyz[3 * cell], xyz[3 * cell + 1], xyz[3 * cell + 2],
        ).azimuth;
        const offset = ((azimuth - passive.from) % 360 + 360) % 360;
        if (offset >= passive.span) active[cell] = 1;
    }
    return {
        active,
        passiveFrom: passive.from,
        passiveTo: passive.from + passive.span,
    };
}
