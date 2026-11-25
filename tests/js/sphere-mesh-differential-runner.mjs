import { readFileSync } from "node:fs";
import Delaunator from "delaunator";

import {
  setDelaunator,
  buildSphere,
  generateFibonacciSphere,
  stereographicProjection,
} from "../../external/planet_heightmap_generation/js/sphere-mesh.js";
import { makeRng } from "../../external/planet_heightmap_generation/js/rng.js";
import {
  withinKm,
  hopDistance,
  floodFrom,
} from "../../external/planet_heightmap_generation/v2/js/core/mesh-bfs.js";
import { ringsOf } from "../../external/planet_heightmap_generation/v2/js/core/mesh-rings.js";
import { spreadKm } from "../../external/planet_heightmap_generation/v2/js/core/min-heap.js";
import { kmToHops } from "../../external/planet_heightmap_generation/v2/js/core/sim-mesh.js";
import {
  cellDistanceKm,
  centroidOfCells,
} from "../../external/planet_heightmap_generation/v2/js/core/sphere-geom.js";
import { SphereIndex } from "../../external/planet_heightmap_generation/v2/js/core/sphere-index.js";
import { Plate } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/plate.js";
import { CrustType } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/crust.js";

setDelaunator(Delaunator);
const cases = [];
for (const [n, seed] of [[12, 99], [24, 1], [48, 17], [96, 2026], [128, 7], [256, 42], [512, 12345]]) {
  const xyz0 = generateFibonacciSphere(n, 0.35, makeRng(seed + 4242));
  const projected = stereographicProjection(xyz0, n);
  const { mesh, r_xyz } = buildSphere(n, 0.35, makeRng(seed + 4242));
  const index = new SphereIndex(r_xyz, mesh, 2);
  const edgeRad = Math.sqrt(8 * Math.PI / (Math.sqrt(3) * mesh.numRegions));
  const edgeKm = edgeRad * 6371;
  const sim = {
    numCells: mesh.numRegions,
    xyz: r_xyz,
    adjOffset: mesh.adjOffset,
    adjList: mesh.adjList,
    edgeRad,
    edgeKm,
  };
  const source = (cell) => cell === 0 || cell === Math.floor(sim.numCells / 2);
  const enter = (cell) => cell % 11 !== 5;
  const dist = new Float64Array(sim.numCells).fill(Infinity);
  dist[0] = 0;
  dist[Math.floor(sim.numCells / 2)] = 0;
  spreadKm(sim, dist, { maxKm: sim.edgeKm * 5, canEnter: (a, b) => b % 17 !== 3 });
  const rings = ringsOf(sim);
  const stride = rings.stride;
  const ringSets = Array.from({ length: sim.numCells }, (_, cell) =>
    Array.from({ length: 4 }, (_, ring) =>
      Array.from(
        rings.cells.slice(
          rings.start[cell * stride + ring],
          rings.start[cell * stride + ring + 1]
        )
      ).sort((a, b) => a - b)
    )
  );
  const distances = new Float64Array(mesh.numRegions).fill(Infinity);
  distances[0] = 0;
  spreadKm(sim, distances, { maxKm: edgeKm * 3.5 });
  const within = withinKm(sim, (cell) => cell === 0, edgeKm * 2.25);
  const hops = hopDistance(sim, (cell) => cell === 0, 3);
  const plate = new Plate(0, mesh.numRegions);
  for (const cell of [0, 1, 2, 4, 7]) {
    if (cell < mesh.numRegions) {
      plate.crust.create(
        cell,
        cell === 0 ? CrustType.CRATON : CrustType.OCEAN,
        -1,
        cell + 1
      );
    }
  }
  plate.refreshExtent(r_xyz);
  const queries = [[1, 0.2, 0.3], [-0.3, 1, 0.4], [0.2, -0.4, 1], [-1, -0.3, -0.2], [0.2, 0.3, -1], [1, 1, 1]];
  const triangleKeys = [];
  for (let i = 0; i < mesh.triangles.length; i += 3) {
    triangleKeys.push(
      [mesh.triangles[i], mesh.triangles[i + 1], mesh.triangles[i + 2]]
        .sort((a, b) => a - b)
        .join(",")
    );
  }
  const neighbours = [];
  for (let cell = 0; cell < mesh.numRegions; cell++) {
    neighbours.push(
      Array.from(
        mesh.adjList.slice(mesh.adjOffset[cell], mesh.adjOffset[cell + 1])
      ).sort((a, b) => a - b)
    );
  }
  cases.push({
    n,
    seed,
    xyz: Array.from(r_xyz),
    projected: Array.from(projected),
    numRegions: mesh.numRegions,
    numTriangles: mesh.numTriangles,
    triangleKeys: triangleKeys.sort(),
    adjOffset: Array.from(mesh.adjOffset),
    neighbours,
    edgeRad,
    edgeKm,
    cellDistance: cellDistanceKm(r_xyz, 0, 2),
    centroid: centroidOfCells(r_xyz, [0, 2, 4]),
    distance: Array.from(distances, (value) => Number.isFinite(value) ? value : "Infinity"),
    within: Array.from(within),
    hops: Array.from(hops),
    plate: {
      stats: plate.stats,
      capCenter: plate.capCenter,
      capCos: plate.capCos,
    },
    indexRes: index.res,
    nearest: queries.map((point) => index.nearest(...point)),
    bfs: {
      within: Array.from(withinKm(sim, source, sim.edgeKm * 2)),
      hops: Array.from(hopDistance(sim, source, 3, enter)),
      flood: Array.from(floodFrom(sim, 0, (cell) => cell % 13 !== 7)),
    },
    ringSets,
    spread: Array.from(dist, (value) => Number.isFinite(value) ? value : "Infinity"),
    kmToHops: [0, sim.edgeKm * 0.49, sim.edgeKm * 0.5, sim.edgeKm * 1.5, sim.edgeKm * 3.1]
      .map((km) => kmToHops(sim, km)),
  });
}
console.log(JSON.stringify(cases));
