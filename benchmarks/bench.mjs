import os from "node:os";
import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { summarize } from "./bench-utils.mjs";
import { FastNoise } from "../tests/reference/v2/core/fast-noise.mjs";
import * as geom from "../tests/reference/v2/core/sphere-geom.mjs";
import { hopDistance, piecesOf, largestPieceOf, withinKm } from "../tests/reference/v2/core/mesh-bfs.mjs";
import { spreadKm } from "../tests/reference/v2/core/min-heap.mjs";
import { CrustRaster, CrustType } from "../tests/reference/v2/tectonics/crust.mjs";

const N_NOISE = 40_000;
const N_GEOM = 80_000;
const N_CELLS = 40_000;
const ROUNDS = Math.max(3, Number(process.env.BENCH_ROUNDS ?? 10));
const WARMUPS = Math.max(1, Number(process.env.BENCH_WARMUPS ?? 2));

function measure(fn) {
  for (let i = 0; i < WARMUPS; i++) fn();
  const samples = [];
  for (let i = 0; i < ROUNDS; i++) {
    const started = performance.now();
    fn();
    samples.push(performance.now() - started);
  }
  return summarize(samples);
}

function makeInputs() {
  const points = Array.from({ length: N_NOISE }, (_, i) => [
    Math.sin(i * 0.013) * 17.0,
    Math.cos(i * 0.007) * 11.0,
    Math.sin(i * 0.003 + 0.4) * 23.0,
  ]);
  const vectors = Array.from({ length: N_GEOM }, (_, i) => {
    const z = 1 - 2 * (i + 0.5) / N_GEOM;
    const angle = i * 2.399963229728653;
    const radius = Math.sqrt(1 - z * z);
    return [radius * Math.cos(angle), radius * Math.sin(angle), z];
  });
  const xyz = new Float32Array(N_CELLS * 3);
  for (let i = 0; i < N_CELLS; i++) {
    const z = 1 - 2 * (i + 0.5) / N_CELLS;
    const angle = i * 2.399963229728653;
    const radius = Math.sqrt(1 - z * z);
    xyz[3 * i] = radius * Math.cos(angle);
    xyz[3 * i + 1] = radius * Math.sin(angle);
    xyz[3 * i + 2] = z;
  }

  const offsets = new Int32Array(N_CELLS + 1);
  const adjacency = new Int32Array(N_CELLS * 6);
  for (let i = 0; i < N_CELLS; i++) {
    offsets[i] = i * 6;
    for (let k = 0; k < 3; k++) {
      adjacency[i * 6 + k] = (i + k + 1) % N_CELLS;
      adjacency[i * 6 + 3 + k] = (i - k - 1 + N_CELLS) % N_CELLS;
    }
  }
  offsets[N_CELLS] = N_CELLS * 6;
  return {
    points,
    vectors,
    mesh: { numCells: N_CELLS, xyz, adjOffset: offsets, adjList: adjacency },
  };
}

function benchmarkJavaScript() {
  const { points, vectors, mesh } = makeInputs();
  const noise = new FastNoise(9876);
  const selectedCells = [];
  for (let start = 0; start < N_CELLS; start += 68) {
    for (let cell = start; cell < Math.min(start + 64, N_CELLS); cell++) {
      selectedCells.push(cell);
    }
  }
  const copySource = new CrustRaster(N_CELLS);
  for (let i = 0; i < N_CELLS; i += 2) {
    copySource.create(i, CrustType.CONTINENT, -100, i + 1);
  }

  return {
    noise_fbm_ms: measure(() => {
      let sum = 0;
      for (const point of points) sum += noise.fbm(...point, 5, 2 / 3);
      globalThis.__benchSink = sum;
    }),
    spherical_geometry_ms: measure(() => {
      let sum = 0;
      for (const point of vectors) {
        const target = geom.destination(point, 0.71, 0.013);
        sum += geom.bearingTo(point, target) + geom.dot(point, target);
      }
      globalThis.__benchSink = sum;
    }),
    mesh_bfs_ms: measure(() => {
      const distance = hopDistance(mesh, (cell) => cell === 0, N_CELLS);
      globalThis.__benchSink = distance[N_CELLS - 1];
    }),
    mesh_components_ms: measure(() => {
      const components = piecesOf(mesh, selectedCells);
      globalThis.__benchSink = components.reduce((sum, part) => sum + part.length, 0);
    }),
    mesh_largest_piece_ms: measure(() => {
      const largest = largestPieceOf(mesh, selectedCells);
      globalThis.__benchSink = largest.length;
    }),
    mesh_within_km_ms: measure(() => {
      const reached = withinKm(mesh, (cell) => cell === 0, 5_000);
      globalThis.__benchSink = reached.reduce((sum, value) => sum + value, 0);
    }),
    mesh_dijkstra_ms: measure(() => {
      const distance = new Float64Array(N_CELLS);
      distance.fill(Infinity);
      distance[0] = 0;
      spreadKm(mesh, distance, { maxKm: 20_000 });
      globalThis.__benchSink = distance[N_CELLS - 1];
    }),
    crust_create_snapshot_ms: measure(() => {
      const crust = new CrustRaster(N_CELLS);
      let sum = 0;
      for (let i = 0; i < N_CELLS; i += 2) {
        crust.create(i, CrustType.CONTINENT, -100, i + 1);
        sum += crust.snapshot(i).length;
      }
      globalThis.__benchSink = sum;
    }),
    crust_copy_ms: measure(() => {
      const crust = new CrustRaster(N_CELLS);
      for (let i = 0; i < N_CELLS; i += 2) {
        crust.copyFrom(i, copySource, i);
      }
      globalThis.__benchSink = crust.count;
    }),
  };
}

function ratio(numerator, denominator) {
  return denominator > 0 ? Number((numerator / denominator).toFixed(3)) : null;
}

function buildComparisons(javascript, python) {
  const comparisons = {};
  for (const [name, jsStats] of Object.entries(javascript)) {
    const pure = python.python_pure.timings[name];
    const accelerated = python.python_numba?.timings[name] ?? null;
    comparisons[name] = {
      javascript_median_ms: jsStats.median_ms,
      python_pure_median_ms: pure.median_ms,
      python_numba_median_ms: accelerated?.median_ms ?? null,
      python_pure_over_js: ratio(pure.median_ms, jsStats.median_ms),
      python_numba_over_js: accelerated
        ? ratio(accelerated.median_ms, jsStats.median_ms)
        : null,
      numba_speedup_over_pure: accelerated
        ? ratio(pure.median_ms, accelerated.median_ms)
        : null,
    };
  }
  return comparisons;
}

const javascript = benchmarkJavaScript();
const pythonRun = spawnSync("python", ["benchmarks/bench_python.py"], {
  encoding: "utf8",
  env: {
    ...process.env,
    BENCH_ROUNDS: String(ROUNDS),
    BENCH_WARMUPS: String(WARMUPS),
  },
  maxBuffer: 20 * 1024 * 1024,
});
if (pythonRun.status !== 0) {
  throw new Error(
    "Python benchmark failed:\n" + pythonRun.stderr + "\n" + pythonRun.stdout,
  );
}
let python;
try {
  python = JSON.parse(pythonRun.stdout);
} catch (error) {
  throw new Error(
    "Could not parse Python benchmark JSON: " + String(error) + "\n" +
      pythonRun.stdout.slice(-4000),
  );
}

const report = {
  schema_version: 2,
  generated_at: new Date().toISOString(),
  configuration: {
    rounds: ROUNDS,
    warmups: WARMUPS,
    noise_points: N_NOISE,
    geometry_points: N_GEOM,
    graph_cells: N_CELLS,
    step_world_cells: python.configuration.step_world_requested_cells,
    timing_unit: "milliseconds",
    statistic: "median, mean, sample standard deviation, p95, min, max, raw samples",
  },
  runtime: {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    cpu_model: os.cpus()[0]?.model ?? null,
    logical_cpus: os.cpus().length,
    python: python.runtime,
  },
  javascript,
  python: {
    numba_available: python.runtime.numba_available,
    pure: python.python_pure,
    numba: python.python_numba,
  },
  comparisons: buildComparisons(javascript, python),
};

console.log(JSON.stringify(report, null, 2));
