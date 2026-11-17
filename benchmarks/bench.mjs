import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { median } from "./bench-utils.mjs";
import { FastNoise } from "../tests/reference/v2/core/fast-noise.mjs";
import * as geom from "../tests/reference/v2/core/sphere-geom.mjs";
import { hopDistance, piecesOf } from "../tests/reference/v2/core/mesh-bfs.mjs";
import { spreadKm } from "../tests/reference/v2/core/min-heap.mjs";
import { CrustRaster, CrustType } from "../tests/reference/v2/tectonics/crust.mjs";

const N_NOISE = 40000, N_GEOM = 80000, N_CELLS = 12000, ROUNDS = 5;
function measure(fn) {
  fn();
  const values = [];
  for (let i = 0; i < ROUNDS; i++) {
    const start = performance.now();
    fn();
    values.push(performance.now() - start);
  }
  return median(values);
}
function makeInputs() {
  const points = Array.from({length:N_NOISE}, (_,i) => [
    Math.sin(i * 0.013) * 17.0,
    Math.cos(i * 0.007) * 11.0,
    Math.sin(i * 0.003 + 0.4) * 23.0,
  ]);
  const vectors = Array.from({length:N_GEOM}, (_,i) => {
    const z = 1 - 2 * (i + 0.5) / N_GEOM;
    const a = i * 2.399963229728653;
    const r = Math.sqrt(1-z*z);
    return [r*Math.cos(a), r*Math.sin(a), z];
  });
  const xyz = new Float32Array(N_CELLS * 3);
  for (let i=0;i<N_CELLS;i++) {
    const z=1-2*(i+0.5)/N_CELLS, a=i*2.399963229728653, r=Math.sqrt(1-z*z);
    xyz[3*i]=r*Math.cos(a); xyz[3*i+1]=r*Math.sin(a); xyz[3*i+2]=z;
  }
  const offsets = new Int32Array(N_CELLS+1), adjacency = new Int32Array(N_CELLS*6);
  for(let i=0;i<N_CELLS;i++) {
    offsets[i]=i*6;
    for(let k=0;k<3;k++){adjacency[i*6+k]=(i+k+1)%N_CELLS;adjacency[i*6+3+k]=(i-k-1+N_CELLS)%N_CELLS;}
  }
  offsets[N_CELLS]=N_CELLS*6;
  return {points,vectors,mesh:{numCells:N_CELLS,xyz,adjOffset:offsets,adjList:adjacency}};
}
const {points,vectors,mesh}=makeInputs();
const noise=new FastNoise(9876);
const selectedCells = [];
for (let start = 0; start < N_CELLS; start += 68) {
  for (let cell = start; cell < Math.min(start + 64, N_CELLS); cell++) selectedCells.push(cell);
}
const copySource = new CrustRaster(N_CELLS);
for (let i=0;i<N_CELLS;i+=2) copySource.create(i,CrustType.CONTINENT,-100,i+1);
const jsResults = {
  noise_fbm_ms: measure(() => { let s=0; for(const p of points) s+=noise.fbm(...p,5,2/3); globalThis.__benchSink=s; }),
  spherical_geometry_ms: measure(() => { let s=0; for(const p of vectors){const q=geom.destination(p,0.71,0.013);s+=geom.bearingTo(p,q)+geom.dot(p,q);} globalThis.__benchSink=s; }),
  mesh_bfs_ms: measure(() => { const d=hopDistance(mesh,c=>c===0, N_CELLS); globalThis.__benchSink=d[N_CELLS-1]; }),
  mesh_components_ms: measure(() => { const pieces=piecesOf(mesh,selectedCells); globalThis.__benchSink=pieces.reduce((n,p)=>n+p.length,0); }),
  mesh_dijkstra_ms: measure(() => { const d=new Float64Array(N_CELLS); d.fill(Infinity); d[0]=0; spreadKm(mesh,d,{maxKm:20000}); globalThis.__benchSink=d[N_CELLS-1]; }),
  crust_create_snapshot_ms: measure(() => { const c=new CrustRaster(N_CELLS); let s=0; for(let i=0;i<N_CELLS;i+=2){c.create(i,CrustType.CONTINENT,-100,i+1);s+=c.snapshot(i).length;} globalThis.__benchSink=s; }),
  crust_copy_ms: measure(() => { const c=new CrustRaster(N_CELLS); for(let i=0;i<N_CELLS;i+=2)c.copyFrom(i,copySource,i); globalThis.__benchSink=c.count; }),
};
const py = spawnSync("python", ["benchmarks/bench_python.py"], {encoding:"utf8"});
if (py.status !== 0) throw new Error("Python benchmark failed:\n"+py.stderr+"\n"+py.stdout);
const pyResults=JSON.parse(py.stdout);
const results={runtime:{node:process.version,python:pyResults.runtime.python,platform:process.platform,arch:process.arch},iterations:{noise:N_NOISE,geometry:N_GEOM,cells:N_CELLS,rounds:ROUNDS},javascript:jsResults,python:pyResults.timings,accelerator_comparisons:pyResults.accelerator_comparisons};
const comparison={};
for(const key of Object.keys(jsResults)){
 const js=jsResults[key], python=pyResults.timings[key];
 comparison[key]={javascript_ms:+js.toFixed(3),python_ms:+python.toFixed(3),python_over_js:+(python/js).toFixed(2),faster:js<python?"javascript":"python"};
}
console.log(JSON.stringify({results,comparison},null,2));
