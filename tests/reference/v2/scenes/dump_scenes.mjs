// Dumps the state of every upstream scene (build step only) for a mesh size and a seed.
// Usage: node dump_scenes.mjs 768 941
// Reads the upstream engine from the git submodule (external/planet_heightmap_generation,
// pinned at ed466f9). Scenes "history" and "cycle" are skipped (whole-history runs).
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
const UP = process.env.UPSTREAM ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../../external/planet_heightmap_generation');
const url = (p) => pathToFileURL(path.join(UP, p)).href;
const Delaunator = (await import('delaunator')).default;  // repo root node_modules (delaunator 5.0.1)
const { setDelaunator } = await import(url('js/sphere-mesh.js'));
setDelaunator(Delaunator);
const { buildSimMesh } = await import(url('v2/js/core/sim-mesh.js'));
const files = ['budget-scenes', 'collision-scenes', 'event-scenes', 'force-scenes',
  'hotspot-scenes', 'margin-scenes', 'rift-scenes', 'terrain-scenes'];
const N = Number(process.argv[2] || 768), SEED = Number(process.argv[3] || 941);
const mesh = buildSimMesh(N, SEED);
const out = {};
for (const f of files) {
  const m = await import(url(`v2/js/tectonics/scenarios/${f}.js`));
  for (const [k, v] of Object.entries(m)) if (Array.isArray(v) && v.length && v[0] && v[0].id)
    for (const s of v) {
      if (s.id === 'history' || s.id === 'cycle' || typeof s.build !== 'function') continue;
      try {
        const w = s.build(mesh, SEED);
        const tcount = {};
        for (const p of w.plates) for (let c = 0; c < mesh.numCells; c++)
          if (p.crust.mask[c]) { const t = p.crust.type[c]; tcount[t] = (tcount[t] || 0) + 1; }
        out[`${k}:${s.id}`] = { plates: w.plates.length,
          cells: w.plates.map(p => p.crust.mask.reduce((a, b) => a + b, 0)), types: tcount };
      } catch (e) { out[`${k}:${s.id}`] = { error: String(e).slice(0, 160) }; }
    }
}
console.log(JSON.stringify(out));
