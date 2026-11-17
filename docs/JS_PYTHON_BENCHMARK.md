# JavaScript / Python performance comparison

## First paired benchmark — 2026-11-17

The benchmark was run by GitHub Actions on the same `ubuntu-latest` runner, using Node.js 22.23.4 and Python 3.12.16. Each implementation was warmed up once, then measured five times; the median elapsed time is reported. Lower is better.

| Workload | Input | JavaScript | Python | Python / JS |
|---|---:|---:|---:|---:|
| 3-D simplex FBM noise | 40,000 points, 5 octaves | 13.804 ms | 664.639 ms | 48.15× slower |
| Spherical geometry | 80,000 destination + bearing calculations | 23.628 ms | 238.610 ms | 10.10× slower |
| Mesh BFS | 12,000 cells, degree 6 | 0.660 ms | 18.871 ms | 28.57× slower |
| Mesh Dijkstra / distance spread | 12,000 cells, degree 6 | 0.722 ms | 1.349 ms | 1.87× slower |
| Crust creation + snapshots | 6,000 cells | 6.895 ms | 153.895 ms | 22.32× slower |

Raw result artifact: [GitHub Actions benchmark artifact](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44142059464/artifacts/14354662616).

## Interpretation

- This first measurement shows that the current Python port is substantially slower than the JavaScript reference in the selected CPU-heavy loops.
- The largest gaps are in simplex/FBM noise, BFS and per-cell crust record/snapshot operations. These are the first optimization targets.
- Dijkstra is much closer, at about 1.9× slower.
- The current Python implementation intentionally prioritizes numerical parity and line coverage; it has not yet received a dedicated performance pass.

## Method and limitations

- Both versions use the same deterministic input formulas, seed, workload counts and graph topology. Each is measured on the same CI runner.
- Timings are medians of five measured runs after one warm-up. They are wall-clock microbenchmarks, not end-to-end planet-generation timings.
- The mesh workload uses a synthetic 12,000-cell degree-6 ring graph. It is useful for comparing the BFS and heap kernels, but it is not a production spherical mesh.
- The test does not measure Delaunay/sphere-mesh generation, full tectonic `step_world`, long geological histories, memory use, or the full planet pipeline.
- Results depend on the hosted runner and should be treated as an initial baseline, not a universal JS/Python ratio.

## Optimization order suggested by this baseline

1. Profile and accelerate `FastNoise.noise3d/fbm` (compiled/vectorized kernel, while retaining JS differential tolerances).
2. Replace Python-level per-cell BFS loops with a compiled or vectorized traversal.
3. Reduce Python object/list creation in spherical geometry and crust record/snapshot hot paths.
4. Re-measure on realistic spherical meshes and complete tectonic steps before selecting further optimizations.
