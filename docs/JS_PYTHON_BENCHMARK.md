# JavaScript / Python performance comparison

## Baseline before Numba — 2026-11-17

The first paired benchmark used the same GitHub Actions `ubuntu-latest` runner, Node.js 22.23.4 and Python 3.12.16. Five measured rounds followed one warm-up.

| Workload | JavaScript | Python before JIT | Python / JS |
|---|---:|---:|---:|
| 3-D simplex FBM, 40,000 points × 5 octaves | 13.804 ms | 664.639 ms | 48.15× slower |
| Spherical geometry, 80,000 points | 23.628 ms | 238.610 ms | 10.10× slower |
| Mesh BFS, 12,000 cells | 0.660 ms | 18.871 ms | 28.57× slower |
| Mesh Dijkstra, 12,000 cells | 0.722 ms | 1.349 ms | 1.87× slower |
| Crust creation + snapshots, 6,000 cells | 6.895 ms | 153.895 ms | 22.32× slower |

## After the first Numba kernels — 2026-11-17

Same runner and workload definitions, with the optional `speed` extra installed. This run measured the Numba FBM batch kernel and Numba BFS kernel.

| Workload | JavaScript | Python + Numba | Python / JS |
|---|---:|---:|---:|
| 3-D simplex FBM, 40,000 points × 5 octaves | 16.493 ms | 14.061 ms | **0.85×** (Python ~15% faster) |
| Spherical geometry, 80,000 points | 30.644 ms | 302.170 ms | 9.86× slower |
| Mesh BFS, 12,000 cells | 0.905 ms | 1.232 ms | **1.36× slower** |
| Mesh Dijkstra, 12,000 cells | 0.981 ms | 1.990 ms | 2.03× slower |
| Crust creation + snapshots, 6,000 cells | 12.810 ms | 208.800 ms | 16.30× slower |

Benchmark artifact: [paired run on GitHub Actions](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44143048224/artifacts/14355401914).

## What changed

- FBM moved from about **48× slower** to slightly faster than the JS reference on this batch workload.
- BFS moved from about **28.6× slower** to about **1.36× slower**.
- These gains come from Numba-compiled loops, not changed noise formulas or graph semantics.
- Geometry and crust snapshots remain major performance gaps. Dijkstra results vary between hosted-runner samples and should not be over-interpreted.

## Numerical-equivalence policy

- Numba kernels use float64 inputs and outputs for noise, preserve the same operation order, and set `fastmath=False`.
- The Numba FBM results are checked bit-for-bit against the retained pure-Python implementation.
- BFS uses the same CSR adjacency, ascending source initialization, FIFO traversal, and int16 distance output. Its compiled path is used only when `can_enter` is absent; arbitrary Python predicates retain the original Python implementation.
- The pure-Python path remains available when Numba is not installed.
- Differential checks against the upstream JavaScript implementation remain part of CI. Performance is accepted only alongside numerical and behavioural tests.

## Method and limitations

- Timings are medians of five measured runs after one warm-up; they are microbenchmarks, not end-to-end planet-generation timings.
- The mesh benchmark uses a synthetic 12,000-cell degree-6 ring graph, not a production spherical mesh.
- The benchmark does not measure Delaunay/sphere-mesh generation, full tectonic `step_world`, long geological histories, memory use, or complete planet generation.
- Python FBM uses one batch call to amortize Python-to-Numba call overhead. Scalar `noise3d` and `fbm` methods are also JIT-dispatched, but real pipeline speed depends on call patterns.
- Results depend on the hosted runner and are a baseline, not a universal JavaScript/Python ratio.

## Next optimization targets

1. Compile/vectorize spherical destination and bearing calculations while retaining JS operation order and tolerances.
2. Optimize crust creation/copy/snapshot paths without changing snapshot field order or typed-array rounding.
3. Benchmark on realistic spherical meshes and full tectonic steps.
