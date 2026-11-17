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

## Confirmation after removing unprofitable scalar JIT paths — 2026-11-17

A later run after keeping only the validated FBM and BFS Numba paths measured:

| Workload | JavaScript | Python | Python / JS |
|---|---:|---:|---:|
| 3-D simplex FBM | 12.642 ms | 10.337 ms | **0.82×** |
| Spherical geometry | 23.363 ms | 249.814 ms | 10.69× slower |
| Mesh BFS | 0.672 ms | 1.272 ms | 1.89× slower |
| Mesh Dijkstra | 0.710 ms | 1.374 ms | 1.94× slower |
| Crust creation + snapshots | 4.559 ms | 159.487 ms | 34.98× slower |

[Confirmation run on GitHub Actions](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44145980375).

The run-to-run variation is substantial on shared hosted runners, especially for geometry, BFS and crust. Treat these numbers as directional, not stable performance guarantees. Scalar JIT wrappers for geometry and per-cell crust initialization were tested and removed: Python-to-Numba call overhead made both paths slower. The retained acceleration is limited to kernels with useful work per compiled call (batched FBM and full BFS traversal).


## After compiling Dijkstra traversal with Numba — 2026-11-17

The Dijkstra workload now uses one compiled kernel for the complete heap traversal whenever neither `can_enter` nor `on_reach` is supplied. The Python implementation remains active for callback-based use.

| Workload | JavaScript | Python + Numba | Python / JS |
|---|---:|---:|---:|
| 3-D simplex FBM, 40,000 points × 5 octaves | 9.673 ms | 9.497 ms | **0.98×** |
| Spherical geometry, 80,000 points | 27.063 ms | 162.268 ms | 6.00× slower |
| Mesh BFS, 12,000 cells | 0.435 ms | 0.679 ms | 1.56× slower |
| Mesh Dijkstra, 12,000 cells | 0.509 ms | 0.0074 ms | **0.015×** (about 69× faster) |
| Crust creation + snapshots, 6,000 cells | 2.299 ms | 115.908 ms | 50.41× slower |

[Benchmark run on GitHub Actions](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44147545592).

The Dijkstra measurement is a five-round median after one warm-up on the synthetic degree-6 ring graph, with `max_km=20,000`. It is not an end-to-end simulation result. Hosted-runner timings for the other workloads remain variable; use these figures as directional measurements only.


## Far-field diffusion kernel — 2026-11-17

The tectonic `spread_push` path now sends its complete weighted Dijkstra traversal to one Numba kernel. It computes great-circle edge costs, applies the crust-type/weak-line multipliers, maintains the source that delivered the strongest arrival, and returns reached cells in heap-pop order. Crust-history recording remains in Python. The original Python traversal remains available when Numba is absent.

The benchmark compares the full `spread_push` path, including Python-side history writes, with Numba enabled and with the Python fallback forced:

| Workload | Numba | Python fallback | Speedup |
|---|---:|---:|---:|
| `spread_push`, 12,000 cells, degree-6 equatorial ring, two sources | 60.856 ms | 203.843 ms | **3.35×** |

[Benchmark run on GitHub Actions](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44149930378).

This is an A/B comparison within Python, not a JavaScript comparison: there is no checked-in JavaScript far-field fixture in this repository. It is a synthetic graph and must not be interpreted as full-history simulation performance. The Numba result is compared against the Python fallback in tests, including reached-cell order, source assignment, distance arrays, and the recorded crust-history arrays.

The same run measured the general callback-free mesh Dijkstra benchmark at 0.522 ms in JavaScript and 0.0106 ms in Python + Numba. That workload is a synthetic ring graph stopped at `max_km=20,000`.

## What changed

- FBM moved from about **48× slower** to near parity or slightly faster than the JS reference on the measured batch workloads.
- BFS moved from about **28.6× slower** to around **1.4–1.9× slower** across the latest runs.
- The full callback-free mesh Dijkstra traversal is now compiled; the latest synthetic benchmark measured about **49× faster** than the JS sample, but that result must not be extrapolated to full-world graphs.
- The real tectonic far-field `spread_push` traversal is also compiled as one batch; its full-path A/B benchmark measured **3.35× faster** than the Python fallback.
- Geometry and crust snapshots remain major performance gaps. Shared-runner results vary and remain directional.

## Numerical-equivalence policy

- Numba kernels use float64 inputs and outputs for noise, preserve the same operation order, and set `fastmath=False`.
- The Numba FBM results are checked bit-for-bit against the retained pure-Python implementation.
- BFS uses the same CSR adjacency, ascending source initialization, FIFO traversal, and int16 distance output. Its compiled path is used only when `can_enter` is absent; arbitrary Python predicates retain the original Python implementation.
- Dijkstra retains the JS heap push/pop ordering, ascending seed insertion, lazy stale-entry checks, max-distance cutoff, and writes the value stored in the destination array back to the heap (including float32 rounding). Its Numba path is used only without `can_enter` or `on_reach`; callback-based calls use Python. Differential tests compare float32 and float64 arrays.
- Far-field diffusion compiles weighted graph traversal, great-circle edge costs, source propagation and reached order; crust history fields are then updated in Python. Its output is checked against the Python fallback.
- The pure-Python path remains available when Numba is not installed.
- Differential checks against the upstream JavaScript implementation remain part of CI. Performance is accepted only alongside numerical and behavioural tests.

## Method and limitations

- Timings are medians of five measured runs after one warm-up; they are microbenchmarks, not end-to-end planet-generation timings.
- The mesh benchmark uses a synthetic 12,000-cell degree-6 ring graph, not a production spherical mesh. The Dijkstra workload also stops at `max_km=20,000`, so it does not visit every cell.
- The benchmark does not measure Delaunay/sphere-mesh generation, full tectonic `step_world`, long geological histories, memory use, or complete planet generation. The far-field A/B uses a synthetic equatorial ring graph and has no direct JavaScript comparison.
- Python FBM uses one batch call to amortize Python-to-Numba call overhead. Scalar `noise3d` and `fbm` methods are also JIT-dispatched, but real pipeline speed depends on call patterns.
- Results depend on the hosted runner and are a baseline, not a universal JavaScript/Python ratio.

## Next optimization targets

1. Add batch geometry operations at real simulation call sites, so one Numba invocation processes many points rather than one scalar.
2. Optimize crust creation/copy/snapshot in batches, preserving snapshot field order and typed-array rounding.
3. Benchmark the far-field kernel and remaining graph kernels on realistic spherical meshes and full tectonic steps.

## Connected-component traversal kernel — 2026-11-17

The connected-component helpers used by collision, subduction and plate-accretion operations now have optional Numba implementations. A full CSR traversal runs inside one compiled call; the wrapper only converts the flattened result back to the existing list-of-lists API. The Python implementation remains the fallback when Numba is not installed or the input contains an out-of-range seed.

The paired benchmark measured a 12,000-cell degree-6 ring graph with selected 64-cell blocks:

| Workload | Numba | Python fallback | Speedup |
|---|---:|---:|---:|
| Connected components, including Python result materialization | 2.571 ms | 22.094 ms | **8.60×** |

[Benchmark run on GitHub Actions](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44158354632).

This is a synthetic CSR-graph microbenchmark, not a full collision or planet-generation timing. Tests compare component contents and traversal ordering against the Python fallback, including duplicate seeds, empty selections and first-component tie handling.

## Paired JavaScript comparison including connected components — 2026-11-17

The benchmark now measures the upstream JavaScript `piecesOf` implementation against the Python Numba `pieces_of` path on the same synthetic 12,000-cell degree-6 ring. Both implementations receive the same ordered selection: 64-cell contiguous blocks separated by 4-cell gaps. Each side includes component construction and result materialization. Five measured rounds follow one warm-up.

| Workload | JavaScript | Python + Numba | Python / JS | Faster |
|---|---:|---:|---:|---|
| 3-D simplex FBM, 40,000 points × 5 octaves | 10.378 ms | 9.743 ms | 0.94× | Python, 1.07× |
| Spherical geometry, 80,000 points | 18.731 ms | 159.392 ms | 8.51× | JavaScript, 8.51× |
| Mesh BFS, 12,000 cells | 0.465 ms | 0.711 ms | 1.53× | JavaScript, 1.53× |
| Connected components, 12,000 cells | 2.403 ms | 1.546 ms | 0.64× | Python + Numba, 1.55× |
| Mesh Dijkstra, 12,000 cells | 0.504 ms | 0.008 ms | 0.02× | Python + Numba, about 63× |
| Crust creation + snapshots, 6,000 cells | 3.860 ms | 119.421 ms | 30.94× | JavaScript, 30.94× |

[Paired benchmark run on GitHub Actions](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44160409232).

These are directional microbenchmark results from one shared GitHub-hosted runner, not end-to-end simulation timings. The connected-component comparison now uses the actual upstream JS helper and the same graph and ordered seed list as Python. The Dijkstra path is callback-free and uses the synthetic ring graph with `max_km=20,000`; the very small Python time should not be generalized to production spherical meshes. Geometry and crust remain the largest measured Python gaps.

## Batched spherical geometry and crust-field kernels — 2026-11-17

The spherical-geometry benchmark now uses destination_many and bearing_to_many over one contiguous (80,000, 3) array. With Numba installed, both operations execute inside compiled batch loops; without Numba, the functions retain a scalar Python fallback. The scalar API remains unchanged.

CrustRaster.create and snapshot now use compiled kernels over their 55 structure-of-arrays fields when Numba is available. The pure-Python loops remain in place as the no-Numba fallback. This avoids changing the snapshot ordering or its public list representation.

| Workload | JavaScript | Python + Numba | Python / JS | Faster |
|---|---:|---:|---:|---|
| 3-D simplex FBM, 40,000 points × 5 octaves | 12.663 ms | 10.438 ms | 0.82× | Python, 1.21× |
| Spherical geometry, 80,000 points | 23.373 ms | 8.727 ms | 0.37× | Python, 2.68× |
| Mesh BFS, 12,000 cells | 0.675 ms | 1.008 ms | 1.49× | JavaScript, 1.49× |
| Connected components, 12,000 cells | 2.820 ms | 2.182 ms | 0.77× | Python + Numba, 1.29× |
| Mesh Dijkstra, 12,000 cells | 0.694 ms | 0.012 ms | 0.02× | Python + Numba, about 56× |
| Crust creation + snapshots, 6,000 cells | 3.095 ms | 80.529 ms | 26.02× | JavaScript, 26.02× |

[Paired benchmark run on GitHub Actions](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44163102291).

For spherical geometry, the previous Python measurement on the same benchmark family was 159.392 ms; the batch implementation measured 8.727 ms, about **18.3× lower**. For crust creation + snapshots, the previous 119.421 ms measurement fell to 80.529 ms, about **1.48× lower**. Crust remains the largest measured gap. The raster benchmark calls the scalar create/snapshot API repeatedly, so it still pays Python-to-Numba dispatch overhead on every cell; a batch API should only be introduced where real simulation call sites can use it.

The paired run was completed on the shared GitHub-hosted runner with Node.js 22.23.4 and Python 3.12.16, five measured rounds after one warm-up. These results are directional microbenchmarks, not end-to-end simulation timings.