# Shull / Rust Delaunay experiment

This branch evaluates shull 0.4.0 as an optional replacement for the SciPy/Qhull
2-D triangulation used by Orogen's spherical mesh builder.

## What has been done

1. Benchmarked Orogen's projected Fibonacci-sphere point clouds at 10k, 40k
   and 80k vertices.
2. Compared SciPy/Qhull and shull on identical float64 projected coordinates.
3. Canonicalized each triangle and now requires exact equality of the complete
   simplex set. Matching only triangle count or vertex coverage is insufficient.
4. Added a full mesh-stage benchmark. It closes the planar convex hull at
   Orogen's extra north-pole vertex, builds the existing cyclic CSR region
   adjacency, checks closed face sets and every vertex's neighbour set, and
   runs Orogen's Numba BFS against both resulting graphs.
5. Reports triangulation, Python post-processing and combined stage timings,
   plus memory occupied by the output triangle and CSR arrays.
6. Leaves SciPy/Qhull as the production default. No shull runtime dependency is
   added to Orogen and this branch does not switch the production backend.

## Run locally

    python -m pip install -e ".[speed]" shull==0.4.0
    python benchmarks/experiment_shull.py
    python benchmarks/experiment_shull_pipeline.py

GitHub Actions runs both scripts on Python 3.12. Timings are medians after
warm-up and should only be compared on the same runner/environment.

## First isolated result

The first isolated run used an unjittered Fibonacci sphere. A more representative
run now uses the production jitter factor (0.35) and deterministic Orogen seed
initialization. On the same GitHub runner, the latest five-round medians were:

| Points | SciPy/Qhull | shull | Speedup |
|---:|---:|---:|---:|
| 10,000 | 26.9 ms | 2.6 ms | 10.43x |
| 40,000 | 154.4 ms | 7.6 ms | 20.36x |
| 80,000 | 344.2 ms | 15.4 ms | 22.40x |

The exact simplex set matched for all three sizes. This is still triangulation-
only; the full mesh-stage result below is much smaller because mesh construction
and SphereIndex are significant costs.

## Full mesh-stage result and Numba composition

The expanded pipeline uses the actual Orogen jitter factor (0.35) and seeded
Fibonacci sphere coordinates. It times triangulation, pole closure, cyclic CSR
construction and SphereIndex construction. Coordinate generation/projection
are prepared once outside the timed region. It then runs the existing Numba
BFS against both graphs and requires identical distances.

| Cells | SciPy + Python CSR | shull + Python CSR | shull + Numba CSR | Combined speedup vs baseline |
|---:|---:|---:|---:|---:|
| 10,001 | 0.3876 s | 0.3609 s | 0.3280 s | **1.18x** |
| 40,001 | 1.5761 s | 1.4444 s | 1.3092 s | **1.20x** |
| 80,001 | 3.2428 s | 2.8978 s | 2.6271 s | **1.23x** |

The experimental Numba CSR builder was **28.56x**, **23.70x** and **17.71x**
faster than the Python cyclic-CSR builder at these sizes. It reproduced the
Python offsets and adjacency order exactly for each backend. After rotating
each cyclic row to a deterministic minimum-neighbour start, both triangulators
also produced identical SphereIndex lookup tables. The Numba BFS distances
matched exactly.

At 80,001 cells, triangulation itself took 323 ms with SciPy and 20.5 ms with
shull, but SphereIndex construction still took about 1.87 s. That is why a
22.4x triangulation gain becomes only a 1.23x measured mesh-stage gain after
adding Numba CSR. SphereIndex and other Python mesh-building work are now the
larger optimization targets.

The Numba CSR implementation currently lives only in the experiment script;
it has not been moved into the production package. The combined result
therefore demonstrates a viable design, not a production speedup already
available in Orogen.

[Full pipeline + Numba benchmark run](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44177187113).

## Why shull and Numba can complement each other

They accelerate different stages:

- shull (Rust) computes the 2-D Delaunay simplices in native code.
- Orogen post-processing closes the planar hull at the extra pole and creates
  the cyclic neighbour CSR required by the spherical mesh.
- Numba accelerates graph work over those CSR arrays: BFS, connected
  components, Dijkstra and other numerical kernels.

Numba cannot JIT through shull's Python/Rust extension call. The useful
composition is a native triangulator followed by Numba kernels over NumPy/CSR
arrays. The new pipeline benchmark checks topology and downstream Numba BFS
equivalence for both triangulators.

shull also exposes vertex_neighbor_vertices CSR built in Rust. It is not yet a
drop-in replacement for Orogen's adjacency: Orogen adds a synthetic pole
vertex and stores neighbours in cyclic order. Any reuse must add the pole
edges and preserve the cyclic-ring contract.

## Constraints

- shull 0.4.0 advertises GPLv3. Orogen is GPL-3.0-only; review the precise
  licence expression and distribution obligations before declaring shull as a
  runtime dependency.
- shull remains optional in this experiment.
- Delaunay triangulations may differ on degenerate/cocircular inputs even when
  both are valid. The deterministic Orogen fixtures intentionally fail on a
  simplex-set mismatch rather than silently accepting it.
- The pipeline benchmark does not yet include Fibonacci coordinate generation,
  SphereIndex construction or a full build_sim_mesh profile.

## Acceptance criteria

Do not switch production until exact simplex topology, closed spherical
neighbour sets, cyclic ordering, Numba outputs, full build_sim_mesh wall time,
peak memory, cross-platform CI and licensing have been validated.
