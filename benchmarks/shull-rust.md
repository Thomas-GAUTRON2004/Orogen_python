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

The first isolated run on GitHub Actions (10k/40k/80k points) measured:

| Points | SciPy/Qhull | shull | Speedup |
|---:|---:|---:|---:|
| 10,000 | 26.0 ms | 1.9 ms | 13.99x |
| 40,000 | 177.1 ms | 8.2 ms | 21.57x |
| 80,000 | 467.3 ms | 15.9 ms | 29.40x |

These are triangulation-only microbenchmarks on one shared runner, not full
build_sim_mesh or planet-generation timings. The new pipeline benchmark is
required before recommending replacement in production.

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
