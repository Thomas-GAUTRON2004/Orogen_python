# Production Delaunay backend: shull

## Decision

Orogen's production spherical mesh builder uses `shull==0.4.0`, a Rust
implementation of 2-D Delaunay triangulation. The dependency is pinned rather
than left on an open-ended version range because triangulation tie-breaking
can affect mesh connectivity and therefore seeded simulation outputs.

The Python package requires Python 3.10+. The published 0.4.0 release provides
prebuilt CPython wheels for Linux, macOS and Windows. Its Rust manifest declares
`GPL-3.0-only`, matching Orogen's declared project licence. The dependency is
not vendored; see [NOTICE](../NOTICE.md) for attribution.

## Production path

1. Orogen generates the same Float32 Fibonacci-sphere coordinates and projects
   them to contiguous Float64 stereographic coordinates.
2. `shull.Delaunay(points).simplices` computes the 2-D triangles in Rust.
3. Orogen normalizes triangle winding and closes the planar convex hull with
   the synthetic north-pole vertex.
4. The existing Orogen cyclic CSR builder creates the directed neighbour rings.
5. `SphereIndex` and `SimMesh` are built through the existing code paths.

The integration deliberately does not use shull's own vertex-neighbour CSR:
Orogen needs a cyclic, consistently oriented ring and one additional pole
region. The public Orogen mesh API and array layouts are unchanged.

SciPy/Qhull was removed from runtime dependencies. It is retained in the
optional `benchmark` extra only as a comparison baseline. Numba remains
optional and its production kernels are unchanged.

## Correctness evidence

The existing seven seeded Delaunator differential fixtures now exercise the
production shull path. They compare triangle connectivity, neighbour sets,
mesh metrics and downstream graph/geometry results. The mesh invariant tests
also cover spherical Euler topology, reciprocal neighbours, cyclic face
orientation, deterministic builds and the north-pole region.

The A/B benchmark additionally checks, for each tested resolution:

- canonicalized simplex-set equality between SciPy and shull;
- closed spherical face-set equality after pole closure;
- identical CSR offsets and cyclic adjacency after canonicalizing ring starts;
- identical `SphereIndex` lookup tables;
- identical BFS distance arrays.

These finite fixtures are not a proof that every degenerate or cocircular point
cloud receives the same triangulation from both algorithms.

## Performance measurements

The previous experimental branch measured isolated triangulation on the same
seeded Orogen point clouds. These results are directional runner medians:

| Points | SciPy/Qhull | shull | Triangulation speedup |
|---:|---:|---:|---:|
| 10,000 | 26.9 ms | 2.6 ms | 10.43× |
| 40,000 | 154.4 ms | 7.6 ms | 20.36× |
| 80,000 | 344.2 ms | 15.4 ms | 22.40× |

The broader pipeline comparison with the production Python CSR builder (not
the experimental Numba CSR prototype) measured:

| Cells | SciPy + Python CSR | shull + Python CSR | Pipeline speedup |
|---:|---:|---:|---:|
| 10,001 | 0.3876 s | 0.3609 s | 1.07× |
| 40,001 | 1.5761 s | 1.4444 s | 1.09× |
| 80,001 | 3.2428 s | 2.8978 s | 1.12× |

The benchmark includes triangulation, pole closure, cyclic CSR and
`SphereIndex`; coordinate generation/projection is prepared outside the
timed region. The old 1.18×–1.23× combined figures used an experimental
Numba CSR builder and are not claimed as production speedups.

## Reproduce the A/B test

Install the optional comparison dependencies and run:

```bash
python -m pip install -e ".[benchmark]"
python benchmarks/bench_shull_pipeline.py
```

The script uses warm-up plus three measured rounds on 10k, 40k and 80k points.
It asserts structural equivalence and prints both stage timings and the actual
`build_sim_mesh` wall time using the production shull backend. Hosted-runner
timings are diagnostic, not hard pass/fail performance thresholds. The
`shull production pipeline benchmark` workflow runs it on GitHub Actions.

## Remaining optimization scope

At 80,001 cells the earlier experiment measured `SphereIndex` construction
at about 1.87 s, so it became the dominant mesh-stage cost after triangulation
was accelerated. Optimizing `SphereIndex` is a separate follow-up. The
experimental Numba CSR prototype remains on the experimental branch and is not
part of this integration.
