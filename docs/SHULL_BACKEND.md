# Production Delaunay backend: shull

## Decision

Orogen's production spherical mesh builder uses `shull>=0.4.0`, a Rust
implementation of 2-D Delaunay triangulation. The lower bound is deliberate:
the target branch already contains the production integration and CI currently
resolves `shull==0.4.0`. A fixed upper pin is not imposed without evidence that
later compatible releases change required topology or behaviour.

The Python package requires Python 3.10+. The dependency is installed as a
separate package and is not vendored. Its upstream project declares
`GPL-3.0-only`; see [NOTICE](../NOTICE.md) for attribution.

## Production path

1. Orogen generates Float32 Fibonacci-sphere coordinates and projects them to
   contiguous Float64 stereographic coordinates.
2. `shull.Delaunay(points).simplices` computes the planar triangles in Rust.
3. Orogen validates the simplex shape and vertex-index bounds, normalizes
   triangle winding, and closes the planar convex hull with a synthetic
   north-pole vertex.
4. The Orogen cyclic CSR builder creates directed neighbour rings.
5. Existing `SphereIndex` and `SimMesh` code paths are used unchanged.

The integration does not use shull's own vertex-neighbour CSR: Orogen needs a
cyclic, consistently oriented ring and one additional pole region. The public
Orogen mesh API and array layouts remain unchanged.

SciPy is retained in the `test` extra for existing test/reference tooling, and
in the optional `benchmark` extra as an A/B baseline. It is not a production
runtime dependency. Numba remains optional.

## Correctness and platform validation

The normal test suite exercises the production shull path. Spherical mesh
checks cover Euler topology, reciprocal neighbours, cyclic face orientation,
deterministic builds, north-pole closure, translation of native exceptions,
invalid simplex shape, out-of-range indices, and invalid convex-hull output.

The `shull platform compatibility` workflow runs the focused mesh test module
on Linux, Windows and macOS with Python 3.10, 3.12 and 3.14. The platform
matrix must be green on the current PR head before the change is considered
validated.

The selected JavaScript differential fixtures are finite evidence only. They
do not prove identical connectivity for every possible seed, resolution,
degenerate point cloud or simulation history.

## A/B pipeline benchmark

Install the optional comparison dependencies and run:

```bash
python -m pip install -e ".[benchmark]"
python benchmarks/bench_shull_pipeline.py
```

The script measures triangulation, pole closure, cyclic CSR and
`SphereIndex` construction at 10k, 40k and 80k points. It compares canonical
simplex sets, closed spherical faces, CSR adjacency, `SphereIndex` lookup
tables and BFS distances. It also measures the production `build_sim_mesh`
path. Timings are diagnostic and are not hard pass/fail performance limits.

A prior benchmark run on the original shull feature branch reported these
three-round medians after warm-up:

| Points | SciPy/Qhull | shull | Triangulation speedup |
|---:|---:|---:|---:|
| 10,000 | 25.6 ms | 2.2 ms | 11.90× |
| 40,000 | 119.3 ms | 9.8 ms | 12.20× |
| 80,000 | 367.7 ms | 21.0 ms | 17.52× |

| Cells | SciPy pipeline | shull pipeline | Pipeline speedup |
|---:|---:|---:|---:|
| 10,001 | 0.4033 s | 0.3662 s | 1.10× |
| 40,001 | 1.5637 s | 1.4452 s | 1.08× |
| 80,001 | 3.2971 s | 2.8801 s | 1.14× |

That historical run also reported a 344,204 KiB peak RSS for the entire
comparison process, including imports and both backends; it is not a
per-backend memory measurement. See the
[historical benchmark workflow](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44183846060).
A fresh benchmark is required after changes to the current branch.

The `shull production pipeline benchmark` workflow runs the comparison on
GitHub Actions and records process peak memory using GNU `/usr/bin/time -v`.

## Remaining optimization scope

The benchmark's broad pipeline includes Python CSR construction and
`SphereIndex`; triangulation is only one part of total mesh-build time.
Optimizing `SphereIndex` is a separate follow-up. The experimental Numba CSR
prototype is not part of this production integration.
