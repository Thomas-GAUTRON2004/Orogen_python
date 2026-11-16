# Orogen Python

A progressive Python port of the **Orogen 3.0.1** engine, based on the upstream
[planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation)
repository.

> **Development status:** The port is not complete. The `port/v3.0.1` branch
> contains the V2 numerical foundation and an initial set of tests. It does not
> yet claim full functional equivalence with the JavaScript engine.

## Version and installation

Development version: `3.0.1.dev0`. Python 3.10 or later.

```bash
python -m pip install -e ".[test]"

# Node.js 22+ and Delaunator are required for differential tests.
npm install --no-save --package-lock=false delaunator@5.0.1

python -m pytest --cov=orogen.v2 --cov-report=term-missing --cov-fail-under=100
```

Current numerical dependencies are NumPy and SciPy (Qhull is used for Delaunay
triangulation). The production triangulation backend differs from JavaScript
Delaunator. Topology and invariants are tested, but matching connectivity on a
finite set of shared seeds is not proof of identical output for every seed and
resolution. Dependencies for future subsystems will be added as those systems
are ported.

## Currently ported components

The `orogen.v2.core` namespace contains initial Python implementations of the
following V2 building blocks:

- Deterministic seed handling and the Park–Miller random number generator.
- 3D simplex noise and fractal Brownian motion (FBM).
- Quaternion rotations.
- An Euler-pole polar frame and small-device memory-budget classification.
- Spherical geometry.
- Fibonacci point generation, stereographic projection, and north-pole mesh closure.
- Delaunay triangulation through SciPy/Qhull and simulation-mesh construction.
- Spherical spatial indexing and mesh reconstruction from arrays.
- Breadth-first search (BFS), hop distances, and connected components.
- Barycentric point location on spherical triangles.
- A min-heap and Dijkstra propagation in kilometres.
- Precomputed neighbourhood rings.

The `orogen.v2.tectonics` namespace currently contains:

- Physical conversion constants for km/Myr and rad/Myr, ridge-segment/fracture-zone
  coordinates, and crust, orogeny, volcanism, and belt-side flags.
- Per-plate crust rasters with typed fields, snapshots and restoration, transfers,
  subduction tracking, and orogeny recording.
- A rigid-plate model, coordinate-frame transformations, culling caps, and
  inter-plate rotations.
- Plate kinematics (Euler-pole and velocity conversions, time-based plate
  rotations) and spatial occupancy indexing.
- Claim gathering, read-only owner reconstruction, and subduction-polarity rules.

These functions are currently low-level building blocks, not a complete
tectonic simulation pipeline. TECTONICS configuration defaults, full world
initialization, the collision/subduction resolver, time-stepping, and the
terrain, geology, climate, finishing, and agriculture systems remain to be
ported.

## Upstream reference

- Upstream release: `v3.0.1`
- Upstream commit: `ed466f9db71292e912d8d0160e689a67a994f2d9`
- License: GPL-3.0-only (see `LICENSE` and `NOTICE.md`).

## Tests and compatibility

The differential test harness runs vendored JavaScript reference modules from
upstream tag `v3.0.1`. It currently compares, among other things:

- Seed parsing, Park–Miller RNG, and simplex/FBM noise.
- Quaternions, spherical geometry, and the polar frame.
- Fibonacci generation, stereographic projection, Delaunay topology, and
  spherical neighbourhoods across seven mesh sizes/seeds.
- Cube-map indexing, reconstruction from arrays, barycentric location, BFS,
  neighbourhood rings, min-heap operations, and Dijkstra propagation.
- Crust-raster fields, copy/transfer/restore operations, plate transforms, and
  initial tectonic-world state including independent RNG streams.

GitHub Actions installs Node.js 22, Delaunator 5.0.1, and Python 3.12. The
latest verified run completed **107 tests successfully**. Line coverage for
`orogen.v2` is **100%** (1,276 statements, none missing). CI enforces this
threshold with `--cov-fail-under=100` and writes a `coverage.xml` report.

Full compatibility has not been established. Dynamic tectonic steps,
multi-step simulations, and the terrain, geology, climate, finishing, and
agriculture systems have not yet been compared against JavaScript outputs.
The seven successful Delaunay topology cases do not cover every seed or
resolution.
