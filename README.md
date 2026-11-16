# Orogen Python

A progressive Python port of the **Orogen 3.0.1** procedural planetary simulation engine. The implementation is based on the upstream [planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation) repository, release `v3.0.1`.

> **Status:** this repository is an incomplete port. The current `orogen.v2` package provides tested numerical and tectonic building blocks; it does not yet generate a complete planet or reproduce the full JavaScript simulation pipeline.

## Documentation

The documentation is organized as a small handbook:

- [Documentation home](docs/README.md) — navigation and project scope.
- [Getting started](docs/GETTING_STARTED.md) — installation and first operations.
- [Architecture](docs/ARCHITECTURE.md) — package structure, data flow, and design decisions.
- [Core API](docs/CORE_API.md) — random numbers, noise, meshes, geometry, and graph algorithms.
- [Tectonics API](docs/TECTONICS_API.md) — crust, plates, kinematics, and world state.
- [Testing and compatibility](docs/TESTING.md) — test commands, differential tests, and compatibility limits.
- [Porting status](docs/PORTING_STATUS.md) — implemented, partial, and outstanding subsystems.

## Requirements

- Python 3.10 or later
- NumPy 1.24 or later
- SciPy 1.10 or later

Install the project and its test dependencies:

```bash
python -m pip install -e ".[test]"
```

The JavaScript differential reference tests additionally require Node.js 22+ and Delaunator 5.0.1:

```bash
npm install --no-save --package-lock=false delaunator@5.0.1
python -m pytest
```

## Quick example

```python
from orogen.v2.core import FastNoise, build_sim_mesh, make_rng

# Deterministic random stream
random = make_rng(42)
print(random())

# Seeded 3-D simplex noise and FBM
noise = FastNoise(42)
print(noise.noise3d(0.1, 0.2, 0.3))
print(noise.fbm(0.1, 0.2, 0.3, octaves=5))

# Build a spherical simulation mesh
mesh = build_sim_mesh(num_cells=128, seed=42)
print(mesh.num_cells, mesh.edge_km, mesh.cell_area_km2)
```

See [Getting started](docs/GETTING_STARTED.md) for a longer walkthrough and important mesh details.

## What is implemented

### Core (`orogen.v2.core`)

Deterministic seed parsing and Park–Miller RNG, 3-D simplex noise and FBM, quaternion operations, spherical geometry, Fibonacci sphere generation, stereographic projection, spherical mesh construction, spatial indexing, simulation-mesh helpers, barycentric point location, breadth-first graph operations, neighbourhood rings, a min-heap, and Dijkstra propagation in kilometres.

### Tectonics (`orogen.v2.tectonics`)

Crust-raster storage and history fields, rigid plate state and coordinate transforms, plate kinematics, occupancy bins, claim gathering, subduction-polarity rules, owner reconstruction, physical conversion constants, and ridge/fracture-zone coordinates.

These are components, not a complete simulation. Tectonic configuration defaults, full world initialization, the collision/subduction resolver, time-stepping orchestration, and terrain, geology, climate, finishing, and agriculture pipelines are not yet ported.

## Numerical backend and reproducibility

The production spherical triangulation uses SciPy/Qhull. The upstream JavaScript implementation uses Delaunator. Differential checks currently confirm matching topology for seven selected seed/resolution cases; this is not a guarantee of identical connectivity for every input.

The code preserves upstream deterministic seed and random-number conventions where implemented. Reproducibility still depends on using the same parameters and supported execution paths. See [Architecture](docs/ARCHITECTURE.md#numerical-and-reproducibility-contract) and [Testing](docs/TESTING.md).

## Validation

The current CI suite has verified **107 passing tests** and **100% line coverage** for `orogen.v2` (1,276 statements). CI enforces the 100% threshold. Coverage measures execution of the implemented Python package; it does not measure feature completeness or JavaScript parity.

## Upstream and license

- Upstream: [raguilar011095/planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation)
- Reference release: `v3.0.1`
- Reference commit: `ed466f9db71292e912d8d0160e689a67a994f2d9`
- License: GPL-3.0-only. See [LICENSE](LICENSE), [NOTICE.md](NOTICE.md), and [UPSTREAM_SOURCE.md](UPSTREAM_SOURCE.md).
