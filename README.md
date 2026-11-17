# Orogen Python

A progressive Python port of the **Orogen 3.0.1** procedural planetary simulation engine. The implementation is based on the upstream [planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation) repository, release `v3.0.1`.

> **Status:** the port contains tested core and tectonic subsystems, including continental rifting and passive-margin subduction initiation in a partial one-step orchestrator. It is not yet a complete port of the JavaScript simulation pipeline and does not yet generate a complete planet.

## Documentation

- [Documentation home](docs/README.md) — navigation and project scope.
- [Getting started](docs/GETTING_STARTED.md) — installation and first operations.
- [Architecture](docs/ARCHITECTURE.md) — package structure, data flow, and design decisions.
- [Core API](docs/CORE_API.md) — random numbers, noise, meshes, geometry, and graph algorithms.
- [Tectonics API](docs/TECTONICS_API.md) — crust, plates, rifting, and tectonic operations.
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

random = make_rng(42)
print(random())

noise = FastNoise(42)
print(noise.noise3d(0.1, 0.2, 0.3))
print(noise.fbm(0.1, 0.2, 0.3, octaves=5))

mesh = build_sim_mesh(num_cells=128, seed=42)
print(mesh.num_cells, mesh.edge_km, mesh.cell_area_km2)
```

See [Getting started](docs/GETTING_STARTED.md) for a longer walkthrough and important mesh details.

## Implemented subsystems

### Core (`orogen.v2.core`)

Deterministic seed parsing and Park–Miller RNG, 3-D simplex noise and FBM, quaternion operations, spherical geometry, Fibonacci sphere generation, stereographic projection, spherical mesh construction, spatial indexing, simulation-mesh helpers, barycentric point location, breadth-first graph operations, neighbourhood rings, a min-heap, and Dijkstra propagation in kilometres.

### Tectonics (`orogen.v2.tectonics`)

The current port includes crust-raster storage and history fields; rigid plate state, coordinate transforms and kinematics; occupancy, claim gathering and ownership reconstruction; first-pass contact resolution and boundary classification; adaptive time-step selection; orogenic-belt record propagation; mantle heat memory; supercontinent analysis and cycle tracking; force collection and a partial plate-motion update; and plume births/deaths, LIP province stamping, moving plume tracks and burst-built volcanic edifices.

The rifting subsystem now includes:
- rift-path search through continental crust, with seeded crust-grain costs and coast/shape constraints;
- partitioning a plate along the chosen path and transferring one half to a new plate;
- rift margins, failed arms, rift-shoulder relief records, plate naming and separation-push directions;
- queued plume-triggered rifts, waiting/expiry, forced overdue breakups and spontaneous tension/heat-driven rifting;
- integration of the rifting stage into `step_world` when the rift configuration is present.

The passive-margin initiation subsystem now includes:
- initiation along sufficiently old, long, non-diverging inter-plate ocean contacts;
- upper/lower plate selection from continental land share and oceanic crust age;
- trench arming and initiation events;
- creation of a new mobile oceanic plate when the passive background ocean is the downgoing side;
- a separate intra-plate old-margin founder operation (`founder_within_plates`) that awaits the upstream slow-step cadence.

Additional collision operations are available as tested standalone helpers: inter-plate crust transfer, passive-ocean plate birth, terrane accretion with fore-arc transfer, narrow-strait/inland-sea closure, and connected-landmass extraction. Arc maturation and arc scraping are also tested standalone helpers. These operations are not yet driven by a collision-event loop; collision-driven accretion/scraping and the slow cadence for arc maturation remain outstanding.

## Remaining porting work

This is not yet the complete tectonic engine. The upstream `TECTONICS` defaults and initial-world/scenario builders are not ported. Collision detection/event tracking and weld/merge, full subduction/slab evolution, event-driven terrane accretion and arc scraping, back-arc/ocean-plate evolution, the complete simulation generator, snapshots and save/load are still outstanding. The full terrain, general geology, climate, finishing and agriculture pipelines also remain to be ported. See [Porting status](docs/PORTING_STATUS.md) for the breakdown.

## Numerical backend and reproducibility

The production spherical triangulation uses SciPy/Qhull. The upstream JavaScript implementation uses Delaunator. Differential checks currently confirm matching topology for seven selected seed/resolution cases; this is not a guarantee of identical connectivity for every input.

The code preserves upstream deterministic seed and random-number conventions where implemented. Reproducibility still depends on using the same parameters and supported execution paths. See [Architecture](docs/ARCHITECTURE.md#numerical-and-reproducibility-contract) and [Testing](docs/TESTING.md).

## Validation

The latest verified push and pull-request Actions runs for commit `a293997f9142251e2ae3583e8b829b6ec8e1d5f4` (runs #629/#630) completed **349 passing tests** with **100% line coverage across 4,573 statements** in `orogen.v2`. CI enforces the 100% threshold. Coverage measures execution of implemented Python code; it does not measure feature completeness or prove whole-engine JavaScript parity.

## Upstream and license

- Upstream: [raguilar011095/planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation)
- Reference release: `v3.0.1`
- Reference commit: `ed466f9db71292e912d8d0160e689a67a994f2d9`
- License: GPL-3.0-only. See [LICENSE](LICENSE), [NOTICE.md](NOTICE.md), and [UPSTREAM_SOURCE.md](UPSTREAM_SOURCE.md).
