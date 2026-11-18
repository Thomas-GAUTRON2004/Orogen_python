# Orogen Python

A progressive Python port of the **Orogen 3.0.1** procedural planetary simulation engine. The implementation is based on the upstream [planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation) repository, release `v3.0.1`.

> **Project status:** selected core and tectonic subsystems are implemented and tested, including rifting, passive-margin subduction initiation, collision tracking and suture welding. These pieces are integrated into a partial tectonic stepper, not the complete upstream simulation pipeline. This repository does not yet provide end-to-end generation of a finished planet or heightmap.

## Documentation

- [Documentation home](docs/README.md) — navigation and project scope.
- [Changelog](CHANGELOG.md) — user-visible changes, fixes and compatibility notes.
- [Getting started](docs/GETTING_STARTED.md) — installation and first operations.
- [Architecture](docs/ARCHITECTURE.md) — package structure, data flow, and design decisions.
- [Core API](docs/CORE_API.md) — random numbers, noise, meshes, geometry, and graph algorithms.
- [Tectonics API](docs/TECTONICS_API.md) — crust, plates, rifting, and tectonic operations.
- [Testing and compatibility](docs/TESTING.md) — test commands, differential tests, and compatibility limits.
- [Porting status](docs/PORTING_STATUS.md) — implemented, partial, and outstanding subsystems.
- [Performance benchmarks](docs/JS_PYTHON_BENCHMARK.md) — numerical-kernel benchmark methodology and limitations.
- [SciPy backend](docs/SCIPY_BACKEND.md) — production triangulation, mesh validation, and numerical compatibility.

## Requirements and installation

**Required runtime dependencies:**
- Python 3.10 or later
- NumPy 1.24 or later
- SciPy 1.10 or later (Qhull-backed Delaunay triangulation)

**Optional acceleration dependency:**
- Numba 0.59 or later — enables the JIT-compiled performance kernels. It is not required to use Orogen; the package falls back to its pure-Python implementations when Numba is not installed.

Install the package and test dependencies (without Numba):

```bash
python -m pip install -e ".[test]"
```

Install with Numba acceleration enabled:

```bash
python -m pip install -e ".[test,speed]"
```

SciPy is a required runtime dependency and supplies the production Qhull-backed Delaunay triangulation. The `test` extra adds pytest and coverage tools; the optional `speed` extra installs `numba>=0.59` for JIT-compiled kernels.

The JavaScript differential reference tests additionally require Node.js 22+ and Delaunator 5.0.1:

```bash
npm install --no-save --package-lock=false delaunator@5.0.1
python -m pytest
```

See [Testing and compatibility](docs/TESTING.md) for the full test and coverage commands.

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

### Core — `orogen.v2.core`

Deterministic seed parsing and Park–Miller RNG; 3-D simplex noise and FBM; quaternion operations; spherical geometry; Fibonacci sphere generation; stereographic projection; spherical mesh construction and spatial indexing; simulation-mesh helpers; barycentric point location; breadth-first graph operations and neighbourhood rings; a min-heap; and Dijkstra propagation in kilometres.

### Tectonics — `orogen.v2.tectonics`

Implemented tectonic building blocks include:

- **State and geometry:** crust-raster storage and history fields, rigid plate state, coordinate transforms and kinematics, occupancy, claim gathering, ownership reconstruction, contact geometry and boundary classification.
- **Forces and long-term state:** adaptive time-step selection, force collection, partial plate-motion updates, mantle heat memory, supercontinent analysis/cycle tracking, orogenic-belt propagation and history-ending/frame-selection helpers.
- **Mantle plumes:** plume births/deaths, LIP province stamping, moving plume tracks and burst-built volcanic edifices.
- **Rifting:** seeded coast-to-coast rift-path search, plate partition/splitting, rift margins and failed arms, shoulder-relief records, queued plume-triggered rifts, waiting/expiry, overdue breakups and spontaneous tension/heat-driven rifting. The rifting stage is integrated into `step_world` when configured.
- **Passive-margin subduction initiation:** detection of sufficiently old, long and non-diverging ocean contacts; upper/lower plate selection; trench arming and initiation events; and creation of a mobile oceanic plate when passive background ocean is the downgoing side. The separate intra-plate old-margin founder helper exists, but its upstream slow-step cadence is not yet integrated.
- **Collision and sutures:** per-pair collision tracking, convergence/shortening and contact-front geometry, underthrusting, arc scraping and terrane docking, collision-belt propagation, far-field reactivation of old sutures/failed rifts, and configured suture closure through plate redraw/merge. Ownership and boundary state are rebuilt after topology-changing operations.

Collision tracking invokes configured arc scraping, terrane docking/accretion, underthrusting, collision-belt propagation and suture welding. Far-field evolution can reactivate old sutures and failed rifts when configured. Arc maturation remains a tested standalone helper whose upstream slow-stage schedule is not yet integrated. These integrations do not mean the full upstream event scheduler or all long-term tectonic interactions have been ported.

## Remaining porting work

The implementation is not the complete V2 engine. Important outstanding areas include:

- upstream tectonic defaults and initial-world/scenario builders;
- the complete `simulate.js` orchestration and stage cadence;
- full subduction and slab evolution, ocean-plate evolution/splitting, back-arc evolution and remaining event-history parity;
- full terrain, general geology, climate, finishing and agriculture pipelines;
- snapshots, persistence and save/load;
- multi-step, multi-seed and whole-simulation differential validation.

See [Porting status](docs/PORTING_STATUS.md) for the subsystem-by-subsystem breakdown. The package should currently be treated as a set of usable numerical and tectonic components, not as a turnkey planet-generation application.

## Numerical backend and reproducibility

Production spherical triangulation uses `scipy.spatial.Delaunay` (Qhull). The planar simplices are closed at the north-pole region and converted to Orogen's cyclic CSR adjacency. `SphereIndex` remains the separate spherical nearest-cell lookup; cKDTree is not used. Qhull can choose different valid diagonals for degenerate or near-degenerate point sets, so compatibility is checked through mesh invariants and selected differential fixtures rather than requiring byte-identical simplex ordering.

The code preserves upstream deterministic seed and random-number conventions where implemented. Reproducibility depends on using the same parameters and supported execution paths. Numba is optional and has a pure-Python fallback. See [Architecture](docs/ARCHITECTURE.md#numerical-and-reproducibility-contract) and [Testing](docs/TESTING.md).

## Validation

SciPy/Qhull sphere-mesh tests are part of the standard Python test suite,
which runs alongside the JavaScript/Python differential tests in the main
[quality PR #9](https://github.com/Thomas-GAUTRON2004/Orogen_python/pull/9).
There is no separate SciPy compatibility matrix.

## Upstream and license

- Upstream: [raguilar011095/planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation)
- Reference release: `v3.0.1`
- Reference commit: `ed466f9db71292e912d8d0160e689a67a994f2d9`
- License: GPL-3.0-only. See [LICENSE](LICENSE), [NOTICE.md](NOTICE.md), and [UPSTREAM_SOURCE.md](UPSTREAM_SOURCE.md).
