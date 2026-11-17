# Orogen 3.0.1 Porting Status

Last updated: 2026-11-16

## Target

- Upstream repository: https://github.com/raguilar011095/planet_heightmap_generation
- Release: `v3.0.1`
- Upstream commit: `ed466f9db71292e912d8d0160e689a67a994f2d9`
- Primary reference: the V2 engine under `v2/js/`

## Initial assessment

The existing Python package is version 1.1.0. It already provides APIs for
world, terrain, geology, climate, and environment generation, but its tectonic
evolution engine is a simplified implementation and does not reproduce the
full Orogen 3.0.1 JavaScript V2 engine.

The upstream V2 engine contains approximately 210 JavaScript files and nearly
46,000 lines. The main areas identified during the initial audit are:

| Area | Files | Approximate lines |
|---|---:|---:|
| Tectonics | 63 | 10,239 |
| Terrain | 17 | 4,661 |
| Climate | 16 | 4,200 |
| Geology | 8 | 2,227 |
| Finishing | 13 | 3,998 |
| Agriculture | 4 | 854 |
| Core | 11 | 683 |
| Save/load | 5 | 519 |

These figures estimate the porting scope; they are not Python test-coverage
measurements.

## Implementation plan

The port is organized into functional batches, with a Git commit for each
verifiable increment:

1. **Python foundation:** package structure, versioning, configuration,
   numerical conventions, and deterministic seed handling.
2. **Core and persistence:** shared types, serialization, loading, and data
   validation.
3. **Tectonics:** plate initialization, geometry, boundaries, velocities,
   operations, and scenarios.
4. **Evolution:** time-stepping engine, tectonic events, rotations, and
   snapshots.
5. **Terrain:** relief, crust, lithology, rifts, ridges, and mountain ranges.
6. **Geology:** records, collisions, subduction, volcanism, and deposits.
7. **Climate:** circulation, temperatures, precipitation, and biomes.
8. **Finishing and agriculture:** hydrology, erosion, ice, rocks, and
   agricultural systems.
9. **Compatibility:** deterministic fixtures, invariant tests, and numerical
   comparisons against the JavaScript engine. An initial differential harness
   already covers seed/RNG, quaternions, crust rasters, and plates; full mesh
   and simulation comparisons remain outstanding.

The order may change as actual dependencies between modules become clearer.

## Validation rules

- Do not mark a feature as ported until it has Python tests.
- Distinguish exact ports from numerical adaptations and approximations.
- Compare outputs using identical seeds, parameters, meshes, and simulation
  times.
- Document numerical tolerances; do not hide discrepancies with overly broad
  thresholds.
- Keep invariant tests even when exact differential comparison is not possible.
- Do not claim full compatibility until all functional areas and reference
  tests have been covered.
- Maintain 100% line coverage for `orogen.v2`, enforced by the
  `pytest-cov` threshold in CI.

## Known initial blockers

The upstream V2 test scripts depend on Node.js packages, including
`delaunator`. Initial attempts to run the reference checks failed before the
checks started because this dependency was missing. CI now installs
Delaunator 5.0.1 explicitly.

## Batch history

| Batch | Status | Notes |
|---|---|---|
| Upstream scope audit | Complete | V2 source tree inspected; domains and approximate sizes recorded. |
| Python 1.1.0 package audit | Complete | Existing implementation identified; full compatibility not established. |
| Python 3.0.1 foundation | In progress | Published package namespace, RNG/seed handling, simplex noise, quaternions, spherical geometry, BFS, spherical indexing, SimMesh, barycentric location, min-heap, Dijkstra, neighbourhood rings, Euler-pole frame, small-device classification, crust raster, rigid-plate model, tectonic-world state container (occupancy/boundary arrays, budgets, events, independent RNG streams), and cached world-frame plate-contact geometry. The world still requires TECTONICS configuration supplied by its caller. **107 tests pass; `orogen.v2` line coverage is 100% (1,276/1,276 statements).** CI now enforces the 100% threshold. |
| V2 core | In progress | Numerical primitives, noise, rotations, spherical geometry, Fibonacci generation, stereographic projection, north-pole mesh closure, SciPy/Qhull Delaunay triangulation, spatial indexing, interpolation, graph and neighbourhood operations. The production backend is SciPy/Qhull. Comparison with Delaunator confirms identical triangles and neighbourhoods for seven resolution/seed pairs (12/99, 24/1, 48/17, 96/2026, 128/7, 256/42, 512/12345). This result is limited to those cases and does not guarantee all seeds or resolutions. |
| V2 domain porting | In progress | Tectonic primitives now include claim gathering, polarity, ownership reconstruction, occupancy bins, ridge-segment coordinates, and cached boundary-contact geometry. The complete tectonic step pipeline remains unported. |
| Compatibility tests | Partial | Differential tests pass against the vendored upstream JavaScript sources; mesh connectivity matches for seven cases. Dynamic TectonicWorld steps and multi-step simulations still need comparison. No claim of full compatibility is made. |

## Next implementation target

The boundary-contact helper from upstream `tectonics/step/contact.js` is now ported as `orogen.v2.tectonics.contact.contact_at`. It computes the foreign-neighbour normal, selects the across-boundary cell, handles degenerate sliver contacts, and caches results against `occupancyStamp`.

The next major tectonics batch remains the upstream `tectonics/step/resolve.js`
resolver. It must be ported with its actual dependencies and reference
semantics, then covered by focused unit and differential tests. The contact
helper is a prerequisite for later boundary classification and force calculations; it does not itself resolve collisions or evolve crust. Expected areas
include plate-claim conflicts, subduction, divergent crust creation, and
provisional-cell ownership; these behaviours must be confirmed from the
upstream source before implementation.


The first resolver pass is now implemented as `orogen.v2.tectonics.resolve.resolve`. It includes claim conflict settlement, oceanic subduction, divergent-gap candidate evaluation and crust creation, fracture-zone coordinates, trench-apron inheritance, and provisional gap ownership. It does not complete the rest of the tectonic step pipeline: boundary classification, force/velocity response, event generation, and simulation orchestration remain outstanding. Differential and invariant coverage of the resolver is being expanded.


Boundary processing has started: `orogen.v2.tectonics.boundaries` now contains the pure motion classifier, connected-segment polarity voting, apron-near-land query, and trench-feed bookkeeping. The integrated boundary pass remains outstanding because it depends on orogeny spreading and far-field deformation, which have not yet been ported.


The boundary module now also exposes `classify_contact_boundaries(world, dt)`, which classifies resolved contacts as ridge, transform, subducting, overriding, or collision and updates convergence, trench flags, slab consumption, and optional segment polarity. It deliberately does not claim full upstream boundary-pass parity: orogeny spreading, flat-slab logic, and far-field deformation remain unported.


Adaptive time-step calculation has also been ported into `orogen.v2.tectonics.simulation`: fastest cap speed, two-plate closing-rate bound, and `choose_step`. The full `simulate.js` loop is still outstanding because it orchestrates multiple unported event, force, terrain, geology, and snapshot stages.


Orogeny porting now includes the spatial `spread_belt` record-propagation pass: source-specific belt reach/work, weak-line weighting, craton-rim limits, cell allow-filters, overprint history, belt geometry/shear, and crustal thickening. The height-building and sediment-cover side effects (`buildHeights` and `thickenCover`) remain outstanding, so this is not full `spreadBelt` parity.


Force-balance progress: `dynamics.py` now ports plate-frame basal drag transformed into the world frame, the 3x3 solve, slab-resistance tensor, collision-normal/shear coupling, smooth speed softening and hard speed cap. These functions accept precomputed loads. The world-wide force-gathering pass, which discovers slab, young-ridge, rift and mantle-flow forces from the resolved plate map, remains outstanding.


Boundary force gathering has begun: `forces.py` now collects slab pull, slab suction, ridge push and partner-rift push from the resolved owner map, including shared-edge length, age/maturity scaling and slab-resistance records consumed by `dynamics.py`. It now also records land-overlap collision contacts, estimates front normals from nearby uncontested land (with a plate-cap fallback), applies craton lock weighting, and excludes fresh rift-margin jitter and arc contacts. Mantle traction and dispersal repulsion are now evaluated by mantle.py and accumulated in the fifth force channel when the world supplies a cycle state. The cycle state producer (landmass cohesion/detection, heat history, seeded upwelling/downwelling and phase clock) remains unported.


A partial `update_plate_motion(world)` now connects the currently ported boundary force collector to the drag/coupling solver. It uses a common start-of-step angular-velocity snapshot for all plate solves, applies soft/hard speed limits, relaxes the target using `lastDt / relaxMyr`, and records force diagnostics. It does not claim complete upstream plate motion because collision contacts and mantle-flow/repulsion forces are not yet gathered.


The cohesive landmass analyser is now ported in `supercontinent.py`. It counts connected continental components above 1 Mkm², separates interiors across necks narrower than `cycleNeckKm`, grows inland cores back through the land by geodesic distance, and reports the largest mass and its share. The cycle clock/transition logic that consumes this result remains outstanding.
