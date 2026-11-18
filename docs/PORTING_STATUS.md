# Orogen 3.0.1 Porting Status

Last updated: 2026-11-18

## Latest porting batch

- Added collision tracking and suture welding (`collision.py`): per-pair records, convergence/shortening, contact-front geometry, underthrusting, arc scraping and terrane docking. With the full belt configuration it spreads plateau-side and foreland-side orogenic belts; with the full suture configuration it samples the sea gap, tracks its best closure and measures enclosed sea. Eligible pairs run the new `ops.merge_plates.merge_plates` redraw, retain the upper plate, retire the lower plate, and emit a `SUTURE` event. `step_world` rebuilds ownership after the merge.
- Added the V2 history-ending predicates and frame selection (`ending.py`):
  `in_stage`, `history_ending`, and `frames_ending`, plus the `ENDINGS` catalogue.
- Added focused tests for supercontinent, breakup and dispersed endings,
  minimum run duration, incomplete histories, and frame trimming.
- Exported the API from `orogen.v2.tectonics`.
- Added far-field weak-line reactivation (`far_field.py`): old sutures and failed rifts can seed rootless intraplate belts when a sufficiently strong push reaches them far from the front. Quiet pre-existing ranges are rejuvenated along their original belt geometry instead of receiving a new belt.
- Added tests for strike estimation, orientation, distance taper, threshold work, old-range rejuvenation, ocean/craton exclusions and integration with the far-field clock.
- Far-field forcing from strongly coupled Andean/flat-slab margins, full collision/event-history parity and the complete simulation-stage cadence remain outstanding.

## Performance optimization (Numba)

- Added an optional `speed` extra and retained a pure-Python fallback.
- Numba compiles batched 3-D simplex FBM, CSR BFS, callback-free mesh Dijkstra, and the weighted far-field traversal used by tectonic `spread_push` (`fastmath=False`). FBM is checked bit-for-bit against Python and upstream JavaScript; BFS retains FIFO/source order and int16 semantics; Dijkstra preserves heap ordering and float32/float64 distance storage.
- Mesh Dijkstra retains the Python path when `can_enter` or `on_reach` callbacks are provided. Far-field diffusion keeps crust-history recording in Python and retains the complete Python graph-traversal fallback. Differential tests compare reached order, source assignment, distance arrays and crust history.
- The latest paired CI sample measured FBM near parity, a large speedup for synthetic mesh Dijkstra, and a **3.35×** full-path speedup for synthetic far-field diffusion versus its Python fallback. Geometry and per-cell crust snapshots remain slower. See `docs/JS_PYTHON_BENCHMARK.md` for method, results and runner variability.
- Scalar JIT wrappers for spherical geometry and per-cell crust resets were benchmarked and removed because call overhead regressed performance. Further work must batch meaningful workloads and retain differential parity.

## Target

- Upstream repository: https://github.com/raguilar011095/planet_heightmap_generation
- Release: `v3.0.1`
- Upstream commit: `ed466f9db71292e912d8d0160e689a67a994f2d9`
- Primary reference: the V2 engine under `v2/js/`

## Initial scope

The upstream V2 engine contains approximately 210 JavaScript files and nearly
46,000 lines. These estimates describe the source-porting scope, not Python
test coverage.

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

## Current implementation

| Area | Status | Implemented scope | Remaining work |
|---|---|---|---|
| Python package and core | Partial | Seed/RNG, simplex noise, rotations, spherical geometry, sphere generation/indexing, SimMesh, interpolation, BFS, rings, heap and Dijkstra. | Persistence, full upstream configuration and remaining shared data models. |
| Tectonic state and primitives | Partial | Crust raster, rigid plates, kinematics, occupancy, claim gathering, owner reconstruction, contact geometry, subduction polarity, first-pass resolution and boundary classification. | Initial-world generation, scenario builders, remaining collision-event parity and remaining plate operations. |
| Tectonic evolution | Partial | Adaptive dt, general orogenic-belt propagation, force collection (slab, rift/ridge classification and torque accumulation are now separate helpers), collision tracking (suture readiness, gap/ring checks and configured welding are isolated), mantle heat, supercontinent-cycle state, partial plate-motion update and `step_world` with rift/subduction-initiation and collision-tracking stages. `spread_belt` now separates front propagation, previous-record preservation, front metadata updates and per-cell geological recording; `rift_continents` separates schedule state, queued plume requests, overdue breakups and spontaneous attempts. Their focused characterization tests remain green. Collision records measure shortening/fronts, call arc/terrane accretion and underthrusting, spread collision belts when the full belt configuration is supplied, and feed shortening into far-field records. Far-field diffusion can reactivate distant quiet sutures/failed rifts and rejuvenate existing ranges. With complete suture and merge configuration, eligible pairs run the raster redraw, sea sealing, suture flags and lower-plate retirement. After topology-changing transfers, the stepper rebuilds ownership and boundary state before force balance. | Full `simulate.js` stage sequence, far-field forcing from strongly coupled Andean/flat-slab margins, complete collision-driven orogeny/event history, full subduction/slab evolution, arc maturation/back-arcs, ocean-plate evolution and snapshots. |
| Rift geometry and plate splitting | Implemented as a subsystem | `rift_partition` assigns the two land blocks and ocean crust; `rift_path` searches and scores coast-to-coast candidates; `rift.split_plate` transfers crust, wings and records to a new plate, and marks margins/failed arms. | Differential tests against full upstream plate histories and interaction with still-unported tectonic stages. |
| Rift execution | Partial, integrated | `rift_continents` consumes queued plume requests, applies waiting/expiry, supercontinent hold, overdue forced breakups and spontaneous tension/heat-driven attempts. `step_world` calls it when rift configuration is present. | Whole-simulation comparison across long histories and full upstream event interactions. |
| Passive-margin subduction initiation | Partial, integrated | `initiate_subduction` groups old, sufficiently long and non-diverging inter-plate ocean contacts, chooses the upper/lower plate, arms trench cells and can create a new mobile plate from passive background ocean. `step_world` calls it when `subductionInitRatePerMyr` is positive and rebuilds the world view after successful topology changes. | Full slab/subduction evolution, intra-plate founder cadence, arc/back-arc evolution and collision/event orchestration. |
| Arc operations | Partial, standalone | `mature_arcs` advances old arc fronts into continental crust; `scrape_arc` transfers/absorbs arc crust onto a host plate. Both have dedicated tests. | Slow-step scheduling for arc maturation and whole-history differential validation. |
| Terrane accretion and sea closure | Partial, standalone | `plate_accretion.accrete_terrane` transfers terrane and fore-arc crust to a host, `close_seas_between` seals narrow straits/small inland seas, and `landmass_of` extracts connected land. The helper is fully covered by tests. | The collision tracker invokes arc scraping, terrane docking and underthrusting, spreads both collision belts, records sea-gap/ring readiness and executes `merge_plates` when a pair is eligible and the full redraw configuration is supplied. Far-field forcing and whole-history event parity remain outstanding. |
| Mantle plumes | Partial | Initial/scheduled/probabilistic plume records, LIP province stamping, tail tracks, burst intervals and volcanic-edifice growth. | Full coupling to the remaining simulation/event pipeline and whole-history comparison. |
| Terrain, geology, climate | Not ported as full V2 pipelines | Some tectonic-side height and sediment-cover helpers exist. | Full terrain relief/lithology, geology/deposition, circulation, temperature, precipitation and biome pipelines. |
| Finishing and agriculture | Not ported | — | Hydrology, erosion, ice, rocks, finishing and agricultural systems. |
| Save/load | Not ported | — | Snapshots, migrations, tree codec and archive I/O. |
| Compatibility | Partial | Differential tests compare selected numerical, mesh and tectonic operations against vendored JavaScript references. | Multi-seed, multi-resolution, multi-step and whole-simulation parity. |

## Validation rules

- Do not mark a feature as ported until it has Python tests.
- Distinguish exact ports from numerical adaptations and approximations.
- Compare outputs using identical seeds, parameters, meshes and simulation times.
- Document numerical tolerances; do not hide discrepancies with overly broad thresholds.
- Keep invariant tests even when exact differential comparison is not possible.
- Do not claim full compatibility until all functional areas and reference tests have been covered.
- Maintain the CI coverage gate for `orogen` (95% total minimum with branch coverage enabled).

## Latest documented verification

The latest PR-head source validation on 2026-11-18 completed successfully:

- **506 tests passed**.
- **99.43% total coverage** for `orogen.v2` (CI minimum: 95%, branch coverage enabled).
- Python tests: [GitHub Actions run 44263575404](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44263575404).
- Platform matrix: [run 44263575402](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44263575402), 9/9 jobs passed.
- A/B pipeline benchmark: [run 44263575444](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44263575444), structural comparisons passed at all three sizes.
- Validated source revision: `b5063ddaa0694d03ea170c78b6076e39818998b2`.

Rerun the workflows after subsequent source changes.

Coverage measures execution of code present; it is not feature completeness.
Differential checks cover selected fixtures and do not prove equivalence for
all seeds, resolutions or simulation histories.

## Next implementation priorities

1. Add far-field forcing from strongly coupled Andean/flat-slab margins and validate complete collision/suture histories differentially; then finish subduction and arc evolution scheduling.
2. Port ocean-plate evolution and splitting, then assemble the complete
   `simulate.js` orchestration and snapshots.
3. Port terrain, general geology, climate, finishing and agriculture as
   distinct tested batches.
4. Implement persistence and multi-step differential fixtures before claiming
   whole-engine compatibility.

## Numerical compatibility limits

Production triangulation uses `shull>=0.4.0`. The selected seeded differential fixtures and mesh-invariant tests exercise finite inputs only; they do not guarantee universal connectivity across all point clouds, seeds or resolutions. The optional SciPy-versus-shull benchmark is a finite A/B comparison, not proof of whole-engine parity.
## Quality refactoring progress

- Continued Phase 2 by extracting the identification of old, untrenched oceanic coast cells and their connected margin runs from `founder_within_plates` into `_old_ocean_margin_runs`. The helper isolates a pure graph/raster stage from the breakup eligibility and plate-transfer orchestration; behavior is intended to remain unchanged. Focused CI validation is pending.
