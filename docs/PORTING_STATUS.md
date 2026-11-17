# Orogen 3.0.1 Porting Status

Last updated: 2026-11-17

## Latest porting batch

- Added collision tracking (`collision.py`): per-pair records, convergence/shortening, contact-front geometry, underthrusting, arc scraping and terrane docking. When the complete belt configuration is present, it spreads plateau-side and foreland-side orogenic belts. `step_world` schedules the stage when collision configuration is present and rebuilds ownership after crust transfers.
- Added the V2 history-ending predicates and frame selection (`ending.py`):
  `in_stage`, `history_ending`, and `frames_ending`, plus the `ENDINGS` catalogue.
- Added focused tests for supercontinent, breakup and dispersed endings,
  minimum run duration, incomplete histories, and frame trimming.
- Exported the API from `orogen.v2.tectonics`.
- This is still a partial compatibility batch: sea-gap/ring checks, plate welding/merge, far-field collision forcing and the complete simulation-stage cadence remain outstanding.

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
| Tectonic state and primitives | Partial | Crust raster, rigid plates, kinematics, occupancy, claim gathering, owner reconstruction, contact geometry, subduction polarity, first-pass resolution and boundary classification. | Initial-world generation, scenario builders, collision/event lifecycle and remaining plate operations. |
| Tectonic evolution | Partial | Adaptive dt, general orogenic-belt propagation, force collection, mantle heat, supercontinent-cycle state, partial plate-motion update and `step_world` with rift/subduction-initiation and collision-tracking stages. Collision records measure shortening/fronts, call arc/terrane accretion and underthrusting, and spread collision belts when the full belt configuration is supplied. After topology-changing transfers, the stepper rebuilds ownership and boundary state before force balance. | Full `simulate.js` stage sequence, collision sea-gap/ring checks, suture welding/merge and far-field forcing, collision-driven orogeny events, full subduction/slab evolution, arcs/back-arcs, ocean-plate evolution and snapshots. |
| Rift geometry and plate splitting | Implemented as a subsystem | `rift_partition` assigns the two land blocks and ocean crust; `rift_path` searches and scores coast-to-coast candidates; `rift.split_plate` transfers crust, wings and records to a new plate, and marks margins/failed arms. | Differential tests against full upstream plate histories and interaction with still-unported tectonic stages. |
| Rift execution | Partial, integrated | `rift_continents` consumes queued plume requests, applies waiting/expiry, supercontinent hold, overdue forced breakups and spontaneous tension/heat-driven attempts. `step_world` calls it when rift configuration is present. | Whole-simulation comparison across long histories and full upstream event interactions. |
| Passive-margin subduction initiation | Partial, integrated | `initiate_subduction` groups old, sufficiently long and non-diverging inter-plate ocean contacts, chooses the upper/lower plate, arms trench cells and can create a new mobile plate from passive background ocean. `step_world` calls it when `subductionInitRatePerMyr` is positive and rebuilds the world view after successful topology changes. | Full slab/subduction evolution, intra-plate founder cadence, arc/back-arc evolution and collision/event orchestration. |
| Arc operations | Partial, standalone | `mature_arcs` advances old arc fronts into continental crust; `scrape_arc` transfers/absorbs arc crust onto a host plate. Both have dedicated tests. | Slow-step scheduling for maturation and collision-driven scraping integration. |
| Terrane accretion and sea closure | Partial, standalone | `plate_accretion.accrete_terrane` transfers terrane and fore-arc crust to a host, `close_seas_between` seals narrow straits/small inland seas, and `landmass_of` extracts connected land. The helper is fully covered by tests. | The collision tracker now invokes arc scraping, terrane docking and underthrusting, and spreads both collision belts when their respective configuration is present. Sea-gap/ring checks, plate welding/merge and far-field forcing remain outstanding. |
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
- Maintain 100% line coverage for `orogen.v2`, enforced by the `pytest-cov` threshold in CI.

## Latest verification

The latest completed successful CI run before the collision-belt integration was run #700 on commit
`22bb2fb6de5fa07efd1e93582da572b7bdcae1e2` (2026-11-17):

- **402 tests passed**.
- `orogen/v2/tectonics/collision.py` reached **100% line coverage**.
- The complete `orogen.v2` package met its enforced **100% coverage threshold**.
- The test job ran the Python suite and JavaScript differential checks.

The belt-integration commit is being validated separately; this section is updated only to the last completed green run.

Coverage measures execution of code present; it is not feature completeness.
Differential checks cover selected fixtures and do not prove equivalence for
all seeds, resolutions or simulation histories.

## Next implementation priorities

1. Complete the collision lifecycle: sea-gap/ring checks, far-field forcing, suture welding/merge and differential history tests; then finish the subduction/arc event cadence.
2. Port ocean-plate evolution and splitting, then assemble the complete
   `simulate.js` orchestration and snapshots.
3. Port terrain, general geology, climate, finishing and agriculture as
   distinct tested batches.
4. Implement persistence and multi-step differential fixtures before claiming
   whole-engine compatibility.

## Numerical compatibility limits

The current SciPy/Qhull production triangulation has matched Delaunator for
seven selected resolution/seed pairs (12/99, 24/1, 48/17, 96/2026, 128/7,
256/42 and 512/12345). This is evidence for those fixtures only, not a
guarantee of identical connectivity for every input. No whole-engine or
whole-history parity is claimed.
