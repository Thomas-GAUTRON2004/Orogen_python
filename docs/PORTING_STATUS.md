# Orogen 3.0.1 Porting Status

Last updated: 2026-11-17

## Target

- Upstream repository: https://github.com/raguilar011095/planet_heightmap_generation
- Release: `v3.0.1`
- Upstream commit: `ed466f9db71292e912d8d0160e689a67a994f2d9`
- Primary reference: the V2 engine under `v2/js/`

## Initial assessment

The upstream V2 engine contains approximately 210 JavaScript files and nearly
46,000 lines. The figures below estimate the porting scope; they are not Python
test-coverage measurements.

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
   comparisons against the JavaScript engine.

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

## Current implementation

| Area | Status | Current scope and remaining work |
|---|---|---|
| Python package and core | Partial | Seed/RNG, simplex noise, rotations, spherical geometry, Fibonacci generation, projection, sphere mesh, spatial indexing, SimMesh, barycentric location, BFS, rings, heap and Dijkstra are implemented. Persistence and full upstream configuration are not. |
| Tectonic state and primitives | Partial | Crust raster, rigid plates, kinematics, occupancy, claim gathering, owner reconstruction, contact geometry, subduction polarity and first-pass resolution are implemented. Initial-world generation, scenario builders and several plate operations remain absent. |
| Tectonic evolution | Partial | Contact-boundary classification, adaptive dt, orogenic record spreading, force collection, mantle heat, supercontinent-cycle state, partial plate-motion update and a partial `step_world` are implemented. The complete upstream event/slow-stage loop is not. |
| Mantle plumes | Partial | Initial/scheduled/probabilistic plume records, LIP province stamping, tail tracks, burst intervals and volcanic edifice growth are implemented and covered by tests. Queued `pendingPlumeRifts` are not executed. |
| Terrain, geology, climate | Not ported as full V2 pipelines | Existing tectonic-side height/geology helpers are not substitutes for the upstream terrain and geology engines. The climate model remains outstanding. |
| Finishing and agriculture | Not ported | Hydrology, erosion, ice, rock finishing and agriculture remain outstanding. |
| Save/load | Not ported | Upstream snapshots, migration, tree codec and archive I/O remain outstanding. |
| Compatibility | Partial | Differential tests compare selected numerical, mesh and tectonic operations against vendored JavaScript references. The latest CI does not establish whole-simulation parity. |

## Latest verification

The latest verified Actions run for commit
`bd4156f002b9aa22d813a099b671236e622f51fb` (run #467, Python tests) completed
successfully on 2026-11-17:

- **236 tests passed**.
- **3,060 statements** in `orogen.v2` were measured.
- **100% line coverage**; CI enforces this threshold.
- The test job installed Node.js 22 and Delaunator 5.0.1 and ran the Python
  and JavaScript differential suite.

Coverage is not feature completeness. Differential checks cover selected
fixtures and do not prove equivalence for all seeds, resolutions or histories.

## Next implementation target — rifting operations

The plume lifecycle already queues requests in `world.pendingPlumeRifts`, but
the stepper does not consume them. The upstream implementation is
`v2/js/tectonics/step/rifting.js` and depends on the plate-cutting operations
`ops/rift.js`, `ops/rift-path.js` and `ops/rift-partition.js`.

The port must preserve the upstream behaviour for:

- plume-triggered junction selection, waiting and expiry;
- candidate rift-path search and shape validation;
- crust partitioning and plate splitting;
- plate naming, rift events and `lastRiftAt`;
- paced spontaneous rifting under tectonic tension and mantle heat.

This is a multi-module dependency, not a missing call that can safely be added
to `step_world` in isolation. The rift operation and its geometry must be
ported and tested before queued plume rifts can be considered complete.

## Compatibility limits

The current SciPy/Qhull production triangulation has matched Delaunator for
seven selected resolution/seed pairs (12/99, 24/1, 48/17, 96/2026, 128/7 and
512/12345, with 256/42 included in the fixture set). This is evidence for those
fixtures only, not a guarantee of identical connectivity for every input.
No whole-engine or whole-history parity is claimed.
