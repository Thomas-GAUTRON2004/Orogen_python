# Upstream JavaScript reference

The upstream engine is a git submodule:

- path: `external/planet_heightmap_generation`
- repository: https://github.com/raguilar011095/planet_heightmap_generation
- pinned commit: `ed466f9db71292e912d8d0160e689a67a994f2d9` (tag `v3.0.1`)

Every `.mjs` file in this directory that corresponds to an upstream module is a
**pointer**: a single `export * from` line into the submodule. Comparisons therefore
run on the upstream code itself, not on a copy.

Reference layout:

- `core/*.mjs` → `v2/js/core/*.js`;
- `js/rng.mjs`, `js/simplex-noise.mjs`, `js/sphere-mesh.mjs` → `js/*.js`;
- `tectonics/*.mjs` → top-level `v2/js/tectonics/*.js`;
- `tectonics/model/*.mjs` → `v2/js/tectonics/model/*.js`;
- `tectonics/step/*.mjs` → `v2/js/tectonics/step/*.js`;
- `tectonics/init/*.mjs` → `v2/js/tectonics/init/*.js`;
- `tectonics/ops/*.mjs` → `v2/js/tectonics/ops/*.js`;
- `tectonics/scenarios/*.mjs` → `v2/js/tectonics/scenarios/*.js`.

The tectonics pointers mirror the complete upstream module tree at the pinned
commit, including model, simulation-step, initialization, operation and scenario
modules. Some Python modules are adaptations or orchestration layers rather than
one-to-one ports; a pointer is not by itself a claim of API equivalence.

Harness files (`*-runner.mjs`, `differential-*.mjs`, `noise-probe.mjs`,
`geology-encoding.mjs`, `scenes/dump_scenes.mjs`) are not upstream modules.
They import the pointers and call upstream functions; they do not reimplement an
upstream step. The existing tectonics differential replay opts in the Python
`plate.py`, `kinematics.py` and `world.py` modules. The focused
`differential-runner.mjs` additionally exercises crust, plate, kinematics and
occupancy semantics. New Python modules should opt in through
`# differential-module:` only when their public API and parameter types can be
replayed faithfully against a matching upstream export; use function-level
`# differential:` markers for export-name or comparison-mode exceptions.

`initial-world-layout-runner.mjs` calls `buildStartWorld` from
`init/start-world.js`; inherited ranges and far-field pushes therefore remain
upstream code. The initial-world runner compares generated orogeny, belt geometry,
relief and far-field arrays. The prescribed-motion scene harness imports the
upstream rigid-drift, Wilson-cycle and start-world-nudge constructors and motion
drivers. The mechanism override runner applies helper outputs to the upstream
`TectonicWorld` constructor.

The sphere-mesh differential tests install Delaunator 5.0.1 in CI and compare
Fibonacci coordinates, stereographic projection, triangle connectivity, region
adjacency, mesh metrics and nearest-cell lookup.

To move to a new upstream commit, update the submodule pin and this file together,
then rerun the comparisons. The upstream project is GPL-3.0; these fixtures are
included under the same license as the Python port.
