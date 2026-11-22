# Upstream JavaScript reference

The upstream engine is a git submodule:

- path: `external/planet_heightmap_generation`
- repository: https://github.com/raguilar011095/planet_heightmap_generation
- pinned commit: `ed466f9db71292e912d8d0160e689a67a994f2d9` (tag `v3.0.1`)

Clone the repository with `git clone --recurse-submodules`, or run
`git submodule update --init` in an existing checkout. CI checks the submodule out
recursively.

Every `.mjs` file in this directory that corresponds to an upstream module is a
**pointer**: a single `export * from` line into the submodule. Comparisons therefore
run on the upstream code itself, not on a copy. The pointers are:

- `core/*.mjs` → `v2/js/core/*.js`;
- `js/rng.mjs`, `js/simplex-noise.mjs`, `js/sphere-mesh.mjs` → `js/*.js`;
- `tectonics/config.mjs` → `v2/js/tectonics/config.js` (the full `TECTONICS` defaults);
- `tectonics/crust.mjs`, `plate.mjs`, `world.mjs`, `geology.mjs` → `v2/js/tectonics/model/*.js`;
- `tectonics/kinematics.mjs`, `occupancy.mjs` → `v2/js/tectonics/step/*.js`;
- `tectonics/init/cratons.mjs`, `landmass.mjs`, `rim-trench.mjs` → `v2/js/tectonics/init/*.js`.

Harness files (`*-runner.mjs`, `differential-*.mjs`, `noise-probe.mjs`,
`geology-encoding.mjs`, `scenes/dump_scenes.mjs`) are not upstream modules. They drive
the upstream code and may reproduce some stages, as described below. Their algorithms
must not be edited to make a Python comparison pass. To move to a new upstream commit,
update the submodule pin and this file together, then rerun the comparisons.

Upstream code is loaded as ES modules. Delaunator 5.0.1 is installed at the repository
root for the sphere-mesh and scene tests.

The initial-world runner also reproduces the inherited-range and far-field
start-push record stages from upstream `v2/js/tectonics/init/start-world.js`,
`v2/js/tectonics/step/orogeny.js`, and `v2/js/tectonics/step/far-field.js`
at the same pinned commit. It compares the generated orogeny, belt geometry,
relief, and far-field arrays; do not relax those comparisons to hide a mismatch.

The sphere-mesh differential tests install Delaunator 5.0.1 in CI and compare
Fibonacci coordinates, stereographic projection, triangle connectivity, region
adjacency, mesh metrics, and nearest-cell lookup.

The upstream project is GPL-3.0. These fixtures are included under the same
license as the Python port.
