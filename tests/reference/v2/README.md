# Upstream JavaScript reference fixtures

These files are copied from the upstream repository
[raguilar011095/planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation),
tag `v3.0.1`, commit `ed466f9db71292e912d8d0160e689a67a994f2d9`.

Vendored upstream modules include:
- `js/rng.js` and `js/sphere-mesh.js`;
- `v2/js/core/seed.js`, `rotation.js`, `fast-noise.js`, `device.js`,
  `sphere-geom.js`, `sphere-index.js`, `mesh-bfs.js`, `mesh-rings.js`,
  `mesh-locate.js`, `min-heap.js`, and `sim-mesh.js`;
- `v2/js/tectonics/model/crust.js`, `plate.js`, and `world.js`.

The JavaScript implementations are retained as the comparison oracle. Only module
extensions, relative import paths, and a minimal empty TECTONICS config shim for
the isolated world-state fixture were adjusted. Algorithms are not rewritten.
Do not edit reference algorithms to make Python comparisons pass; rebase them only
when intentionally moving to a new upstream commit, and record that commit here.

The sphere-mesh differential tests install Delaunator 5.0.1 in CI and compare
Fibonacci coordinates, stereographic projection, triangle connectivity, region
adjacency, mesh metrics, and nearest-cell lookup.

The upstream project is GPL-3.0. These fixtures are included under the same
license as the Python port.
