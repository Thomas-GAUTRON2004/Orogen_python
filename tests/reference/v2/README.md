# Upstream JavaScript reference fixtures

These files are copied from the upstream repository
[raguilar011095/planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation),
tag `v3.0.1`, commit `ed466f9db71292e912d8d0160e689a67a994f2d9`.

Source files:
- `js/rng.js`
- `v2/js/core/seed.js`
- `v2/js/core/rotation.js`
- `v2/js/tectonics/model/crust.js`
- `v2/js/tectonics/model/plate.js`

The JavaScript implementations are retained as the comparison oracle. Only module
extensions and relative import paths were adjusted so the isolated fixtures run
as Node ES modules from this directory. Do not edit their algorithms to make
Python comparisons pass; update them only when intentionally rebasing to a
different upstream commit, and record that commit here.

The upstream project is GPL-3.0. These fixtures are included under the same
license as the Python port.
