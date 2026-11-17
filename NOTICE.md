# Attribution and provenance

This repository ports parts of the Orogen 3.0.1 engine.

- Upstream project: https://github.com/raguilar011095/planet_heightmap_generation
- Upstream release: v3.0.1
- Upstream commit: ed466f9db71292e912d8d0160e689a67a994f2d9
- Upstream license: GNU General Public License v3.0 (see LICENSE)

The Python implementation is a derivative work where it translates or adapts
upstream algorithms. Ported modules retain source references in their
documentation and should be compared against the upstream implementation.


## Runtime dependency attribution

- `shull` 0.4.0: https://github.com/schlegelp/shull
- Copyright holder: Philipp Schlegel.
- License: GNU General Public License v3.0 only (`GPL-3.0-only`), as declared
  by the dependency's `Cargo.toml` and `LICENSE`.
- `shull` is installed as a separate package dependency; its source is not
  vendored into this repository. Its licence and source distribution remain
  available from the upstream project.
