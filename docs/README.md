# Orogen Python documentation

This documentation describes the Python code currently present in this repository. It is intended for users who want to call the implemented numerical components and developers who want to extend the port.

## Start here

1. Read [Getting started](GETTING_STARTED.md) to install the package and run a first mesh/noise example.
2. Read [Architecture](ARCHITECTURE.md) to understand the package boundaries and numerical conventions.
3. Use the [Core API reference](CORE_API.md) for mesh and numerical operations.
4. Use the [Tectonics API reference](TECTONICS_API.md) for crust and plate data structures.
5. Read [Testing and compatibility](TESTING.md) before relying on numerical equivalence with upstream.
6. Read [SciPy backend](SCIPY_BACKEND.md) for production triangulation and mesh compatibility.
7. Consult [Porting status](PORTING_STATUS.md) for the current implementation boundary.

## Scope and terminology

- **Upstream** means the JavaScript project `raguilar011095/planet_heightmap_generation`, tag `v3.0.1`.
- **V2** means the upstream second-generation engine under `v2/js/` and the corresponding Python namespace `orogen.v2`.
- **Ported** means that a Python implementation exists and has tests. It does not automatically mean complete end-to-end equivalence.
- **Differential test** means a test that executes a vendored upstream JavaScript reference and compares its result with Python.
- **Invariant test** means a test of a documented property or edge case where a direct reference comparison is not available.

## Current limitation

The V2 package is a collection of low-level components. It is not currently a turnkey world-generation API. In particular, there is no supported public function that runs a complete tectonic history and emits a finished heightmap. Do not infer that functionality from the existence of `TectonicWorld`, `Plate`, or `CrustRaster`.
