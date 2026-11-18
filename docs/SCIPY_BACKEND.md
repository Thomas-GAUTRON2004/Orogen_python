# Production Delaunay backend: SciPy/Qhull

## Decision

Orogen uses `scipy.spatial.Delaunay` as its production planar triangulator.
SciPy is a required runtime dependency (`scipy>=1.10`). The project prefers
the mature Qhull-backed implementation and its established Python packaging
over the small end-to-end speed advantage measured for the previously tested
Rust-backed `shull` alternative. The difference in total mesh-build time was
modest because triangulation is only one stage of the full pipeline.

The previous `shull` integration and its comparison benchmark have been
removed. The discarded cKDTree experiment is also not part of this backend.

## Production path

1. Orogen generates Float32 Fibonacci-sphere coordinates.
2. The points are stereographically projected to contiguous Float64 planar
   coordinates.
3. `scipy.spatial.Delaunay(points).simplices` computes the planar triangles
   with Qhull.
4. Orogen validates simplex shape and vertex indices, normalizes triangle
   winding, and closes the planar convex hull with a synthetic north-pole
   vertex.
5. The Orogen cyclic CSR builder creates directed neighbour rings.
6. `SphereIndex` remains the separate spherical nearest-cell lookup used by
   `SimMesh`; it is not replaced by cKDTree.

The public Orogen mesh API and array layouts remain unchanged.

## Correctness and compatibility

The focused mesh tests cover Euler topology, reciprocal neighbours, cyclic
face orientation, deterministic builds, north-pole closure, small point counts,
and normalization of Qhull failures into the stable mesh `ValueError`
interface. Selected JavaScript differential fixtures provide finite evidence
for those inputs; they do not prove identical connectivity for every seed,
resolution or degenerate point cloud.

The `scipy sphere-mesh compatibility` workflow exercises Python 3.10, 3.12
and 3.14 on Ubuntu and Windows. macOS is intentionally omitted from this
matrix.

## Installation

SciPy is installed with the package itself, including a normal editable install:

```bash
python -m pip install -e ".[test]"
```

There is no separate triangulation backend extra to enable.
