# Architecture and design

## Package map

```text
orogen/
├── __init__.py
└── v2/
    ├── __init__.py
    ├── core/
    │   ├── seed.py             # seed parsing and deterministic RNG
    │   ├── fast_noise.py       # simplex noise and FBM
    │   ├── rotation.py         # quaternion operations
    │   ├── sphere_geometry.py  # spherical/vector geometry
    │   ├── sphere_mesh.py      # point generation and closed sphere mesh
    │   ├── sphere_index.py     # spatial lookup
    │   ├── sim_mesh.py        # simulation mesh and metrics
    │   ├── mesh_bfs.py         # graph traversal and components
    │   ├── mesh_rings.py       # neighbourhood rings
    │   ├── mesh_locate.py      # barycentric point location
    │   └── min_heap.py         # heap and weighted propagation
    └── tectonics/
        ├── crust.py            # per-plate crust raster
        ├── plate.py            # rigid plate state and transforms
        ├── kinematics.py       # angular motion and velocities
        ├── occupancy.py        # spatial occupancy bins
        ├── gather.py           # plate/world claim gathering
        ├── polarity.py         # subduction polarity rules
        ├── reconstruct.py      # owner/claim reconstruction
        ├── segments.py         # ridge/fracture-zone coordinates
        ├── config.py           # physical conversion constants
        └── world.py            # mutable tectonic state container
```

The exact exported names are listed in the package `__init__.py` files and summarized in the API reference pages. Internal helpers beginning with an underscore are not public API.

## Layer responsibilities

### Core

Core modules provide deterministic and geometry-oriented primitives. They do not know about a complete geological scenario.

- Seed and noise modules provide reproducible scalar fields.
- Geometry and rotation modules handle unit vectors, spherical coordinates, and rigid rotations.
- Sphere and simulation mesh modules build the discrete planetary surface.
- Graph modules operate on mesh adjacency and provide unweighted or weighted propagation.

### Tectonics

Tectonics modules store and transform a tectonic state. They do not yet orchestrate a complete simulation step.

- `CrustRaster` stores one record per mesh cell in a plate-local frame.
- `Plate` stores a rigid orientation, angular velocity, and its crust raster.
- Kinematics functions convert Euler-pole motion into rotations and surface velocities.
- Occupancy and gathering helpers map plate-local crust claims into world-frame cells.
- `TectonicWorld` stores world arrays, plate collections, budgets, event records, and independent random streams.

## Coordinate conventions

- Spherical positions are represented as Cartesian `(x, y, z)` vectors on the unit sphere unless a function explicitly takes latitude/longitude or physical distances.
- Latitude and longitude helpers use degrees.
- Quaternions represent rigid orientations; plate matrices are derived from the quaternion.
- `Plate.to_world` maps plate-frame vectors to world coordinates.
- `Plate.to_plate_frame` maps world vectors back to plate coordinates using the transpose of the rotation matrix.
- Angular velocities are expressed in radians per million years where used by the kinematics layer.
- Physical surface distances are expressed in kilometres.
- Time values in tectonic state are in millions of years (Myr), as indicated by the source semantics and field names.

## Mesh representation

A `SimMesh` is a structure-of-arrays object:
- `xyz`: flattened Float32 Cartesian positions; reshape to `(num_cells, 3)` for vectorized work.
- `adj_offset`: CSR row offsets of length `num_cells + 1`.
- `adj_list`: concatenated neighbour cell indices.
- `index`: spherical spatial index.
- `edge_rad`, `edge_km`, `cell_area_km2`: characteristic mesh metrics.

CSR storage avoids one Python list per cell and matches the compact adjacency representation used by the upstream engine.

## Numerical and reproducibility contract

The port aims to preserve upstream semantics where they are implemented:
- Park–Miller random-number generation and seed hashing use the upstream constants.
- Typed NumPy arrays mirror JavaScript typed-array storage for many tectonic fields.
- Arithmetic order is kept close to upstream in noise and rotation code where practical.
- Tests compare Python outputs with vendored JavaScript references for selected operations.

There are also explicit implementation differences:
- SciPy/Qhull builds the production Delaunay triangulation; upstream uses Delaunator.
- NumPy and SciPy may have different low-level numerical behaviour from JavaScript typed arrays and math functions.
- Differential agreement for selected fixtures is evidence for those fixtures only, not a universal equivalence proof.

For deterministic work, keep the seed, parameters, mesh resolution, and random draw order fixed. Do not assume that changing the triangulation backend or dependency versions is output-neutral.

## Tectonic state lifecycle

The current tectonics layer provides data structures and individual operations. A typical future orchestration will need to initialize configuration and plates, gather claims, resolve boundaries and collisions, update crust histories, advance plate poses, and record events. The resolver and full time-step pipeline are not implemented yet; callers must not treat the current state container as a complete lifecycle manager.

## Extension principles

1. Verify the upstream implementation and its dependencies before porting behaviour.
2. Keep Python naming idiomatic only where it does not obscure upstream semantics.
3. Preserve array dtype, index width, sentinel values, and mutation behaviour deliberately.
4. Add invariant tests for edge cases and differential tests when an executable reference exists.
5. Document numerical adaptations and tolerances explicitly.
6. Do not label a subsystem complete solely because its lines are covered.
