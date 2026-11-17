# Architecture and design

## Package map

The package is split into deterministic numerical primitives and stateful
tectonic operations. The current implementation includes:

- `orogen.v2.core`: seed/RNG, noise, rotations, spherical geometry and meshes,
  spatial indexing, interpolation, graph traversal, neighbourhood rings, heap
  and weighted propagation.
- `orogen.v2.tectonics`: crust and plate state, kinematics, occupancy and claim
  gathering, resolution and boundary classification, force/dynamics helpers,
  orogenic-belt spreading, mantle heat, supercontinent cycle, plume lifecycle,
  rift-path search, plate splitting and rift orchestration, passive-margin subduction initiation, ocean-plate birth, inter-plate crust transfer, standalone arc operations, and terrane-accretion/sea-closure helpers.

The exact exported names are listed in the package `__init__.py` files and
summarized in the API reference pages. Internal helpers beginning with an
underscore are not public API.

## Layer responsibilities

### Core

Core modules provide deterministic and geometry-oriented primitives. They do
not know about a complete geological scenario.

- Seed and noise modules provide reproducible scalar fields.
- Geometry and rotation modules handle unit vectors, spherical coordinates,
  and rigid rotations.
- Sphere and simulation mesh modules build the discrete planetary surface.
- Graph modules operate on mesh adjacency and provide unweighted or weighted
  propagation.

### Tectonics

Tectonics provides state structures and several working evolution subsystems,
but not yet a complete simulation engine.

- `CrustRaster` stores one record per mesh cell in a plate-local frame.
- `Plate` stores a rigid orientation, angular velocity, and its crust raster.
- Kinematics functions convert Euler-pole motion into rotations and surface
  velocities.
- Occupancy and gathering helpers map plate-local crust claims into
  world-frame cells.
- The resolver and boundary classifier rebuild ownership and classify
  contacts after plate movement.
- Force and dynamics helpers estimate plate loads and compute partial
  force-balanced motion.
- Orogeny, mantle heat, supercontinent-cycle and plume modules update their
  respective records and fields.
- The `ops` rift modules partition a candidate cut, search for a valid
  coast-to-coast path, split the plate and record new rift margins.
- `rift_continents` handles queued plume rifts, paced/overdue breakups and
  spontaneous rifting.
- `subduction_init` starts passive-margin trenches on old non-diverging contacts;
  it can carve a new mobile ocean plate from the passive background.
- `arc_maturation` and `ops.scrape_arc` provide separately tested arc
  maturation and host-plate transfer operations; their slow/collision cadence
  is not yet orchestrated.
- `ops.plate_accretion` transfers terranes and fore-arc crust onto a host,
  seals narrow straits/small inland seas, and extracts connected landmasses.
  It is a tested operation but is not yet called by a collision tracker.
- `step_world` connects the currently ported movement, resolution,
  classification, plume, cycle, rifting, subduction-initiation and motion stages.
  It rebuilds ownership/boundaries after rifts or new subduction systems before
  force balancing.

The default upstream `TECTONICS` configuration and initial-world builders are
not yet ported. The collision detection/tracking and weld/merge pipeline, event-driven terrane
accretion and arc scraping, slab/subduction propagation, arc maturation slow
cadence, back-arcs, ocean-plate evolution, terrain/geology stages, snapshots
and run/resume handling remain outstanding.

## Coordinate conventions

- Spherical positions are represented as Cartesian `(x, y, z)` vectors on the
  unit sphere unless a function explicitly takes latitude/longitude or physical
  distances.
- Latitude and longitude helpers use degrees.
- Quaternions represent rigid orientations; plate matrices are derived from
  the quaternion.
- `Plate.to_world` maps plate-frame vectors to world coordinates.
- `Plate.to_plate_frame` maps world vectors back to plate coordinates using
  the transpose of the rotation matrix.
- Angular velocities are expressed in radians per million years where used by
  the kinematics layer.
- Physical surface distances are expressed in kilometres.
- Time values in tectonic state are in millions of years (Myr), as indicated by
  the source semantics and field names.

## Mesh representation

A `SimMesh` is a structure-of-arrays object:
- `xyz`: flattened Float32 Cartesian positions; reshape to
  `(num_cells, 3)` for vectorized work.
- `adj_offset`: CSR row offsets of length `num_cells + 1`.
- `adj_list`: concatenated neighbour cell indices.
- `index`: spherical spatial index.
- `edge_rad`, `edge_km`, `cell_area_km2`: characteristic mesh metrics.

CSR storage avoids one Python list per cell and matches the compact adjacency
representation used by the upstream engine.

## Numerical and reproducibility contract

The port aims to preserve upstream semantics where they are implemented:
- Park–Miller random-number generation and seed hashing use the upstream
  constants.
- Typed NumPy arrays mirror JavaScript typed-array storage for many tectonic
  fields.
- Arithmetic order is kept close to upstream in noise and rotation code where
  practical.
- Tests compare Python outputs with vendored JavaScript references for selected
  operations.

There are also explicit implementation differences:
- `shull==0.4.0` builds the production 2-D Delaunay triangulation in Rust;
  upstream uses Delaunator. Orogen retains its own pole closure and cyclic CSR
  construction rather than using shull's non-cyclic adjacency.
- SciPy is an optional benchmark-only dependency and is not imported by the
  production package.
- Differential agreement for selected fixtures is evidence for those
  fixtures only, not a universal equivalence proof.

For deterministic work, keep the seed, parameters, mesh resolution, and random
draw order fixed. Do not assume that changing the triangulation backend or
dependency versions is output-neutral.

## Tectonic state lifecycle

The current `step_world` function is a partial one-step orchestration. It
advances plate poses, resolves world ownership, classifies contacts, updates
plumes and mantle heat, advances the supercontinent cycle, processes
continental rifting and passive-margin subduction initiation when their
configuration is present, and optionally updates plate motion. After a rift or
new subduction system it resolves ownership/boundaries again before the force
solve. It does not run the complete upstream collision/orogeny, slab evolution,
arc/back-arc, ocean-plate, slow terrain/geology or snapshot sequence. It should not be treated as a full `simulate` generator.

## Extension principles

1. Verify the upstream implementation and its dependencies before porting
   behaviour.
2. Keep Python naming idiomatic only where it does not obscure upstream
   semantics.
3. Preserve array dtype, index width, sentinel values, and mutation behaviour
   deliberately.
4. Add invariant tests for edge cases and differential tests when an
   executable reference exists.
5. Document numerical adaptations and tolerances explicitly.
6. Do not label a subsystem complete solely because its lines are covered.
