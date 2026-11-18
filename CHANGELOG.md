# Changelog

All notable user-visible changes to Orogen Python are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project currently has no published Python package release; entries below
describe the repository state and changes recorded during the V2 port.

## [Unreleased]

### Added
- Collision tracking with configured terrane/arc transfer, underthrusting,
  collision-belt propagation and suture welding through plate redraw/merge.
- Far-field propagation capable of reactivating old sutures and failed rifts.
- Strongly coupled Andean and flat-slab subduction margins feed the far-field deformation record, using slab-age/plate-advance classification and hysteresis.
- Added age-dependent slab rollback, controlled back-arc strip splitting, finite-duration strip steering, and rejoining with oceanic seam bridging.
- Boundary classification spreads local Andean and Laramide belts from subduction fronts, caps steep-slab arc belts, suppresses arc spreading above flat slabs, and spreads transpression belts along oblique closing land contacts.
- SciPy/Qhull-backed spherical Delaunay triangulation through
  `scipy.spatial.Delaunay`.
- Focused sphere-mesh tests for topology, reciprocal neighbours, cyclic
  adjacency, pole closure, deterministic construction and invalid Qhull output.
- JavaScript/Python differential reference tests for selected core and
  tectonic operations.
- Documentation covering architecture, core and tectonics APIs, testing,
  compatibility limits, the SciPy backend and porting status.

### Changed
- The partial `step_world` orchestrator now conditionally includes collision
  tracking and far-field evolution, and rebuilds ownership after topology changes.
- The production spherical triangulation backend is SciPy/Qhull. Orogen retains
  its north-pole closure, cyclic CSR adjacency and separate `SphereIndex`
  lookup.
- NumPy and SciPy are runtime dependencies; Numba remains optional and provides
  accelerated kernels with pure-Python fallbacks.
- The project documentation now distinguishes implemented components from
  incomplete end-to-end V2 functionality.

### Removed
- The experimental `shull` triangulation backend and the discarded cKDTree
  lookup experiment are not part of the production dependency or runtime path.

### Compatibility
- The Python port remains partial. Differential tests cover selected operations
  and fixtures; they do not establish full engine or whole-simulation parity.
- Qhull may select a different valid diagonal for degenerate point sets.
  Sphere-mesh compatibility is therefore checked through topology and
  adjacency invariants rather than byte-identical simplex ordering.
