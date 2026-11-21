# Changelog

All notable user-visible changes to Orogen Python are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project currently has no published Python package release; entries below
describe the repository state and changes recorded during the V2 port.

## [Unreleased]

### Added
- Complete shared V2 tectonic configuration defaults and mechanism controls, with isolated per-world overrides and validation.
- Added the packed, noise-shaped initial supercontinent builder with connected proto-continent assembly, terranes, sutures, ocean apron, active/passive trench sectors, inherited ranges and far-field start records. Vendored JavaScript differential fixtures compare layout, landmass, assembled plate/crust state, inherited range/relief records and far-field push records on 512- and 768-cell meshes. Added deterministic prescribed-motion scenes and all 37 entries across the event, budget, force, margin, rift, terrain, hotspot and collision scenario catalogues.
- Shared geology flags and compact frame encoders/decoders matching upstream JavaScript rounding.
- Raised the required whole-package branch coverage threshold from 95 % to 99 %, with new back-arc edge-case regression tests.
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
- Initial-world configuration now rejects zero-width margins and singular
  growth settings, bounds latitude sampling, uses stable mesh-cell tie breaks,
  and reports an unmet continental-fraction target instead of returning a
  silently degraded start world.
- Initial-world and fixture top-k mesh selection now resolves ties by cell index
  while using partition-based selection to avoid sorting the entire mesh.
- Paired JavaScript/Python graph benchmarks now exercise 40,000 mesh cells.
- The partial `step_world` orchestrator now conditionally includes collision
  tracking and far-field evolution, and rebuilds ownership after topology changes.
- The production spherical triangulation backend is SciPy/Qhull. Orogen retains
  its north-pole closure, cyclic CSR adjacency and separate `SphereIndex`
  lookup.
- NumPy and SciPy are runtime dependencies; Numba remains optional and provides
  accelerated kernels with pure-Python fallbacks.
- The project documentation now distinguishes implemented components from
  incomplete end-to-end V2 functionality.
- With Numba available, `CrustRaster.create_many` clears cells block by dtype
  instead of unrolling over each of the 55 fields, with identical results. On the
  single-core benchmark host the `crust_create_snapshot_ms` median went from
  19.9 ms to 1.24 ms (6k cells, 10 rounds, 2 warm-ups). The pure-Python
  fallback is unchanged.

### Removed
- The experimental `shull` triangulation backend and the discarded cKDTree
  lookup experiment are not part of the production dependency or runtime path.

### Compatibility
- The Python port remains partial. Differential tests cover selected operations
  and fixtures; they do not establish full engine or whole-simulation parity.
- Qhull may select a different valid diagonal for degenerate point sets.
  Sphere-mesh compatibility is therefore checked through topology and
  adjacency invariants rather than byte-identical simplex ordering.
