# Tectonics API reference

Import tectonic components from `orogen.v2.tectonics`. This namespace exposes
state models and tested individual operations, including a partial
one-step orchestrator. It is not the full upstream tectonic evolution engine.

## Enumerations and flags

The port represents upstream enumerations as dictionaries of string keys to
integer values.

### Crust types

`CrustType`:
- `OCEAN`: 0
- `CONTINENT`: 1
- `CRATON`: 2
- `ARC`: 3

### Other classifications

- `OrogenyType`: NONE, ANDEAN, LARAMIDE, URAL, HIMALAYAN, ARC, RIFT,
  TRANSPRESSION, INTRAPLATE.
- `VolcanicType`: NONE, TRACK, LIP.
- `BeltSide`: NONE, UPPER, LOWER.
- `CrustFlag`: bit flags including TRENCH, SUTURE, FAILED_RIFT, RIFT_MARGIN,
  LIP, APRON, MARGIN_ISLE, and EROSIVE.
- `BoundaryKind`, `EventKind`, and `Provisional` are defined in `world.py`
  and exported from the package.

Use the named constants instead of embedding numeric values in application
code.

## CrustRaster

`CrustRaster(num_cells)` stores crust in a structure-of-arrays layout for one
plate-local mesh.

Important attributes:
- `numCells`: number of cells.
- `mask`: Uint8-like occupancy mask.
- `count`: number of occupied cells.
- Per-cell arrays: `type`, `flags`, `formedAt`, `terrane`, `slabKm`,
  `thickKm`, orogeny and belt fields, volcanic fields, cover fields, and other
  history attributes.
- `fields`: ordered list of field arrays used by snapshot/copy operations.

Main methods:
- `create(cell, crust_type, formed_at, terrane=0)`
- `copy_from(cell, other, source, rot=None)`
- `snapshot(cell)` and `restore(cell, values)`
- `remove(cell)`
- `arm_trench(cell, slab_km)`
- `record_subduction(cell, km)`
- `record_orogeny(cell, oro_type, work_km, time_myr, quiet_myr)`

Example:

```python
from orogen.v2.tectonics import CrustRaster, CrustType

crust = CrustRaster(num_cells=100)
crust.create(cell=12, crust_type=CrustType["OCEAN"], formed_at=-25.0)
snapshot = crust.snapshot(12)
crust.arm_trench(12, slab_km=80.0)
crust.restore(12, snapshot)
```

The arrays are mutable. Directly editing `mask` or other fields can make the
cached `count` inconsistent; prefer the methods when changing occupancy.

## Plate

`Plate(plate_id, num_cells, born=0.0)` stores a rigid plate pose and a
`CrustRaster`.

Key attributes include `id`, `name`, `alive`, `born`, `quat`, `mat`,
`omega`, `crust`, `stats`, and spatial culling-cap fields.

Methods:
- `set_quat(quaternion)` updates the quaternion and its rotation matrix.
- `to_world(x, y, z)` transforms a plate-frame vector into world coordinates.
- `to_plate_frame(x, y, z)` applies the inverse rotation.
- `cell_at(index, x, y, z)` transforms a world vector into plate coordinates
  and queries the supplied spatial index.
- `cap_in_world(mesh)` returns the plate extent cap in world coordinates.
- `mark_reshaped()` marks the plate as changed.
- `refresh_extent(xyz)` recomputes occupied-cell statistics and the bounding
  cap.

`frame_rotation(source, target)` returns the rotation matrix from source
plate coordinates into target plate coordinates.

## Kinematics

The kinematics module exports:

- `lat_lon_to_xyz(lat_deg, lon_deg)`: degrees to unit Cartesian coordinates.
- `xyz_to_lat_lon(x, y, z)`: unit Cartesian coordinates to degree-valued
  latitude/longitude.
- `omega_from_euler_pole(pole_xyz, speed_cm_yr)`: Euler-pole vector and
  angular speed conversion.
- `surface_velocity(omega, px, py, pz, out=None)`: local surface velocity
  vector.
- `speed_cm_yr_at(omega, px, py, pz)`: local speed in centimetres per year.
- `advance_plates(world, dt)`: integrate the pose of each living plate and
  advance world time/step.
- `pose_at_constant_omega(q0, omega, t)`: evaluate a quaternion pose at
  constant angular velocity.

Conversion constants:
- `CM_PER_YR_TO_KM_PER_MYR = 10.0`
- `CM_PER_YR_TO_RAD_PER_MYR = 10.0 / 6371.0`

The radius used by these conversions is 6,371 km.

## Occupancy and claim operations

- `BinIndex`, `bin_index`, `mark_bin`, and `may_have_crust_near`
  implement spatial occupancy bins.
- `clear_world_view` and `for_each_claim` gather plate crust claims into the
  world frame.
- `overrides` applies the currently ported subduction-polarity decision
  rules.
- `reconstruct` rebuilds ownership and claim counters from the available
  plate data without mutating crust.

These functions are low-level. They do not resolve all plate collisions,
create a complete tectonic boundary network, or execute a full simulation
step.

## TectonicWorld

`TectonicWorld(sim_mesh, seed, config, overrides=None)` creates a mutable
world-state container.

The constructor requires an explicit configuration mapping. The upstream
`TECTONICS` default configuration has not yet been ported; there is no
supported built-in default to substitute here.

The world contains:
- mesh reference, seed, configuration, time, and step counters;
- a list of plates and event records;
- independent deterministic random streams for separate tectonic mechanisms;
- per-cell ownership, claim, collider, boundary, convergence, and
  provisional-state arrays;
- budgets and auxiliary collections for collisions, arc scrapes, hotspots,
  and plumes.

Methods include `add_plate(name)`, `next_terrane_id()`,
`name_by_cratons(plate, fallback)`, and `log(kind, text, data=None)`.

The container does not initialize a physically complete planet, resolve
boundaries, or run the upstream tectonic step pipeline. Supply a configuration
only if you know the fields expected by the specific downstream operation you
are integrating.

## Ridge segments

- `PLANET_RADIUS_KM = 6371.0`
- `SEGMENT_KM = 400.0`
- `fracture_coordinate(px, py, pz, omega)` computes a ridge-floor coordinate
  in segment lengths about a spreading pole. It returns 0.5 for a near-zero
  angular-velocity vector; zero is reserved for cells without a recorded
  coordinate.

## Rift operations

The rift helpers are available from `orogen.v2.tectonics.ops`.

### Partition and score a proposed rift

Module: `orogen.v2.tectonics.ops.rift_partition`

- `masked_cells(crust)` returns occupied plate cells in index order.
- `partition(mesh, crust, path, cells=None)` marks left/right land blocks
  around a proposed rift path and propagates oceanic crust to the nearest
  block. It returns a side array (1 or 2), land-cell counts and unique craton
  counts.
- `sides_of(mesh, crust, path, cells=None)` summarizes the smaller land-block
  share, maximum block elongation and whether both sides retain cratons.
- `exterior_sea(mesh, crust)` marks ocean connected to the outside of the
  plate, or the largest ocean basin when the plate covers the whole mesh.
- `elongation_of(mesh, side, which, cells=None)` estimates a block's
  long/short axis ratio from position covariance.

### Find a coast-to-coast path

Module: `orogen.v2.tectonics.ops.rift_path`

- `grain_of(world)` returns a seeded, cached crust-grain multiplier.
- `rift_context(world, plate)` caches the plate cells, exterior sea,
  distance from the outer coast and grain field for repeated junction trials.
- `find_rift(world, plate, junction, pull_axis=None,
  need_cratons_both_sides=True, ctx=None)` scores coast-to-coast paths from
  one junction. Cratons cannot be traversed; old sutures/failed-rift margins
  receive lower costs. Candidate paths are checked for coast-hugging,
  overhang, block share, craton retention and block elongation.

The result contains `path`, `failed_arm`, three `arms`, the `across`
normal and a comparative `value`. A result of `None` means that no
candidate passed the configured constraints.

### Split a plate and execute a rift

Module: `orogen.v2.tectonics.ops.rift`

- `choose_junction(world, plate, rng)` selects a weighted interior land cell,
  favouring weak lines and excluding cratons.
- `plume_junctions(world, plate, xyz)` samples eligible land cells near a
  plume head.
- `split_plate(world, plate, path, failed_arm, name_for, across=None)`
  transfers one partition to a new `Plate`, copies pose and crust records,
  moves wing tips, flags rift margins/failed arms, spreads rift-shoulder
  records, updates rift partners and calculates separation-push directions.
- `try_rift(world, plate, junctions, cause,
  need_cratons_both_sides=True)` tests the supplied junctions, chooses the
  lowest-valued valid cut, splits the plate and records the rift event.
- `pull_axis_of(plate, mesh)` estimates the principal axis of trench pull.
- `names_after_split(world, plate_a, plate_b, before)` uses craton letters
  where distinct, otherwise allocates unused numbered names.

### Run the rifting stage

Module: `orogen.v2.tectonics.rifting`

- `pace_of(age, config)` computes the age-dependent breakup pace.
- `search_junctions(world, request)` evaluates a batch of junctions in order.
- `rift_continents(world, dt)` processes queued plume requests (including
  waiting and expiry), temporarily held supercontinent pieces, overdue forced
  rifts and spontaneous tension/heat-driven breakup attempts.

`step_world(world, dt=None, update_motion=True, update_cycle=True)`
integrates this stage after plume and supercontinent updates when
`riftRatePerMyr` exists in the world configuration or plume-rift requests are
pending. It reports the number of plate splits in the `rifts` field. The same
stepper can also run passive-margin subduction initiation, collision tracking
and far-field evolution when their respective configuration keys are present.

Example:

```python
from orogen.v2.tectonics import TectonicWorld, step_world

world = TectonicWorld(mesh, seed=123, config=tectonics_config)
diagnostic = step_world(world, dt=1.0)
print(diagnostic["rifts"])
```

This is a tested rifting subsystem within a partial stepper, not the entire
`simulate.js` generator. Collision tracking and configured suture welding are
also integrated into the stepper; far-field propagation runs when
`farFieldEveryMyr` is configured. Strongly coupled Andean/flat-slab far-field
forcing, complete subduction/slab evolution, scheduled arc maturation,
ocean-plate evolution, terrain, geology, climate and snapshot stages remain
outstanding.


## Passive-margin subduction initiation

Module: `orogen.v2.tectonics.subduction_init`

- `initiate_subduction(world, dt)` groups old oceanic contacts between plates,
  filters short, young, stale and strongly diverging boundaries, applies the
  configured initiation probability, chooses the overriding/downgoing side,
  arms trench cells and logs a `SUBDUCTION_INIT` event.
- If one side is the passive background-ocean plate, it first calls
  `birth_ocean_plate` to create a mobile plate from reachable background
  oceanic crust.
- `choose_foundering(world, plate_a, plate_b, segment)` prefers the side
  with more continental land as the overriding plate; an ocean/ocean tie is
  resolved by oceanic crust age.
- `arm_margin(world, upper, lower, segment)` records the trench/slab depth
  on the overriding plate.
- `is_on_segment(world, lower, cell, upper)` tests whether a cell in the
  upper plate touches lower-plate crust.
- `founder_within_plates(world, dt)` is the separate slow-stage operation
  that can split a mobile plate along an old oceanic margin; it is exported,
  but is not yet scheduled by `step_world` because the upstream slow-stage
  cadence has not been ported.

`step_world` runs `initiate_subduction` when
`subductionInitRatePerMyr > 0`. If a rift or subduction changes plate
topology, the stepper rebuilds ownership and classifies boundaries again
before the force-balance stage. This is initiation only, not the complete
subduction/slab evolution model.

## Arc operations

Arc maturation and arc scraping have different integration status:


- `orogen.v2.tectonics.arc_maturation.mature_arcs(world, dt)` advances a
  distance-based front and converts mature island-arc cells touching
  continental/cratonic crust into continental crust. Margin-island cells are
  excluded, and other crust-history fields are retained. This helper is tested
  but its upstream slow-stage cadence is not yet scheduled by `step_world`.
- `orogen.v2.tectonics.ops.scrape_arc.scrape_arc(world, host, donor, cells)`
  transfers arc raster cells into a host plate using the plate-frame rotation,
  absorbs cells where the host already has land, records a suture, and retires
  an empty donor plate. Collision tracking invokes this operation when the
  configured collision conditions are met.

## Plate transfer and passive-ocean birth

Module: `orogen.v2.tectonics.ops`

- `transfer_cells(world, source, target, cells)` transfers crust history
  between plate frames, transforms directional far-field records, avoids
  overwriting existing host land/ocean, and removes the source cells.
- `free_cell_beside(target, mesh, cell, world_point, taken)` finds the nearest
  available target-frame neighbour when transferred land overlaps host land.
- `birth_ocean_plate(world, background, seed_cells, reach_km, name)` carves a
  connected region from the passive background ocean, subject to the minimum
  area setting, and records a `PLATE_BORN` event.
- `ocean_plate_name(world)` returns the next generated `Thalassa` name.

These are plate operations used by subduction initiation and intra-plate
foundering. They do not yet replace the full upstream back-arc/ocean-plate
lifecycle. `step_world` runs `evolve_far_field` when `farFieldEveryMyr` is
present; this propagates recorded push and can reactivate old sutures/failed
rifts, but strongly coupled Andean/flat-slab forcing is not yet ported.


## Terrane accretion and sea closure

Module: `orogen.v2.tectonics.ops.plate_accretion`

- `accrete_terrane(world, host, donor, terrane, rate_cm_yr)` transfers a
  donor's terrane into the host plate's frame, gathers a fore-arc strip,
  resolves overlap against existing host land, removes the transferred donor
  cells, closes narrow straits and eligible enclosed seas, marks the suture
  and spreads an Ural-type orogenic belt on both sides.
- A fragment below `terraneMinAreaKm2` is treated as scrap and does not
  receive a full suture-belt treatment.
- `close_seas_between(out, origin, mesh, upper, config, time)` identifies sea
  cells reached from both sides within `sealStraitKm`, preserves ocean
  connected to the far-side open sea, and fills enclosed sea components below
  `inlandSeaMaxKm2`. Filled cells inherit nearby land history, are dated at
  the accretion time and receive a suture flag.
- `landmass_of(plate, mesh, seeds)` returns the connected land component
  containing the supplied land seeds.

These operations are now used by the collision tracker when their
configuration is present. Terrane docking handles small fragments; collision
tracking also underthrusts the lower plate, spreads collision belts, measures
sea-gap and enclosed-ring geometry, and invokes the plate-welding operation
for eligible sutures.

Example:

```python
from orogen.v2.tectonics.ops import accrete_terrane, landmass_of

component = landmass_of(donor, world.mesh, terrane_seed_cells)
result = accrete_terrane(world, host, donor, component, rate_cm_yr=4.0)
print(result["cells"], result["sealed"])
```


## History endings

Module: `orogen.v2.tectonics.ending` (also exported from
`orogen.v2.tectonics`).

- `ENDINGS` lists the four supported history-ending choices: none,
  supercontinent, breakup and dispersed.
- `in_stage(state, stage, config)` evaluates a live world or captured frame
  against the upstream cycle/event rules. Breakup age is measured from the
  first rift since the latest supercontinent formation window.
- `history_ending(min_myr, stage, config)` returns the maximum run horizon
  and a `stopAt` predicate. It never changes the simulation trajectory.
- `frames_ending(frames, min_myr, stage, config, complete=False)` trims a
  recorded history at the first eligible frame, or returns `None` while an
  incomplete history needs to run further.

The helpers are tested independently. They do not yet imply that the full
upstream `simulate.js` stage sequence or snapshot codec has been ported.


## Collision tracking and suture welding

`track_collisions(world, dt)` gathers continent-on-continent overlaps by plate
pair, creates and updates collision records, determines upper/lower plates,
measures convergence and contact-front length, and advances shortening and
stall counters. Where enabled by configuration, it also accounts for
lower-plate underthrusting and routes touching terranes or overlapping arcs
through the accretion helpers. Positive shortening spreads an Ural-type belt
on both plates for a short front, or a Himalayan plateau belt on the upper
plate and a foreland belt on the lower plate when the front meets
`sutureMinContactKm`.

With the suture-gap settings present, the tracker periodically records open
sea area/width, the best observed closure and enclosed sea area. It distinguishes
`weldCandidate`, `weldBlockedByRing` and `mergeEligible`. For an eligible
pair, the full merge configuration enables `merge_plates(world, upper, lower,
overlap_world_cells, collision)`, implemented in
`orogen.v2.tectonics.ops.merge_plates` and exported from
`orogen.v2.tectonics.ops` and `orogen.v2.tectonics`.

The merge redraws the crust in the upper plate's frame, deforms the overlap
zone, stamps cratons as rigid blocks, closes narrow straits and sufficiently
small enclosed seas, marks the suture zone, reapplies Himalayan/foreland belt
records, keeps the upper plate and retires the lower plate. The tracker then
logs a `SUTURE` event and returns `changed=True`; `step_world` rebuilds
ownership and boundaries after the merge.

The merge is intentionally gated on a complete configuration group,
including belt geometry and spread settings, indent/falloff/escape parameters,
suture width and sea-sealing thresholds. A readiness marker without these
settings does not mutate the plates. Far-field collision forcing, full
collision-history parity and the complete upstream `simulate.js` cadence
remain outstanding. A `sutureCandidate` or `weldCandidate` by itself does
not imply that a plate merge has happened.


## Initial-world generation and scenario catalogues

`build_start_world(sim_mesh, seed, overrides=None)` returns a mapping with a
`TectonicWorld` and start-layout metadata. On meshes with at least 512 cells,
it uses the upstream craton packing and noise-shaped landmass assembly:
craton bodies and clusters, connected continental growth, terranes, inner/ring/
major sutures, a smoothed ocean apron and active/passive trench sectors. It
records pre-history ranges and far-field shortening. Meshes below 512 cells
retain a smaller deterministic fixture builder to keep low-resolution unit
fixtures inexpensive and spatially meaningful.

The Python/JavaScript reference runner receives the exact Python mesh arrays
(`xyz`, CSR adjacency offsets and adjacency list), then runs the pinned
upstream craton placement, landmass shaping and rim/trench construction. The
production differential test covers 512- and 768-cell meshes with different
seeds. It compares layout search decisions, craton geometry, the landmass
`type`, `terrane`, `suture` and `owner` arrays, and the assembled
continent/background-ocean masks, crust types, flags, terranes, formation
times, trench slab lengths, rim cells and passive-sector metadata. Old-range
(`oroType`/relief) and far-field record arrays are checked through Python
invariant tests but are not yet included in the JavaScript differential
comparison; multi-step tectonic histories are outside this fixture.

### Controlled test fixtures (not public API)

The minimal continental, oceanic and mixed worlds live in `tests/scenario_fixtures.py`.
They are test fixtures, not part of the `orogen.v2.tectonics` package API, and they
have no upstream JavaScript counterpart (see `docs/ADAPTATIONS.md`, section 5).

### Prescribed-motion scenes

- `build_rigid_drift_world` / `set_rigid_drift_time`
- `build_wilson_world` / `drive_wilson`
- `build_nudge_scene`

These scenes use deterministic geometry and prescribed Euler-pole motion;
they are separate from the named rule-testing catalogues below.

### Rule-testing scene catalogues

The builders return deterministic scenes by identifier. Their catalogues are
ported from the pinned upstream v3.0.1 scenario files:

| Builder | Supported identifiers |
|---|---|
| `build_event_scene` | `wings`, `terrane`, `arc`, `arc-collision` |
| `build_budget_scene` | `erosion`, `india`, `arcs`, `history` |
| `build_force_scene` | `one-sided`, `tug`, `wilson`, `india`, `start` |
| `build_margin_scene` | `margins`, `laramide`, `mariana`, `japan`, `history` |
| `build_rift_scene` | `plume`, `breakup`, `founder` |
| `build_terrain_scene` | `andes`, `laramide`, `india`, `mariana`, `japan`, `hawaii`, `traps`, `rift`, `cratons`, `farfield`, `history`, `cycle` |
| `build_hotspot_scene` | `hawaii`, `deccan`, `history` |
| `build_collision_scene` | `zipper` |

The eight catalogues contain 37 entries in total (some reuse the same
underlying scene in different testing contexts). Tests verify catalogue
identifiers, deterministic construction, complete plate partitions and
family-specific metadata. These are scenario fixtures and rule-testing
scenes, not a claim that the full upstream tectonic time-stepping engine or
the complete interactive application has been ported.
