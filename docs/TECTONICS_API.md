# Tectonics API reference

Import tectonic components from `orogen.v2.tectonics`. This namespace currently exposes state models and individual operations. It does not contain the full upstream tectonic evolution engine.

## Enumerations and flags

The port represents upstream enumerations as dictionaries of string keys to integer values.

### Crust types

`CrustType`:
- `OCEAN`: 0
- `CONTINENT`: 1
- `CRATON`: 2
- `ARC`: 3

### Other classifications

- `OrogenyType`: NONE, ANDEAN, LARAMIDE, URAL, HIMALAYAN, ARC, RIFT, TRANSPRESSION, INTRAPLATE.
- `VolcanicType`: NONE, TRACK, LIP.
- `BeltSide`: NONE, UPPER, LOWER.
- `CrustFlag`: bit flags including TRENCH, SUTURE, FAILED_RIFT, RIFT_MARGIN, LIP, APRON, MARGIN_ISLE, and EROSIVE.
- `BoundaryKind`, `EventKind`, and `Provisional` are defined in `world.py` and exported from the package.

Use the named constants instead of embedding numeric values in application code.

## CrustRaster

`CrustRaster(num_cells)` stores crust in a structure-of-arrays layout for one plate-local mesh.

Important attributes:
- `numCells`: number of cells.
- `mask`: Uint8-like occupancy mask.
- `count`: number of occupied cells.
- Per-cell arrays: `type`, `flags`, `formedAt`, `terrane`, `slabKm`, `thickKm`, orogeny and belt fields, volcanic fields, cover fields, and other history attributes.
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

The arrays are mutable. Directly editing `mask` or other fields can make the cached `count` inconsistent; prefer the methods when changing occupancy.

## Plate

`Plate(plate_id, num_cells, born=0.0)` stores a rigid plate pose and a `CrustRaster`.

Key attributes include `id`, `name`, `alive`, `born`, `quat`, `mat`, `omega`, `crust`, `stats`, and spatial culling-cap fields.

Methods:
- `set_quat(quaternion)` updates the quaternion and its rotation matrix.
- `to_world(x, y, z)` transforms a plate-frame vector into world coordinates.
- `to_plate_frame(x, y, z)` applies the inverse rotation.
- `cell_at(index, x, y, z)` transforms a world vector into plate coordinates and queries the supplied spatial index.
- `cap_in_world(mesh)` returns the plate extent cap in world coordinates.
- `mark_reshaped()` marks the plate as changed.
- `refresh_extent(xyz)` recomputes occupied-cell statistics and the bounding cap.

`frame_rotation(source, target)` returns the rotation matrix from source plate coordinates into target plate coordinates.

## Kinematics

The kinematics module exports:

- `lat_lon_to_xyz(lat_deg, lon_deg)`: degrees to unit Cartesian coordinates.
- `xyz_to_lat_lon(x, y, z)`: unit Cartesian coordinates to degree-valued latitude/longitude.
- `omega_from_euler_pole(pole_xyz, speed_cm_yr)`: Euler-pole vector and angular speed conversion.
- `surface_velocity(omega, px, py, pz, out=None)`: local surface velocity vector.
- `speed_cm_yr_at(omega, px, py, pz)`: local speed in centimetres per year.
- `advance_plates(world, dt)`: integrate the pose of each living plate and advance world time/step.
- `pose_at_constant_omega(q0, omega, t)`: evaluate a quaternion pose at constant angular velocity.

Conversion constants:
- `CM_PER_YR_TO_KM_PER_MYR = 10.0`
- `CM_PER_YR_TO_RAD_PER_MYR = 10.0 / 6371.0`

The radius used by these conversions is 6,371 km.

## Occupancy and claim operations

- `BinIndex`, `bin_index`, `mark_bin`, and `may_have_crust_near` implement spatial occupancy bins.
- `clear_world_view` and `for_each_claim` gather plate crust claims into the world frame.
- `overrides` applies the currently ported subduction-polarity decision rules.
- `reconstruct` rebuilds ownership and claim counters from the available plate data without mutating crust.

These functions are low-level. They do not resolve all plate collisions, create a complete tectonic boundary network, or execute a full simulation step.

## TectonicWorld

`TectonicWorld(sim_mesh, seed, config, overrides=None)` creates a mutable world-state container.

The constructor requires an explicit configuration mapping. The upstream `TECTONICS` default configuration has not yet been ported; there is no supported built-in default to substitute here.

The world contains:
- mesh reference, seed, configuration, time, and step counters;
- a list of plates and event records;
- independent deterministic random streams for separate tectonic mechanisms;
- per-cell ownership, claim, collider, boundary, convergence, and provisional-state arrays;
- budgets and auxiliary collections for collisions, arc scrapes, hotspots, and plumes.

Methods include `add_plate(name)`, `next_terrane_id()`, `name_by_cratons(plate, fallback)`, and `log(kind, text, data=None)`.

The container does not initialize a physically complete planet, resolve boundaries, or run the upstream tectonic step pipeline. Supply a configuration only if you know the fields expected by the specific downstream operation you are integrating.

## Ridge segments

- `PLANET_RADIUS_KM = 6371.0`
- `SEGMENT_KM = 400.0`
- `fracture_coordinate(px, py, pz, omega)` computes a ridge-floor coordinate in segment lengths about a spreading pole. It returns 0.5 for a near-zero angular-velocity vector; zero is reserved for cells without a recorded coordinate.

## Boundary contact geometry

Module: `orogen.v2.tectonics.contact`

`contact_at(world, cell, out)` finds the neighbouring world cell across a plate boundary. It writes a four-value result to `out`: the averaged unit normal `(nx, ny, nz)` pointing toward foreign plate neighbours, followed by the number of shared foreign-neighbour edges. It returns `-1` for an interior cell.

The result is cached per world cell and is valid until `world.occupancyStamp` changes. The normalized direction of each mesh adjacency edge is cached per mesh. Increment the stamp whenever the owner map is rebuilt; otherwise callers may reuse a previous contact result.

```python
import numpy as np
from orogen.v2.tectonics import contact_at

normal = np.zeros(4)
across = contact_at(world, world_cell, normal)
if across >= 0:
    nx, ny, nz, shared_edges = normal
```


## Tectonic convergence and divergence resolver

Module: `orogen.v2.tectonics.resolve`

`resolve(world)` rebuilds the current world-frame claims and settles plate overlap. It applies the upstream ocean/continent and ocean/ocean polarity rules, leaves continental collisions overlapping for later boundary/event handling, attempts symmetric oceanic crust growth into divergent gaps, assigns provisional owners to unresolved gap cells, updates creation/consumption budgets, and refreshes dirty plate extents. It returns integer counts `created`, `consumed`, and `collisionCells`.

The diagnostic helpers `candidate_cells(world, grower, gaps)` and `decide_birth(world, grower, cell, everyone)` expose the candidate and birth-rule stages. This is the first port of the resolver pass; the wider tectonic event, boundary-classification, force, and time-stepping pipeline remains separate and is not implied by this function.

```python
from orogen.v2.tectonics import resolve

result = resolve(world)
# {"created": ..., "consumed": ..., "collisionCells": ...}
```


## Boundary-motion primitives

Module: `orogen.v2.tectonics.boundaries`

The current port exposes four deterministic building blocks from the upstream boundary pass:

- `classify_motion(closing, shear, active_speed, oblique_limit)` distinguishes opening, closing, sliding, and locked contacts.
- `vote_segments(world, contacts)` computes a single weighted polarity vote for each connected ocean-ocean contact segment.
- `near_land(plate, mesh, cell, config)` checks for non-ocean crust within the configured subduction-apron reach.
- `feed_trench(world, world_cell, crust, cell, subducted_km)` starts or extends a trench only when this cell or an adjacent world cell consumed crust.

These are helper ports, not yet the complete `classifyBoundaries(world, dt)` pass. Full boundary classification also requires the still-unported orogeny and far-field deformation systems.


### Contact boundary classification

`classify_contact_boundaries(world, dt)` performs the currently ported contact-state pass over a resolved world view. It sets per-cell boundary kinds and convergence, handles collision and interior trench-flag bookkeeping, resolves closing-contact polarity, records trench consumption, and optionally votes ocean-ocean contact segments when `segmentPolarity` is enabled.

This remains a partial port of upstream `classifyBoundaries`: mountain-belt spreading, transpression/orogeny records, flat-slab decisions, and far-field force recording are not yet included.


## Adaptive simulation stepping

Module: `orogen.v2.tectonics.simulation`

- `fastest_speed(omega, plate, centre)` computes the fastest angular surface speed over a plate's spherical cap.
- `closing_rate(world)` sums the two highest living-plate cap speeds.
- `choose_step(world)` applies the upstream adaptive time-step limit in mesh-cell units, with the configured maximum-step and oceanic-speed floor.

These helpers port the time-step calculation from `simulate.js`; they do not yet implement the full simulation generator or event pipeline.


## Orogenic belt geometry

Module: `orogen.v2.tectonics.orogeny`

- `belt_width_km(kind, work_km, rate_cm_yr, config)` reproduces the upstream belt-width rule, including speed clamps and maximum width.
- `belt_shape(head_on, slab_age_myr, config)` applies obliquity and slab-age scaling.
- `obliquity_deg(velocity, normal, position, closing)` returns signed dextral/sinistral obliquity.

`spread_belt(plate, mesh, sources, belt, config, allow=None)` propagates a front across connected land cells, with source-specific reach/work, weak-line costs, craton-rim restrictions, and optional cell filtering. It records orogeny history, previous-belt overprints, belt geometry/shear, and crustal thickening, and calls the ported `build_heights` helper to update root/uplift/ruggedness. When `config['geology']['records']` is enabled, it also calls `thicken_cover` for existing sediment. It returns the reached continental cell indices. Time-dependent erosion, ridge/crest synthesis, and full sediment deposition remain outstanding.


## Orogenic relief and sediment response

Modules: `orogen.v2.tectonics.heights` and `orogen.v2.tectonics.geology`

- `belt_relief_km(type, reach_km, heights)` scales mature relief by belt reach.
- `broad_profile(type, side, distance_km, reach_km)` evaluates the smoothed cross-belt uplift profile.
- `build_heights(crust, cell, type, side, u, reach_km, work_km, thickened_km, config, amp=1)` updates root, broad uplift, and ruggedness for one cell.
- `thicken_cover(crust, cell, thickened_km, time_myr, geology)` increases existing sediment thickness as the underlying crust thickens and updates maximum burial and unroofing records.

The erosion/evolution pass, full terrain synthesis, ridge/crest construction, and full basin sediment-deposition model remain separate unported stages.


## Force-balance primitives

Module: `orogen.v2.tectonics.dynamics`

- `drag_tensor(plate, mesh, config)` computes the basal-drag tensor in the world frame, weighting oceanic, continental and cratonic crust separately.
- `solve3(matrix, vector)` solves the 3x3 linear system used for angular velocity, with a zero-vector fallback for singular systems.
- `balanced_rotation(plate, mesh, load, omega_of, config)` solves driving torque against basal drag, slab resistance and land-collision coupling. The load must be supplied by a force-gathering stage.
- `soften_speed` and `clamp_speed` apply the upstream smooth and hard angular-speed limits.

These are balance-stage primitives, not a complete plate-dynamics update: the world-wide force gathering pass (`gatherLoads`), mantle-flow forcing and simulation integration are still outstanding.


## Boundary-force gathering (partial)

Module: `orogen.v2.tectonics.forces`

`gather_boundary_loads(world)` gathers slab pull, slab suction, young-ridge push, and young-rift push from a resolved world owner map. Its records are compatible with `balanced_rotation` and include per-kind torque/vector totals plus slab-resistance records.

The gatherer also records land-overlap collision contacts with opposing front normals and optional craton lock weighting; fresh conjugate rift margins and arc contacts are excluded from collision locking. It now also gathers mantle flow and dispersal repulsion when an existing cycle state enables them. The cycle tracker that discovers landmasses and advances cycle phases remains a separate outstanding stage.


## Plate angular-velocity update (partial)

`update_plate_motion(world)` connects boundary-force gathering to the torque-balance solver. It solves each living non-background plate from the same start-of-step angular velocities, applies smooth and hard speed limits, then relaxes toward the solved target over `relaxMyr`. It also records a compact per-plate force diagnostic.

The update uses the partial boundary-force gatherer above; collision-overlap forces and mantle-flow/repulsion are not yet included. Therefore it is not a full upstream `updatePlateMotion` replacement.


## Mantle flow and continental dispersal

Module: orogen.v2.tectonics.mantle

- flow_of_cycle(cycle, config, x, y, z, out=None) evaluates tangent traction from an existing cycle state: outward from the heated upwelling, toward the downwelling during gathering, and local hold-inflow over the assembled footprint.
- mantle_flow_at(world, x, y, z, out=None) evaluates the same field from world state.
- dispersal_field(world) builds large-continent centres and touching-pair exclusions during the dispersing phase.
- repel_at(field, plate_id, x, y, z, out=None) accumulates a quadratic falloff push away from other nearby large continents.

gather_boundary_loads applies these vectors only to continental and cratonic cells and scales them by cell area. The field evaluator and force gatherer are ported; supercontinent landmass detection, heat evolution, seeded cycle formation and phase transitions are not yet implemented.


## Cohesive continental landmasses

Module: `orogen.v2.tectonics.supercontinent`

- `is_continental(world, cell)` classifies continental and cratonic cells, excluding island arcs and unowned cells.
- `landmasses(world)` counts connected continental components of at least 1 Mkm² and returns the largest cohesive mass, its share of total continental cells, and the large-component count. Narrow necks are split by growing inland cores back through continental crust using geodesic distance.

The result feeds `track_cycle(world)`, which consumes rift events, waits for sustained continental-share thresholds, places seeded upwellings/downwellings, and advances assembled, dispersing and gathering phases. It reads mantle heat from `world.mantleHeat` but does not evolve that field; heat production/decay remains a separate hotspot/thermal stage.


`track_cycle(world)` advances the world cycle state. It uses `world.rngCycle` for deterministic downwelling orientation and cycle duration, logs supercontinent/dispersal events, records the large plates of a clump, computes the upwelling heat average, and updates `phase`, `gather`, `clump`, and `hold`. Required cycle parameters are read from `world.config`.
