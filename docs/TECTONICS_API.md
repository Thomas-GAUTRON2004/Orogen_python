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
