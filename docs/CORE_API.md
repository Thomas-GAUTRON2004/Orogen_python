# Core API reference

Import core operations from `orogen.v2.core`. The package re-exports the names listed below.

## Deterministic seeds and random numbers

Module: `orogen.v2.core.seed`

| Name | Description |
|---|---|
| `SEEDS` | Maximum supported seed-range constant used by the upstream convention. |
| `is_word_seed(value)` | Returns whether a non-empty value is treated as a word seed rather than a finite numeric string. |
| `seed_of(value)` | Parses numeric seed text or hashes word text using FNV-1a. |
| `random_seed()` | Generates a seed in the supported range using system entropy. |
| `make_rng(seed=0)` | Returns a stateful Park–Miller random function producing values in `[0, 1)`. |
| `make_rand_int(seed=0)` | Returns a deterministic integer sampler; calling it with `n` samples `[0, n)`. |

Example:

```python
from orogen.v2.core import make_rand_int, make_rng, seed_of

seed = seed_of("my-world")
random = make_rng(seed)
rand_index = make_rand_int(seed)
print(random(), rand_index(100))
```

A generator is stateful. Two generators created from the same seed start identically, but their outputs diverge if their calls are interleaved differently.

## Noise

Class: `FastNoise(seed=0)`

- `noise3d(x, y, z)`: seeded 3-D simplex noise.
- `fbm(x, y, z, octaves=5, persistence=2/3)`: normalized fractal Brownian motion formed by summing octave-scaled noise.

```python
from orogen.v2.core import FastNoise

field = FastNoise(1234)
value = field.noise3d(0.2, 0.3, 0.4)
detail = field.fbm(0.2, 0.3, 0.4, octaves=6, persistence=0.65)
```

The class builds its permutation table from the Orogen random generator. Reuse an instance when sampling a field; constructing another instance rebuilds the same table for the same seed.

## Quaternion rotations

The following functions are exported:

- `quat_identity()`
- `quat_normalize(q)`
- `quat_mul(a, b)`
- `quat_from_axis_angle(x, y, z, angle)`
- `quat_integrate(q, omega, dt)`
- `quat_to_mat3(q)`
- `quat_to_euler_pole(q)`

They implement quaternion construction, composition, normalization, integration, and conversion. Follow the argument order shown above; the axis-angle constructor takes four scalar arguments, not an axis tuple.

## Spherical geometry

Exported operations:

| Function | Purpose |
|---|---|
| `normalize(x, y, z)` | Normalize a vector. |
| `dot(...)`, `cross(...)` | Vector products. |
| `angle_between(...)` | Angular separation. |
| `destination(...)`, `bearing_to(...)` | Spherical destination and bearing operations. |
| `tangent_basis(...)`, `tangent_toward(...)` | Local tangent-frame operations. |
| `polar_frame(...)` | Build the polar Euler frame used by the engine. |
| `centroid_of_cells(...)` | Compute a centroid from mesh cells. |
| `cell_distance_km(...)` | Convert angular cell separation to kilometres. |
| `cap_fraction(...)` | Spherical-cap area fraction. |
| `random_point(...)` | Generate a random spherical point. |

Latitude/longitude conversion helpers are in `orogen.v2.tectonics.kinematics`; their input and output units are degrees.

## Sphere generation and simulation meshes

Module: `orogen.v2.core.sphere_mesh`

- `generate_fibonacci_sphere(num_points, jitter, rng)` generates flattened Float32 unit-sphere coordinates.
- `stereographic_projection(xyz, num_points)` projects the positions from the north pole.
- `build_sphere(num_points, jitter, rng)` returns a closed `SphereMesh` and its coordinates.
- `SphereMesh` contains triangles, CSR adjacency, region count, and triangle count.

Module: `orogen.v2.core.sim_mesh`

```python
from orogen.v2.core import build_sim_mesh, mesh_from_arrays, km_to_hops

mesh = build_sim_mesh(256, seed=42)
hops = km_to_hops(mesh, distance_km=800)
```

- `build_sim_mesh(num_cells, seed)` constructs a deterministic simulation mesh.
- `mesh_from_arrays(*, num_cells, xyz, adj_offset, adj_list)` reconstructs a mesh from transferred arrays without rebuilding Delaunay topology.
- `km_to_hops(sim_mesh, distance_km, minimum=1)` converts a distance to a characteristic hop count using JavaScript-style rounding.
- `SimMesh` exposes `num_cells`, `xyz`, `adj_offset`, `adj_list`, `index`, and mesh metrics.

The builder adds a north-pole closure region. Inspect `mesh.num_cells` after construction.

## Spatial indexing and point location

- `SphereIndex` provides spherical spatial lookup over the mesh.
- `locate` performs barycentric point location on spherical triangles.

These are lower-level APIs. Their accepted array shapes and return conventions are defined in their module docstrings and should be followed exactly; in particular, `SimMesh.xyz` is a flat array unless explicitly reshaped.

## Graph operations

The exported graph helpers include:

- `flood_from`, `hops_from`, `hop_distance`
- `pieces_of`, `largest_piece_of`
- `touches`, `reaches_within`, `within_km`
- `rings_of` and `RING_COUNT`

They operate on mesh adjacency. BFS/hop operations measure graph steps. The weighted distance utilities are separate.

## Weighted propagation

- `MinHeap` is the heap primitive.
- `spread_km` performs weighted propagation over mesh edges using kilometre costs.
- `forget_edge_km` is the companion edge-distance operation used by the propagation layer.

Use weighted propagation when the cost should reflect physical distance rather than treating every graph edge as exactly one hop.

## Device classification

`SMALL_DEVICE` and `is_small_device` expose the small-device classification used by the port. This is a memory-budget heuristic, not a hardware capability detector.
