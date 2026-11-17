# Getting started

## 1. Requirements

Orogen Python requires:

| Dependency | Minimum | Purpose |
|---|---:|---|
| Python | 3.10 | Runtime |
| NumPy | 1.24 | Typed arrays and vector operations |
| shull | 0.4.0 | Rust-backed 2-D Delaunay triangulation (required runtime dependency) |
| pytest | 8 | Test runner (optional extra) |
| pytest-cov | 5 | Coverage reporting (optional extra) |

Node.js 22+ and Delaunator 5.0.1 are needed only for the JavaScript differential reference tests.

SciPy 1.10+ is optional and is used only as the comparison baseline by the `benchmark` extra. It is not used by the production package.

## 2. Install from a checkout

```bash
git clone https://github.com/Thomas-GAUTRON2004/Orogen_python.git
cd Orogen_python
python -m venv .venv
```

Activate the virtual environment, then install:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

To run the optional A/B benchmark against SciPy:

```bash
python -m pip install -e ".[benchmark]"
python benchmarks/bench_shull_pipeline.py
```

The package metadata currently reports version `3.0.1.dev0`. This is a development port, not a claim that all upstream 3.0.1 features are available.

## 3. Build a deterministic simulation mesh

```python
from orogen.v2.core import build_sim_mesh

mesh = build_sim_mesh(num_cells=128, seed=42)

print("cells:", mesh.num_cells)
print("mean edge (km):", mesh.edge_km)
print("cell area (km²):", mesh.cell_area_km2)
print("xyz array:", mesh.xyz.shape)
print("CSR offsets:", mesh.adj_offset.shape)
print("adjacency entries:", mesh.adj_list.size)
```

The requested point count is the input to the spherical point generator. The returned `mesh.num_cells` is the actual region count after the mesh builder adds its north-pole closure region. Use the returned count and arrays rather than assuming the requested count is the final cell count.

The mesh stores unit-sphere Cartesian coordinates in `xyz`, plus adjacency in compressed sparse row (CSR) form:
- `adj_offset` has one extra entry; neighbours of cell `i` are `adj_list[adj_offset[i]:adj_offset[i + 1]]`.
- `edge_rad` and `edge_km` are characteristic edge-length estimates.
- `cell_area_km2` is the equal-area estimate (4\pi R^2/N), not an individual polygon area measurement.

## 4. Use deterministic random numbers and noise

```python
from orogen.v2.core import FastNoise, make_rng, seed_of

seed = seed_of("archipelago-01")
rng = make_rng(seed)
noise = FastNoise(seed)

sample = rng()
height_signal = noise.fbm(0.12, -0.4, 1.7, octaves=5)
print(sample, height_signal)
```

The same seed and call sequence produce the same random stream. A random generator is stateful: adding or removing a draw changes all subsequent values from that generator. The tectonic world therefore has separate streams for different mechanisms.

## 5. Query graph neighbourhoods

`SimMesh` exposes CSR adjacency. Graph helpers operate on a mesh-like object with `num_cells`, `adj_offset`, and `adj_list`:

```python
from orogen.v2.core import hops_from, hop_distance

# Starting from cell 0; the exact returned structure is documented
# by the function's type/docstring in core/mesh_bfs.py.
distances = hop_distance(mesh, 0)
reachable = hops_from(mesh, 0, 2)
```

Graph distances are hop counts, not kilometres. Use `km_to_hops(mesh, distance_km)` for the mesh's characteristic edge scale, or the weighted `spread_km` operation when physical path distance is required.

## 6. Run tests

```bash
python -m pytest
python -m pytest --cov=orogen.v2 --cov-report=term-missing --cov-fail-under=100
```

For the full differential suite, install Node and Delaunator as described in [Testing](TESTING.md).

## 7. Import paths

The V2 API is deliberately namespaced:

```python
from orogen.v2 import core, tectonics
from orogen.v2.core import build_sim_mesh
from orogen.v2.tectonics import Plate, CrustRaster
```

The historical package namespace and V2 port are not interchangeable. Prefer explicit `orogen.v2` imports when working with the new implementation.
