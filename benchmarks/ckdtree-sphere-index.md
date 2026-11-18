# experiment/ckdtree-sphere-index

Isolated experiment based on the current `port/v3.0.1` production path. No
production backend is changed by this PR.

## Production-shaped benchmark

Run:

```bash
pip install -e ".[benchmark]"
python benchmarks/bench_ckdtree_production.py
```

The benchmark uses `build_sim_mesh(num_cells, seed)`, including the production
Fibonacci-sphere jitter (0.35), deterministic RNG, shull triangulation, CSR
adjacency, and production `SphereIndex` construction. The candidate build
repeats the same sphere/CSR construction and substitutes cKDTree for the
SphereIndex. It then compares:

- complete mesh/index build time;
- `SphereIndex` and cKDTree construction time on the same production vertices;
- scalar nearest-neighbour calls, with cKDTree `workers=1`;
- batched cKDTree queries, reported separately from the scalar comparison;
- the complete `mesh_locate.locate` operation using a drop-in cKDTree adapter;
- nearest-cell identity, locate triplets, interpolation-weight differences, and
  angular excess calculated from normalized float64 coordinates.

All query methods use the same deterministic set of 5,000 unit directions.
Timings below are medians after warmup (three repetitions for index/query
measurements; two for complete builds) on the GitHub Actions Ubuntu runner.

### Recorded run

Run: [GitHub Actions #44276768870](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44276768870)
Commit: `24b3c932426b67f607478808c87a0595ce0dff8e`

| Requested cells | Production build | cKDTree candidate build | SphereIndex scalar | cKDTree scalar | SphereIndex locate | cKDTree locate |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 623 ms | 267 ms | 52.5 ms | 117.6 ms | 155.2 ms | 252.0 ms |
| 40,000 | 2,579 ms | 1,075 ms | 53.4 ms | 118.3 ms | 154.3 ms | 249.4 ms |
| 80,000 | 5,362 ms | 2,398 ms | 54.7 ms | 121.1 ms | 153.3 ms | 256.4 ms |

The candidate index makes construction substantially faster because cKDTree
construction is much cheaper than building the spatial lookup table used by
SphereIndex. However, cKDTree scalar queries are about 2.2x slower, and the
complete `locate` operation is about 1.6x slower. The current production
lookup path is scalar, so the faster batched cKDTree result is not a drop-in
performance win.

| Requested cells | Scalar nearest IDs identical | Locate triplets identical | Mean abs. weight diff | Max abs. weight diff | Max angular excess |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 5,000/5,000 | 5,000/5,000 | 0 | 0 | 0° |
| 40,000 | 5,000/5,000 | 4,998/5,000 | 0.0000934 | 0.428756 | 0° |
| 80,000 | 4,998/5,000 | 4,997/5,000 | 0.0001418 | 0.441054 | 0.000184° |

The nearest directions are effectively equivalent angularly, but a small
number of different selected cells can produce large maximum interpolation
weight differences. This needs explicit correctness review before any backend
replacement.

Batch cKDTree queries took 2.8–3.8 ms for 5,000 points, versus 118–121 ms for
5,000 scalar cKDTree calls. This is a separate vectorized workload and does not
represent the present scalar `locate` API.

## Decision

**Do not replace SphereIndex with cKDTree in the current production lookup
path.** The candidate's build-time improvement is real, but repeated scalar
lookup and complete `locate` performance regress materially. Batched queries
could be revisited only alongside an API/workflow change that actually batches
the production workload, and after the interpolation differences are
characterized.

The earlier `experiment_ckdtree.py` is retained as a historical isolated
microbenchmark. Its zero-jitter mesh and parallel batched query are not
representative of the production scalar lookup path and must not be used alone
to justify integration.

The benchmark reports measurements rather than asserting that different
nearest-neighbour algorithms must return identical cell IDs. Keep the
production backend unchanged until a future proposal demonstrates an
end-to-end benefit and acceptable interpolation correctness.
