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
adjacency, and production `SphereIndex` construction. It then compares:

- production mesh build time;
- `SphereIndex` and cKDTree construction time on the same production vertices;
- scalar nearest-neighbour calls, with cKDTree `workers=1`;
- batched cKDTree queries, reported separately from the scalar comparison;
- the complete `mesh_locate.locate` operation using a drop-in cKDTree adapter;
- nearest-cell identity, locate triplets, interpolation-weight differences, and
  angular excess calculated from normalized float64 coordinates.

All query methods use the same deterministic set of unit directions. Timings are
medians after warmup. The benchmark reports measurements rather than asserting
that different nearest-neighbour algorithms must return identical cell IDs.

The earlier `experiment_ckdtree.py` is retained as a historical isolated
microbenchmark. Its zero-jitter mesh and parallel batched query are not
representative of the production scalar lookup path and must not be used alone
to justify integration.

## Acceptance criteria

Do not replace `SphereIndex` based only on the isolated nearest-query speedup.
An integration proposal must account for index construction, the complete
`locate` path, correctness/quality differences, and the additional SciPy
runtime dependency. Keep the production backend unchanged until these results
are reviewed.
