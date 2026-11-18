# Testing and compatibility

## Test layers

The repository uses several complementary forms of validation.

### Python unit and invariant tests

These cover edge cases, array invariants, graph behaviour, geometry, state mutation, and error handling. Run:

```bash
python -m pytest
```

### Spherical mesh backend tests

The production sphere-mesh builder uses `scipy.spatial.Delaunay`
(Qhull). Focused tests cover spherical topology, cyclic CSR adjacency,
deterministic builds, small point counts, Qhull exception translation,
invalid simplex shapes and out-of-range indices. The
`scipy sphere-mesh compatibility` workflow checks Python 3.10, 3.12 and
3.14 on Ubuntu and Windows. See [SciPy backend](SCIPY_BACKEND.md) for the
production path and compatibility contract.

### JavaScript differential tests

The differential harness executes vendored JavaScript reference modules from upstream tag `v3.0.1`. It compares selected Python operations against the reference implementation; it is not a whole-engine comparison.

Requirements:
- Node.js 22+
- Delaunator 5.0.1

Install the JavaScript dependency without creating a project lockfile:

```bash
npm install --no-save --package-lock=false
python -m pytest
```

Reference files are under `tests/reference/v2/`. They are test fixtures, not a separately supported JavaScript package. The exact upstream source revision is recorded in [UPSTREAM_SOURCE.md](../UPSTREAM_SOURCE.md).

### Audit des tests sans instruction `assert`

L’audit statique recense 29 fonctions de test sans instruction `assert` Python.
23 utilisent néanmoins `pytest.raises` ou `numpy.testing` comme mécanisme
d’assertion. Les 6 fonctions restantes de
`tests/test_v2_numba_kernel_coverage.py` sont des tests de couverture de branches :
elles exécutent explicitement les corps Python des kernels Numba (`.py_func`)
pour faire remonter les arcs de couverture ; elles sont maintenant documentées
comme telles. Les tests de parité et d’invariants du comportement observable
restent dans les suites Numba et raster dédiées.

### Coverage enforcement

Run the CI-equivalent coverage command:

```bash
NUMBA_JIT_COVERAGE=1 python -m pytest --cov=orogen --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=95
```

CI installs Python 3.12, Node.js 22, the test extras, and Delaunator before running the suite. It enables branch coverage, sets `NUMBA_JIT_COVERAGE=1`, and enforces a 95% total-coverage minimum for the full `orogen` package.

## Historical CI verification before the SciPy backend migration

The previous PR-head source validation on 2026-11-18 completed successfully using the former shull backend. These results are historical and do not validate the current SciPy backend:

- **506 tests passed**;
- **99.43% total coverage** for `orogen.v2` (CI minimum: 95%, branch coverage enabled);
- Python tests: [GitHub Actions run 44263575404](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44263575404);
- platform matrix: [run 44263575402](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44263575402), 9/9 jobs passed;
- benchmark: [run 44263575444](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44263575444), all structural comparisons passed.

The tested source revision was `b5063ddaa0694d03ea170c78b6076e39818998b2`. Rerun the workflows after subsequent source changes.

The differential suite covers selected core, geometry, mesh and tectonic operations. Production mesh tests exercise SciPy/Qhull directly and check spherical mesh invariants, Qhull error translation, simplex shape and index bounds. The historical shull A/B benchmark has been removed after selecting SciPy as the production backend.

## What the checks establish

- Python tests establish the tested invariants and behaviours of the implemented modules.
- JavaScript differential tests establish agreement for the specific operations and inputs exercised by those tests.
- Numba-vs-Python tests establish agreement between the accelerated implementation and its Python fallback; by themselves, they do not establish JavaScript parity.
- The historical seven Delaunay fixtures and current mesh tests exercise SciPy/Qhull on selected inputs; they do not establish universal connectivity for every point cloud.

These results do not establish that every possible input matches upstream.

## Compatibility limits

The following have not been demonstrated:
- complete JavaScript/Python API equivalence;
- universal Delaunay connectivity equivalence for all seeds and resolutions;
- end-to-end tectonic simulation parity over multiple time steps;
- full terrain, geology, climate, finishing, agriculture or save/load pipeline parity;
- compatibility with every upstream configuration or serialized world.

Some implementation details can also differ while preserving current tested behaviour. In particular, `CrustRaster` groups fields into NumPy backing allocations and exposes per-field views. The fields remain separately writable and retain their names, dtypes, indexing and field order, but their underlying buffer ownership/aliasing details are not identical to independent JavaScript typed-array allocations. No current internal consumer has been identified as depending on those low-level allocation properties.

Coverage is not a feature-completeness metric. A module can have 100% line coverage while substantial upstream functionality remains unported.

## Adding a ported feature

For each new subsystem:

1. Identify the exact upstream source files and transitive dependencies.
2. Record upstream version and commit.
3. Port data types, sentinel values, mutation semantics, and random-stream behaviour.
4. Add focused tests for normal cases and edge cases.
5. Add differential tests when the upstream operation can be executed independently.
6. Document deviations and numerical tolerances.
7. Keep the coverage threshold green.
8. Update the relevant API page and [Porting status](PORTING_STATUS.md).

Avoid broad tolerances that make real differences invisible. For mesh and geometry work, compare both numerical values and structural invariants (orientation, connectivity, valid indices, and closure).
