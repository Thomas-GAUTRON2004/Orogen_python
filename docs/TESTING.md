# Testing and compatibility

## Test layers

The repository uses several complementary forms of validation.

### Python unit and invariant tests

These cover edge cases, array invariants, graph behaviour, geometry, state mutation, and error handling. Run:

```bash
python -m pytest
```

### Spherical mesh backend tests

The production sphere-mesh builder uses the native `shull` Delaunay backend.
Focused tests cover spherical topology, cyclic CSR adjacency, deterministic
builds, exception translation, invalid simplex shapes and out-of-range
indices. The `shull platform compatibility` workflow checks Python 3.10,
3.12 and 3.14 on Linux, Windows and macOS.

For an optional SciPy-versus-shull pipeline benchmark, install
`python -m pip install -e ".[benchmark]"` and run
`python benchmarks/bench_shull_pipeline.py`. SciPy remains a test and
benchmark dependency, not a production runtime dependency.

### JavaScript differential tests

The differential harness executes vendored JavaScript reference modules from upstream tag `v3.0.1`. It compares selected Python operations against the reference implementation; it is not a whole-engine comparison.

Requirements:
- Node.js 22+
- Delaunator 5.0.1

Install the JavaScript dependency without creating a project lockfile:

```bash
npm install --no-save --package-lock=false delaunator@5.0.1
python -m pytest
```

Reference files are under `tests/reference/v2/`. They are test fixtures, not a separately supported JavaScript package. The exact upstream source revision is recorded in [UPSTREAM_SOURCE.md](../UPSTREAM_SOURCE.md).

### Coverage enforcement

Run the CI-equivalent coverage command:

```bash
NUMBA_JIT_COVERAGE=1 python -m pytest --cov=orogen.v2 --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=95
```

CI installs Python 3.12, Node.js 22, the test extras, and Delaunator before running the suite. It enables branch coverage, sets `NUMBA_JIT_COVERAGE=1`, and enforces a 95% total-coverage minimum for `orogen.v2`.

## Latest documented verification

The last successful baseline run on `port/v3.0.1`, before the backend-benchmark refresh, was:

- **506 tests passed**;
- **99.43% total coverage** for `orogen.v2` (CI minimum: 95%, branch coverage enabled);
- run: [GitHub Actions — Python tests](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44260359489);
- validated revision: `090c8e3f22d592c9f1bcc4d50c9cb635e36e2942`.

This is a historical verified run, not a claim that the current branch head has been tested. The run is associated in the project history with commit `6ea20d647169979ccbdf107732ddb7f2535695b9`; rerun CI after subsequent code changes to establish the current status.

The differential suite covers selected core, geometry, mesh and tectonic operations. Production mesh tests now exercise shull directly and check spherical mesh invariants, error translation, simplex shape and index bounds. The optional A/B benchmark compares SciPy and shull on selected deterministic point sets; neither finite test set proves equivalence for all inputs.

## What the checks establish

- Python tests establish the tested invariants and behaviours of the implemented modules.
- JavaScript differential tests establish agreement for the specific operations and inputs exercised by those tests.
- Numba-vs-Python tests establish agreement between the accelerated implementation and its Python fallback; by themselves, they do not establish JavaScript parity.
- The historical seven Delaunay fixtures establish SciPy/Qhull agreement only; shull must be evaluated against its own CI results.

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
