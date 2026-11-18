# Testing and compatibility

## Test layers

The repository uses several complementary forms of validation.

### Python unit and invariant tests

These cover edge cases, array invariants, graph behaviour, geometry, state mutation, and error handling. Run:

```bash
python -m pytest
```

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
python -m pytest --cov=orogen.v2 --cov-report=term-missing --cov-report=xml --cov-fail-under=100
```

CI installs Python 3.12, Node.js 22, the test extras, and Delaunator before running the suite. It enforces 100% line coverage for `orogen.v2`.

## Latest documented verification

The latest successful test run explicitly recorded in the benchmark documentation is:

- **476 tests passed**;
- **100% line coverage** for `orogen.v2`;
- run: [GitHub Actions — Python tests](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44168460148).

This is a historical verified run, not a claim that the current branch head has been tested. The run is associated in the project history with commit `6ea20d647169979ccbdf107732ddb7f2535695b9`; rerun CI after subsequent code changes to establish the current status.

The differential suite covers selected core, geometry, mesh and tectonic operations. The seven historical Delaunay fixture matches (12/99, 24/1, 48/17, 96/2026, 128/7, 256/42, 512/12345) were measured with SciPy/Qhull and do not establish equivalent connectivity for the new shull backend. New shull-specific tests check spherical mesh invariants across multiple sizes.

## What the checks establish

- Python tests establish the tested invariants and behaviours of the implemented modules.
- JavaScript differential tests establish agreement for the specific operations and inputs exercised by those tests.
- Numba-vs-Python tests establish agreement between the accelerated implementation and its Python fallback; by themselves, they do not establish JavaScript parity.
- The seven Delaunay fixtures establish agreement only for those resolution/seed pairs.

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
