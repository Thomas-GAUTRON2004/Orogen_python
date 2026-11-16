# Testing and compatibility

## Test layers

The repository uses several complementary forms of validation.

### Python unit and invariant tests

These cover edge cases, array invariants, graph behaviour, geometry, state mutation, and error handling. Run:

```bash
python -m pytest
```

### JavaScript differential tests

The differential harness runs vendored JavaScript reference modules adapted from upstream tag `v3.0.1`. It compares selected Python operations against the reference implementation.

Requirements:
- Node.js 22+
- Delaunator 5.0.1

Install the JavaScript dependency without creating a project lockfile:

```bash
npm install --no-save --package-lock=false delaunator@5.0.1
python -m pytest
```

Reference files are under `tests/reference/v2/`. They are test fixtures, not a separately supported JavaScript package.

### Coverage enforcement

Run the CI-equivalent coverage command:

```bash
python -m pytest --cov=orogen.v2 --cov-report=term-missing --cov-report=xml --cov-fail-under=100
```

CI installs Python 3.12, Node.js 22, the test extras, and Delaunator before running the suite. It enforces 100% line coverage for `orogen.v2`.

## What the current checks establish

The latest verified baseline recorded in the project documentation is:
- 107 passing tests;
- 1,276 statements in `orogen.v2`;
- 100% line coverage, with no missing statements;
- differential comparisons for selected core and tectonic primitives;
- matching Delaunay triangles/neighbourhoods for seven tested resolution/seed pairs: 12/99, 24/1, 48/17, 96/2026, 128/7, 256/42, and 512/12345.

These results establish that the covered Python code executes under the tested conditions and that the selected comparisons agree. They do not establish that every possible input matches upstream.

## What is not established

The following have not been demonstrated:
- complete JavaScript/Python API equivalence;
- universal Delaunay connectivity equivalence for all seeds and resolutions;
- end-to-end tectonic simulation parity over multiple time steps;
- terrain, geology, climate, finishing, or agriculture parity;
- compatibility with every upstream configuration or serialized world.

Coverage is not a feature-completeness metric. A small, incomplete module can have 100% line coverage.

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
