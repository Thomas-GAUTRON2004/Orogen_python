# JavaScript/Python differential testing policy

## Oracle and scope

The pinned upstream JavaScript implementation is the sole behavioral oracle for
Python-port tests. A test that validates behavior must execute both implementations
against the same input and compare their observable outputs. The JavaScript side
must import the upstream source through the reference pointers in
`tests/reference/v2`; it must not contain a second implementation of the algorithm.

## Required shape of a behavioral test

1. Construct a deterministic input payload (mesh arrays, seed, configuration and
   operation sequence).
2. Pass that exact payload to an upstream JavaScript runner.
3. Pass the same payload to the Python implementation.
4. Normalize only representation differences required by the JSON boundary
   (typed arrays / NumPy arrays, typed scalars and non-finite-number encoding).
5. Compare the complete observable outputs. Use a documented numeric tolerance
   only where the JS/Python numeric representations require it.
6. On mismatch, report the output path and both values so the divergence is
   actionable.

Use `tests/_differential.py` for process execution and recursive comparisons.
Prefer one runner invocation for a complete scenario over launching Node once per
individual field.

## What is not an acceptable behavioral test

Do not add tests whose behavioral oracle is:
- a literal expected number, array, snapshot, event string or configuration value;
- an independently reconstructed Python expectation;
- an invariant asserted only against Python output (for example, “coverage sums
  to one”, “the vector is normalized”, or “the result is reproducible”);
- a mock or monkeypatch that substitutes the behavior being tested.

These may be useful during development, but they are not conformance tests and
must not be added to the required behavioral suite. Test data may of course contain
literal inputs: the restriction applies to expected outputs, not to deterministic
scenario inputs.

## Exceptions

Tests for packaging, import behavior, CI wiring, performance, static analysis and
coverage measurement are infrastructure checks, not behavioral tests. They may
validate their own explicit contracts. They must not be presented as evidence of
JavaScript/Python behavioral parity.

## Failure policy

Node.js and the pinned upstream submodule are mandatory in CI. A missing runtime,
missing reference module, malformed reference output, timeout or non-zero runner
exit must fail the CI job. Local developer runs may skip when Node.js is absent,
but that skip must never be accepted in CI.

## Migration rule

When converting an existing Python-only behavioral test, first add or extend the
matching upstream runner so that it accepts the same scenario input and emits the
observable state. Then replace Python-only expected-value assertions with a
comparison against the JavaScript output. Do not weaken or delete a mismatch
comparison merely to make a test pass. Keep coverage instrumentation, but do not
treat coverage assertions as behavioral conformance.
