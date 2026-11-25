# Generic Python/JavaScript Differential Tests

This guide describes the target format for adding differential tests to
Orogen_python: one central Python configuration file lists the Python/JavaScript
function pairs and their test cases, while one generic pytest module executes
every configured pair.

> **Status:** this is the proposed generic format. The current test suite still
> contains domain-specific JavaScript runners and pytest modules. The shared
> helpers in `tests/_differential.py` are already available and should be reused.
> Do not assume that the generic registry or runner described below already
> exists.

## 1. Purpose and definition

A differential test checks whether the Python port behaves like the pinned
upstream JavaScript implementation for the same input.

A test is genuinely differential only when it:

1. Builds one input case.
2. Passes equivalent input data to the Python implementation and the JavaScript
   reference.
3. Executes the real Python implementation.
4. Executes the actual upstream JavaScript implementation.
5. Compares the two returned values directly.

These are **not** differential tests:

- checking the Python result against a hard-coded expected value and separately
  checking JavaScript against another hard-coded expected value;
- comparing two reimplementations that do not call the production functions;
- testing only that both implementations execute without raising an exception.

Hard-coded expected values remain useful for ordinary unit tests, but they do
not replace a comparison between the two implementations.

## 2. Proposed directory layout

The generic format centralizes the function pairs and their cases:

```text
tests/
├── _differential.py
├── differential/
│   ├── __init__.py
│   ├── config.py
│   ├── cases.py
│   └── test_all_pairs.py
└── js/
    └── differential-runner.mjs
```

Responsibilities:

| File | Responsibility |
| --- | --- |
| `_differential.py` | Shared process execution and recursive result comparison. |
| `differential/config.py` | The central registry of Python/JavaScript pairs. |
| `differential/cases.py` | Reusable, deterministic input cases. |
| `differential/test_all_pairs.py` | Generic pytest parametrization and execution. |
| `js/differential-runner.mjs` | Explicit mapping from stable IDs to upstream JavaScript functions. |

The registry is Python code, not JSON. It needs to hold Python callables,
adapters and tolerances, which cannot be represented reliably in a plain JSON
file. Test input data itself should remain JSON-serializable.

## 3. Define one uniform input protocol

Every test case should expose:

- a stable `id`, used in pytest output;
- an `input` object, containing JSON-serializable data.

Both language adapters receive that same input object. This avoids ambiguity
between positional arguments, keyword arguments and domain-specific objects.

For example, a case can be declared as:

```python
# tests/differential/cases.py

CASES = {
    "distance.basic": {
        "id": "basic",
        "input": {
            "point_a": [1.0, 0.0, 0.0],
            "point_b": [0.0, 1.0, 0.0],
            "radius": 6371.0,
        },
    },
    "distance.antipodal": {
        "id": "antipodal",
        "input": {
            "point_a": [1.0, 0.0, 0.0],
            "point_b": [-1.0, 0.0, 0.0],
            "radius": 6371.0,
        },
    },
}
```

Use deterministic inputs. If a function uses randomness, include the seed in
the input and make sure both implementations use the same seed and equivalent
random-number semantics. Do not rely on ambient global state.

## 4. Create the central configuration

Each registry entry describes one function pair. A practical configuration
record contains:

- `name`: human-readable pair name;
- `javascript_id`: stable ID understood by the JS runner;
- `python_call`: Python wrapper accepting the case input and calling the real
  production function;
- `cases`: one or more case definitions;
- `rtol` and `atol`: numeric comparison tolerances.

Example:

```python
# tests/differential/config.py

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from orogen.v2.core.sphere_geometry import cell_distance_km

from .cases import CASES


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class DifferentialPair:
    name: str
    javascript_id: str
    python_call: Callable[[JsonObject], Any]
    cases: tuple[JsonObject, ...]
    rtol: float = 2e-7
    atol: float = 2e-9


def _python_distance(data: JsonObject) -> float:
    # Adapt the JSON case to the production Python function's real signature.
    # This example is illustrative: use the actual function signature and
    # required domain objects in the real adapter.
    return cell_distance_km(
        data["xyz"],
        data["first"],
        data["second"],
    )


PAIRS = (
    DifferentialPair(
        name="Spherical cell distance",
        javascript_id="sphere.cellDistanceKm",
        python_call=_python_distance,
        cases=(
            {
                "id": "basic",
                "input": {
                    "xyz": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    "first": 0,
                    "second": 1,
                },
            },
        ),
        atol=1e-7,
    ),
)
```

The example adapter illustrates the structure only. Before adding a real pair,
check the actual Python and upstream JavaScript signatures and build valid
inputs for both.

### Why use a Python wrapper?

Python functions often require NumPy arrays, mesh objects, crust objects or
other domain structures. The case data should stay simple and serializable.
The wrapper converts that data into the structures required by the real Python
function.

The JavaScript side performs the equivalent conversion. Neither wrapper should
reimplement the algorithm being tested.

### Normalizing outputs

If the two APIs intentionally expose different representations, normalize the
outputs to the same observable data structure before comparison. Keep
normalization limited to representation differences, such as converting a
NumPy array to a list or renaming a field. Do not normalize away a behavioural
difference.

If normalization is needed for a particular pair, add an explicit
`python_result_adapter` or equivalent field to the configuration. Document
why it is necessary.

## 5. Register upstream JavaScript functions explicitly

The generic JavaScript runner should import the pinned upstream implementation
and expose a fixed registry of supported function IDs.

Conceptually:

```javascript
// tests/js/differential-runner.mjs

import { cellDistanceKm } from
  "../../external/planet_heightmap_generation/v2/js/core/sphere-geometry.js";

const FUNCTIONS = {
  "sphere.cellDistanceKm": (input) =>
    cellDistanceKm(input.xyz, input.first, input.second),
};

const request = JSON.parse(await readStdin());
const fn = FUNCTIONS[request.function];

if (typeof fn !== "function") {
  throw new Error(`Unknown differential function: ${request.function}`);
}

const result = await fn(request.input);
process.stdout.write(JSON.stringify(result));
```

This is a structural example: replace the import path and exported symbol with
the exact path and function name in the pinned upstream source.

Rules for the JavaScript registry:

- Import the real upstream function; do not copy its implementation into the
  test runner.
- Use stable, explicit IDs such as `sphere.cellDistanceKm`.
- Do not use `eval`, user-controlled dynamic imports, or arbitrary module paths.
- Keep standard output reserved for the single JSON result. Diagnostics belong
  on standard error.
- Ensure every result is JSON-serializable. Convert typed arrays or other
  special values explicitly where necessary.
- If a JavaScript function is asynchronous, await it before serializing the
  result.
- Treat `null` as an explicit result. Do not allow an `undefined` result to
  produce empty or invalid JSON output.

The actual upstream module paths and exports must be verified before adding
each function.

## 6. Implement the generic pytest executor

The generic test module expands every pair into one pytest item per case.

Example:

```python
# tests/differential/test_all_pairs.py

from __future__ import annotations

import pytest

from _differential import assert_js_python_equal, run_js_reference

from .config import PAIRS


RUNNER = "tests/js/differential-runner.mjs"


def _parameters():
    for pair in PAIRS:
        for case in pair.cases:
            yield pytest.param(
                pair,
                case,
                id=f"{pair.javascript_id}::{case['id']}",
            )


@pytest.mark.parametrize(("pair", "case"), list(_parameters()))
def test_python_matches_upstream_javascript(pair, case):
    data = case["input"]

    python_result = pair.python_call(data)
    javascript_result = run_js_reference(
        RUNNER,
        {
            "function": pair.javascript_id,
            "input": data,
        },
    )

    assert_js_python_equal(
        python_result,
        javascript_result,
        rtol=pair.rtol,
        atol=pair.atol,
    )
```

This executor deliberately has no knowledge of geometry, tectonics, collisions
or any other domain. Domain-specific setup belongs in the adapters and in the
explicit JavaScript registry.

The existing `run_js_reference` helper serializes the request as JSON, invokes
Node.js and parses its JSON output. The existing
`assert_js_python_equal` helper recursively compares dictionaries, arrays,
booleans and numeric values, including NumPy-backed Python outputs. Reuse these
helpers rather than introducing a second comparison implementation.

## 7. Handling complex functions

Some functions cannot be called from a few scalar values. They may require a
mesh, a world, a crust model, a seeded random generator or mutable state.

For these functions:

1. Define a compact JSON input specification.
2. Build the Python domain objects in `python_call`.
3. Build equivalent JavaScript domain objects in the JS registry wrapper.
4. Call the real production function in each language.
5. Return the same observable fields from both sides.
6. Compare the returned structures.

For stateful functions, compare equivalent initial states and observable final
states. If the function mutates an object, include the relevant post-call state
in the returned result. Do not compare only the direct return value if important
behaviour is expressed through mutation.

For random or noise-based functions, record the seed and all configuration
values in the case. Confirm that both implementations use compatible random
number generation before expecting bit-for-bit equality.

For large structures, return only the fields relevant to the behaviour under
test. Keep enough information to detect meaningful differences; do not reduce
the result to a single success flag.

## 8. Numeric tolerances and special values

Use the shared comparator's default tolerances unless the pair has a documented
reason to use different values.

- `rtol` controls relative floating-point tolerance.
- `atol` controls absolute floating-point tolerance.
- Use tighter tolerances where the algorithms and data types allow it.
- Use looser tolerances only when justified by known floating-point differences.
- Do not increase tolerances simply to make a failing test pass.

The current JSON request helper uses `allow_nan=False`. Therefore, test inputs
must not contain Python NaN or infinity. If a function's observable output can
contain non-finite values, define and document an explicit cross-language
representation for them before comparing. Do not silently convert them to
ordinary numbers or `null`.

The comparator should continue to distinguish booleans from numbers and report
the exact result path that differs. A useful failure should identify the
function pair, case ID and differing field or array index.

## 9. Adding a new function pair

For every new pair, follow this sequence:

1. **Locate the Python implementation.** Identify the production function and
   the domain objects it requires.
2. **Locate the upstream JavaScript implementation.** Verify the exact module,
   export and upstream revision. The reference must be the pinned upstream
   source, not a locally rewritten copy.
3. **Choose observable behaviour.** Decide which return values and mutations
   must match.
4. **Write deterministic cases.** Include ordinary inputs, boundaries and
   relevant edge cases. Give every case a stable ID.
5. **Write the Python adapter.** Convert the case input to the real Python
   function's arguments and call it.
6. **Register the JavaScript adapter.** Convert the same input to the JS
   function's arguments and call the real upstream function.
7. **Add one entry to `PAIRS`.** Set the tolerances only when justified.
8. **Run the targeted pytest item.** Inspect failures; do not adjust the
   reference or loosen tolerances without understanding the discrepancy.
9. **Run the relevant wider test suite.** Preserve the repository's coverage,
   lint, formatting and typing requirements.

A new function pair should normally require changes only to the cases,
configuration and JavaScript registry. The generic pytest executor should not
need to change for each new domain function.

## 10. What belongs outside the generic registry?

The generic engine is for direct Python-versus-JavaScript behavioural
comparisons. It does not replace every test in the repository.

Keep separate tests where they serve a different purpose, including:

- pure-Python fallback and Numba kernel coverage;
- branch-coverage tests for internal implementation paths;
- integration tests involving full simulations or multiple time steps;
- invariants that have no direct JavaScript counterpart;
- tests for Python-specific error handling or infrastructure;
- tests whose setup or lifecycle cannot be represented cleanly by the generic
  input protocol.

A test that checks a Python invariant is valuable, but should not be labelled
as a differential test unless it actually compares the two implementations.

## 11. Migration plan

Do not convert all existing tests in one large change.

1. Add the generic registry and runner while keeping existing tests intact.
2. Validate the framework with a small number of simple functions.
3. Add a function that returns arrays or nested objects.
4. Add a complex function that needs domain-object adapters.
5. Migrate existing differential pairs incrementally, preserving their cases
   and assertions.
6. Keep Python-only and integration tests in their appropriate modules.
7. Update CI only after the generic suite has equivalent coverage and has been
   verified locally and in CI.
8. Remove an old domain-specific runner only when every comparison it provided
   has been migrated and the replacement is demonstrably equivalent.

The generic format must not reduce the number of meaningful cases or weaken
existing comparisons during migration.

## 12. Acceptance checklist

Before considering a pair complete, verify:

- [ ] The Python adapter calls the real production Python function.
- [ ] The JavaScript adapter calls the real pinned upstream function.
- [ ] Both sides receive equivalent data from the same case input.
- [ ] The case is deterministic, including its random seed where applicable.
- [ ] The comparison covers the relevant return values and observable mutations.
- [ ] Any output normalization changes representation only, not behaviour.
- [ ] Numeric tolerances are justified and no wider than necessary.
- [ ] The pytest ID identifies the function pair and case.
- [ ] The test fails when a real Python/JavaScript discrepancy is introduced.
- [ ] Existing unit, integration, fallback and coverage tests remain in place.

## 13. Repository quality requirements

Follow the repository instructions in `AGENT.md` and `QUALITY.md`.

In particular:

- Never modify a differential test merely to make a mismatch pass. Investigate
  and document the discrepancy first.
- Do not lower the 99% branch-coverage threshold.
- Preserve the pure-Python fallback for Numba-accelerated paths.
- Do not add runtime dependencies for the test framework without an explicit
  decision.
- Do not bypass CI, lint, formatting, typing or security checks.
- Do not claim that compatibility is complete based on a small number of
  differential cases.

The generic registry is intended to make adding comparisons easier and more
consistent. It is not a substitute for representative cases, correct adapters,
or review of the actual upstream behaviour.
