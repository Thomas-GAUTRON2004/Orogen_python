# Generated JavaScript/Python differential tests

The generated differential suite uses Hypothesis to explore inputs and sends
each generated value to the Python implementation and to one or more JavaScript
backends. The Node.js runner is the reference oracle; JSPyBridge is an
experimental second access path into the same pinned upstream ES module.

## Install and run

Install the normal project test dependencies, Node.js 22, Hypothesis and the
optional JSPyBridge package:

```bash
python -m pip install -e ".[test]"
python -m pip install hypothesis
python -m pip install "javascript==1!1.2.6"
npm install --no-save --package-lock=false
python -m pytest tests/differential -q
```

JSPyBridge requires Node.js 18 or newer. It is deliberately kept out of the
runtime dependencies: it is an optional test backend. If the `javascript`
package is not installed, only the JSPyBridge parametrizations are skipped;
the Node.js oracle tests still run.

## Backends

- `node`: starts the persistent `property-runner.mjs` process. This remains
  the official reference path and preserves the existing JSON-lines protocol.
- `jspybridge`: loads `tests/differential/jspybridge-loader.cjs` through
  JSPyBridge. That CommonJS adapter exposes the same exports from the pinned
  `tests/reference/v2/core/seed.mjs` pointer using Node 22's synchronous
  CommonJS-to-ESM interop. JSPyBridge manages its own persistent Node.js process.

The two backends are not treated as independent JavaScript engines: both
execute Node.js. The comparison isolates the bridge/transport path, not V8
semantics versus another JavaScript engine. Node remains the authority if the
two backends disagree.

## Configuration

Edit `tests/config.json`:

- `settings.max_examples`: generated examples per target and backend.
- `settings.deadline_ms`: per-example deadline in milliseconds.
- `settings.seed`: fixed Hypothesis seed; use `null` for derandomized runs.
- `settings.backends`: any combination of `node` and `jspybridge`.
- `targets`: one entry per generated comparison test.

Targets come from two sources.

Configured targets are listed explicitly under `targets`:

- `id` is the snake_case name of a public function in the module (`seed_of`).
- `module` (optional, default `seed`) selects one of the modules allowlisted in
  `TARGET_MODULES` (`test_generated.py`).
- `strategy` selects a Hypothesis strategy from `STRATEGIES`. For functions with
  several inputs, use `strategies` with one entry per argument.

Generated targets come from `generate`:

- `modules` lists the modules to introspect. For each public function, a test is
  created when a same-named camelCase export exists in the pinned JavaScript
  module (`quat_mul` maps to `quatMul`), and every required parameter has a
  strategy.
- A parameter's strategy is taken from `param_strategies` by parameter name
  first, then from its annotation (`float` gives `unit_float`, `object` gives
  `seed_text`). Parameters with default values are left to their defaults.
- `exclude` lists functions that must not be generated, for example
  non-deterministic ones or known divergences. Each exclusion should carry a
  comment explaining why.
- `comparison` applies to all generated tests.

Node and JSPyBridge both receive `module`, `target` and `args`, and resolve the
JavaScript export by the same camelCase rule. Results cross the bridge as JSON
text, so both backends use the same wire format.

Comparison modes currently supported are:

- `exact`: exact equality, suitable for seeds, flags and discrete state.
- `approx`: scalar numeric comparison using configurable `rel_tol` and
  `abs_tol`.

To add a module to the generation, add it to `TARGET_MODULES`, add its
re-export under `tests/reference/v2/core/`, register it in both JavaScript
runners, and list it under `generate.modules`. Tests for its public functions
are then created automatically. A new input domain needs a strategy in
`STRATEGIES` and, when it is not implied by the annotation, an entry in
`param_strategies`.

Hypothesis drives the same generated input through both implementations and
shrinks a failing input to a smaller counterexample. JSPyBridge is an
experimental backend: it is not a replacement for the persistent Node runner
or the official compatibility oracle.
