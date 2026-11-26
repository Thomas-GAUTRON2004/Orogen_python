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

Each target is resolved automatically:

- `id` is the snake_case name of a public function in the module (`seed_of`).
- `module` (optional, default `seed`) selects one of the modules allowlisted in
  `TARGET_MODULES` (`test_generated.py`). The Node runner and the JSPyBridge
  loader expose the same module names.
- The JavaScript export is derived from `id` in camelCase (`seed_of` becomes
  `seedOf`). No per-function registration is needed on the JavaScript side.
- `strategy` selects a Hypothesis strategy from `STRATEGIES`, since input
  domains are the only part that must be written by hand.

A target that does not resolve to a public function fails its generated test
with an explicit error on the Python side, or on the JavaScript side for the
Node and JSPyBridge backends.

Comparison modes currently supported are:

- `exact`: exact equality, suitable for seeds, flags and discrete state.
- `approx`: scalar numeric comparison using configurable `rel_tol` and
  `abs_tol`.

To add a comparison, make sure the function is exported with the same name
(snake_case in Python, camelCase in the pinned JavaScript module), add it to
`tests/config.json`, and add a strategy to `STRATEGIES` if the existing ones
do not fit.

Hypothesis drives the same generated input through both implementations and
shrinks a failing input to a smaller counterexample. JSPyBridge is an
experimental backend: it is not a replacement for the persistent Node runner
or the official compatibility oracle.
