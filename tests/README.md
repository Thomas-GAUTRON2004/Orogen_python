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

Configured targets are listed explicitly under `targets`. Each names a Python
module (`module`, a file stem under `generate.source`, default `seed`), a
public function (`id`), and one strategy per argument (`strategy`, or
`strategies` for several arguments).

Generated targets are discovered from the source directory `generate.source`:

- `modules` selects Python modules by file stem, or `"all"`.
- A function gets a test when a JavaScript module exists for its file (same
  stem, or with `_` replaced by `-`, or set explicitly in `js_modules`) and that
  module exports the camelCase name (`quat_mul` maps to `quatMul`).
- Parameters get a strategy from `param_strategies` by name first, then from
  their annotation (`float` gives `unit_float`, `object` gives `seed_text`).
  Parameters with default values keep their defaults.
- `exclude` lists functions to skip globally, with a comment explaining why.

A comment directly above a function refines its generated test:

```python
# differential: js=quatMul strategies=quaternion,quaternion mode=approx
# differential: skip
```

- `js=` names a different JavaScript export. It fails if the export is missing.
- `strategies=` gives the strategy per argument, overriding inference.
- `mode=` sets the comparison for this function.
- `skip` excludes the function. Use it for non-deterministic functions such as
  `random_seed`.

Node and JSPyBridge both receive a module file name, an export name and `args`.
Both load the file from `tests/reference/v2/core` and reject any other path.
Results cross the bridge as JSON text, so both backends use the same wire format.

Comparison modes currently supported are:

- `exact`: exact equality, suitable for seeds, flags and discrete state.
- `approx`: scalar numeric comparison using configurable `rel_tol` and
  `abs_tol`.

To add a module, put its Python file in `generate.source` and its JavaScript
re-export in `tests/reference/v2/core/`. If the names differ, add a
`js_modules` entry. No runner changes are needed. Tests appear on the next run.
Check the list with `python -m pytest tests/test_generated.py --collect-only -q`.

Hypothesis drives the same generated input through both implementations and
shrinks a failing input to a smaller counterexample. JSPyBridge is an
experimental backend: it is not a replacement for the persistent Node runner
or the official compatibility oracle.
