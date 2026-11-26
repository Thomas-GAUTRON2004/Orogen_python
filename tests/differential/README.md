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
- `jspybridge`: imports the same `tests/reference/v2/core/seed.mjs` pointer
  through JSPyBridge and invokes its exported functions directly. JSPyBridge
  starts and manages its own persistent Node.js process.

The two backends are not treated as independent JavaScript engines: both
execute Node.js. The comparison isolates the bridge/transport path, not V8
semantics versus another JavaScript engine. Node remains the authority if the
two backends disagree.

## Configuration

Edit `tests/differential/config.json`:

- `settings.max_examples`: generated examples per target and backend.
- `settings.deadline_ms`: per-example deadline in milliseconds.
- `settings.seed`: fixed Hypothesis seed; use `null` for derandomized runs.
- `settings.backends`: any combination of `node` and `jspybridge`.
- `targets`: target IDs, strategy IDs and comparison settings.

The Python callable, JavaScript callable and Hypothesis strategy are selected
through static registries in `test_generated.py` and
`tests/reference/v2/property-runner.mjs`. Configuration cannot import
arbitrary code.

Comparison modes currently supported are:

- `exact`: exact equality, suitable for seeds, flags and discrete state.
- `approx`: scalar numeric comparison using configurable `rel_tol` and
  `abs_tol`.

To add a target, add its Python callable to `TARGETS`, its JavaScript export to
`JSPyBridgeSession`, its Node counterpart to the Node runner, and a suitable
strategy to `STRATEGIES`. Then register it in the JSON configuration.

Hypothesis drives the same generated input through both implementations and
shrinks a failing input to a smaller counterexample. JSPyBridge is an
experimental backend: it is not a replacement for the persistent Node runner
or the official compatibility oracle.
