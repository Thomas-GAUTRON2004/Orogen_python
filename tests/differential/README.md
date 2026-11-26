# Generated JavaScript/Python differential tests

The generated differential suite uses Hypothesis to explore inputs and sends
each generated value to both the Python implementation and a persistent Node.js
process running the pinned upstream JavaScript reference.

## Install and run

Install the normal project test dependencies, Node.js 22, and Hypothesis:

```bash
python -m pip install -e ".[test]"
python -m pip install hypothesis
npm install --no-save --package-lock=false
python -m pytest tests/differential -q
```

Hypothesis is intentionally installed separately in this first iteration. The
repository CI workflows and dependency lockfile are not changed by this
addition.

## Configuration

Edit `tests/differential/config.json`:

- `settings.max_examples`: generated examples per target.
- `settings.deadline_ms`: per-example deadline in milliseconds.
- `settings.seed`: fixed Hypothesis seed; use `null` for derandomized runs.
- `targets`: target IDs, strategy IDs and comparison settings.

The Python callable, JavaScript callable and Hypothesis strategy are selected
through static registries in `test_generated.py` and
`tests/reference/v2/property-runner.mjs`. Configuration cannot import
arbitrary code.

Comparison modes currently supported are:

- `exact`: exact equality, suitable for seeds, flags and discrete state.
- `approx`: scalar numeric comparison using configurable `rel_tol` and
  `abs_tol`.

To add a target, add its Python callable to `TARGETS`, its JavaScript
counterpart to the Node runner, and a suitable strategy to `STRATEGIES`.
Then register it in the JSON configuration.

Hypothesis is the initial generator because it can drive the same generated
input through both implementations and shrink a failing input to a smaller
counterexample. Pynguin can complement this later by generating Python unit-test
candidates, but it does not replace the paired JavaScript execution or the
cross-language comparison layer.
