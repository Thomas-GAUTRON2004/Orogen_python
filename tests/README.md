# Differential tests: Python port against the JavaScript reference

Property-based comparisons between the Python port in `orogen/v2/core` and the
pinned upstream JavaScript. Hypothesis draws inputs, each input is sent to the
Python function and to the JavaScript reference, and the results are compared.
Node.js is the reference oracle. JSPyBridge is a second access path into the
same Node.js runtime, not an independent JavaScript engine.

The reference modules are re-exports under `tests/reference/v2/core/`. They have
no dependencies of their own, so no `npm install` is needed.

## Run

```bash
python -m pip install -e ".[test]" "hypothesis>=6.100,<7" "javascript==1!1.2.6"
python -m pytest tests/test_generated.py -q
```

Node.js 22 is required. JSPyBridge needs Node.js 18 or newer. If the
`javascript` package is missing, only its backend is skipped.

List the generated tests without running them:

```bash
python -m pytest tests/test_generated.py --collect-only -q
```

CI runs the same command in `.github/workflows/differential-jspybridge.yml`.

## Configuration lives in comments

There is no configuration file.

**Global settings** are one comment line in `tests/test_generated.py`. Missing
keys keep their defaults:

```python
# differential-settings: source=orogen/v2/core max_examples=250 deadline_ms=1000 seed=20261126 backends=node,jspybridge pynguin_seconds=60 pynguin_seed=1
```

- `source`: Python directory scanned for opted-in modules.
- `max_examples`: Hypothesis examples per generated test. Higher finds rarer
  divergences, at a linear cost in time.
- `deadline_ms`: maximum time for one example. Raise it for slow functions.
- `seed`: fixed Hypothesis seed for reproducible runs, or `none` for
  derandomized runs.
- `backends`: any combination of `node` and `jspybridge`.
- `pynguin_seconds`, `pynguin_seed`: default search budget and seed used when a
  Pynguin corpus is regenerated. Neither changes the tests themselves.

A deeper run with a longer corpus search:

```python
# differential-settings: max_examples=2000 deadline_ms=2000 pynguin_seconds=300 pynguin_seed=7
```

**A module opts in** with a comment line near the top of its file. Only opted-in
modules are tested:

```python
# differential-module: js_module=sphere-geom.mjs mode=approx rel_tol=1e-9 abs_tol=1e-9 params=q:quaternion,omega:vec3
```

- `js_module`: JavaScript file name, when it is not derived automatically. The
  automatic lookup tries `<stem>.mjs`, then `<stem with _ replaced by ->.mjs`.
- `mode`: `exact` for seeds, flags and discrete values, or `approx` for floats.
  `rel_tol` and `abs_tol` apply only in `approx`.
- `params`: strategy per parameter name, as `name:strategy` pairs separated by
  commas. Parameters not listed fall back to their annotation: `float` gives
  `unit_float`, `object` gives `seed_text`. A parameter with no strategy makes
  its function get no test.

**A function can refine its test** with a comment directly above its `def`:

```python
# differential: js=quatMul strategies=quaternion,quaternion mode=approx
# differential: skip
```

- `js`: JavaScript export to call, when it is not the camelCase name. The test
  fails if that export does not exist.
- `strategies`: one strategy per required argument, overriding inference.
- `mode`: comparison for this function only.
- `skip`: no test. Use it for non-deterministic functions, such as `random_seed`,
  and for known divergences. Put the reason in a comment above the marker.

A function gets a test only if its name does not start with `_` and its
JavaScript module exports the matching name. Missing exports are ignored
silently, unless the marker names one explicitly.

**Example.** `orogen/v2/core/rotation.py` opts in with `params=`, so its
quaternion arguments get quaternion strategies. `quat_to_mat3` is skipped with a
stated reason: the upstream requires an `out` argument that the Python port
makes optional.

## Adding a module

1. Make sure its JavaScript re-export exists in `tests/reference/v2/core/`, under
   the same name as the Python file or with `-` in place of `_`. Use
   `js_module=` if the name differs.
2. Add the module comment to the Python file. Use `params=` for any argument
   whose name is not implied by its annotation.
3. Run `--collect-only` and check the list.

No runner or loader change is needed. The runners load only validated file
names from `tests/reference/v2/core`.

## Backends

- `node`: `property-runner.mjs`, a persistent process speaking JSON lines.
- `jspybridge`: `jspybridge-loader.cjs`, loaded through JSPyBridge.

Both backends receive the same JSON request and return the same JSON result.
Results are compared recursively. Non-finite numbers are encoded as
`{"__number__": "NaN" | "Infinity" | "-Infinity"}` on both sides.

## Pynguin corpus

Pynguin produces inputs, not expected values. `tests/pynguin_corpus.py` runs it
on one module and records the arguments of every call, plus whether Python
raised. The result is `tests/corpus/<module>.json`, which is data, not
configuration.

Install Pynguin, then regenerate the corpus:

```bash
python -m pip install pynguin
PYNGUIN_DANGER_AWARE=1 python tests/pynguin_corpus.py rotation --seconds 60 --seed 1
```

The budget and seed default to `pynguin_seconds` and `pynguin_seed` on the
settings line. The command-line options override them for one run. Pynguin
executes the project's code, so the environment variable is required. Its own
assertions are not kept: they would freeze whatever Python returns.

`test_corpus_replay` replays each recorded call against the JavaScript
reference, with the same comparison as the generated tests. An entry is reported
as skipped, with its reason, when:

- its argument count differs from the target's strategies;
- an argument does not have the type the strategy expects (for example bytes
  where a quaternion is expected);
- an argument is not finite, since NaN and infinities cannot be sent on the
  JSON wire.

If Python raises for an entry, the test passes only when the JavaScript target
also raises.

Only functions that are generated targets are replayed. Other recorded calls,
such as internal helpers or excluded functions, stay in the file but are not
replayed. Regenerate a corpus when the code changes. The test suite never runs
Pynguin.
