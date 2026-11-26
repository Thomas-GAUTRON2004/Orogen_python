# Generated JavaScript/Python differential tests

Property-based comparisons between the Python port in `orogen/v2/core` and the
pinned upstream JavaScript. Hypothesis generates inputs, each input is sent to
the Python function and to one or more JavaScript backends, and the results are
compared. Node.js is the reference oracle. JSPyBridge is a second access path
into the same Node.js runtime, not an independent JavaScript engine.

## Run

```bash
python -m pip install -e ".[test]" hypothesis "javascript==1!1.2.6"
npm install --no-save --package-lock=false
python -m pytest tests/test_generated.py -q
```

List the generated tests without running them:

```bash
python -m pytest tests/test_generated.py --collect-only -q
```

Node.js 22 is required. JSPyBridge needs Node.js 18 or newer. If the
`javascript` package is missing, only its backend is skipped.

## Configuration lives in comments

There is no configuration file.

**Global settings** are one comment line in `tests/test_generated.py`. Missing
keys keep their defaults:

```python
# differential-settings: source=orogen/v2/core max_examples=250 deadline_ms=1000 seed=20261126 backends=node,jspybridge
```

- `source`: Python directory scanned for opted-in modules.
- `max_examples`, `deadline_ms`: Hypothesis limits per example and per test.
- `seed`: fixed seed for reproducible runs, or `none` for derandomized runs.
- `backends`: any combination of `node` and `jspybridge`.

**A module opts in** with a comment line near the top of its file. Only opted-in
modules are tested:

```python
# differential-module: js_module=sphere-geom.mjs mode=approx rel_tol=1e-9 abs_tol=1e-9 params=q:quaternion,omega:vec3
```

- `js_module`: JavaScript module file, when the name is not derived automatically.
  The automatic lookup tries `<stem>.mjs`, then `<stem with _ replaced by ->.mjs`.
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

Functions whose name starts with `_` are never tested. A function gets a test
only if its JavaScript module exports the matching name. Missing exports are
ignored silently, unless the marker names one explicitly.

## Example

`orogen/v2/core/rotation.py` opts in with `params=` so its quaternion arguments
get quaternion strategies. Its `quat_to_mat3` is skipped with a stated reason,
because the upstream requires an `out` argument that the Python port makes
optional.

## Adding a module

1. Put its JavaScript re-export in `tests/reference/v2/core/`, under the same
   name as the Python file or with `-` in place of `_`. Use `js_module=` if the
   name differs.
2. Add the module comment to the Python file. Use `params=` for any argument
   whose name is not implied by its annotation.
3. Run `--collect-only` and check the list.

No runner or loader changes are needed. Runners only load validated file names
from `tests/reference/v2/core`.

## Backends

- `node`: `property-runner.mjs`, a persistent process speaking JSON lines.
- `jspybridge`: `jspybridge-loader.cjs`, loaded through JSPyBridge. It returns
  JSON text, so both backends use the same wire format.

Results are compared recursively. Non-finite numbers are encoded as
`{"__number__": "NaN" | "Infinity" | "-Infinity"}` on both sides.
