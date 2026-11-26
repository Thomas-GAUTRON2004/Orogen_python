# Differential tests: Python port against the JavaScript reference

`tests/test_generated.py` replays calls recorded from Pynguin against the
pinned upstream JavaScript. Each recorded call goes to the Python function and
to the JavaScript export, and the results are compared. Node.js is the
reference oracle. JSPyBridge is a second access path into the same Node.js
runtime, not an independent JavaScript engine.

Pynguin only produces inputs. Its own assertions are not used: they would freeze
whatever Python returns.

The reference modules are re-exports under `tests/reference/v2/core/`. They have
no dependencies of their own, so no `npm install` is needed.

## Run

```bash
python -m pip install -e ".[test]" "javascript==1!1.2.6"
python -m pytest tests/test_generated.py -q
```

Node.js 22 is required. JSPyBridge needs Node.js 18 or newer. If the
`javascript` package is missing, only its backend is skipped.

Show the reason for each skipped entry:

```bash
python -m pytest tests/test_generated.py -q -rs
```

CI runs the same replay in `.github/workflows/differential-jspybridge.yml`.

## Regenerate a corpus

The test suite never runs Pynguin. A corpus is a file committed in
`tests/corpus/<module>.json`, regenerated when the Python code changes:

```bash
python -m pip install pynguin
PYNGUIN_DANGER_AWARE=1 python tests/pynguin_corpus.py rotation
PYNGUIN_DANGER_AWARE=1 python tests/pynguin_corpus.py seed
```

Pynguin executes the project's code, so the environment variable is required.
The budget and seed default to `pynguin_seconds` and `pynguin_seed` on the
settings line. `--seconds` and `--seed` override them for one run.

Regenerating a corpus only changes the replayed inputs. It does not change the
tests.

## Configuration lives in comments

There is no configuration file.

**Global settings** are one comment line in `tests/test_generated.py`. Missing
keys keep their defaults:

```python
# differential-settings: source=orogen/v2/core backends=node,jspybridge pynguin_seconds=60 pynguin_seed=1
```

- `source`: Python directory scanned for opted-in modules.
- `backends`: any combination of `node` and `jspybridge`.
- `pynguin_seconds`, `pynguin_seed`: default budget and seed for regenerating a
  corpus.

**A module opts in** with a comment line near the top of its file. Only
opted-in modules are replayed:

```python
# differential-module: js_module=sphere-geom.mjs mode=approx rel_tol=1e-9 abs_tol=1e-9 params=q:quaternion,omega:vec3
```

- `js_module`: JavaScript file name, when it is not derived automatically. The
  automatic lookup tries `<stem>.mjs`, then `<stem with _ replaced by ->.mjs`.
- `mode`: `exact` for seeds, flags and discrete values, or `approx` for floats.
  `rel_tol` and `abs_tol` apply only in `approx`.
- `params`: type per parameter name, as `name:type` pairs separated by commas.
  Types are `unit_float`, `angle`, `time_step` (numbers), `quaternion` (4
  numbers), `vec3` (3 numbers) and `seed_text` (a string). Parameters not listed
  fall back to their annotation: `float` gives `unit_float`, `object` gives
  `seed_text`. A parameter with no type makes its function get no replay.

**A function can refine its replay** with a comment directly above its `def`:

```python
# differential: js=quatMul mode=approx
# differential: skip
```

- `js`: JavaScript export to call, when it is not the camelCase name. The
  replay fails if that export does not exist.
- `mode`: comparison for this function only.
- `skip`: no replay. Use it for non-deterministic functions, such as
  `random_seed`, and for known divergences. Put the reason in a comment above
  the marker.

A function is replayed only if its name does not start with `_` and its
JavaScript module exports the matching name. Missing exports are ignored
silently, unless the marker names one explicitly.

**Example.** `orogen/v2/core/rotation.py` opts in with `params=`, so its
quaternion arguments are recognised. `quat_to_mat3` is skipped with a stated
reason: the upstream requires an `out` argument that the Python port makes
optional.

## Adding a module

1. Make sure its JavaScript re-export exists in `tests/reference/v2/core/`, under
   the same name as the Python file or with `-` in place of `_`. Use
   `js_module=` if the name differs.
2. Add the module comment to the Python file. Use `params=` for any argument
   whose type is not implied by its annotation.
3. Regenerate its corpus with `tests/pynguin_corpus.py`, then run the replay.

No runner or loader change is needed. The runners load only validated file
names from `tests/reference/v2/core`.

## Replay

Each corpus entry becomes one test per backend. An entry is reported as skipped,
with its reason, when:

- its argument count differs from the function's parameters;
- an argument does not have the type its parameter expects (for example bytes
  where a quaternion is expected);
- an argument is not finite, since NaN and infinities cannot be sent on the
  JSON wire.

If Python raises for an entry, the test passes only when the JavaScript target
also raises.

Only functions that are replay targets are replayed. Other recorded calls, such
as internal helpers or excluded functions, stay in the corpus file but are not
replayed.

## Backends

- `node`: `property-runner.mjs`, a persistent process speaking JSON lines.
- `jspybridge`: `jspybridge-loader.cjs`, loaded through JSPyBridge.

Both backends receive the same JSON request and return the same JSON result.
Results are compared recursively. Non-finite numbers are encoded as
`{"__number__": "NaN" | "Infinity" | "-Infinity"}` on both sides.
