# Differential tests: Python port against the JavaScript reference

`tests/test_generated.py` replays recorded calls against the pinned upstream
JavaScript. Each recorded call goes to the Python function and to the JavaScript
export, and the results are compared. Node.js is the reference oracle. JSPyBridge
is a second access path into the same Node.js runtime, not an independent
JavaScript engine.

The recorded calls come from CrossHair. CrossHair only supplies arguments: the
corpora hold no expected values, so the JavaScript reference alone decides.

The reference modules are re-exports under `tests/reference/v2/`. They have no
dependencies of their own, so no `npm install` is needed.

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

The differential CI job also writes an HTML coverage report to `htmlcov/`
and uploads it as the `differential-coverage-html-<run id>` workflow artifact
(retained for 14 days). On pushes to `main`, the same report is published to
GitHub Pages by a dedicated deployment job. Pull-request runs only build and
upload the artifact; they do not publish a public site. The published report is
available at:

`https://thomas-gautron2004.github.io/Orogen_python/`

The Pages deployment uses the `github-pages` environment and the
`actions/deploy-pages` workflow. The repository's Pages source must be set to
**GitHub Actions** in Settings → Pages.

CI runs the corpus generator before the replay in
`.github/workflows/differential-jspybridge.yml`. The workflow restores
`tests/corpus/` from a branch-scoped GitHub Actions cache, regenerates only
modules whose source fingerprint changed, then saves the updated corpus under
a new cache key for the next run. The checked-in corpus remains the fallback
when no cache exists. Cache entries are subject to GitHub Actions cache
retention and eviction policies; the checked-in files are the durable baseline.

## Regenerate the corpora

The test suite never runs CrossHair. Each opted-in module has a corpus file,
committed as `tests/corpus/<dotted module name>.json`, for example
`tests/corpus/orogen.v2.core.rotation.json`.

```bash
python -m pip install "crosshair-tool==0.0.114"
python tests/crosshair_corpus.py
```

With no argument, the tool regenerates only the opted-in modules whose
fingerprint changed. Name one or more modules to restrict it:
`python tests/crosshair_corpus.py rotation`. Other options:

- `--dry-run`: list what would be regenerated, without running CrossHair.
- `--force`: regenerate everything.
- `--timeout N`: CrossHair time per condition, in seconds. Defaults to
  `crosshair_timeout` on the settings line.

CrossHair reads the annotations of each public function and reports one example
per path, as an argument dictionary. The tool replays each example through the
Python function, to record whether Python raised. A module's corpus is
regenerated only when its fingerprint changes.

**Fingerprint.** Each corpus records a SHA-256 over its module's source, the
orogen files it imports (followed transitively), the parent packages'
`__init__.py` files, and the CrossHair budget. Editing `seed.py` regenerates
`seed` and every module that imports it, and leaves `rotation` alone. Editing a
package `__init__.py` regenerates every module under it.

**Stale corpora.** The tool reports corpus files whose module no longer opts in.
Delete them by hand.

## Configuration lives in comments

There is no configuration file.

**Global settings** are one comment line in `tests/test_generated.py`. Missing
keys keep their defaults:

```python
# differential-settings: source=orogen backends=node,jspybridge crosshair_timeout=5
```

- `source`: directory scanned for opted-in modules, recursively. Defaults to
  `orogen`.
- `backends`: any combination of `node` and `jspybridge`.
- `crosshair_timeout`: CrossHair time per condition, in seconds, when a corpus
  is regenerated.

**A module opts in** with a comment line near the top of its file. Only
opted-in modules are replayed. A module must live under `orogen/v2/`, so that
its reference can be found:

```python
# differential-module: js_module=sphere-geom.mjs mode=approx rel_tol=1e-9 abs_tol=1e-9 params=q:quaternion,omega:vec3
```

- `js_module`: JavaScript file, relative to `tests/reference/v2/`. Without a
  slash it is relative to the module's directory: `sphere-geom.mjs` means
  `core/sphere-geom.mjs` for a module in `orogen/v2/core/`. With a slash it is a
  path from `tests/reference/v2/`, so `js/simplex-noise.mjs` names a reference
  outside the module's directory. Without `js_module`, the lookup tries
  `<stem>.mjs`, then `<stem with _ replaced by ->.mjs`.
- `mode`: `exact` for seeds, flags and discrete values, or `approx` for floats.
  `rel_tol` and `abs_tol` apply only in `approx`.
- `params`: type per parameter name, as `name:type` pairs separated by commas.
  Types are `unit_float`, `angle`, `time_step` (numbers), `integer`, `boolean`,
  `string`, `quaternion` (4 numbers), `vec3` (3 numbers), `number_list`,
  `integer_list`, `coordinate_buffer` (flat xyz triples), `cell_index`,
  `cell_indices`, `seed_text` and `seed_input`. Parameters not listed fall back
  to their annotation: `float`, `int | float` and `float | int` map to
  `unit_float`; `int`, `bool`, `str`, `SeedInput`, `Sequence[float]` and
  `Iterable[int]` map to their corresponding kinds. A parameter with no
  supported type makes its function get no replay.

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

**Classes.** A public class whose JavaScript class is exported gets one replay per
public method. The replay builds the instance from the constructor arguments
recorded with each example, then calls the method, in Python and with `new` in
JavaScript. A marker above a method renames its JavaScript member, for example
`# differential: js=noise3D` above `def noise3d`.

A function is replayed only if its name does not start with `_` and its
JavaScript module exports the matching name. Missing exports are ignored
silently, unless the marker names one explicitly. A marked module with no
reference fails collection, and the error names the reference to add, if one
exists under another directory.

## Adding a module

1. Make sure its JavaScript re-export exists in the matching directory under
   `tests/reference/v2/` (for example `core/` for `orogen/v2/core/`), under the
   same name as the Python file or with `-` in place of `_`. Use `js_module=` if
   the name differs.
2. Add the module comment to the Python file. Use `params=` for any argument
   whose type is not implied by its annotation.
3. Run `python tests/crosshair_corpus.py <module>` to create its corpus, then run
   the replay.

No runner or loader change is needed. The runners load only validated paths
under `tests/reference/v2/`.

## Replay

Each corpus entry becomes one test per backend. An entry is reported as skipped,
with its reason, when:

- its argument count differs from the function's parameters;
- an argument does not have the type its parameter expects (for example a
  quaternion of the wrong length);
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

Both load reference modules by path under `tests/reference/v2/`, for example
`core/seed.mjs`. Any other path is rejected. Both receive the same JSON request
and return the same JSON result. Results are compared recursively. Non-finite
numbers are encoded as `{"__number__": "NaN" | "Infinity" | "-Infinity"}` on
both sides.
