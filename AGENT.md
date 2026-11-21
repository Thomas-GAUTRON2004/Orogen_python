
# Agent Instructions — Orogen_python
 
This file is for any agent (AI or automation) that modifies this repository.
Read it and `QUALITY.md` before making any change.
 
## 1. What This Project Is
 
Orogen_python ports the JavaScript V2 engine of Orogen 3.0.1 to Python. Upstream
source: `raguilar011095/planet_heightmap_generation`, tag `v3.0.1`, commit
`ed466f9db71292e912d8d0160e689a67a994f2d9`.
 
The port is **partial**. Never claim that a module is "complete" or "compatible"
without evidence (differential tests over several time steps).
 
Sources of truth, in order of precedence:
1. `docs/PORTING_STATUS.md`: porting status and priorities.
2. `quality.md`: quality rules.
3. `docs/SCIPY_BACKEND.md`: production triangulation choice and its limits.
4. `UPSTREAM_SOURCE.md` (on `main`): exact upstream reference.
5. `NOTICE.md`: licence and attribution.
## 2. Branches and Scope
 
- Porting reference branch: `port/v3.0.1`.
- Quality and tooling changes are proposed on dedicated branches and reviewed
  through pull requests. Do not assume a particular shared quality branch is current.
- Never push directly to `port/v3.0.1`, `main` or a shared quality branch. Create
  a dedicated branch (`docs/...`, `fix/...`, `feat/...`, `refactor/...`) and open
  a pull request against the branch the change belongs to.
- Never rewrite the history of a shared branch (`push --force`, rebasing a
  published branch, `reset` on a pushed commit).
- Do not delete `experiment/*` branches. They hold the measurements cited in the
  documentation.
## 3. Before Changing Code
 
1. Read the section for the affected module in `docs/PORTING_STATUS.md`.
2. Read the existing tests for the module and the associated differential tests.
3. Fetch the remote and check the current head of the branch (`git fetch`). The
   branch may have moved since your last read.
4. If the behaviour is ported, locate the JavaScript reference function in
   `v2/js/` of the upstream repository.
## 4. While Changing Code
 
- Keep changes small and focused: one commit, one intention.
- Do not mix refactoring, bug fixes, formatting and new features.
- Preserve behaviour during a refactoring. Existing tests must pass **without
  modification**. If a test has to change, stop and report it.
- Do not modify a differential test against the JavaScript reference to make a
  result pass.
- Do not add a runtime dependency without an explicit request.
- Do not remove SciPy from the runtime dependencies, and do not replace
  `scipy.spatial.Delaunay` (Qhull) with another triangulator, without a
  correctness benchmark and a maintainer decision (see `docs/SCIPY_BACKEND.md`).
- Do not replace `SphereIndex` without a correctness benchmark (see the
  `experiment/ckdtree-sphere-index` branch).
- Keep a pure-Python fallback for any path accelerated by Numba.
- Use float64 for spherical geometry and point location.
## 5. Prohibitions
 
- Do not disable or skip a CI step (`--no-verify`, removing `bandit`, `gitleaks`,
  `ruff`, `mypy`, `xenon`, or `uv lock --check`).
- Do not lower the 99 % coverage threshold.
- Do not add `# type: ignore` without a precise code and a justification.
- Do not add a bare `except:`, nor an `except Exception` without justification.
- Do not add `# noqa` without a precise rule code.
- Do not add mypy diagnostics. Reducing the count is welcome.
- Do not commit any secret, token, password or key, even in a test file or an
  example.
- Do not change the licence (`LICENSE`, `NOTICE.md`, the `license` field of
  `pyproject.toml`) without a human decision.
- Do not claim a performance result without a reproducible measurement on the
  same host.
- Do not invent a citation, a reference or a benchmark result. If a measurement
  is missing, say so.
## 6. Commands to Run Before Proposing a Change
 
```bash
npm install --no-save --package-lock=false       # upstream Delaunator reference
python -m pip install -e ".[test,speed,quality]"
uv lock --check                                  # if dependencies changed
python -m pytest --cov=orogen --cov-branch --cov-fail-under=99
ruff check .                                    # match the repository-wide CI lint gate
ruff check <changed files>                       # faster local feedback
ruff format --check orogen scripts tests
mypy orogen                                      # must add no diagnostic
bandit -q -r orogen
xenon --max-absolute C <new files>
```
 
If a command cannot be run (network, missing tool), say so explicitly in the
report. Do not present that step as validated.
 
## 7. Commit Messages
 
Format: `type(scope): description`, in the imperative mood, in English.
 
Allowed types: `feat`, `fix`, `test`, `docs`, `perf`, `bench`, `refactor`,
`build`, `ci`, `style`, `chore`.
 
Examples:
- `refactor(collision): isolate suture lifecycle stage`
- `fix(sphere): preserve north-pole closure orientation`
- `docs: record SciPy benchmark results`
## 8. Documentation to Update
 
- `CHANGELOG.md`: any user-visible change, under `[Unreleased]`.
- `docs/PORTING_STATUS.md`: any change to the porting status.
- `QUALITY_REFACTOR_REPORT.md`: any change to the quality plan.
- `docs/SCIPY_BACKEND.md`: any change to triangulation or its measurements.
Do not create a new summary document without a request. Update the existing ones.
 
## 9. When to Stop and Ask
 
Stop and ask for a human decision before:
- changing a compatibility-critical algorithm (sphere, plates, collisions, BFS);
- adding or changing a runtime dependency (including SciPy), or the licence;
- removing a module or a public function;
- modifying a differential test or a reference fixture;
- merging a pull request whose CI is red, or whose step was bypassed;
- touching secrets, CI configuration, or repository permissions.
## 10. Expected Report
 
Every agent deliverable states:
1. what was changed, file by file, and why;
2. the commands that were run and their actual results;
3. the steps that were not run, with the reason (network, missing tool, time);
4. known risks and discrepancies, including those that contradict an earlier
   report;
5. the state of the branch (commit, branch, pull request) and what remains to be
   done.
When an earlier analysis is out of date (for example because the branch has
moved on), say so and repeat the measurement before drawing conclusions.

## 11. Platform Dependency Bundles (Windows and Linux)

The workflow `.github/workflows/python-platform-bundles.yml` builds four
platform-specific ZIP artifacts for **CPython 3.12**:

| Artifact | Purpose | Contents |
|---|---|---|
| `orogen-ci-windows-py312` | Local Python CI/test dependencies on Windows | Python wheels for `.[test,speed,quality]`, Ruff 0.17.5, Radon, and the Delaunator 5.0.1 Node dependency |
| `orogen-ci-linux-py312` | Local Python CI/test dependencies on Linux | Same dependency set, built on Linux |
| `orogen-runtime-windows-py312` | Runtime installation on Windows | Orogen wheel and runtime dependency wheels only |
| `orogen-runtime-linux-py312` | Runtime installation on Linux | Orogen wheel and runtime dependency wheels only |

These are **dependency bundles**, not source-code archives. The CI bundles do
not contain the repository's test files or application source; check out the
repository separately. They contain wheels for the selected OS and Python
version, so do not use a Windows bundle on Linux, or vice versa. They target
CPython 3.12 and the architecture of the corresponding GitHub-hosted runner.

### Building and downloading

The workflow runs on a pull request when `pyproject.toml`, `package.json` or
the workflow itself changes, and can also be started manually with
`workflow_dispatch`. It uploads four GitHub Actions artifacts and retains them
for 14 days. Download them from the workflow run's **Artifacts** section.
Artifact names are the four names in the table above.

### Installing from a CI bundle

Extract the matching `orogen-ci-<platform>-py312.zip`. From the repository
root, install the Python dependencies without accessing a package index:

```bash
python -m pip install --no-index --find-links ./ci/wheelhouse ".[test,speed,quality]"
python -m pip install --no-index --find-links ./ci/wheelhouse "ruff==0.17.5" "radon>=6"
```

The archive stores the JavaScript dependency in `ci/node_modules`. For
JavaScript/Python differential tests, use Node.js 22 and make that directory
available as the repository-root `node_modules` directory. For example, on
Linux:

```bash
cp -a ci/node_modules ./node_modules
```

On Windows PowerShell:

```powershell
Copy-Item -Recurse ci/node_modules ./node_modules
```

Then run the relevant checks from `.github/workflows/python-tests.yml`. The
main test and coverage command is:

```bash
python -m pytest --cov=orogen --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=99
```

The bundle supplies Python wheels and Delaunator, but it does **not** bundle
Python itself, Node.js/npm, Git, uv, or Gitleaks. Those tools must be installed
separately for the corresponding CI steps. In particular, `uv lock --check`
requires uv, and the secret scan requires the Gitleaks executable. Network
access is still required by any workflow step that downloads external tools or
dependencies not present in the bundle.

### Installing the runtime-only bundle

Extract `orogen-runtime-<platform>-py312.zip`. It contains the Orogen wheel and
runtime wheels under `runtime/wheelhouse`. Install without an index:

```bash
python -m pip install --no-index --find-links ./runtime/wheelhouse orogen
```

This runtime bundle intentionally excludes pytest, coverage, Ruff, mypy,
Bandit, Xenon, Radon and JavaScript test dependencies.

### Maintenance

When project dependencies or CI commands change, update
`.github/workflows/python-platform-bundles.yml` and this section together.
Keep the bundle dependency lists aligned with `pyproject.toml` and
`.github/workflows/python-tests.yml`. Do not describe the bundle as a complete
offline CI environment unless every external tool and workflow prerequisite is
also included and verified.
