
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
- Do not lower the 95 % coverage threshold.
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
python -m pytest --cov=orogen --cov-branch --cov-fail-under=95
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
 
