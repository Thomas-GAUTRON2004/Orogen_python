
# Quality Rules — Orogen_python
 
This document lists the quality rules that every change to the repository must
follow. It applies to humans and to agents alike. Rules that CI can check are
enforced by CI; the others are enforced during code review.
 
Context: Orogen_python is a Python port of the JavaScript V2 engine of Orogen
3.0.1 (`raguilar011095/planet_heightmap_generation`, tag `v3.0.1`, commit
`ed466f9`). Compatibility is partial (see `docs/PORTING_STATUS.md`).
 
## 1. Tests
 
- All tests pass: `python -m pytest`.
- Every bug fix and every newly ported behaviour comes with a regression or
  differential test.
- Differential tests against the JavaScript reference are never modified to make
  a result pass. If a discrepancy appears, it is documented and analysed first.
- Tests must not depend on execution order or on global state.
- Property tests (physical invariants: surface conservation, plate identifier
  consistency, mesh topology) are preferred over tests that freeze a particular
  implementation.
## 2. Coverage
 
- Branch coverage of the whole `orogen` package, with a **99 %** threshold, as
  enforced by CI:
  `python -m pytest --cov=orogen --cov-branch --cov-fail-under=99`.
- The threshold is never lowered. A coverage drop in a pull request is justified
  in its description.
- The pure-Python fallback (used when Numba is absent) stays covered.
## 3. Lint and Format
 
- Configuration: `[tool.ruff]` in `pyproject.toml`. Enabled rule sets: `E`, `F`,
  `I`, `UP`, `B`, `SIM`; line length 88.
- **CI enforces lint on the whole repository** with `ruff check .`. Run the same
  command locally to reproduce the CI lint gate; checking only changed files is
  useful for faster feedback but is not equivalent.
- **CI enforces formatting on the whole package**:
  `ruff format --check orogen scripts tests`. To format, run `ruff format <files>`
  and do not mix formatting with functional changes in the same commit.
- No `# noqa` without a precise rule code and a justifying comment.
## 4. Typing
 
- The package is typed (`orogen/py.typed`). Public modules are annotated.
- Check: `mypy orogen`.
- **CI fails only if a pull request adds diagnostics** compared with its base
  branch, using `scripts/check_mypy_regressions.py`. Diagnostics are reduced,
  never increased. The base branch `port/v3.0.1` currently reports 118 diagnostics
  in 19 files; the quality branch reports 62 in 13 files. Re-measure before quoting
  either figure.
- A `# type: ignore[code]` targets a precise error code and is justified by a
  comment. A bare `# type: ignore` is forbidden.
- Prefer `dataclass` or `TypedDict` over anonymous dictionaries and arrays.
## 5. Complexity
 
- **CI applies `xenon --max-absolute C` to newly added Python files only.**
- For code that is modified, the target is a complexity below 15 per function.
  No new function of complexity F is accepted.
- A function longer than 150 lines is split when it is modified.
- Splitting preserves behaviour: differential tests and existing tests pass
  without modification, and the benchmark of the affected path is compared before
  and after.
## 6. Security and Secrets
 
- CI runs `bandit -q -r orogen`, which must report no issue.
- CI runs `gitleaks`, which must detect no secret. No token, password or key is
  committed, even temporarily.
- No bare `except:`. An `except Exception` is justified by a comment and does not
  hide a data error.
- No `eval`, `exec`, or dynamic import of a string supplied by a user.
## 7. Dependencies
 
- Runtime dependencies are `numpy>=1.24,<3` and `scipy>=1.10`. Nothing else is
  added without an explicit decision.
- SciPy (`scipy.spatial.Delaunay`, Qhull-backed) is the production planar
  triangulator. Replacing it requires a correctness benchmark and a maintainer
  decision (see `docs/SCIPY_BACKEND.md`). `shull` is not a dependency: it was
  evaluated and removed.
- Numba is optional (`speed` extra). Any path accelerated by Numba has a tested
  pure-Python fallback.
- `delaunator` is pinned in `package.json` (version 5.0.1), for the upstream
  differential tests.
- Any dependency change updates `uv.lock`. CI runs `uv lock --check`, so the
  lockfile must be current.
- A new runtime dependency or a licence change is discussed before it is
  implemented (see `NOTICE.md`).
## 8. Numerical Compatibility
 
- Spherical geometry and point location are computed in **float64**, to match the
  JavaScript reference.
- Mesh comparisons use **topological invariants** (Euler characteristic,
  reciprocal neighbours, cyclic orientation, pole closure), not identity of
  simplices: Qhull may choose a different diagonal in degenerate cases.
- No claim of "complete" compatibility is made without differential tests over
  several time steps. The status is updated in `docs/PORTING_STATUS.md`.
## 9. Performance
 
- A change to a hot path (triangulation, CSR construction, `SphereIndex`,
  simulation loops) comes with a before/after measurement on the same host, using
  medians after warm-up.
- An optimisation is accepted only if it is measurable and the benchmark's
  correctness checks pass. Timings alone are not sufficient.
- An alternative is not adopted without its own correctness benchmark. Example:
  a cKDTree-based lookup is rejected until its interpolation-weight differences
  are explained (branch `experiment/ckdtree-sphere-index`).
## 10. Documentation and Traceability
 
- Any change to the porting status is recorded in `docs/PORTING_STATUS.md`.
- Any user-visible change is recorded in `CHANGELOG.md`, under `[Unreleased]`.
- Important technical decisions (choice of a tool, a dependency or a threshold)
  are documented in `docs/`, with the measurements that justify them.
- Commit messages follow `type(scope): description`, with `feat`, `fix`, `test`,
  `docs`, `perf`, `bench`, `refactor`, `build`, `ci`, `style` or `chore`.
## 11. Git and Review
 
- No direct commit to `port/v3.0.1` or to a shared quality branch. Work on a
  dedicated branch and open a pull request.
- A pull request does not mix refactoring, bug fixes and formatting.
- CI must be green before merging. A red CI is not bypassed (no `--no-verify`,
  no disabling of a step).
## Pre-merge Checklist
 
- [ ] `python -m pytest` passes, branch coverage ≥ 95 %
- [ ] `ruff check .` passes; `ruff format --check` passes on
      `orogen scripts tests`
- [ ] `mypy orogen` adds no diagnostic compared with the base branch
- [ ] `xenon --max-absolute C` passes on new files
- [ ] `bandit -q -r orogen` reports nothing; no secret detected
- [ ] `uv lock --check` passes if dependencies changed
- [ ] Differential tests are unchanged, or their discrepancies are documented
- [ ] Before/after benchmark attached if a hot path is touched
- [ ] `CHANGELOG.md` and `docs/PORTING_STATUS.md` are up to date
 
