# Execution report — Orogen Python quality plan

Target branch: `port/v3.0.1`  
Base commit: `0afae616a78d318ac1c77e6b52b5118059d98f5f`

## Implemented in this working copy

### Quality guard rails

- Added explicit Ruff configuration for `E`, `F`, `I`, `UP`, `B` and `SIM`, with an 88-character line length and an explicit formatting style.
- Added a `quality` optional dependency group for Ruff, mypy, Xenon and Bandit.
- Added a mypy diagnostic-baseline comparator. It compares path, error code and message (including duplicate counts), while ignoring line-number shifts.
- Added CI steps for changed-file Ruff linting, formatting of the new diagnostic checker, and a Xenon complexity ceiling for newly added Python files.
- Changed the CI coverage target from `orogen.v2` to the complete `orogen` package, retaining branch coverage and the 95% threshold.
- Added Bandit and Gitleaks CI scans.

### Complexity pilot

- Decomposed `spread_belt` into smaller stages for front propagation, preserving earlier belt records, updating front metadata and recording per-cell geology.
- Decomposed `rift_continents` into schedule/state preparation, queued plume-request handling, overdue breakups and spontaneous attempts. Added `_RiftSchedule` to hold shared timing and area rules.
- Updated `PORTING_STATUS.md` to record these refactorings.

### Typing, dependencies and test hygiene

- Added `orogen/py.typed`, configured package data so it is included in distributions, and annotated the top-level version constant.
- Bounded runtime dependencies to `numpy>=1.24,<3` and SciPy/Qhull `scipy>=1.10` for production Delaunay triangulation.
- Added `package.json` with `delaunator` pinned to `5.0.1`; CI now installs from that manifest.
- Audited the 29 test functions with no direct Python `assert`: 23 use `pytest.raises` or NumPy testing assertions, while six are branch-coverage smoke tests for Numba `.py_func` bodies. The latter are now documented.
- Documented why the sphere-mesh adapter catches backend-specific exceptions before translating them to the stable mesh API.

## Validation performed

- 78 focused tests passed across `test_v2_spread_belt.py`, `test_v2_tectonic_rifting.py`, `test_v2_far_field.py` and `test_v2_tectonic_orogeny.py`.
- After the boundary-force extraction, 34 tests passed across `test_v2_tectonic_forces.py`, `test_v2_tectonic_dynamics.py` and `test_v2_tectonic_stepper.py`, using a temporary SciPy-backed adapter because `shull` is unavailable in this local environment.
- Extracted the suture readiness, gap/ring checks and configured welding stage from `track_collisions`; 105 focused collision, force, rifting, spread-belt and orogeny tests passed with the same temporary adapter.
- `compileall` succeeded for all Python files under `orogen`, `scripts` and `tests`.
- `pyproject.toml`, `package.json` and `.github/workflows/python-tests.yml` parsed successfully.
- A wheel was built successfully and checked to contain `orogen/py.typed`.
- The patch was applied to a clean copy of the supplied source archive without conflicts.

**Historical test limitation:** at the time of the initial local checks, `shull` was not installed in the offline environment, so focused tests used a temporary SciPy-backed adapter and the complete suite did not finish within the local execution limit. This limitation predates the current switch to SciPy/Qhull as the actual production backend; current validation is delegated to the SciPy-focused GitHub Actions workflows.

Ruff, mypy, Xenon and Bandit were not available locally and could not be installed because the environment had no package-index network access. Their CI steps are configured but have not yet been observed passing on this change. The full Ruff formatting pass, mypy diagnostic comparison and complete coverage threshold therefore remain to be confirmed by CI.

## Additional work after the initial quality pilot

- Extracted slab pull/suction classification, young-rift/ridge classification and force accumulation from `gather_boundary_loads` into dedicated helpers. The existing force, dynamics and stepper tests pass (34 tests) with a temporary SciPy-backed test adapter.
- Fixed the Ruff findings in `rift_continents` and restored the missing cKDTree benchmark script from the experiment branch after the PR CI exposed the missing path.

## Additional work after the initial quality pilot

- Extracted slab pull/suction classification, young-rift/ridge classification and force accumulation from `gather_boundary_loads` into dedicated helpers. The existing force, dynamics and stepper tests pass (34 tests) with a temporary SciPy-backed test adapter.
- Extracted suture readiness, gap/ring checks and configured welding from `track_collisions` into `_update_suture_state`; the targeted collision/force/rifting/spread/orogeny suite passes 105 tests with the same adapter.
- Ran the full Ruff formatting pass in a dedicated formatting-only commit (`1a359d8`), covering the Python package, scripts and tests.
- Generated `uv.lock` on GitHub Actions with CPython 3.12; uv resolved 44 packages and `uv lock --check` passed. The lockfile is committed separately (`5015af8`).
- Fixed the initial Ruff findings and restored the missing cKDTree benchmark script exposed by the first PR CI run. Applied Ruff's safe autofixes to the changed source and test files; the complete recheck is pending.

## Remaining plan items

- Run the updated GitHub Actions workflows against the production SciPy/Qhull backend and confirm full-package branch coverage is at least 95%.
- Verify that mypy introduces no new diagnostics relative to the PR base, then work down the existing baseline in `crust.py`, `ops/merge_plates.py`, `_numba_kernels.py` and `ops/rift_path.py`.
- Measure Radon/Xenon complexity and Numba performance before/after; the local environment did not provide a usable Radon/Xenon installation or a complete benchmark comparison.
- Continue decomposing `track_collisions` (pair orchestration, convergence and front geometry), then refactor `merge_plates`, `find_rift`, `founder_within_plates`, and others. Add the proposed physical-property tests and cover the residual branches in `collision.py` and `contact.py`.

The report is included with the proposed quality-refactor changes. The associated pull request and its CI runs are the publication and validation record.

- Extracted old-ocean coast detection and connected-run labeling from `founder_within_plates` into `_old_ocean_margin_runs`. The change is committed; focused native-backend CI validation is pending.
- Replaced the Gitleaks Action's failing PR revision-range integration with a pinned Gitleaks CLI scan over the full commit range from the merge base, avoiding the action's invalid-range failure while retaining history scanning.
- Extracted lower-plate raster resampling from `merge_plates` into `_resample_lower_cells`, alongside old-ocean margin-run discovery in `founder_within_plates`. Both refactorings are included in the SciPy-migration PR-head full-suite validation below.

## Numerical backend selection

- cKDTree remains discarded; its dedicated workflow and comparison script were removed. No cKDTree lookup is used in production.
- Production Delaunay triangulation now uses `scipy.spatial.Delaunay` (Qhull), and SciPy is a required runtime dependency.
- The previous `shull` runtime dependency, platform workflow, A/B benchmark workflow and comparison script were removed.
- `SphereIndex` remains the separate spherical nearest-cell lookup structure; it is not a triangulation backend.
- The SciPy platform matrix is limited to Ubuntu and Windows for Python 3.10, 3.12 and 3.14.


## SciPy production backend migration — 2026-11-18

- Replaced the production planar triangulator with `scipy.spatial.Delaunay` (Qhull); `scipy>=1.10` is now a required runtime dependency and `shull` has been removed from the dependency graph.
- Preserved the Orogen spherical closure, cyclic CSR adjacency, `SphereIndex` lookup, and public mesh array layouts. The three-point minimum is handled by its single possible simplex; Qhull is used for four or more points.
- Removed the former shull platform and comparison workflows/scripts. cKDTree remains excluded. The SciPy compatibility matrix contains six jobs: Ubuntu/Windows × Python 3.10/3.12/3.14; macOS is excluded.
- GitHub Actions on PR-head `0ac5ed9c6e9d4791fce457cca7f64eaa9844b43a`: **517 tests passed**, **99.44% branch coverage** (95% minimum), and **6/6** SciPy platform jobs passed.
- Ruff, Bandit, Gitleaks, Xenon, `uv lock --check` and mypy-baseline comparison passed. Mypy diagnostics fell from 118 on the base to 62 on the PR head, with no new diagnostics.
- Validation runs: [Python tests](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44324665297) and [SciPy platform matrix](https://github.com/Thomas-GAUTRON2004/Orogen_python/actions/runs/44324665407).
