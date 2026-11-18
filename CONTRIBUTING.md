
# Contributing to Orogen_python
 
Thank you for your interest in Orogen_python. This project is a Python port of
the JavaScript V2 engine of Orogen 3.0.1. The port is partial, and compatibility
with the upstream engine is checked with differential tests, so every change
needs care. This guide explains how to set up the project, how to propose a
change, and what is expected from it.
 
Before you start, read:
 
- [`quality.md`](quality.md): the quality rules every change must follow.
- [`agent.md`](agent.md): working rules that also apply to automated agents.
- [`docs/PORTING_STATUS.md`](docs/PORTING_STATUS.md): what is ported, what is
  not, and the current priorities.
- [`docs/SCIPY_BACKEND.md`](docs/SCIPY_BACKEND.md): the production triangulation
  backend and its limits.
- [`NOTICE.md`](NOTICE.md) and [`LICENSE`](LICENSE): licence and attribution.
## Ways to Contribute
 
- **Report a bug or a discrepancy** with the upstream JavaScript engine. A
  reproducible case (seed, resolution, expected and observed output) is the most
  useful report.
- **Port a missing piece** from the priorities in `docs/PORTING_STATUS.md`.
- **Add tests**, especially property tests on physical invariants and
  differential tests against the JavaScript reference.
- **Improve the documentation**, including the porting status when you change a
  module.
- **Review pull requests.**
For a large change (a new module, a change of algorithm, a new runtime
dependency), open an issue first to agree on the approach.
 
## Setting Up a Development Environment
 
Requirements:
 
- Python 3.10 or later;
- Node.js and npm, used by the upstream differential tests (`delaunator`);
- `uv`, used to check the dependency lockfile.
```bash
git clone https://github.com/Thomas-GAUTRON2004/Orogen_python.git
cd Orogen_python
npm install --no-save --package-lock=false      # upstream Delaunator reference
python -m pip install -e ".[test,speed,quality]"
python -m pytest
```
 
`numba` is optional at runtime. The `speed` extra installs it so that the
accelerated paths are exercised. Every accelerated path has a pure-Python
fallback, which must keep passing.
 
## Branches and Pull Requests
 
- The porting reference branch is `port/v3.0.1`. Quality work is done on a
  dedicated branch (currently `quality/phase-1-refactor-2026-11-18`).
- Create your own branch with a descriptive name: `fix/...`, `feat/...`,
  `refactor/...` or `docs/...`. Fork the repository if you do not have write
  access.
- Keep each branch focused on one intention.
- Open a pull request against the branch the change is meant for. Do not push
  directly to `port/v3.0.1`, `main` or a shared quality branch.
- Describe in the pull request what changed, why, how it was tested, and any
  discrepancy with the upstream reference.
- Wait for CI to pass. Address review comments in new commits, and do not
  rewrite a branch that is already under review.
A pull request does not mix refactoring, bug fixes and formatting. Send them as
separate pull requests, or as separate commits that a reviewer can read one by
one.
 
## Commit Messages
 
Use the form `type(scope): description`, in the imperative mood:
 
```
fix(sphere): preserve north-pole closure orientation
refactor(collision): isolate suture lifecycle stage
docs: record SciPy benchmark results
```
 
Allowed types: `feat`, `fix`, `test`, `docs`, `perf`, `bench`, `refactor`,
`build`, `ci`, `style`, `chore`.
 
## Before You Open a Pull Request
 
Run the checks that CI runs:
 
```bash
uv lock --check                                    # if dependencies changed
python -m pytest --cov=orogen --cov-branch --cov-fail-under=95
ruff check <changed files>
ruff format --check orogen scripts tests
mypy orogen                                        # must not add diagnostics
bandit -q -r orogen
xenon --max-absolute C <new files>
```
 
Notes on what CI enforces today:
 
- Ruff lint runs on the Python files changed by the pull request, not on the
  whole repository. A few legacy files are excluded from lint until a dedicated
  cleanup; do not add lint debt to them.
- `ruff format --check` runs on the whole package, scripts and tests.
- The mypy check fails only if the pull request adds diagnostics compared with
  its base branch. The current baseline is recorded in the quality report.
- Xenon applies only to newly added Python files.
The full requirements are in [`quality.md`](quality.md). In short:
 
- all tests pass, and branch coverage stays at or above 95 %;
- the code is formatted with `ruff format`, and changed files pass `ruff check`;
- mypy reports no new diagnostic compared with the base branch;
- new Python files have complexity C or below;
- no secret is committed;
- `uv.lock` is updated when dependencies change;
- `CHANGELOG.md` is updated under `[Unreleased]`, and
  `docs/PORTING_STATUS.md` is updated if the porting status changed.
If you cannot run a check, say so in the pull request. Do not present it as done.
 
## Tests and Correctness
 
- Differential tests compare the Python port with the JavaScript reference. Do
  not change a differential test or its fixtures to make a result pass. If you
  find a discrepancy, document it and open an issue.
- Mesh tests check topological invariants (Euler characteristic, reciprocal
  neighbours, cyclic orientation, pole closure). They do not require the same
  simplices as the JavaScript reference: Qhull may choose a different valid
  diagonal in degenerate cases.
- Spherical geometry and point location use float64.
- Do not claim compatibility with the upstream engine beyond what the tests
  show. The porting status must reflect what is actually verified.
## Performance Changes
 
A change to a hot path (triangulation, CSR construction, `SphereIndex`,
simulation loops) needs a before/after measurement on the same host, with medians
after warm-up. Timings alone are not enough: the benchmark's correctness checks
must also pass.
 
Alternatives are explored on `experiment/*` branches and are not adopted
without their own correctness benchmark. For example, a cKDTree-based lookup is
not accepted until its interpolation-weight differences are explained.
 
## Dependencies and Licence
 
- Runtime dependencies are `numpy` (`>=1.24,<3`) and `scipy` (`>=1.10`).
  Adding a runtime dependency requires an issue and a discussion first.
- SciPy (`scipy.spatial.Delaunay`, Qhull-backed) is the production Delaunay
  backend. Replacing it requires an issue, a correctness benchmark and a
  maintainer decision, documented in
  [`docs/SCIPY_BACKEND.md`](docs/SCIPY_BACKEND.md).
- `delaunator` is pinned in `package.json` for the upstream differential tests.
- Project licence: GPL-3.0-only. Upstream attribution is in
  [`NOTICE.md`](NOTICE.md). Do not change the licence or the attribution without
  a maintainer decision.
- By submitting a contribution, you agree that it is licensed under the
  project's licence (GPL-3.0-only).
## Security and Secrets
 
Do not report security issues, secrets or leaked credentials in a public issue.
Use GitHub's private vulnerability reporting if it is enabled on the repository,
or contact the repository owner directly.
 
Never commit a secret, even temporarily. The CI runs `gitleaks`, and a detected
secret blocks the merge.
 
## Review Process
 
- A maintainer reviews every pull request. Expect questions on correctness,
  compatibility and performance, not only on style.
- A pull request is merged only when CI is green and the review is approved.
- Keep discussions factual. A measurement, a reference line or a failing test
  settles a disagreement better than an opinion.
## Questions
 
If something in this guide is unclear, or a rule seems to block a legitimate
change, open an issue. Improving this guide is als a welcome contribution.
 
