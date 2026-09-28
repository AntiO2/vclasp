# Contributing to VClasp

Contributions to the Rust core, Python bindings, tests, examples, and
documentation are welcome. Start with the [README](README.md),
[public API](docs/API.md), and [source guide](src/README.md).

## Discuss a change

Open a bug report with a minimal reproduction, or a feature request describing
the application that needs it. Discuss changes to the file format, public API,
or execution architecture before investing in a large implementation. Small
fixes and documentation improvements can go directly to a pull request.

For a security issue, follow [SECURITY.md](SECURITY.md). For participation
standards and conduct reporting, see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Prepare the environment

Install the native dependencies listed in the [setup guide](docs/SETUP_AND_RUNBOOK.md).
Then, from the repository root:

```bash
micromamba create -f environment/ci-native.yml
micromamba activate vclasp-native
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"
export PATH="$CONDA_PREFIX/bin:$PATH"
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" python scripts/doctor.py
python -m pip install -r requirements-dev.txt
python -m pip install -e .
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" cargo test --locked --features ffmpeg --lib --test rust_session_api
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" python -m pytest
```

The Python extension is built with the `ffmpeg` feature by `pyproject.toml`.
Rebuild it after changing Rust code before running Python tests.

Keep credentials in an untracked `.env` file or environment variables.
[.env.example](.env.example) documents the example variables. Tests do not
automatically load that file.

## Work on a branch

`main` is protected. Submit changes through a pull request from a feature
branch or fork. Keep each pull request focused on one change and use an
existing checkout when practical. Additional worktrees are optional, not a
requirement of this project.

Only **Squash and merge** is allowed. Merge commits and rebase merges are
disabled in repository settings. Do not merge PRs automatically or bypass branch
protection. After a PR is squash-merged, fetch `origin/main` and start the next
PR branch from that revision; do not keep pushing follow-up work to the merged
PR branch. Check the remote PR state before every submission.

Describe the problem, the resulting behavior, and the checks you ran. Call out
any public API or file-format change, including whether existing chunks must
be rebuilt. Preserve unrelated work in the working tree.

## Issue and pull request titles

Use a specific English summary describing the problem or change, not a task
number, branch name, or a generic title such as "update" or "fix bugs".

Issue titles keep the prefixes supplied by the issue templates:

- `[Bug]: <observed problem>`
- `[Feature]: <requested capability>`
- `[Performance]: <operation and observed regression>`

Pull request titles use `<type>(<scope>): <summary>`. The scope is optional;
types are `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`,
`chore`, and `revert`. Use a short imperative summary without a trailing period.
Choose the type for the main change, and describe secondary changes in the PR
body. For a breaking API or format change, add `!` before the colon and explain
the migration in the PR body.

Examples:

- `feat(core)!: unify the session API and chunk format`
- `fix(storage): reject ranges beyond the payload`
- `ci: run Rust and Python unit tests in parallel`
- `docs: document local build dependencies`

Link related issues in the body using `Closes #123` only when the PR fully
resolves them; otherwise use `Refs #123`. Check the title before opening or
merging a PR. Squash-merge commit titles should follow the same PR convention.

## Keep the core reusable

- Applications submit logical targets through `VClaspSession`; the backend is
  selected when the session is constructed.
- Derive execution choices from dependencies, byte extents, available state,
  measured costs, and resource limits. Do not route by benchmark or workload name.
- Keep codec and I/O operations in Rust. PyO3 adapters convert arguments and
  results without implementing a second planner or decoder.
- Preserve duplicate targets, batch boundaries, and output order. Fetching a
  gap must not cause its unneeded access units to enter the decoder.
- Keep mechanism controls behind `experiment-controls`. Production readers
  must not require callers to select a historical executor.
- Keep datasets, credentials, cluster configuration, and large experiment
  outputs outside this repository.

## Validate the change

Run the checks relevant to the files you changed:

```bash
cargo fmt --all --check
cargo fmt --manifest-path object-store-transport/Cargo.toml --all --check
python -m ruff format --check tests examples scripts/ci
cargo build --locked --features ffmpeg --lib --examples
cargo test --locked --features ffmpeg --lib --test rust_session_api
cargo test --locked --manifest-path object-store-transport/Cargo.toml --lib
cargo check --locked --features experiment-controls --all-targets
python -m pytest
git diff --check
```

GitHub Actions runs **Format** independently. **Build** compiles the FFmpeg-enabled
core and public API tests, the transport tests, and the Python extension. It
publishes the executables and extension as one artifact. **Rust unit tests** and
**Python unit tests** both depend only on **Build** and run in parallel on separate
runners. Neither test job invokes Cargo or maturin; Python imports the prebuilt
extension directly, so Rust tests do not wait for wheel packaging. CI installs
build dependencies only in **Build** and codec runtime libraries in test jobs. It does not
require a GPU, a dataset, or a running object store. Local-storage tests create
their files with Rust `tempfile` or pytest `tmp_path` under the runner's temporary
directory and remove them when the tests finish. Fixture-dependent Rust tests
remain ignored, and Python tests marked `integration` are excluded by default.

This CI covers the production FFmpeg configuration and unit tests. The additional
default-feature, example, and experiment-control checks above remain local
development checks. Wheel packaging is not part of this unit-test workflow.

Use `cargo fmt --all` and `python -m ruff format tests examples scripts/ci scripts/doctor.py` to apply the
formatting checked by CI. CI pins Rust and Python in
[the workflow](.github/workflows/ci.yml) and Python development tools in
[requirements-dev.txt](requirements-dev.txt).

The `experiment-controls` compile check is separate from its test execution:
the current PyO3 `extension-module` configuration can leave Python symbols
unresolved when linking a standalone test executable with those controls.
Do not describe a successful compile check as a passing test suite.

Some codec and chunk tests are ignored because they require explicit fixtures.
Provide a newly built chunk in the current format and run the applicable test
with `--ignored`; inspect its documented environment variables first. Report
which fixture tests ran and which were skipped.

Add focused regression coverage for behavioral fixes. Documentation-only
changes need link and command review, not a full performance run.

## Performance reports

Report the source revision, build profile, backend, hardware, request trace,
concurrency, cache state, and repetition count. Distinguish logical targets,
clips, and batches. Keep raw measurements and explain how summaries were
computed. A lower GET count alone does not establish a latency improvement.

Never change target semantics, resource accounting, or correctness checks to
improve a benchmark result. Large experiments belong in the separate artifact
harness; small deterministic fixtures belong in this repository.

## License

Contributions are provided under the project's existing
[AGPL-3.0-only license](LICENSE). Preserve third-party notices and identify
the source and license of any imported code or fixture.
