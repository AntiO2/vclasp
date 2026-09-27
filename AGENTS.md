# Repository instructions

## Start with the current code

Read `README.md`, `CONTRIBUTING.md`, `src/README.md`, and the relevant part of
`docs/API.md` before a substantial change. Check `git status`, the current
branch, existing tests, and recent history. Search for an existing mechanism
before implementing another one.

Use the current checkout and feature branch. Do not create additional worktrees
unless the user explicitly requests them. `main` is protected: submit changes
through pull requests and never bypass protection or force-push it. Preserve
unrelated and uncommitted work.

## Architecture

- `VClaspSession` is the production request interface. Local, S3, and AIStore
  backends change construction, not target semantics.
- Keep dependency resolution, planning, I/O, codec execution, and bounded
  state ownership in Rust. PyO3 is an adapter, not a second implementation.
- Base execution choices on targets, actual codec dependencies, byte extents,
  available state, measured costs, and declared resource budgets. Never infer
  a benchmark/workload name or require a workload-specific reader.
- Preserve logical IDs, duplicates, batch boundaries, and output order across
  synchronous, concurrent, and pipeline calls.
- Fetch and decode plans are separate. Gap bytes may be read; unneeded gap AUs
  must not be submitted to the decoder.
- Respect shared I/O/decode limits, memory budgets, and pipeline backpressure.
  Account for persistent decoder state separately from encoded-byte caches.
- Keep historical mechanisms under `experiment-controls`. Do not expose them
  as alternative default readers.

## Format and API changes

Keep the schema, checked-in schema adapter, reader, writer, documentation, and
tests consistent. The current format uses identifier `VCSP`, magic `VClasp`,
and version 1. Do not add automatic legacy-format fallback without an explicit
requirement. State when a change requires rebuilding chunks.

Prefer existing public types and helpers. Keep imports and examples aligned
with actual Rust exports and Python registrations. Do not prescribe new
configuration objects or aliases unless they solve an identified interface
problem.

## Validation

Use the commands in `CONTRIBUTING.md`. Choose tests for the changed behavior;
run Python API checks against an extension rebuilt from the current source.
Distinguish tests that passed, tests that were skipped, compile-only checks,
and unavailable fixture or service tests. Do not hide linker or runtime errors.

For performance changes, use the existing harness with explicit inputs and
resource settings. Keep raw measurements traceable. Do not fabricate missing
metrics or alter correctness requirements to improve results. Do not start a
large experiment when a focused regression test answers the question.

## Repository scope

Keep core code, focused tests, small examples, and user documentation here.
Keep baseline ports, paper sources, datasets, deployment-specific configuration,
large results, and experiment status reports in the artifact or experiment
workspace. Never commit credentials, private endpoints, machine-specific paths,
local environments, or build outputs.

Use concise comments for non-obvious invariants. Keep changes focused and
document behavior rather than the conversation that led to it.
