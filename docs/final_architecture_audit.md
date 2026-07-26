# Final Architecture Audit

This document records the production architecture contract for VClasp 0.1 and
the evidence used to guard it. Benchmark results are not part of this audit.

## Requirements and evidence

| Requirement | Production implementation | Verification |
| --- | --- | --- |
| Logical targets only | `session::Target` contains only `sample_id`, `video_id`, and `frame_idx` | Source tests reject benchmark workload labels in the executor |
| One process-wide control plane | `session::VClaspSession` owns admission, catalog, planner, feedback, AU cache, live cursors, and budgets | Public Python API exposes only `VClaspSession` as an executor |
| All API shapes share global planning | Concurrent `execute` and `execute_window` calls become `SessionSubmission`s in one admission stream | `mixed_batch_and_window_submissions_restore_each_api_boundary` plus Local and MinIO mixed-call smoke |
| One cost decision before I/O | The executor flattens the visible logical window, unions closures, removes resident records, then resolves one immutable range plan | `execute_window` and `execute_stateless_incremental_window` each call `resolve_plan` once before dispatch |
| No backend-side second plan | Local, default S3, and AIStore report exact-span execution; S3 vectored coalescing is experiment-only | `exact_backend_contract_rejects_a_second_physical_plan` and `default_s3_backend_is_an_exact_span_executor` |
| Parallel but globally bounded execution | S3 uses one shared connection pool and semaphore; decode operations acquire weighted permits from one `DecodeBudget` | MinIO smoke observed eight in-flight GETs; weighted budget unit tests pass |
| Cross-request compressed state | One session owns the encoded-AU cache and live libavcodec cursors/DPB state | Sequential cross-call smoke records live-cursor hits without a workload hint |
| State selection follows dependencies | Visible-window closure liveness drives cursor admission, pinning, and eviction | Resident-cache and window-liveness unit tests |
| Execution semantics are explicit | Internal `RecordSelection` distinguishes closure, region, and prefix records; metric labels never control execution | Production source contains no workload classifier or string-based mode dispatch |
| Codec state comes from ground truth | The index stores closure IDs, decode order, PTS, DTS, and target output ordinal | Ingest validation and timestamp-release unit tests |
| Exact logical semantics | Physical execution restores duplicates, API boundaries, batch boundaries, IDs, and original order | Mixed-submission test and byte-identical Local/MinIO smoke |
| Backend-neutral execution | `open_local`, `open_s3`, and `open_aistore` differ only in backend construction | Pure Rust `rust_session_api` tests and common Python Session methods |
| Registered S3 object mapping | S3 startup HEAD-checks the remote object size against the local chunk that supplies its catalog/index; this is a length gate, not a content hash | Unit coverage rejects length mismatch; real MinIO construction validates the registered object before measured execution |
| No production bypass reader | Raw transports, direct decoders, tier readers, forced actions, and retired executors require `experiment-controls` | `tests/test_public_api.py` |

## Public API boundary

The default Python module contains:

```text
VClaspChunk
VClaspSession
VClaspBuildStats
build_vclasp_chunk
build_vclasp_chunk_profiled
plan_byte_ranges
```

`VClaspChunk.video_ids()` and `frame_indices(video_id)` expose valid logical
targets from the embedded catalog. The default module does not register raw
Range readers, direct H.264 decoders, legacy tier executors, forced actions, or
per-worker entry points.

The Rust execution API is `session::VClaspSession`. `SessionConfig` declares:

- admission width and pending-call backpressure;
- one enforced object-store concurrency contract;
- per-operation decoder threads and a process-wide weighted decode budget;
- range bounds and runtime feedback;
- encoded-AU, read-ahead, and live-cursor state budgets.

Resident cursor capacity is a state-memory limit. It is intentionally
independent of the active decoder-thread budget; every cursor advance still
acquires a weighted decode permit.

## Bounded runtime validation

A current-format fixture containing eight videos and 60 frames per video was
read through both Local mmap and a three-node MinIO deployment.

| Trace shape | Samples | MinIO final ranges | Observed in-flight GETs |
| --- | ---: | ---: | ---: |
| Cross-video random | 48 | 23 | 8 |
| Same-video sparse | 20 | 5 | 3 |
| Consecutive frames | 32 | 2 | 2 |

For all three traces, Local and MinIO returned byte-identical RGB buffers with
identical IDs, dimensions, batch boundaries, and order. Splitting the
consecutive trace across four separate `execute` calls produced live-cursor
hits, confirming that encoded and decoder state survive API-call boundaries.

A separate mixed-call smoke submitted one `execute` and one `execute_window`
concurrently. Both reported the same joint-submission count, while physical
range and byte work was attributed exactly once.

## Verification commands

```bash
cargo fmt --check
cargo check --features ffmpeg
cargo check --features ffmpeg,experiment-controls
cargo test --release --features ffmpeg --lib
cargo test --release --features ffmpeg --test rust_session_api
git diff --check
```

The release library suite currently reports 75 passed and 10 fixture-dependent
ignored tests. The pure Rust Session API suite reports 2 passed.

## Non-production controls

The `experiment-controls` feature retains mechanism baselines and diagnostic
transports required by the research artifact. Those controls are intentionally
not alternate production readers and must not be used to describe the default
runtime architecture.
