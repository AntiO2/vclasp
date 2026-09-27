# Source guide

The core turns logical video targets into reads and decoder submissions while
preserving the caller's output order. Start with
[`session.rs`](execution/session.rs) for the application-facing Rust API and
[`lib.rs`](lib.rs) for Rust exports and Python bindings.

## Modules

| Directory | Responsibility | Main entry points |
| --- | --- | --- |
| `format/` | Chunk headers, embedded index, and physical record metadata | [`chunk.rs`](format/chunk.rs), [`index.rs`](format/index.rs) |
| `ingest/` | Controlled encoding, AU discovery, dependency construction, and chunk writing | [`hierarchical_ingest.rs`](ingest/hierarchical_ingest.rs) |
| `planning/` | Closure resolution, range candidates, cost comparison, and runtime feedback | [`hierarchical_layout.rs`](planning/hierarchical_layout.rs), [`planner.rs`](planning/planner.rs), [`runtime_feedback.rs`](planning/runtime_feedback.rs) |
| `execution/` | Shared session, bounded admission, fetch/decode scheduling, state reuse, and ordered delivery | [`session.rs`](execution/session.rs), [`hierarchical_scheduler.rs`](execution/hierarchical_scheduler.rs), [`resident_cache.rs`](execution/resident_cache.rs) |
| `storage/` | Local, S3-compatible, and AIStore backend adapters | [`backend.rs`](storage/backend.rs) |
| `codec/` | H.264 encoding and libavcodec decoding | [`encoder.rs`](codec/encoder.rs), [`decoder.rs`](codec/decoder.rs) |
| `controls/` | Alternative mechanisms used in experiments | [`mod.rs`](controls/mod.rs) |

`lib.rs` uses `#[path = ...]` declarations to connect these directories to the
crate module tree. Directory names do not imply public Rust module paths;
use the exports documented in the [API guide](../docs/API.md).

The shared Rust/C object-store transport is in
[`object-store-transport/`](../object-store-transport/README.md).
The x264 C adapter is in [`native/`](../native/).

## Read path

1. The session receives logical targets or already-sampled batches.
2. The index resolves targets to sufficient codec dependencies and byte extents.
3. Planning selects physical reads under the configured resource limits.
4. Storage fetches ranges; execution extracts required AU records from them.
5. The codec layer receives legal decode order. Transfer-only gaps are skipped.
6. Execution restores requested identities, duplicates, batch boundaries, and order.

Physical byte order, decode order, and output order are distinct. A layout
change must update offsets without changing dependency semantics or silently
submitting extra records to the decoder.

The session owns reusable state and shared budgets. Independent callers should
share a session when they need one global I/O and decode budget. Pipeline
submission applies backpressure; ordered consumption does not require physical
execution to finish in submission order.

## Build features

- The default feature set provides metadata and range-planning support.
- `ffmpeg` enables production ingestion, decode, and `VClaspSession`.
- `experiment-controls` includes `ffmpeg` and exposes diagnostic mechanisms.
  It does not change the normal application request interface.

## Where to make changes

- A backend change belongs in `storage/` or the transport crate, without
  redefining the logical request API.
- A physical layout change belongs in ingestion and record metadata. Planning
  and execution must consume actual extents, not infer physical order from a GOP.
- A dependency change must be constructed and validated at ingestion, then
  consumed through the same index interface.
- A Python API change belongs in the PyO3 adapters, with a corresponding
  check in [`test_public_api.py`](../tests/test_public_api.py).
- A chunk-format change must keep [`chunk_v1.fbs`](../schemas/chunk_v1.fbs),
  [`chunk_schema.rs`](format/chunk_schema.rs), the reader/writer, and format
  tests consistent.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for validation commands and
[AGENTS.md](../AGENTS.md) for repository instructions for coding agents.
