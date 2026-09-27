# VClasp Execution Architecture

## Contract

VClasp accepts logical targets, never workload labels:

```text
(sample_id, video_id, frame_index)
```

`execute(batch)` and `execute_window(batches)` are blocking convenience APIs.
`submit(batch)` and `submit_window(batches)` admit the same requests and return
one-shot handles. A producer can therefore expose bounded future work before it
waits for ordered results; no thread-per-request adapter is required.
Concurrent application callers use one `VClaspSession`; callers do not provide
an identity, route, worker, or workload label. `max_callers` bounds one
admission cohort, `max_pending_calls` provides process-wide backpressure, and
`max_inflight_windows` bounds overlapping immutable execution windows.
`execute` and `execute_window` enter the same admission stream. Concurrent
calls of either shape are flattened into one visible logical window before
closure resolution, then restored to their original API and batch boundaries.

The default training adapter is a bounded ordered pipeline over this same
session:

```text
sampler/producer -> submit(batch) -> shared session -> ordered take() -> training
                        ^                                  |
                        +----------- backpressure ---------+
```

Production, I/O, decode, and model consumption overlap. The adapter does not
introduce a second planner or a workload-specific route. It bounds submitted
but unconsumed batches and targets, allows physical work to finish out of
order, and serializes only the delivery boundary so `take()` matches submission
order.

Uniform, Zipf, Same-video, Clip, and Sequential are benchmark trace names. They
must not appear in production planning or execution code.

## Session Control Plane

```text
concurrent logical submissions
          |
          v
bounded admission window
          |
          v
bounded immutable execution windows
          |
          v
target lookup and closure union
          |
          v
deduplicate dependencies and resident state
          |
          v
enumerate physical span candidates
          |
          v
one cost decision per admitted request set
          |
          v
immutable execution plan
```

The session coordinator owns:

- admission and API-boundary restoration;
- shared runtime cost feedback;
- global I/O and weighted decode budgets;
- session-wide physical accounting.

There is no per-caller planner, cache, or decoder pool. The session combines
concurrently admitted requests before span selection.
Logical batch boundaries constrain result restoration, not physical planning.
Cross-request dependency deduplication, resident-state selection, and range
coalescing therefore happen in one decision that accounts for bytes, requests,
decoder work, and shared resource pressure.

Several immutable windows may execute concurrently. They share the same object
store client and connection semaphore, runtime-feedback estimator, weighted
decode budget, and metrics owner. Each window is planned exactly once after
admission. Before physical dispatch, the S3 range broker can deduplicate and
coalesce immutable spans from concurrently executing windows. It never changes
codec closures and never revisits a span after that span has been dispatched.

Execution lanes retain lane-local encoded AUs and live decoder cursors. The
declared state budget is partitioned across lanes, and single-video streams use
deterministic lane affinity so compatible DPB state reaches the same owner.
Mixed-video cohorts remain eligible for joint cross-video span planning.

Range GET completion and decoding form a completion-driven pipeline both within
and across concurrently executing windows. The broker publishes every logical
range as soon as its containing physical span completes; it does not wait for
the other ranges in that caller's request. Persistent decoder workers can
therefore consume ready closures while later GETs remain in flight. One
accounting owner retains each shared physical GET and byte count, so completion
fan-out does not duplicate physical statistics.

The admission quiet window is a session resource parameter. A single-caller
session does not wait for speculative peers. Multi-caller sessions collect
requests until the configured quiet interval expires or the declared caller
capacity is reached. `session_admission_ns` and
`session_joint_submissions` make this latency visible.

While all S3 slots are occupied, the range broker may collect intents for the
next physical wave. Its wait bound starts from the calibrated request-wave
estimate and follows an EWMA of observed Range GET service time. It is not a
workload-specific timeout; it adapts to current backend execution state.

## Parallel Data Plane

Global planning does not serialize physical execution:

```text
final spans
  -> bounded parallel Range GETs
  -> extract required AU bytes
  -> discard transfer-only gaps
  -> ready decode groups
  -> weighted parallel decoder slots
  -> ordered completion
```

The session first produces immutable per-window spans. The S3 range broker may
then combine overlapping or nearby spans from concurrent callers under the
declared gap and maximum-range bounds. This transport-level operation has no
codec metadata and cannot add records to the decoder closure. The S3 transport
executes the resulting physical spans through one shared connection pool and
semaphore. A completed physical span is sliced back into its original logical
ranges before completion is published.

The executor may fetch gap bytes to reduce object-store request waves, but only
registered closure AUs enter libavcodec. This is the `overfetch without
overdecode` contract.

`execute_window` records per-batch readiness separately from ordered delivery.
Physical completion order may differ from logical order, while sample IDs,
duplicates, batch boundaries, and output order remain exact.

## Cost Decision

For every visible request set, the planner derives candidates from registered
AU offsets and closures. The model accounts for:

- request waves and current global I/O pressure;
- fetched and overfetched bytes;
- submitted access units and decoder work;
- bounded I/O/decode overlap;
- resident encoded AUs and compatible live decoder state;
- backend-calibrated request and bandwidth terms.

Runtime observations update one shared robust estimator. Cross-candidate
selection remains a calibrated heuristic; fixed span-count minimum-byte covers
remain deterministic. Neither path receives a benchmark workload name.

Admission wait, assembly/copy CPU, and ordered-readiness penalties are reported
separately. They must not be hidden inside backend latency.

## Resident Codec State

VClasp has two compressed-state layers:

1. The encoded-AU cache avoids repeated reads. A cached AU may still need to be
   submitted to a decoder.
2. The live cursor cache retains an FFmpeg decoder and its DPB for one
   `(video, GOP/anchor group)` stream. A compatible monotonic request submits
   only newly required AUs.

The index retains packet PTS and DTS. A live cursor feeds through the first
packet whose DTS exceeds the requested PTS, plus the configured frame-thread
pipeline depth. This is the decoder output-release contract; VClasp does not
guess a B-frame margin from GOP size or workload shape.

Closure liveness over the visible window determines whether retaining a cursor
has a future consumer. States with remaining consumers are pinned; exhausted
states become probationary. Capacity eviction is policy-driven and globally
bounded. Resident cursor capacity is a state-memory budget, independent of the
weighted semaphore that limits concurrently active decoder threads. Leaf
targets do not justify persistent decoder state.

VClasp does not serialize FFmpeg decoder checkpoints. libavcodec does not expose
a portable DPB export/import contract.

## Backend Boundary

All backends implement the same logical execution contract:

- Local SSD uses mmap-backed byte ranges.
- S3/MinIO uses explicit final Range GETs through `object_store`.
- AIStore uses its native multi-range request transport.

Backend adapters may change transport mechanics and physical request
accounting. They may not perform dependency resolution, choose a workload path,
or apply another VClasp span policy.
S3 session construction performs a HEAD size check against the registered local
chunk before accepting its catalog-to-object mapping. This startup integrity
check is outside measured request execution. It rejects an obvious stale or
truncated mapping; it is not a cryptographic object-identity check.

## Experiment Controls

Whole-GOP, Keyframe-Prefix, Pair, Normalized, and forced span policies are
mechanism controls. They compile only with the
`experiment-controls` feature and are absent from the default Python module.
They are not alternative production readers.

## Invariants

- Every physical span comes from immutable planned spans admitted to the
  session-global range broker.
- Required closure records are deduplicated before I/O and decode.
- Transfer-only gaps never enter the decoder.
- I/O and decode concurrency use process-global budgets.
- Encoded cache and live DPB state persist across requests in one session.
- Duplicates, batch boundaries, and logical order are restored exactly.
- Production behavior does not branch on workload names.
- Calls may arrive during execution, but cohort formation, planning, and
  resident-state mutation remain serialized within each execution lane.
- Range completion, closure assembly, and decode overlap across concurrently
  executing windows; completed batches need not wait for unrelated ranges.
- Session-wide I/O/decode limits and feedback are shared across overlapping
  windows. Queued cross-window spans may be deduplicated or coalesced, while
  already-dispatched spans are never recalled.
