# Public API

This guide covers the supported VClasp `0.1` workflow:

```text
source videos -> build_vclasp_chunk -> logical targets -> dependency closures
              -> byte spans -> Range GET -> selective decode -> RGB frames
```

## Install

Build the Python extension with the FFmpeg-backed codec path:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
maturin develop --release --features ffmpeg
```

For Rust-only planning and chunk metadata, the default feature set is enough:

```bash
cargo build --release
```

## Range planning

### Rust

```rust
use vclasp::{plan_byte_ranges, PlannedRecord, RangePlan, RecordRange};

let records = vec![
    RecordRange { record_id: 10, offset: 4096, length: 800 },
    RecordRange { record_id: 11, offset: 6000, length: 900 },
];

// Merge positive gaps up to 16 KiB. None disables the maximum range size.
let plans: Vec<RangePlan> = plan_byte_ranges(&records, Some(16 * 1024), None)?;
assert_eq!(plans.len(), 1);
# Ok::<(), String>(())
```

`RangePlan.offset` and `RangePlan.length` address the physical fetch. Each
`PlannedRecord` stores the record ID, relative offset, and length inside that
fetch. Gap bytes may be transferred but are never submitted to the decoder.

### Python

```python
import vclasp

plans = vclasp.plan_byte_ranges(
    [(10, 4096, 800), (11, 6000, 900)],
    merge_threshold_bytes=16 * 1024,
    max_range_bytes=None,
)

# [(range_offset, range_length,
#   [(record_id, relative_offset, record_length), ...]), ...]
```

## Build a chunk

`build_vclasp_chunk` accepts ordinary source videos. Rust performs controlled encode,
access-unit parsing, dependency indexing, payload placement, and final chunk
assembly.

```python
import vclasp

videos = [
    ("video-0001", "ApplyEyeMakeup", "/data/ApplyEyeMakeup/v_0001.mp4"),
    ("video-0002", "ApplyEyeMakeup", "/data/ApplyEyeMakeup/v_0002.mp4"),
]

stats = vclasp.build_vclasp_chunk(
    videos=videos,
    output_path="ucf10.vclasp",
    gop_size=16,
    max_frames=512,
    width=320,
    height=240,
    fps=25,
    crf=23,
    preset="veryfast",
    dependency_policy="hierarchical_b",
    max_b_frames=7,
    b_pyramid="strict",
    b_adapt=0,
    rc_lookahead=0,
    reference_frames=1,
)
```

Each input tuple is `(video_id, class_name, source_path)`. Video IDs must be
unique. The returned tuple reports videos, records, logical targets, payload
bytes, index bytes, total chunk bytes, and maximum closure size.

For `dependency_policy="hierarchical_b"`, `max_b_frames` must be smaller than
`gop_size`; `b_pyramid` is `strict` or `none`; `b_adapt` is 0, 1, or 2; and
adaptive placement requires enough `rc_lookahead` to cover the configured
B-frame run. `reference_frames` is in `1..=16`. Other dependency policies
reject non-default Hierarchical-B parameters so an irrelevant knob cannot be
silently accepted.

## Inspect a chunk

```python
import vclasp

chunk = vclasp.VClaspChunk("ucf10.vclasp")
print(chunk.format_version())       # 1
print(chunk.record_count())
print(chunk.codec_info())
```

The corresponding Rust entry point is:

```rust
use std::path::Path;
use vclasp::chunk::ChunkReader;

let mut chunk = ChunkReader::open(Path::new("ucf10.vclasp"))?;
let codec_config = chunk.read_sps_pps()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Execute logical requests

A target is `(sample_id, video_id, frame_index)`. `sample_id` is returned
unchanged and controls duplicate/order restoration.

### Rust session

The Rust API owns chunk metadata, backend transport, planning, caches, decoder
state, and process-wide budgets. Applications construct one session and submit
only logical targets:

```rust
use std::time::Duration;
use vclasp::session::{
    CostModel, ResidentStateBudget, RuntimeFeedbackConfig, SessionConfig,
    Target, VClaspSession,
};

let config = SessionConfig {
    cost_model: CostModel {
        request_latency_ns: 1_000.0,
        bandwidth_bytes_per_ns: 4.0,
        io_concurrency: 1,
        wave_request_overhead_ns: vec![],
        selection_tolerance_ns: 1_000.0,
        decode_fixed_ns: 0.0,
        decode_access_unit_ns: 100_000.0,
        fetch_decode_overlap: 0.0,
    },
    runtime_feedback: RuntimeFeedbackConfig::default(),
    max_callers: 8,
    max_pending_calls: 32,
    max_inflight_bytes: 256 << 20,
    max_inflight_windows: 8,
    admission_quiet: Duration::from_micros(250),
    max_merge_gap_bytes: Some(16 * 1024),
    max_range_bytes: None,
    decoder_threads: 1,
    global_decode_threads: 8,
    cursor_decoder_threads: 1,
    resident_state: ResidentStateBudget {
        encoded_bytes: 8 << 20,
        read_ahead_bytes: 64 << 10,
        live_cursors: 8,
    },
};

let session = VClaspSession::open_local("ucf10.vclasp", config)?;
let (frames, stats) = session.execute(vec![Target {
    sample_id: 100,
    video_id: "video-0001".into(),
    frame_idx: 32,
}])?;
assert_eq!(frames[0].sample_id, 100);
# Ok::<(), String>(())
```

`open_s3` and `open_aistore` change only construction. Their sessions expose
the same blocking and submitted request methods. The S3 constructor rejects a
cost-model I/O concurrency that differs from the transport's enforced
concurrency budget.
Concurrent calls to either method share one admission stream and one physical
plan; callers do not need to coordinate which API shape their peers use.

`execute` can return frames before the containing physical window finishes
accounting. After stopping producers and consuming their results, call
`session.synchronize()` before reading session counters or changing execution
trace phases. This barrier waits for all earlier commands on every execution
lane, including metric and trace updates; it preserves resident reader state.
`metrics_snapshot()` itself remains a nonblocking snapshot. The same barrier
is available on the Python session and releases the GIL while waiting.
Session `resident_encoded_evictions` counts actual LRU removals during completed
physical windows, including insertion and capacity reduction. Disabled caches
and rejected oversized AUs do not count as evictions. Subtract phase snapshots
after synchronization to exclude warmup; this is a count, not a memory estimate.

`max_pending_calls` bounds admitted calls through execution and result
consumption, not just an ingress queue. A cancelled handle releases its permit
only after execution also releases it. Submit and consume concurrently when
using more work than this capacity; do not fill an unbounded list of handles
before waiting for any result.

`max_inflight_bytes` is a session-wide transient working-buffer reservation
(Python default: 256 MiB). It includes planned range buffers, conservative
Annex-B/packet expansion and RGB staging/restoration. Oversized windows return
an error; smaller competing windows wait for capacity. Decoder jobs keep the
reservation until completion, including cancellation/error cleanup. This is
not a process RSS limit: allocator/container overhead, preloaded metadata,
persistent encoded caches, codec-private DPBs and caller-owned outputs remain
separate. Reservations describe planned logical working buffers; a shared
broker response can pin a larger physical allocation, so this counter must not
be presented as an exact live-allocation counter. Live cursors no longer convert
or retain unrequested RGB. An already
emitted output that was not requested requires replay if requested later.

The current AU catalog is parsed once per chunk reader and shared across
session lanes. Tier-control indexes are parsed only through explicit control
access. Invalid current catalogs are errors, not an implicit legacy fallback.

The transport's Rust completion API owns `bytes::Bytes`; broker consumers
receive shared slices. APIs returning `Vec<u8>` still return independently
owned bytes, and PyO3 still converts results to Python bytes. Cache entries
hold compact Annex-B AUs, not small slices pinning entire GET buffers.

AIStore native GetBatch delivers validated tar members incrementally and has
a 10-second connection timeout and 120-second request deadline. Its request
count is one per GetBatch, not one per member. Member arrival time is not
reported as full-request service time. Returned payload-byte accounting does
not include tar headers or HTTP framing.

Ingestion prepares at most `workers` encoded/probed videos before ordered
serialization. `prepare_seconds` is preparation wall time; `encode_seconds`
and `au_parse_seconds` are sums of per-video elapsed times and can overlap.
They are not CPU-seconds or additive wall-time stages. Frame probing uses
FFmpeg frame decoding; closure construction is restricted to the supported
controlled encodings, not arbitrary H.264 reference-graph discovery. Closure
graph validation does not substitute for decoded RGB reference validation.

Python local, S3 and AIStore session constructors release the GIL during
index loading and backend setup, so independent Python monitoring threads can
observe initialization. Argument conversion and validation retain the GIL.

### Python session

The chunk catalog is the source of truth for valid logical targets:

```python
chunk = vclasp.VClaspChunk("ucf10.vclasp")
video_ids = chunk.video_ids()
frame_indices = chunk.frame_indices(video_ids[0])
```

No caller-side GOP arithmetic or codec dependency inference is required.

The executor's bootstrap cost model is explicit:

```python
cost_model = {
    "request_latency_ns": 800_000.0,
    "bandwidth_bytes_per_ns": 0.25,
    "io_concurrency": 8.0,
    "selection_tolerance_ns": 50_000.0,
    "decode_fixed_ns": 80_000.0,
    "decode_access_unit_ns": 35_000.0,
    "fetch_decode_overlap": 0.5,
    "runtime_feedback_enabled": 1.0,
}
wave_overhead_ns = [800_000.0] * 8
```

`bandwidth_bytes_per_ns` is aggregate transfer bandwidth shared by concurrent
GETs, not the bandwidth of each individual GET. Without wave calibration, the
model charges one startup latency per wave of up to `io_concurrency` requests,
plus the total transferred bytes divided by that aggregate bandwidth.

`selection_tolerance_ns` is a prediction uncertainty band. Within that band,
selection prefers fewer fetched bytes, then less decoder work and fewer GETs
among eligible candidates. The existing predicted-cost bound against the
Fixed 16 KiB reference still applies. This ranking is a cost heuristic, not a simulation of the live I/O
queue; global request/decode limits and configured range caps remain execution
constraints. No workload label is needed.

The session's `admission_quiet_us` is the total waiting budget for collecting
one joint request window. It is not restarted for each arriving caller. Ready
requests may still be collected without waiting when this budget is zero;
barriers fence later submissions off from the current window.

Runtime feedback is enabled by default. Successful windows update a shared
request/byte/decode estimator after a minimum-observation and prediction-error
gate. The estimator does not receive a backend or workload name. Optional
numeric keys are:

```text
runtime_feedback_min_observations
runtime_feedback_forgetting_factor
runtime_feedback_huber_multiplier
runtime_feedback_max_relative_update
runtime_feedback_activation_ape_threshold
runtime_feedback_learning_rate
runtime_feedback_activation_stable_observations
```

Set `runtime_feedback_enabled` to `0` for a static-model control. See
[`runtime_cost_feedback.md`](runtime_cost_feedback.md) before changing the
safety parameters.

Training uses the bounded `VClaspPipeline` by default. Producers may continue
submitting already-sampled batches while the training loop consumes earlier
batches. Capacity is measured over submitted but not yet consumed work, so a
slow consumer applies backpressure without creating an unbounded frame queue.
Physical execution may complete out of order; `take()` restores submission
order.

```python
reader = vclasp.VClaspSession.local(
    "ucf10.vclasp",
    cost_model,
    wave_overhead_ns,
    max_merge_gap_bytes=16 * 1024,
)

pipeline = reader.pipeline(
    max_outstanding_batches=8,
    max_outstanding_targets=8 * 32,
)

from threading import Thread

def produce():
    try:
        for batch in sampler:
            pipeline.submit(batch)  # blocks only when capacity is full
    finally:
        pipeline.close()

producer = Thread(target=produce)
producer.start()
while (item := pipeline.take()) is not None:
    sequence, frames, stats, mode, predicted_ns, residence_ns = item
    train_step(frames)
producer.join()
```

`close()` rejects new submissions but does not discard admitted work; `take()`
returns `None` only after the pipeline has been closed and drained. A single
ordered consumer is the intended interface. Pipeline metrics report submitted,
delivered, outstanding, and peak batch/target counts plus producer
backpressure time.

`execute`, `execute_window`, `submit`, and `submit_window` remain lower-level
interfaces for synchronous applications and diagnostics. A single batch is an
`L=1` request; callers never choose a workload-specific reader.

Fetched range buffers are released after their last assembly consumer has
copied the required AUs into decoder-owned input. Shared ranges remain alive
until all their consumers are done; encoded-cache ownership is separate.
The I/O-to-execution completion queue is bounded using the I/O/decode operation
capacities, so a slow consumer applies backpressure. This bounds queue entries,
not total bytes: range sizes, encoded inputs, decoder state and RGB outputs have
their own memory costs.

`incremental_range_buffer_peak_bytes` reports the largest total of raw fetched
buffers held by one incremental assembly window. Session aggregation takes the
maximum rather than adding peaks across windows or lanes. It excludes bytes
in transport/completion queues, encoded caches, decoder inputs and RGB outputs;
it is neither total RSS nor a process-wide memory limit. Non-incremental paths
do not contribute to this metric.

Execution also reports the following counters in returned stats
and `metrics_snapshot()`. Python names have the `incremental_` prefix; Rust
exposes the same fields through `ExecutionStats.incremental` and
`SessionMetrics.incremental` (`session::IncrementalExecutionMetrics`).

| Suffix after `incremental_` | Meaning |
| --- | --- |
| `range_buffer_received_bytes` | Logical range-buffer bytes delivered to assembly; not unique physical network bytes |
| `range_buffer_released_bytes` | Delivered bytes released after the last consumer, including unused late delivery |
| `range_completion_blocked_sends` | Range completion sends that encountered a full execution queue |
| `range_completion_block_ns` | Time those range producers spent in blocking send calls |
| `decode_queue_ns` | Time from decode-job submission until its worker receives it |
| `decode_resource_wait_ns` | Worker time acquiring the shared decode budget and decoder-slot lock |
| `decode_service_ns` | Time in the decoder helper, including its RGB materialization |
| `decode_timing_samples` | Successfully completed jobs contributing decode timings |
| `decoded_rgb_bytes` | RGB bytes produced by those jobs, before logical duplicate restoration |
| `output_restore_ns` | Time restoring logical RGB outputs; may overlap other jobs |
| `output_clone_bytes` | RGB bytes cloned during logical output restoration, excluding Python copies |

Session aggregation sums these counters. Decode timing covers successful
stateless and resident-cursor jobs. The `incremental_` names identify the
existing metric group, not an execution-policy requirement. A zero sample
count means no decode-timing sample, not that decoder work was free.
Range send counters exclude decoder-completion sends and the final completion
notification. Decoder queue/resource/service times exclude Python conversion
and output delivery. The original `decode_ns` retains its existing worker-time
scope, including resource waiting; do not add it to the new decode timings.

Parallel-job timing sums are not a mutually exclusive wall-clock breakdown and
do not provide queue P95/P99 distributions. Logical buffer bytes may differ
from attributed physical GET bytes when one GET serves several consumers.
Do not derive exact overfetch by blindly subtracting per-window useful bytes
from physical bytes; use the existing extent and consumer trace for attribution.

### Optional codec profiling

`session.set_stage_profiling(True)` enables detailed timing for subsequent jobs;
it is off by default. Stop producers and call `synchronize()` before changing
the setting for an overhead comparison. Returned stats and session snapshots
include `codec_profile_jobs`, `codec_accepted_packets`, `codec_received_frames`,
and `codec_<stage>_{wall_ns,thread_cpu_ns,calls,cpu_samples,bytes}`. Stages are
`decoder_setup`, `packet_copy`, `send`, `receive`, `flush`,
`rgb_convert`, and `rgb_copy`. Accepted packets and received frames are distinct
from closure size and the existing submitted-AU counters. Packet copy bytes
count FFmpeg-owned input copies; RGB copy bytes count packed RGB materialization,
not logical duplicate restoration or Python output copies.

CPU samples use the calling thread's CPU clock on Linux. They exclude codec
worker-thread CPU, which requires process-level measurement. Zero CPU samples
mean unavailable or disabled, not zero CPU cost. Job aggregates cover completed
successful jobs; a failed request is not a successful timing sample.

`take_decode_job_trace()` returns `(events, dropped_events)`. Events contain
lane/job IDs, submitted and requested record IDs, codec metrics, and monotonic
timestamps for queue submission, worker start, resource acquisition and service
completion. The buffer holds at most 8,192 events per execution lane and must
be drained periodically; lost events are counted. IDs are scoped to the lane.
Linux timestamps use `CLOCK_MONOTONIC`; other platforms report the clock as
unavailable. These are job timelines, not a serialized critical-path breakdown.
Events also identify their logical consumer sample IDs, stateless/resident
execution path and decoder context address. Addresses are process-local debug
identifiers and may be reused after a decoder is destroyed; use opening/flush
counts and job order rather than treating addresses as permanent IDs.
`codec_decoder_opens` counts successful decoder construction under profiling.

Python batch/window results separately report `python_output_wall_ns`,
`python_output_bytes`, and available calling-thread CPU samples for conversion
to Python-owned bytes. These adapter metrics are not Rust-session counters.
Admission metrics include current/peak pending calls and cumulative admission
wait. Snapshot capacities and current occupancy are gauges; lifetime peaks must
not be differenced or presented as post-warmup interval maxima.

`resident_state_snapshot()` returns one coherent cache/cursor snapshot per
execution lane. It is published at construction and after an execution returns,
including errors. `updated_monotonic_ns` identifies its observation time;
`scope="last_completed_execution_per_lane"` means it is not an instantaneous
inspection of an active lane. Fields include cache hits/misses/evictions,
encoded-AU and read-ahead payload bytes, the byte budget, resident entries,
live/pinned/probationary cursor counts and cursor capacity. Payload byte counts
are not allocator usage or codec DPB memory. Sample process RSS/PSS separately.

S3-compatible applications should create one process-wide session. The session
owns the planner, caches, live decoder state, and global resource budgets.
Credentials should come from the environment, not source code:

```python
import os
import vclasp

session = vclasp.VClaspSession(
    "ucf10.vclasp",                  # local header/index copy
    "datasets/ucf10.vclasp",         # object key
    os.environ["VCLASP_S3_ENDPOINT"],
    os.environ["VCLASP_S3_BUCKET"],
    os.environ["VCLASP_S3_ACCESS_KEY"],
    os.environ["VCLASP_S3_SECRET_KEY"],
    cost_model,
    wave_overhead_ns,
    8,                               # maximum concurrent callers
    global_io_concurrency=8,
    global_decode_concurrency=8,
    admission_quiet_us=250,
    max_merge_gap_bytes=16 * 1024,
)
batches, ready_ns, delivery_ns, stats, mode, predicted_ns = (
    session.execute_window(window)
)
```

Independent application threads may call `session.execute(batch)`
concurrently. VClasp admits those logical requests into one bounded planning
window before issuing physical I/O; callers provide neither worker IDs nor
routes. `max_callers` is an admission bound, not a pool size.
`session_admission_ns` and `session_joint_submissions` expose the batching
cost and the number of jointly planned submissions.

On S3-compatible storage, concurrently executing windows also share a
completion-driven range broker. It coalesces only their already planned byte
spans, publishes each logical range when its physical GET completes, and
attributes every shared GET exactly once. Decode can begin before the remaining
ranges in the same submitted request finish.

For lower-level asynchronous producers, `submit` and `submit_window` return
`PendingBatch` and `PendingWindow`. Calling `result()` waits without holding the
Python GIL and consumes the result exactly once. `VClaspPipeline` manages this
bounded outstanding-work discipline for the standard training path. VClasp
never infers a workload class.

Use `VClaspSession.aistore(...)` to select the AIStore transport. Backend
selection changes construction only; `execute` and `execute_window` are
identical across local, S3-compatible, and AIStore storage.

Each frame is `(sample_id, rgb_bytes, width, height)`. `stats` includes request,
byte, decoder-submission, planning, fetch, decode, and reorder counters/timers.

## Object-store transport

The standalone `vclasp-object-store` crate is the shared transport layer:

- Rust: `S3ObjectStoreClient::fetch_object_ranges`;
- C FFI: `object-store-transport/include/vclasp_object_store.h`;
- Python diagnostics: `vclasp.S3ObjectStoreReader`, available only with
  `experiment-controls`.

It preserves input order, rejects short reads, and uses one bounded connection
pool. It does not perform dependency resolution or decode.

With `experiment-controls`, an S3-backed `VClaspSession` can call
`start_execution_trace(max_range_events, max_decode_events)` before a measured
interval and `take_execution_trace()` after all work completes. The latter
returns member-range events and actual submitted AU records, with independent
expected counts and dropped-event counts, in the order
`(ranges, GETs, dropped_ranges, AUs, submitted_AUs, dropped_AUs)`.
Events sharing `physical_request_id` belong to one actual GET; combine their
`consumer_sample_ids` instead of charging that GET to each member. The physical
object offset and length may differ from a member's planned payload range when
concurrent requests are coalesced. A submitted AU can likewise serve more than
one logical target. A nonzero dropped count makes the corresponding trace
incomplete. This diagnostic is not part of the default reader API.

To stream long runs, Python can call `take_execution_trace(stop=False)` while
work continues; Rust uses `drain_execution_trace()`. Each drain atomically
returns complete observed windows, resets their counts and frees buffer capacity,
while preserving capture and unique window IDs. Sum expected and dropped counts
across drains, and combine shared physical request consumers across all drains.
After synchronizing the reader, perform a final default `take_execution_trace()`
to stop capture. A window larger than the configured capacity can still lose
events, which must invalidate the run.

The diagnostic `decoder_activity()` method samples each distinct shared decode
budget directly, without waiting behind execution commands. Each entry reports
`active_jobs`, `queued_jobs`, `available_threads`, `lifetime_peak_active_jobs`,
`completed_jobs`, and cumulative `acquisition_wait_ns`. Active jobs count held
decode permits, not FFmpeg threads or allocated decoder contexts; queue counts
cover threads waiting to acquire those permits. Acquisition time includes mutex
and capacity waits. Configuration validation does not acquire a decode permit.
Lifetime peaks include initialization and warmup and must not be subtracted to
infer measurement-phase peaks; preserve the timestamped samples instead.

The diagnostic `io_activity()` method samples each distinct session backend.
S3 entries report the shared transport's `active_requests`,
`outstanding_requests`, `queued_requests`, configured `max_concurrency`, and
service-time EWMA. `S3ObjectStoreReader.pressure_snapshot()` exposes the same
transport measurement for baseline adapters. Active counts represent held I/O
permits; queued counts represent outstanding transport tasks awaiting those
permits. They exclude planner and broker admission queues. Fields are read
independently and can briefly disagree during transitions. Local backends
return `None`; a missing gauge is not interpreted as zero activity.

On Linux, each exact physical range event also contains `monotonic_timing`:
the start, first-byte, and completion timestamps in nanoseconds from
`CLOCK_MONOTONIC`. These timestamps share Python's `time.monotonic_ns()` clock
and remain comparable across execution windows and broker consumers.
The existing `started_ns`, `first_byte_ns`, and `completed_ns` values measure
execution-window timing. Unsupported platforms and aggregate transport modes
return `None` for `monotonic_timing`; those events cannot establish exact
cross-window GET concurrency. No chunk format changes are required.

## Mechanism controls

Anchor/Delta, Pair, Prefix, adaptive portfolio, materialization, and forced
actions are mechanism controls. Raw transport readers, direct decoder adapters,
and tier-oriented chunk methods are controls as well. They are absent from the
default Python module; diagnostic hooks require the explicit
`experiment-controls` Cargo feature. The paper artifact pins the experiment
source and configuration used for registered ablations.

## Errors and invariants

- The reader accepts only FlatBuffer identifier `VCSP`, magic `VClasp`, and
  format version 1.
- Logical duplicates and request order are preserved.
- Positive-gap bytes may be fetched but never decoded.
- Out-of-bounds ranges, short object reads, missing closures, and unsupported
  codecs return errors instead of partial results.
