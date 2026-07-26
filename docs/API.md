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
use vclasp::{plan_ranges, PlannedRecord, RangePlan, RecordRange};

let records = vec![
    RecordRange { record_id: 10, offset: 4096, length: 800 },
    RecordRange { record_id: 11, offset: 6000, length: 900 },
];

// Merge positive gaps up to 16 KiB. None disables the maximum range size.
let plans: Vec<RangePlan> = plan_ranges(&records, Some(16 * 1024), None)?;
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
)
```

Each input tuple is `(video_id, class_name, source_path)`. Video IDs must be
unique. The returned tuple reports videos, records, logical targets, payload
bytes, index bytes, total chunk bytes, and maximum closure size.

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

## Mechanism controls

Anchor/Delta, Pair, Prefix, adaptive portfolio, materialization, and forced
actions are mechanism controls. Raw transport readers, direct decoder adapters,
and tier-oriented chunk methods are controls as well. They are absent from the
default Python module; diagnostic hooks require the explicit
`experiment-controls` Cargo feature. The paper artifact pins the experiment
source and configuration used for registered ablations.

## Errors and invariants

- The reader accepts only FlatBuffer identifier `VCL1`, magic `VCL`, and
  format version 1.
- Logical duplicates and request order are preserved.
- Positive-gap bytes may be fetched but never decoded.
- Out-of-bounds ranges, short object reads, missing closures, and unsupported
  codecs return errors instead of partial results.
