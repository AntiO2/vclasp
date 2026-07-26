# VClasp Core

VClasp Core is a Rust library for dependency-aware access to predictively
encoded video on local filesystems and S3-compatible object stores. It turns an
already-sampled window of logical video targets into codec closures, physical
byte spans, bounded Range GETs, and ordered RGB outputs.

The caller never labels a request as sequential, random, Zipf, or same-video.
VClasp derives dependency reuse and physical locality from the request window
and the chunk index, then compares legal execution plans with one
request/byte/decode cost model. The supplied model is a safe bootstrap:
post-execution measurements update it online after an accuracy gate, without
backend names or workload labels.

## What is in this repository

- Rust ingestion, access-unit parsing, closure construction, and validation;
- one-copy chunk payload and embedded Parquet target index;
- exact and coalesced span planning;
- bounded local and S3-compatible reads through `object_store`;
- libavcodec decode with exact logical-order restoration;
- bounded encoded-AU and live-decoder state;
- Python bindings through PyO3;
- unit and integration tests.

Paper runners, cluster deployment, baseline ports, datasets, and result files
belong in the separate reproducibility artifact, not this reusable core.

## Requirements

Verified on Ubuntu x86-64 with Rust stable, Python 3.12, and FFmpeg 6.x.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config clang libclang-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libx264-dev flatbuffers-compiler ffmpeg python3-dev

pkg-config --modversion libavcodec libavformat libavutil libswscale
```

## Build and test

```bash
cargo test --release --features ffmpeg
cargo build --release --features ffmpeg
cp target/release/libvclasp.so vclasp.so
PYTHONPATH="$PWD" python -c 'import vclasp; print(vclasp.__file__)'
```

Fixture-backed decode tests are ignored unless `VCLASP_TEST_CHUNK` is set. The
default suite does not require private datasets.

## Python quick start

### 1. Build a chunk

Python supplies paths and immutable encoding parameters; Rust performs video
decode, controlled H.264 encode, AU parsing, closure construction, validation,
index serialization, and chunk writing.

```python
from pathlib import Path
import vclasp

videos = [
    ("video-0001", "class-a", "/data/class-a/video-0001.mp4"),
    ("video-0002", "class-b", "/data/class-b/video-0002.mp4"),
]

vclasp.build_vclasp_chunk(
    videos=videos,
    output_path="dataset.vclasp",
    gop_size=16,
    max_frames=512,
    width=320,
    height=240,
    fps=25,
    crf=23,
    preset="veryfast",
    workers=8,
)
```

### 2. Read an already-sampled request window

A request is `(sample_id, video_id, frame_index)`. A window is a list of
logical batches whose IDs and order have already been chosen by the application
or data loader. Visibility permits cross-batch deduplication and state reuse;
it never permits VClasp to change sample order.

`VClaspChunk.video_ids()` and `VClaspChunk.frame_indices(video_id)` enumerate
the valid logical targets directly from the embedded closure catalog.

```python
cost_model = {
    "request_latency_ns": 200_000.0,
    "bandwidth_bytes_per_ns": 1.0,
    "io_concurrency": 8.0,
    "selection_tolerance_ns": 10_000.0,
    "decode_fixed_ns": 50_000.0,
    "decode_access_unit_ns": 150_000.0,
    "fetch_decode_overlap": 0.0,
    # Runtime feedback is enabled by default. Set this to 0 only for a
    # static-model control.
    "runtime_feedback_enabled": 1.0,
}

reader = vclasp.VClaspSession.local(
    "dataset.vclasp",
    cost_model,
    [],                         # optional calibrated request-wave overheads
    decoder_threads=1,
    global_decode_concurrency=8,
    resident_encoded_bytes=8 << 20,
    resident_read_ahead_bytes=64 << 10,
    resident_cursor_capacity=8,
)

window = [
    [(0, "video-0001", 7), (1, "video-0002", 31)],
    [(2, "video-0001", 8), (3, "video-0002", 47)],
]

pipeline = reader.pipeline(
    max_outstanding_batches=8,
    max_outstanding_targets=256,
)
```

Training should run its sampler producer and model consumer concurrently:

```python
from threading import Thread

def produce():
    try:
        for batch in data_sampler:
            pipeline.submit(batch)
    finally:
        pipeline.close()

producer = Thread(target=produce)
producer.start()
while (item := pipeline.take()) is not None:
    sequence, frames, stats, mode, predicted_ns, residence_ns = item
    train_step(frames)
producer.join()
```

`submit()` blocks when the declared batch or target capacity is full. Execution
may complete out of order, but `take()` returns submission order and drains all
admitted work after `close()`. This bounds prefetched state while overlapping
sampling, object I/O, decode, and model compute.

Use `execute(batch)` as the synchronous `L=1` convenience form and
`execute_window` when an application already owns one finite request window.

For S3-compatible storage, create one process-wide `VClaspSession`.
Concurrent callers submit logical targets to that session; they do not select
planner workers, cache partitions, or workload identities. The configured
caller count bounds an admission cohort. `max_inflight_windows` independently
bounds overlapping physical execution.
Concurrent calls are jointly admitted within each cohort; the session restores
each call's original batch boundaries and output order.

`submit` and `submit_window` remain lower-level one-shot handle APIs.
Visibility is supplied by already-sampled work, not inferred from a workload
name.
Select local or AIStore transport with
`VClaspSession.local(...)` or `VClaspSession.aistore(...)`; execution preserves
the same request-window and output-order contract.

## Execution contract

The production path is:

```text
request window
  -> target lookup
  -> sufficient closure union
  -> physical span candidates
  -> costed closure / region / resident-cursor choice
  -> bounded range reads
  -> discard transfer-only gaps
  -> codec-order decode
  -> restore duplicates, batches, and logical order
```

Resident decoder state is selected only when both conditions hold:

1. registered closures prove a future consumer in the visible window; and
2. its monotonic suffix is cheaper than ordinary window execution under the
   same current cost model and resource limits.

The encoded-byte budget and live-cursor count are separate resources. FFmpeg's
private DPB allocation is not serializable or directly byte-accounted, so
process RSS should also be measured in memory-sensitive deployments.

Historical Prefix, Pair, and Normalized implementations live under
`src/controls/` solely for mechanism tests. They are absent from the default
Python module, are not alternative production readers, and are never selected
from a workload name. Diagnostic hooks require the explicit
`experiment-controls` feature; the paper artifact pins the experiment source
and configuration used for reproduction.

See [execution architecture](docs/execution_architecture.md) for the detailed
planner, cache, and fallback invariants. See
[runtime cost feedback](docs/runtime_cost_feedback.md) for the online update
model, safety gate, telemetry, and diagnostic runner. The
[final architecture audit](docs/final_architecture_audit.md) maps every
production invariant to its code and verification evidence.

## Development

Before submitting a change:

```bash
cargo test --features ffmpeg --lib
cargo test --release --features ffmpeg
cargo fmt --check
```

Do not add benchmark-name branches to the core. New execution choices must be
derived from logical targets, registered codec dependencies, physical extents,
runtime observations, and explicit resource budgets.

## License

VClasp is licensed under the
[GNU Affero General Public License v3.0](LICENSE), using the SPDX identifier
`AGPL-3.0-only`. Linked third-party components retain their own licenses.
