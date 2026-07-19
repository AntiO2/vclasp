# Public API

This guide covers the supported VClasp `0.1` workflow:

```text
source videos -> build_chunk -> logical targets -> dependency closures
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

`build_chunk` accepts ordinary source videos. Rust performs controlled encode,
access-unit parsing, dependency indexing, payload placement, and final chunk
assembly.

```python
import vclasp

videos = [
    ("video-0001", "ApplyEyeMakeup", "/data/ApplyEyeMakeup/v_0001.mp4"),
    ("video-0002", "ApplyEyeMakeup", "/data/ApplyEyeMakeup/v_0002.mp4"),
]

stats = vclasp.build_chunk(
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

The executor's measured cost model is explicit:

```python
cost_model = {
    "request_latency_ns": 800_000.0,
    "bandwidth_bytes_per_ns": 0.25,
    "io_concurrency": 8.0,
    "selection_tolerance_ns": 50_000.0,
    "decode_fixed_ns": 80_000.0,
    "decode_access_unit_ns": 35_000.0,
    "fetch_decode_overlap": 0.5,
}
wave_overhead_ns = [800_000.0] * 8
```

Local execution:

```python
reader = vclasp.LocalBatchExecutor(
    "ucf10.vclasp",
    cost_model,
    wave_overhead_ns,
    max_merge_gap_bytes=16 * 1024,
)

frames, stats, mode, predicted_ns = reader.execute([
    (100, "video-0001", 32),
    (101, "video-0002", 96),
])
```

S3-compatible execution uses a local metadata copy and the remote chunk
object. Credentials should come from the environment, not source code:

```python
import os
import vclasp

reader = vclasp.S3BatchExecutor(
    "ucf10.vclasp",                  # local header/index copy
    "datasets/ucf10.vclasp",         # object key
    os.environ["VCLASP_S3_ENDPOINT"],
    os.environ["VCLASP_S3_BUCKET"],
    os.environ["VCLASP_S3_ACCESS_KEY"],
    os.environ["VCLASP_S3_SECRET_KEY"],
    cost_model,
    wave_overhead_ns,
    max_concurrency=8,
    max_merge_gap_bytes=16 * 1024,
)
frames, stats, mode, predicted_ns = reader.execute(targets)
```

Each frame is `(sample_id, rgb_bytes, width, height)`. `stats` includes request,
byte, decoder-submission, planning, fetch, decode, and reorder counters/timers.

## Object-store transport

The standalone `vclasp-object-store` crate is the shared transport layer:

- Rust: `S3ObjectStoreClient::fetch_object_ranges`;
- C FFI: `object-store-transport/include/vclasp_object_store.h`;
- Python: `vclasp.S3ObjectStoreReader`.

It preserves input order, rejects short reads, and uses one bounded connection
pool. It does not perform dependency resolution or decode.

## Experimental policy APIs

The following classes support paper-artifact experiments but are not part of
the `0.1` stability promise: Anchor/Delta, Pair, Prefix, adaptive portfolio,
materialization, shared-resource saturation, and forced-ablation executors.
Their exact experiment configs and invocation are maintained in
`vclasp-artifact`, not duplicated here.

## Errors and invariants

- The reader accepts only FlatBuffer identifier `VCL1`, magic `VCLASP`, and
  format version 1.
- Logical duplicates and request order are preserved.
- Positive-gap bytes may be fetched but never decoded.
- Out-of-bounds ranges, short object reads, missing closures, and unsupported
  codecs return errors instead of partial results.
