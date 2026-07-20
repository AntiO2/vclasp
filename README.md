# VClasp Core

VClasp is a Rust library for reading predictive video from object storage at
ML-sample granularity. It resolves codec dependencies, turns the required
access units into byte ranges, coalesces those ranges for the storage backend,
and decodes only the required records.

This repository contains the reusable library and Python bindings. Benchmark
drivers, baseline ports, deployment recipes, and released result tables live in
[`vclasp-artifact`](https://github.com/AntiO2/vclasp-artifact).

> **Release status:** `0.1.0` is the first public format and API. The repository
> remains private until the release license is selected.

## What VClasp provides

- a self-describing chunk with H.264 payloads and a Parquet record index;
- dependency-closure resolution for logical frame and clip requests;
- exact byte-range planning and configurable range coalescing;
- local, S3-compatible, and AIStore transports with bounded concurrency;
- libavcodec-backed selective decode and logical-order restoration;
- deterministic parallel ingestion with profiled closure-index construction;
- Rust APIs for data-plane integration and PyO3 bindings for ML loaders.

## Quick start

### 1. Install native dependencies

Ubuntu 22.04/24.04:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config clang libclang-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libx264-dev flatbuffers-compiler ffmpeg
```

### 2. Run the Rust planner example

```bash
cargo run --example plan_ranges
```

The public planner accepts physical record extents and returns the contiguous
ranges to fetch plus each requested record's location inside its range:

```rust
use vclasp::{plan_byte_ranges, RecordRange};

let records = vec![
    RecordRange { record_id: 1, offset: 0, length: 1024 },
    RecordRange { record_id: 2, offset: 4096, length: 1024 },
];
let ranges = plan_byte_ranges(&records, Some(4096), None)?;
# Ok::<(), String>(())
```

### 3. Install the Python extension

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
maturin develop --release --features ffmpeg
python -c 'import vclasp; print(vclasp.__version__)'
```

Then plan ranges without moving dependency or planning logic into Python:

```python
import vclasp

ranges = vclasp.plan_byte_ranges(
    [(1, 0, 1024), (2, 4096, 1024)],
    merge_threshold_bytes=4096,
)
print(ranges)
```

See [Public API](docs/API.md) for chunk ingestion, local execution, S3
execution, return schemas, and configuration fields.

## Repository layout

```text
src/
  format/       chunk header, Parquet index, generated FlatBuffer bindings
  codec/        x264 ingestion adapter and libavcodec decode
  ingest/       source-video ingestion and closure-index construction
  planning/     closure representations, span planning, cost policies
  execution/    batch schedulers, deduplication, ordering, caches
  storage/      local, S3-compatible, and AIStore backends
native/         small C bridge for controlled x264 reference behavior
schemas/        VClasp on-disk FlatBuffer schema
object-store-transport/  standalone Rust/C Range GET transport crate
examples/       minimal Rust and Python examples
```

The filesystem is grouped by responsibility while the Rust facade keeps the
short public paths `vclasp::chunk`, `vclasp::index`, and the crate-root planner
types.

## Chunk format

VClasp `0.1.0` writes format version 1 with FlatBuffer identifier `VCL1` and
header magic `VCLASP`:

```text
[u32 header length]
[FlatBuffer header]
[codec configuration bytes]
[encoded access-unit payload]
[Parquet record and closure index]
```

This is the first public format. Pre-release research chunks are not accepted;
regenerate them with the public VClasp builder. No compatibility code or schema
is shipped for internal prototype formats.

## Build and test

```bash
cargo fmt --check
cargo test --release --no-default-features
cargo test --release --features ffmpeg
cargo test --manifest-path object-store-transport/Cargo.toml
```

Fixture-backed codec tests are ignored unless `VCLASP_TEST_CHUNK` points to a
format-v1 VClasp chunk. Full setup and troubleshooting are in
[Setup and development](docs/SETUP_AND_RUNBOOK.md).

## Scope and stability

The supported `0.1` surface is the chunk reader/writer, range planner,
object-store transport, `build_chunk`, and the Local/S3/AIStore batch
executors. Research policy classes remain available for artifact reproduction
but are explicitly marked experimental in the API guide.

## License

The current `LICENSE` is a release-staging notice and grants no redistribution
rights. It must be replaced with the selected open-source license before the
repository is made public.
