# VClasp Core

VClasp Core is the reusable Rust implementation of codec-closure construction,
object-store span planning, range retrieval, and libavcodec execution. The
paper artifact, baseline ports, deployment recipes, and result corpus live in
the separate
[`vclasp-artifact`](https://github.com/AntiO2/vclasp-artifact) repository.

## Responsibilities

- ingest controlled H.264 video and parse access-unit metadata;
- write one-copy payloads and embedded target-to-closure indexes;
- union and deduplicate closures over a bounded request window;
- enumerate exact minimum-byte span candidates and select a calibrated plan;
- issue bounded-concurrency local or S3-compatible range reads;
- keep transfer-only gap records out of decoder submission;
- decode required access units through libavcodec and restore logical order;
- expose orchestration bindings through PyO3.

Python must not reimplement dependency resolution, byte-range planning, or
decode. It supplies immutable catalogs, logical requests, and experiment
configuration.

## System dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config clang libclang-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libx264-dev flatbuffers-compiler ffmpeg

pkg-config --modversion libavcodec libavformat libavutil libswscale
flatc --version
```

The currently verified FFmpeg development ABI is 6.x. A non-standard install
must expose its `.pc` files through `PKG_CONFIG_PATH`.

## Build

```bash
cargo test --release --features ffmpeg
cargo build --release --features ffmpeg
cp target/release/libvclasp.so ../vclasp.so
```

The extension is imported as:

```python
import vclasp

chunk = vclasp.VClaspChunk("/path/to/layout.chunk")
print(chunk.format_version(), chunk.record_count())
```

Current ingestion and execution are exposed through the
`build_hierarchical_*` functions and `PyLocalHierarchicalBatchExecutor` /
`PyS3HierarchicalBatchExecutor`. See [the API guide](docs/API.md) and
`examples/plan_ranges.rs` for stable entry points.

## On-disk compatibility

The existing FlatBuffer schema identifier and magic are `HVS1` and `HVS`.
They remain unchanged so registered chunks and raw results stay readable.
Likewise, historical manifest keys such as `hvs_rs_sha256` are provenance
fields, not current product names.

## Tests requiring fixtures

Most tests are self-contained. Fixture-backed decode tests are ignored unless
`VCLASP_TEST_CHUNK` is set to a private chunk. The ignored real-video Anchor-P
test accepts `VCLASP_ANCHOR_P_TEST_CHUNK`.

Core-only setup and tests are documented in `docs/SETUP_AND_RUNBOOK.md`.

## License status

This worktree is a public-release staging candidate. The author must replace
the staging notice in `LICENSE` with the selected open-source license before
the repository is made public.
