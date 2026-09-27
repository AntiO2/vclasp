# Setup and development

## Supported host

The verified environment is Linux x86-64, Rust stable, Python 3.9+, and
FFmpeg/libavcodec 6.x.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config clang libclang-dev cmake \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libx264-dev flatbuffers-compiler ffmpeg

rustc --version
cargo --version
pkg-config --modversion libavcodec libavformat libavutil libswscale
flatc --version
```

For a non-standard FFmpeg install, expose its `.pc` files through
`PKG_CONFIG_PATH` before invoking Cargo.

## Rust build

```bash
cargo fmt --check
cargo test --release --no-default-features
cargo test --release --features ffmpeg
cargo build --release --features ffmpeg
```

The default suite validates format, index, closure, and span-planning logic.
The `ffmpeg` feature adds libavcodec-backed decode and execution.

## Python development install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
maturin develop --release --features ffmpeg
python -c 'import vclasp; print(vclasp.__version__)'
pytest -q tests/test_logical_scheduler.py
```

Do not copy or rename the compiled shared library manually; `maturin` installs
the correctly named extension into the active environment. In particular, a
root-level `vclasp.so` shadows the installed extension when Python starts in
the checkout and can silently run stale code. Diagnostic builds belong outside
the source tree and must be identified by SHA-256.

## Format schema

The source schema is `schemas/chunk_v1.fbs`. The checked-in Rust binding is
`src/format/chunk_schema.rs`. Both must use file identifier `VCSP`. The runtime
header also validates magic `VClasp` and format version 1.

This first public release intentionally has no reader for pre-release research
chunks. Any schema change after `0.1.0` must increment the format version and
include an explicit migration decision.

## Patched x264 option

The normal build uses the system x264 API. If a locally patched x264 exports the
reference-control hooks used by the optional reference-page encoder, build with:

```bash
VCLASP_PATCHED_X264=1 cargo build --release --features ffmpeg
```

## Object-store transport

```bash
cargo test --manifest-path object-store-transport/Cargo.toml
cargo build --release --manifest-path object-store-transport/Cargo.toml
```

Copy `.env.example` to `.env` for local endpoints and credentials. `.env` is
ignored and must never be committed.

## Fixture-backed tests

Most tests generate their own data. Tests requiring a real encoded chunk are
ignored unless `VCLASP_TEST_CHUNK` points to a VClasp format-v1 chunk. The
Anchor-P fixture test uses `VCLASP_ANCHOR_P_TEST_CHUNK`.

## Release checklist

1. `cargo fmt --check` and both Rust test configurations pass.
2. The Python wheel builds with `maturin build --release --features ffmpeg`.
3. Examples run from a clean clone.
4. No prototype format identifier or compatibility schema is present.
5. Package metadata declares `AGPL-3.0-only` and the repository contains the
   complete GNU AGPLv3 license text.
