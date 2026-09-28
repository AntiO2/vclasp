# Setup and development

## Supported host

The verified environment is Linux x86-64, Rust stable, Python 3.9+, and
FFmpeg 7.1 (libavcodec 61, libavformat 61, libavutil 59, libswscale 8).
Plain `cargo test --no-default-features` does not need FFmpeg or x264.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config clang libclang-dev cmake flatbuffers-compiler

rustc --version
cargo --version
flatc --version
```

For media-enabled builds, provide FFmpeg 7.1 and x264 from the same prefix.
Do not combine system headers with Conda runtime libraries. Use the supplied
environment definition, or activate an existing compatible environment such as
`vclasp-training`:

```bash
micromamba create -f environment/ci-native.yml
micromamba activate vclasp-native
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"
export PATH="$CONDA_PREFIX/bin:$PATH"
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" python scripts/doctor.py
```

Run `scripts/doctor.py --import-extension` after installing the Python wheel.
The script checks library major versions, the x264 entry-point symbol, and the
FFmpeg executable. `PKG_CONFIG_PATH` selects build headers/libraries;
`LD_LIBRARY_PATH` selects runtime libraries for a development build. Set it
only on commands that load the extension or run linked tests; do not export it
globally because it can affect unrelated system tools.
For CI the equivalent dependencies are in `environment/ci-native.yml`.

## Rust build

```bash
cargo fmt --check
cargo test --release --no-default-features
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" cargo test --release --features ffmpeg
cargo build --release --features ffmpeg
```

The default suite validates format, index, closure, and span-planning logic.
The `ffmpeg` feature adds libavcodec-backed decode and x264-backed ingestion.
The default build does not compile or link the native x264 bridge.

On a new Linux host, first run the doctor command and build the wheel there.
Copying a locally compiled `.so` without its matching FFmpeg/x264 libraries is
not a supported deployment method. Check `ldd` on the built extension before
moving it, and run the doctor and import check again on the target host.

## Python development install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
maturin develop --release --features ffmpeg
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" python -c 'import vclasp; print(vclasp.__version__)'
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" pytest -q tests/test_logical_scheduler.py
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
