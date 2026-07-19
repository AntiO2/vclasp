# VClasp Core setup

## Host dependencies

The verified development environment is Ubuntu x86-64, Rust stable, Python
3.12, and FFmpeg/libavcodec 6.x.

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

## Build and test

```bash
cargo test --release --no-default-features
cargo test --release --features ffmpeg
cargo build --release --features ffmpeg
```

The no-default-features suite validates chunk, index, closure, and planner
logic. The FFmpeg feature adds libavcodec-backed decode tests. Tests that need a
real video fixture are ignored unless `VCLASP_TEST_CHUNK` is set.

## Python extension

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cargo build --release --features ffmpeg
cp target/release/libvclasp.so ./vclasp.so
PYTHONPATH="$PWD" python -c 'import vclasp; print(vclasp.__file__)'
```

Local paths, endpoints, and credentials belong in an ignored `.env`. Copy
`.env.example` and supply values locally. Never commit `.env`.

## Object-store transport

The standalone `object-store-transport` crate provides the common Rust and C
Range GET data plane used by VClasp and the source-grounded baseline ports.

```bash
cargo test --manifest-path object-store-transport/Cargo.toml
cargo build --release --manifest-path object-store-transport/Cargo.toml
```

Deployment, datasets, formal benchmark configs, and paper reproduction belong
to `vclasp-artifact`, not this reusable core repository.
