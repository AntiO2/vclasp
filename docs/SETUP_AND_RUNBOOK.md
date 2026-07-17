# VClasp setup and runbook

## 1. Host dependencies

Supported development environment: Ubuntu x86-64, Rust stable, Python 3.12,
FFmpeg/libavcodec 6.x, and an NVIDIA GPU only for the later training gate.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config clang libclang-dev cmake \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libx264-dev flatbuffers-compiler ffmpeg git-lfs

rustc --version
cargo --version
pkg-config --modversion libavcodec libavformat libavutil libswscale
ffmpeg -version | head -1
flatc --version
```

Do not install VMAF for the current evidence gate. The installed FFmpeg exposes
PSNR and SSIM; VMAF is optional only when an already verified libvmaf toolchain
is available.

## 2. Source and submodules

```bash
git clone --recurse-submodules <repository> cloud-video-layout-preexp
cd cloud-video-layout-preexp
git submodule update --init --recursive
git submodule status --recursive
```

Pinned submodules include AIStore, Lance, TorchCodec, VSS, VStore, and
WebDataset. VSS and VStore contain source-grounded object-store ports; they are
not unmodified upstream object-store implementations.

## 3. Python environment

For a fresh open-source checkout, use a standard virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Torch and TorchCodec must use mutually compatible CPU or CUDA wheels. If the
default index does not provide the pinned pair, install both from the wheel
channel documented for the selected TorchCodec release, then install the rest
of `requirements.txt`.

The authoritative package contract is `configs/systems_environment.json`.
The paper experiment machine uses `.conda-baselines/bin/python`. Preserve that
environment until CPU Torch/TorchCodec wheel URLs and hashes are exported.

Minimum verification, after loading `.env` or activating `.venv`:

```bash
PYTHON=${VCLASP_PYTHON:-.venv/bin/python}
"$PYTHON" - <<'PY'
import av, lance, numpy, pandas, pyarrow, torch, torchcodec, webdataset
print(torch.__version__, torchcodec.__version__)
PY
```

Build the native extension after activating or selecting the target Python
environment:

```bash
cd vclasp-rs
cargo test --release --features ffmpeg
cargo build --release --features ffmpeg
cp target/release/libvclasp.so ../vclasp.so
cd ..
PYTHON=${VCLASP_PYTHON:-.venv/bin/python}
PYTHONPATH="$PWD" "$PYTHON" -c \
  'import vclasp; print(vclasp.__file__)'
```

Local paths and credentials belong in `.env`, which is ignored by Git. Start
from `.env.example`, then load it before commands that need those values:

```bash
cp .env.example .env
set -a
source .env
set +a
```

## 4. Test gates

```bash
cd vclasp-rs
cargo test --release --features ffmpeg
cd ..

PYTHON=${VCLASP_PYTHON:-.venv/bin/python}
PYTHONPATH="$PWD:$PWD/scripts" "$PYTHON" -m pytest -q \
  tests/test_reproducibility.py \
  tests/test_aistore_saturation_cell.py \
  tests/test_run_object_store_capacity_formal.py \
  tests/test_analyze_hierarchical_cumulative_ablation.py \
  tests/test_analyze_hierarchical_g16_unified.py \
  tests/test_run_hierarchical_lookahead_gate.py \
  tests/test_run_hierarchical_main_cell.py \
  tests/test_run_exact_order_mp4_trace.py
```

The legacy Rust decode fixture is intentionally not committed. Mount or link
the registered `layouts/` directory in an isolated worktree before running all
tests.

## 5. Distributed object stores

The managed deployment uses cds35--37. Scripts only manage their own pidfiles
and roots; they do not format disks or kill unrelated services. Review
`deploy/distributed_backends/README.md` before use.

MinIO on each storage node:

```bash
cd /path/to/cloud-video-layout-preexp
deploy/distributed_backends/minio_node.sh start
deploy/distributed_backends/minio_node.sh status
deploy/distributed_backends/minio_node.sh stop
```

AIStore on each storage node:

```bash
cd /path/to/cloud-video-layout-preexp
python deploy/distributed_backends/aistore_node.py configure
python deploy/distributed_backends/aistore_node.py start
python deploy/distributed_backends/aistore_node.py status
python deploy/distributed_backends/aistore_node.py stop
```

Run benchmark clients on the local compute host. cds35--37 are storage nodes;
do not install the training or benchmark Python environment there. Credentials
and endpoints must come from the existing environment/configuration and must
not be committed.

Before a baseline matrix, run the shared-transport parity smoke. Timed VClasp,
VSS, and VStore paths use the Rust transport; boto3 is allowed only as an
untimed byte oracle.

```bash
PYTHON=${VCLASP_PYTHON:-.venv/bin/python}
PYTHONPATH="$PWD:$PWD/scripts" "$PYTHON" \
  scripts/object_store_transport_parity_smoke.py --help
```

## 6. Formal-run provenance

Formal runs must use a clean commit or one immutable, explicitly applied patch.
Before execution, record:

```bash
git rev-parse HEAD
git status --short
git submodule status --recursive
sha256sum RUNNER CONFIG
```

Every registry row must contain the commit, clean/dirty state, patch hash when
applicable, runner hash, config hash, and raw artifact hash. Do not treat paper
text or a snapshot-only summary as stronger than raw artifacts, manifests, and
the formal registry.

The current formal branch is `codex/strong-accept-evidence`; its dedicated
worktree path is recorded in `evidence/source_state_before_formal_runs.md`.

## 7. Paper and evidence boundaries

- `paper/` is the current modular paper.
- `paper_legacy/` is read-only historical material.
- `PROJECT_STATUS.md` summarizes frozen facts but does not override raw data.
- Snapshot-only reruns marked `conflicting / not paper-authoritative` must not
  enter the Abstract, Introduction, Evaluation, or Conclusion.
- `HVS1`, `hvs_rs_sha256`, and historical `layouts/hvs_*` paths are retained
  solely to reconstruct registered artifacts.
