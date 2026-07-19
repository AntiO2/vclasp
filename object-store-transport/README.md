# VClasp object-store transport

This standalone Rust crate is the common S3-compatible data plane used by
VClasp and the object-store ports of VSS and VStore. It has no Python, PyO3, or
video-decoder dependency.

```bash
cargo test --manifest-path vclasp-rs/object-store-transport/Cargo.toml
cargo build --release \
  --manifest-path vclasp-rs/object-store-transport/Cargo.toml
```

Interfaces:

- Rust: `S3ObjectStoreClient::fetch_object_ranges`.
- C/C++: `include/vclasp_object_store.h`; returned buffers must be released
  with `vclasp_buffers_free` and errors with `vclasp_error_free`.
- Python harness: `vclasp.S3ObjectStoreReader`, a single-call batch adapter
  over the same Rust client.

The transport preserves input order, validates every `(key, offset, length)`,
uses one persistent connection pool, bounds concurrent Range GETs, and rejects
short reads. Layout planning, dependency resolution, and decoding are outside
this crate.

`scripts/object_store_transport_parity_smoke.py` is the required gate before a
baseline workload run. `boto3` appears there only as an untimed byte oracle; it
must not be used in timed VSS, VStore, or VClasp data paths.
