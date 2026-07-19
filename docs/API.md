# Public API guide

VClasp Core exposes three layers.

## Rust planning API

`vclasp::planner` provides `RecordRange`, `RangePlan`,
`plan_byte_ranges`, `plan_group_spans`, and the bounded byte cache. These APIs
accept physical record extents and return ordered ranges plus record-relative
offsets. They do not perform codec dependency resolution.

`vclasp::hierarchical_layout` and `vclasp::hierarchical_scheduler` expose the
codec-closure index and batch executor used by the paper system. A caller
submits logical targets; the executor resolves sufficient closures, deduplicates
dependencies, plans ranges, decodes in codec order, and restores logical order.

## Object-store transport API

`vclasp-object-store` provides:

- Rust: `S3ObjectStoreClient::fetch_object_ranges`;
- C: `object-store-transport/include/vclasp_object_store.h`;
- Python: `PyS3ObjectStoreReader` through the main extension.

All interfaces preserve input order, reject short reads, and share one bounded
connection pool. Layout planning and decoding remain outside the transport
crate.

## Python API

The PyO3 extension exposes chunk readers, local and S3 hierarchical executors,
the decoder pool, and range readers. Python supplies configuration and logical
requests; dependency resolution, planning, retrieval, and decoding stay in
Rust. The authoritative signatures are registered in `src/lib.rs`.

The on-disk FlatBuffer identifier remains `HVS1` for compatibility with
registered research artifacts. This historical identifier is not a product
name.
