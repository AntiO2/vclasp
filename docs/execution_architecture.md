# VClasp Execution Architecture

## One Request Interface

The production reader accepts an already sampled request window:

```text
read_window([
  [(sample_id, video_id, frame_idx), ...],
  ...
])
```

The caller does not label the window as Uniform, Zipf, Clip, Same-video, or
Sequential. Those names belong to benchmark trace generation only. The reader
derives locality and dependency geometry from the request IDs and embedded
chunk index.

```text
logical request window
  -> target lookup
  -> sufficient closure union
  -> request/layout geometry
  -> production action selection
  -> Range GET
  -> filter transfer-only gaps
  -> codec-order decode
  -> restore duplicates and request order
```

The production implementation is
`hierarchical_scheduler::HierarchicalBatchExecutor`. Its canonical operation is
`execute_window`; `execute` is exactly the one-batch (`L=1`)
convenience form. Local, S3, AIStore, and the partitioned S3 executor pool expose
the same window semantics. In particular, monotonic requests may reuse a
compatible bounded decoder cursor, but the caller never selects a separate
sequential reader.

## Production Actions

`HierarchicalBatchExecutor` may select a sparse closure plan, contiguous region
plan, or resident decoder cursor from observable request and physical-layout
facts. Cursor admission has two independent gates: dependency liveness proves
that reuse exists, then the same current request/byte/decode model compares the
monotonic cursor suffix against ordinary adaptive execution over the visible
requests. Reuse alone does not force cursor execution.
The selection code must not receive a benchmark workload name.

The model supplied by the application is a bootstrap. Successful fetch/decode
measurements update a shared robust online estimator after an accuracy gate;
the estimator has no backend or workload-name input. See
[`runtime_cost_feedback.md`](runtime_cost_feedback.md).

Forced actions exist only inside the Rust mechanism-control boundary. The
default Python extension cannot select them. Registered ablations use the same
payload, index, transport, and decoder as the production reader.

## Bounded Resident State

The production executor has two independent compressed-state layers:

1. The encoded-AU cache avoids repeated object reads. A hit still belongs to
   the sufficient closure and may need to be submitted to a decoder.
2. The live decoder-cursor cache retains an FFmpeg decoder and its DPB for a
   particular `(video, representation, GOP/anchor group)`. A compatible
   monotonic request submits only newly required AUs.

The catalog stores encoder/bitstream-grounded sufficient closures, not inferred
direct edges. At runtime, VClasp derives remaining consumers from the closure
union in the visible request window. A cursor is admitted only when this window
proves reuse; states with a future consumer are pinned, while states whose last
visible consumer completed become probationary. Capacity eviction uses LRU
among probationary states and farthest-next-use only when every state is pinned.
A leaf target does not itself justify a cursor. This policy uses neither a
workload label nor a target-count threshold, and it does not claim that FFmpeg
exposes individual DPB entries; reference marking remains a codec property.

The compressed-byte budget is shared by encoded AU bytes and cursor read-ahead,
and the live cursor count is separately capped by the configured decoder-slot
budget. These are different resources: encoded bytes are byte-accounted, while
FFmpeg's private DPB allocation is not. Cache misses, backward requests,
cost-ineffective suffixes, evicted state, and stream-key mismatches fall back to
one global sufficient-closure plan. Formal memory studies must therefore report
measured process RSS in addition to the logical encoded-byte and cursor-count
limits.

VClasp does not serialize FFmpeg decoder checkpoints. Public libavcodec state
contains private pointer-rich H.264 structures and has no stable export/import
API. An isolated FFmpeg 6.1.1 prototype demonstrates version-pinned in-memory
context cloning through FFmpeg's private thread-context copy implementation;
see `experiments/ffmpeg_checkpoint`. The clone is process-local, requires an
already-open single-thread decoder, and is not part of the production executor.
A portable checkpoint would instead require a stateless decoder whose client
explicitly owns the DPB and all associated codec state.

## Historical Controls

The following implementations are retained to reproduce registered ablations:

| Control | Rust implementation | Paper role |
| --- | --- | --- |
| Prefix | `controls::closed_record` | Scanner-style/stream control |
| Pair | `controls::closed_record` | materialization control |
| Normalized | `controls::normalized` | historical representation control |
| Packed GOP8 | `fragment_scheduler` | physical-layout control |

They are not aliases for VClasp and are not part of the default extension. The
formal artifact builds explicit controls separately and rejects their method
IDs unless mechanism controls are enabled.

Control variants are explicit as well. Prefix streaming requires
`--prefix-control-streaming`; a trace named `sequential` cannot enable it.

## Benchmark Boundary

Every adapter implements `execute_window(batches)` and returns
ordered batches plus a common metrics record. `execute(batch)` delegates to an
`L=1` window. A trace generator may create different access patterns, but trace
names cannot select a different VClasp implementation. The constant
`VCLASP_PRIMARY_METHOD` in `scripts/method_registry.py` is the sole mapping from
the paper method name to executable code.

New paper-facing runners must import this registry instead of spelling a
VClasp method ID directly.
