# VClasp Representation Policy Sweep

## Question

Can VClasp spend additional retained storage to reduce predictive-codec
dependency work and improve object-store application throughput?

This experiment varies representation construction inside the same VClasp
format and executor. It does not compare a new reader with an old reader.

## Policies

All policies use the same source frames, resolution, frame rate, CRF, preset,
chunk/index schema, closure planner, Range GET implementation, libavcodec
decoder, request traces, and resource limits.

- `closed_gop`: the current production FFmpeg/libx264 closed GOP with I/P/B
  frames. Closures are derived from the encoded packet graph.
- `chained_p`: a controlled no-B stream with ordinary `I -> P1 -> P2 -> ...`
  references.
- `shared_anchor`: a controlled no-B stream in which every P target references
  only the GOP's I anchor. The anchor is stored once; a target closure contains
  `anchor + target`.

`chained_p` and `shared_anchor` use the same native encoder path and differ only
in reference invalidation. This control separates dependency-graph effects from
the absence of B frames.

## Independent Variables

- dependency policy: `closed_gop`, `chained_p`, `shared_anchor`;
- GOP size: 4, 8, 16 frames;
- execution visibility: L1 for the primary gate; L8 is excluded because the
  existing finite embedding job establishes L1 as its application operating
  point.

GOP sizes above 16 are excluded from the first Shared-Anchor gate because stock
x264 bounds the retained short-term reference set. They remain legal for
`closed_gop`.

## Gate Corpus and Workload

Gate:

- corpus: existing 200-video, 64-frame UCF prefix;
- traces: matched Uniform, Epoch, Zipf, Same-video, Clip-8, and Sequential where
  existing traces resolve against every generated representation;
- primary application: frame-embedding replay without model retraining;
- one development seed for admission, followed by three matched seeds only for
  non-dominated candidates.

Expansion:

- UCF101 10-class, 2,048-frame embedding trace;
- three matched seeds;
- only the current operating point and gate candidates that add a
  non-dominated storage/performance point.

## Metrics

- retained payload, index, and total chunk bytes;
- source-referenced PSNR/SSIM distribution when representations do not decode
  byte-identically;
- complete-job and steady-state throughput;
- first-batch, batch P50/P95;
- GETs, request waves, fetched/useful/overfetch bytes;
- submitted and decoded AUs;
- CPU-seconds, effective cores, process peak RSS, and GPU utilization;
- closure mean, median, P95, and maximum;
- ingestion wall time and CPU-seconds.

## Correctness Gate

Before performance measurement:

1. every logical target is present in the embedded index;
2. every closure is in codec decode order and remains within one GOP;
3. `shared_anchor` non-anchor targets have exactly two logical dependencies;
4. target-only closure decode succeeds without intermediate P frames;
5. decoded outputs match the policy's full-stream reference decode;
6. IDs, duplicates, batch boundaries, and output order match across policies;
7. planner and executor contain no dependency-policy or workload-name branch.

## Decision Rules

A candidate advances only if it:

- passes all correctness checks;
- is non-dominated in retained storage versus at least one primary endpoint
  (low-locality geometric-mean throughput or complete embedding-job
  throughput); and
- does not regress any locality boundary by more than 20% without an explicit
  selector-supported crossover.

If `shared_anchor` is dominated by `chained_p`, the dependency policy is a
negative result. If a smaller GOP alone reaches the same point, the claim is GOP
tuning rather than a new dependency representation. If Shared-Anchor creates a
new point, it becomes an optional high-storage representation selected through
the existing representation-independent planner.

## Artifacts

- code worktree:
  `/home/antio2/experiments/vclasp/worktrees/representation-policy-sweep-310bc88`
- layouts:
  `/home/antio2/experiments/vclasp/layouts/representation-policy-sweep`
- results:
  `/home/antio2/experiments/vclasp/results/representation-policy-sweep`

Experiment Gate: PASS for bounded correctness and 200-video admission smoke.
Full UCF101 three-seed execution remains conditional on the admission result.
