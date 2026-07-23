# Runtime Cost Feedback

## Why Runtime Feedback

VClasp must choose among legal byte-span covers before issuing object-store
requests. A deployment-specific offline calibration can become stale when
network contention, connection-pool behavior, object-store software, or request
size changes. It also makes a reusable library depend on a benchmark-specific
setup step.

The production reader therefore treats the caller's cost model as a bootstrap,
not as permanent backend truth. It records actual fetch and decode work after
each successful request window and updates a small robust model shared by all
workers in an executor pool. Neither the estimator nor the planner receives a
backend type or workload label.

This design follows the feedback principle of DB2's Learning Optimizer (LEO):
execution observations improve later estimates while preserving the existing
optimizer and legal plan space. It is intentionally less invasive than Eddies,
SkinnerDB, or a learned plan selector: VClasp does not reorder logical samples,
explore a deliberately slower candidate, or replace deterministic span
enumeration.

References:

- M. Stillger et al., "LEO - DB2's LEarning Optimizer," VLDB 2001.
  <https://www.vldb.org/conf/2001/P019.pdf>
- R. Avnur and J. Hellerstein, "Eddies: Continuously Adaptive Query
  Processing," SIGMOD 2000.
  <https://db.cs.berkeley.edu/papers/sigmod00-eddy.pdf>
- I. Trummer et al., "SkinnerDB: Regret-Bounded Query Evaluation via
  Reinforcement Learning," SIGMOD 2019.
  <https://arxiv.org/abs/1901.05152>
- R. Marcus et al., "Bao: Making Learned Query Optimization Practical,"
  SIGMOD 2021. <https://people.csail.mit.edu/tatbul/publications/bao_sigmod21.pdf>

## Observation Model

For each successful window, the reader observes:

```text
physical range count
fetched bytes
fetch time
submitted access units
decode time
```

The I/O estimator has three non-negative terms:

```text
fetch time =
    request-wave count * wave fixed cost
  + in-wave extra requests * incremental request cost
  + fetched bytes * byte cost
```

The decode estimator uses a fixed term plus a per-access-unit term. Features
are normalized before a projected normalized-LMS update. A Huber-clipped
residual, bounded per-observation parameter movement, and parameter limits
prevent one remote-latency spike from destabilizing the planner.

Cache-only observations with no physical I/O do not update the I/O model.
Failed executions never update either model.

## Throughput And Latency Objective

Bounded visibility improves throughput by sharing dependencies and issuing more
I/O concurrently, but it can delay the first batch in a visible window. VClasp
therefore does not maximize throughput without a latency constraint. For the
caller-provided candidate horizons, the selector solves:

```text
maximize    expected useful samples / second
subject to  risk-adjusted first-batch latency <= caller SLO
```

The expected runtime model ranks feasible candidates. A separate rolling P95
model, built from observed actual/predicted latency ratios, is used only for SLO
admission. Bootstrap convergence residuals are discarded when online feedback
activates; until enough post-activation tail observations exist, the selector
uses a conservative no-lookahead fallback. This keeps backend and workload
labels out of the execution path.

The SLO applies to first-batch readiness, not to completion of every batch in a
lookahead window. Larger horizons can therefore improve aggregate throughput
while increasing later ordered-delivery latency. Applications that require a
bound on every delivered batch should use `L=1` or provide a tighter visibility
limit; the current selector must not be described as enforcing an
ordered-delivery P95 SLO.

## Activation And Fallback

Online parameters are not used immediately. The executor continues using the
bootstrap model until both estimators have:

1. at least 16 valid observations by default; and
2. four consecutive predictions within the configured error threshold.

After passing this gate, activation is latched. Later observations continue to
adapt the parameters, but a single noisy request cannot switch execution back
and forth between the learned and bootstrap models. Disabling
`runtime_feedback_enabled` produces the static-model control.

The feedback path is passive: it learns only from the selected plan. It does
not directly measure unselected candidates and therefore does not claim
oracle-optimal selection or regret guarantees.

## Telemetry

Every result exposes:

```text
runtime_feedback_io_observations
runtime_feedback_decode_observations
runtime_feedback_rejected_observations
runtime_feedback_active
runtime_feedback_io_ape_ppm
runtime_feedback_decode_ape_ppm
runtime_feedback_io_tail_multiplier_ppm
runtime_feedback_decode_tail_multiplier_ppm
```

Executor-pool workers share one feedback state, so observations are not lost
when requests are partitioned across persistent workers.

## Bounded Diagnostic

`examples/runtime_feedback_benchmark.py` compares the same payload, generated
trace, decoder, resource limits, and bootstrap under three modes:

- `static`: unbounded candidate space with a fixed bootstrap;
- `online`: the same candidate space with runtime feedback;
- `capped16`: a strong 16-KiB candidate-gap control.

The diagnostic alternates dispersed and physically adjacent video groups to
exercise different range/byte trade-offs. It is not a paper-formal workload.

On short three-repetition runs, online feedback recovered from the generic
bootstrap and approached the strong capped control:

| Backend | Static | Online | Capped 16 KiB | Online / capped |
| --- | ---: | ---: | ---: | ---: |
| Three-node MinIO | 278.4 | 461.1 | 469.3 | 0.983x |
| AIStore S3 endpoint | 303.6 | 632.9 | 643.4 | 0.984x |

Values are useful samples/s. The online model used the same initial constants
and no backend label over the same generated Long-UCF200-prefix trace in both
runs. These diagnostics support a narrow engineering conclusion: runtime
feedback can remove the mandatory offline backend-calibration step and recover
within 1.6--1.7% of a strong fixed policy's throughput on the tested
deployments. They do not show that online feedback dominates a well-chosen
fixed policy.

An additional single-repetition MinIO smoke used candidate horizons
`L={1,2,4,8}` and a 100-ms first-batch SLO after 128 warmup windows:

| Policy | samples/s | first-batch P50 | first-batch P95 | ordered-delivery P95 | mean L |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static bootstrap | 332.0 | 304.4 ms | 690.6 ms | 781.4 ms | 8.00 |
| Tail-aware online | 577.8 | 48.4 ms | 95.2 ms | 325.8 ms | 3.56 |
| Fixed 16 KiB, L8 | 647.7 | 43.0 ms | 47.6 ms | 312.8 ms | 8.00 |

The online selector satisfies the measured first-batch P95 target in this
bounded smoke and improves throughput by 1.74x over the stale bootstrap. It is
10.8% below the fixed control and does not dominate ordered-delivery latency.
The result is diagnostic, uses one repetition, and is not a paper-authoritative
experiment.
