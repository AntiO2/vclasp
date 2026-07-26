# Runtime Cost Feedback

## Why Runtime Feedback

VClasp must choose among legal byte-span covers before issuing object-store
requests. A deployment-specific offline calibration can become stale when
network contention, connection-pool behavior, object-store software, or request
size changes. It also makes a reusable library depend on a benchmark-specific
setup step.

The production reader therefore treats the caller's cost model as a bootstrap,
not as permanent backend truth. It records actual fetch and decode work after
each successful request window and updates the model owned by the process-wide
VClasp session. Neither the estimator nor the planner receives a backend type
or workload label.

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
per-range dispatch, first-byte, and completion timestamps
realized maximum in-flight requests
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

The registered bootstrap additionally models the factors that AnyBlob's object
store study identifies as first-order: fixed request latency, transferred
bytes, the number of outstanding requests, and concurrency saturation. VClasp
fits request-wave overheads and an aggregate byte rate from held-out probes.
The registered MinIO calibration also tested a largest-range term. Its fitted
coefficient was zero, while the untouched mixed-size cells had 1.81% P50 MAPE;
the deployed Rust projection therefore retains request waves and aggregate
bytes rather than adding an unsupported range-shape coefficient.

The physical candidate set enumerates every legal largest-gap cut unless the
caller supplies an explicit maximum range or gap. It also constructs an
explicit Fixed 16-KiB reference plan. Selection tolerance may prefer fewer
requests among statistically indistinguishable candidates, but the selected
plan must have predicted cost no greater than the Fixed 16-KiB reference. This
is a model-space safety property. Remote noise and unobserved counterfactuals
make a per-request realized-latency guarantee impossible without executing both
plans.

References:

- T. Durner et al., "Exploiting Cloud Object Storage for High-Performance
  Analytics," PVLDB 16(11), 2023.
  <https://www.vldb.org/pvldb/vol16/p2769-durner.pdf>
- AWS, "Performance guidelines for Amazon S3."
  <https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance-guidelines.html>

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

A neural selector is deliberately not used. The legal plans have a small,
structured feature space; online labels require costly counterfactual requests;
backend latency shifts over time; and inference lies on the reader's critical
path. A monotone generalized additive model or small tree ensemble is a future
option only if held-out residuals show stable structure that the wave model
cannot explain. Such a replacement must beat the wave model on video-, object-,
and window-disjoint data, improve plan regret rather than only latency fit, and
retain the explicit Fixed 16-KiB fallback.

## Telemetry

Every result exposes:

```text
planner_candidate_count
predicted_total_ns
fixed16_reference_ranges
fixed16_reference_fetched_bytes
fixed16_reference_predicted_ns
range_dispatch_ns_sum
range_ttfb_ns_sum
range_service_ns_sum
range_max_in_flight
range_timing_samples
runtime_feedback_io_observations
runtime_feedback_decode_observations
runtime_feedback_rejected_observations
runtime_feedback_active
runtime_feedback_io_ape_ppm
runtime_feedback_decode_ape_ppm
runtime_feedback_io_tail_multiplier_ppm
runtime_feedback_decode_tail_multiplier_ppm
```

Concurrent callers contribute to one session-global feedback state and one
joint planning decision. Feedback is therefore associated with the physical
plan that actually executed, rather than reconstructed from per-worker plans.

`range_max_in_flight` is an observation, not an additive latency term. A plan
with fewer ranges than configured slots is expected to report a smaller value.
Only batches with enough independent ranges can diagnose backend saturation;
otherwise lowering the model's effective concurrency would misread a compact
plan as transport serialization.

## Saturation-Aware Model Gate

A 32-batch Uniform diagnostic with eight concurrent clients and eight shared
I/O slots exposed a limitation of the current online feature set. All rows
observed seven or eight in-flight GETs, but the last-observation I/O APE had a
27.5% median (decode APE: 8.4%). Per-executor range count and bytes do not fully
describe queueing caused by other executors sharing the semaphore.

The transport now records the shared state visible when a plan is made:

```text
candidate features:
  range count, total bytes, largest range, size buckets
runtime pressure:
  configured slots, active requests, outstanding requests, queued requests
decode features:
  submitted AUs, decode groups, decoder resets
```

These pressure counters are diagnostics rather than an unvalidated additive
cost term. The intended model remains decomposable:

```text
T_io = waves(ranges, effective concurrency) * L(pressure)
     + total bytes / BW(pressure)
T_decode = open/group cost * decode groups
         + AU cost * submitted AUs
         + reset cost * decoder resets
T_total = T_io + T_decode - measured overlap
```

Calibration should use bounded probes over request sizes and concurrency, then
adapt only low-dimensional residuals online. Effective concurrency may be
updated only from observations containing at least as many independent ranges
as configured slots. A compact Sequential plan with two ranges is evidence of
two available tasks, not evidence that the backend supports only concurrency
two.

The production confidence rule should be conservative: always retain the
Fixed 16-KiB candidate and select a different plan only when its risk-adjusted
prediction is no worse. If sustained prediction error exceeds the registered
threshold, fall back to Fixed 16 KiB and schedule a separate calibration probe;
do not learn counterfactual costs by slowing a user request.

The selected backend calibration also defines the model's request-size support.
Unless the application explicitly supplies a stricter or independently
validated limit, the planner caps every physical range at the largest calibrated
request size. This prevents a low-dimensional bandwidth model fitted over
1 KiB--1 MiB requests from extrapolating to multi-megabyte spans. The explicit
Fixed 16-KiB candidate remains present after applying this cap, and the
model-space selector cannot choose a plan with a larger predicted cost.

The online estimator enters a Fixed 16-KiB fallback only after sustained
prediction error; one noisy remote observation does not switch policy. Recovery
also requires a sustained run of accurate observations. Entering fallback
resets the low-dimensional estimator to its registered bootstrap instead of
retaining a model contaminated by an unsupported or transient regime.

The Fixed 16-KiB rule is a model-space safety property, not a deterministic
latency bound. Object-store contention can reverse two plans whose predicted
costs are close, and executing both plans on every user request would defeat the
optimization. Diagnostic repetitions must therefore report variability and
actual plan regret; they must not reinterpret the prediction guard as measured
dominance.

We do not use a neural network in the online path. Candidate evaluation has a
small, monotone feature space and receives sparse counterfactual labels: only
the selected plan is executed. A neural model would add training and
distribution-shift failure modes without fixing missing queue-pressure labels.
A monotone GAM or shallow tree is justified only if a held-out residual study
shows repeatable structure beyond request waves, bytes, calibrated size support,
decode work, and observed pressure.

A leave-one-run-out MinIO diagnostic over 192 existing batch observations found
that adding plan-time pressure to a ridge model reduced median/P90 I/O APE from
11.1%/27.7% to 10.4%/23.8%. A depth-two histogram tree reached 10.0%/23.2%.
The small incremental gain does not justify a neural selector. Pressure remains
registered telemetry until a backend- and workload-held-out test demonstrates
lower candidate-selection regret, not merely a slightly better latency fit.

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

## Historical Scheduler Diagnostic

An earlier, now-retired multi-worker prototype issued ranges concurrently
inside one call but submitted one executor call per GOP. A bounded MinIO
diagnostic motivated whole-window planning, shared range waves, and
dependency-liveness-based cursor retention:

- Sequential reached 1,696 samples/s, up from the earlier 1,199 samples/s path.
- Same-video reached 3,263 samples/s, up from 2,218 samples/s before the
  liveness fixes.
- Uniform reached 564 samples/s versus 574 samples/s before that prototype change,
  a 1.8% one-run difference within observed MinIO variability.
- Uniform batches produced roughly 30 ranges and observed six to eight
  concurrent GETs. Sequential batches commonly produced only two or three
  ranges, so they could not fill eight I/O slots.

An additional 256-sample Uniform provenance smoke verified that every batch had
a non-zero Fixed 16-KiB reference, the selected predicted cost was never larger
than that reference (ratios 0.936--1.000), and realized GET concurrency was
six to eight. These are implementation diagnostics, not replacements for
formal rows. They separate transport concurrency from plan geometry; they do
not establish a new paper result and do not describe the current architecture.
The production `VClaspSession` has no worker routing or per-worker planner:
concurrent batch and window calls enter one process-wide admission and planning
path.

A subsequent 1,024-sample diagnostic exposed why calibration support must be
enforced. With no maximum range, the model selected 6--9 MiB spans despite an
offline calibration whose largest request was 1 MiB. The run fetched 33.7 MB
and achieved 294 samples/s even though GET concurrency reached eight. Applying
the 1-MiB calibration-derived cap reduced fetched data to 14.0 MB and returned
the run to the prior normal throughput range. Matched high-locality checks
reached 3,389 steady-state samples/s for Same-video and 1,706 samples/s for
Sequential, so the support guard did not disable the resident-cursor path.
