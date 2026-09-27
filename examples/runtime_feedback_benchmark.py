#!/usr/bin/env python3
"""Bounded diagnostic for VClasp's runtime cost feedback.

This runner keeps requests, payload, executor resources, and bootstrap costs
identical. The only changed variable is whether post-execution observations may
update the cost model. It is intentionally a diagnostic, not a formal paper
benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import vclasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=Path, required=True)
    parser.add_argument("--object-key", required=True)
    parser.add_argument("--endpoint", default=os.getenv("VCLASP_S3_ENDPOINT"))
    parser.add_argument("--bucket", default=os.getenv("VCLASP_S3_BUCKET"))
    parser.add_argument("--access-key", default=os.getenv("VCLASP_S3_ACCESS_KEY"))
    parser.add_argument("--secret-key", default=os.getenv("VCLASP_S3_SECRET_KEY"))
    parser.add_argument("--videos", type=int, default=1_374)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--video-id-template", default="{}")
    parser.add_argument("--warmup-windows", type=int, default=32)
    parser.add_argument("--measured-windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20_260_723)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--lookahead-candidates",
        default="1",
        help="comma-separated bounded visibility candidates; must include 1",
    )
    parser.add_argument(
        "--first-batch-slo-ms",
        type=float,
        default=100.0,
        help="maximize predicted throughput subject to this first-batch latency SLO",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("endpoint", "bucket", "access_key", "secret_key"):
        if not getattr(args, name):
            parser.error(
                f"--{name.replace('_', '-')} or its environment variable is required"
            )
    args.lookahead_candidates = sorted(
        {int(value) for value in args.lookahead_candidates.split(",")}
    )
    if not args.lookahead_candidates or args.lookahead_candidates[0] != 1:
        parser.error("--lookahead-candidates must include 1")
    if any(value <= 0 for value in args.lookahead_candidates):
        parser.error("--lookahead-candidates must be positive")
    if args.first_batch_slo_ms <= 0:
        parser.error("--first-batch-slo-ms must be positive")
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def make_trace(
    seed: int, windows: int, videos: int, frames: int, video_id_template: str
) -> list[list[tuple[int, str, int]]]:
    rng = random.Random(seed)
    batch_sizes = (8, 16, 32, 48)
    result: list[list[tuple[int, str, int]]] = []
    sample_id = 0
    for window_index in range(windows):
        batch_size = batch_sizes[window_index % len(batch_sizes)]
        # Alternate dispersed and physically adjacent video groups so the same
        # trace contains both one-range-per-record and legal coalescing choices.
        if window_index % 2 == 0:
            video_ids = rng.sample(range(videos), batch_size)
        else:
            start = rng.randrange(videos - batch_size + 1)
            video_ids = list(range(start, start + batch_size))
            rng.shuffle(video_ids)
        batch = []
        for video_id in video_ids:
            batch.append(
                (
                    sample_id,
                    video_id_template.format(video_id),
                    rng.randrange(frames),
                )
            )
            sample_id += 1
        result.append(batch)
    return result


def trace_hash(trace: list[list[tuple[int, str, int]]]) -> str:
    payload = json.dumps(trace, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def cost_model(feedback_enabled: bool) -> dict[str, float]:
    return {
        "request_latency_ns": 1_000_000.0,
        "bandwidth_bytes_per_ns": 0.25,
        "io_concurrency": 8.0,
        "selection_tolerance_ns": 50_000.0,
        "decode_fixed_ns": 80_000.0,
        "decode_access_unit_ns": 35_000.0,
        "fetch_decode_overlap": 0.0,
        "runtime_feedback_enabled": float(feedback_enabled),
        "runtime_feedback_min_observations": 16.0,
        "runtime_feedback_activation_stable_observations": 4.0,
    }


def make_reader(args: argparse.Namespace, method: str):
    feedback_enabled = method == "online"
    max_merge_gap_bytes = 16 * 1024 if method == "capped16" else None
    return vclasp.S3VClaspExecutor(
        str(args.chunk),
        args.object_key,
        args.endpoint,
        args.bucket,
        args.access_key,
        args.secret_key,
        cost_model(feedback_enabled),
        [],
        max_concurrency=8,
        max_merge_gap_bytes=max_merge_gap_bytes,
        decoder_threads=1,
        incremental_decode_slots=8,
        resident_encoded_bytes=0,
        resident_read_ahead_bytes=0,
        resident_cursor_capacity=0,
    )


def execute_trace(
    reader: Any,
    trace: list[list[tuple[int, str, int]]],
    measured_start: int,
    lookahead_candidates: list[int],
    first_batch_slo_ms: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    window_index = 0
    max_lookahead = max(lookahead_candidates)
    while window_index < len(trace):
        # Do not let one execution group straddle the warmup/measurement
        # boundary. The logical trace remains identical across methods.
        boundary = measured_start if window_index < measured_start else len(trace)
        available = min(max_lookahead, boundary - window_index)
        visible = trace[window_index : window_index + available]
        candidates = [value for value in lookahead_candidates if value <= available]
        selected, selected_under_slo, alternatives = reader.choose_lookahead(
            visible,
            candidates,
            first_batch_slo_ms,
        )
        planned = visible[:selected]
        started = time.perf_counter_ns()
        output, batch_ready_ns, ordered_delivery_ns, stats, mode, predicted_ns = (
            reader.execute_window(planned)
        )
        elapsed_ns = time.perf_counter_ns() - started
        expected_ids = [[sample[0] for sample in batch] for batch in planned]
        actual_ids = [[sample[0] for sample in batch] for batch in output]
        if actual_ids != expected_ids:
            raise RuntimeError(f"output order mismatch at window {window_index}")
        if window_index >= measured_start:
            rows.append(
                {
                    "window": window_index - measured_start,
                    "batches": selected,
                    "samples": sum(len(batch) for batch in planned),
                    "selected_lookahead": selected,
                    "selected_under_slo": bool(selected_under_slo),
                    "lookahead_alternatives": alternatives,
                    "elapsed_ns": elapsed_ns,
                    "first_batch_ready_ns": batch_ready_ns[0],
                    "ordered_delivery_ns": ordered_delivery_ns,
                    "predicted_ns": predicted_ns,
                    "mode": mode,
                    "physical_ranges": stats["physical_ranges"],
                    "fetched_bytes": stats["fetched_bytes"],
                    "fetch_ns": stats["fetch_ns"],
                    "decode_ns": stats["decode_ns"],
                    "submitted_access_units": stats["submitted_access_units"],
                    "feedback_active": bool(stats["runtime_feedback_active"]),
                    "feedback_io_observations": stats[
                        "runtime_feedback_io_observations"
                    ],
                    "feedback_io_ape": stats["runtime_feedback_io_ape_ppm"] / 1e6,
                    "feedback_decode_ape": stats["runtime_feedback_decode_ape_ppm"]
                    / 1e6,
                    "feedback_io_tail_multiplier": stats[
                        "runtime_feedback_io_tail_multiplier_ppm"
                    ]
                    / 1e6,
                    "feedback_decode_tail_multiplier": stats[
                        "runtime_feedback_decode_tail_multiplier_ppm"
                    ]
                    / 1e6,
                }
            )
        window_index += selected
    elapsed = [row["elapsed_ns"] / 1e6 for row in rows]
    first_batch_ready = [row["first_batch_ready_ns"] / 1e6 for row in rows]
    ordered_delivery = [
        value / 1e6 for row in rows for value in row["ordered_delivery_ns"]
    ]
    total_samples = sum(row["samples"] for row in rows)
    total_seconds = sum(row["elapsed_ns"] for row in rows) / 1e9
    return {
        "samples_per_second": total_samples / total_seconds,
        "latency_ms_p50": percentile(elapsed, 0.50),
        "latency_ms_p95": percentile(elapsed, 0.95),
        "first_batch_latency_ms_p50": percentile(first_batch_ready, 0.50),
        "first_batch_latency_ms_p95": percentile(first_batch_ready, 0.95),
        "ordered_delivery_latency_ms_p95": percentile(ordered_delivery, 0.95),
        "mean_selected_lookahead": statistics.fmean(
            row["selected_lookahead"] for row in rows
        ),
        "slo_feasible_fraction": statistics.fmean(
            row["selected_under_slo"] for row in rows
        ),
        "gets_per_sample": sum(row["physical_ranges"] for row in rows) / total_samples,
        "bytes_per_sample": sum(row["fetched_bytes"] for row in rows) / total_samples,
        "active_windows": sum(row["feedback_active"] for row in rows),
        "final_io_observations": rows[-1]["feedback_io_observations"],
        "final_io_ape": rows[-1]["feedback_io_ape"],
        "final_decode_ape": rows[-1]["feedback_decode_ape"],
        "final_io_tail_multiplier": rows[-1]["feedback_io_tail_multiplier"],
        "final_decode_tail_multiplier": rows[-1]["feedback_decode_tail_multiplier"],
        "rows": rows,
    }


def summarize(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "samples_per_second",
        "latency_ms_p50",
        "latency_ms_p95",
        "first_batch_latency_ms_p50",
        "first_batch_latency_ms_p95",
        "ordered_delivery_latency_ms_p95",
        "mean_selected_lookahead",
        "slo_feasible_fraction",
        "gets_per_sample",
        "bytes_per_sample",
        "active_windows",
        "final_io_ape",
        "final_decode_ape",
        "final_io_tail_multiplier",
        "final_decode_tail_multiplier",
    )
    return {
        key: {
            "mean": statistics.fmean(run[key] for run in repetitions),
            "stdev": statistics.stdev(run[key] for run in repetitions)
            if len(repetitions) > 1
            else 0.0,
        }
        for key in keys
    }


def main() -> None:
    args = parse_args()
    total_windows = args.warmup_windows + args.measured_windows
    trace = make_trace(
        args.seed,
        total_windows,
        args.videos,
        args.frames,
        args.video_id_template,
    )
    runs: dict[str, list[dict[str, Any]]] = {
        "static": [],
        "online": [],
        "capped16": [],
    }
    for repetition in range(args.repetitions):
        order = (
            ("static", "online", "capped16")
            if repetition % 2 == 0
            else ("capped16", "online", "static")
        )
        for method in order:
            reader = make_reader(args, method)
            runs[method].append(
                execute_trace(
                    reader,
                    trace,
                    args.warmup_windows,
                    args.lookahead_candidates,
                    args.first_batch_slo_ms,
                )
            )
    result = {
        "diagnostic": "runtime-feedback-static-vs-online",
        "trace_sha256": trace_hash(trace),
        "chunk_sha256": hashlib.sha256(args.chunk.read_bytes()).hexdigest(),
        "object_key": args.object_key,
        "endpoint": args.endpoint,
        "warmup_windows": args.warmup_windows,
        "measured_windows": args.measured_windows,
        "repetitions": args.repetitions,
        "lookahead_objective": {
            "candidates": args.lookahead_candidates,
            "first_batch_slo_ms": args.first_batch_slo_ms,
            "policy": "maximize predicted samples/s subject to first-batch SLO",
        },
        "resources": {
            "io_slots": 8,
            "decode_slots": 8,
            "resident_encoded_bytes": 0,
            "resident_read_ahead_bytes": 0,
            "resident_cursor_capacity": 0,
        },
        "summary": {method: summarize(values) for method, values in runs.items()},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
