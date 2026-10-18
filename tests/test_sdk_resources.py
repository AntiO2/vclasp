"""SDK resource/correctness regressions using generated temporary media."""

import shutil
import subprocess

import pytest
import vclasp


@pytest.fixture
def media(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg CLI is required for generated fixtures")
    source = tmp_path / "source.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=25",
            "-frames:v",
            "64",
            "-threads",
            "1",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    chunk = tmp_path / "fixture.vclasp"
    other_source = tmp_path / "other.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            "hflip",
            "-threads",
            "1",
            "-c:v",
            "ffv1",
            str(other_source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    baseline = tmp_path / "mp4"
    stats = vclasp.build_vclasp_chunk_profiled(
        videos=[("a", "fixture", str(source)), ("b", "fixture", str(other_source))],
        output_path=str(chunk),
        baseline_mp4_dir=str(baseline),
        gop_size=16,
        max_frames=64,
        width=64,
        height=64,
        fps=25,
        crf=23,
        preset="veryfast",
        workers=2,
    )
    assert stats.records == 128
    assert stats.prepare_seconds > 0
    decoded = {}
    for index, video in enumerate(("a", "b")):
        decoded[video] = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(baseline / f"{index:06}.mp4"),
                "-threads",
                "1",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        assert len(decoded[video]) == 64 * 64 * 64 * 3
    assert decoded["a"] != decoded["b"]
    return chunk, decoded


def reader(chunk, **options):
    model = {
        "request_latency_ns": 1_000_000.0,
        "bandwidth_bytes_per_ns": 1.0,
        "io_concurrency": 2.0,
        "selection_tolerance_ns": 0.0,
        "decode_fixed_ns": 100_000.0,
        "decode_access_unit_ns": 100_000.0,
        "fetch_decode_overlap": 0.0,
        "runtime_feedback_enabled": 0.0,
    }
    return vclasp.VClaspSession.local(
        str(chunk),
        model,
        [],
        max_callers=2,
        decoder_threads=1,
        global_decode_concurrency=2,
        **options,
    )


def test_readers_match_external_ffmpeg_after_parallel_ingestion(media):
    chunk, reference = media
    stateless = reader(chunk, resident_encoded_bytes=0, resident_cursor_capacity=0)
    resident = reader(chunk, resident_encoded_bytes=8 << 20, resident_cursor_capacity=2)
    requests = [
        [(0, "a", 0), (1, "a", 1), (2, "a", 2)],
        [(3, "a", 3), (4, "a", 4), (5, "a", 5)],
        [(6, "a", 15), (7, "a", 7), (8, "a", 7)],
        [(9, "a", 63), (10, "b", 31), (11, "a", 16)],
        [(12, "a", 0), (13, "a", 63)],
    ]
    for batch in requests:
        expected = [
            (sample_id, reference[video][frame * 12288 : (frame + 1) * 12288], 64, 64)
            for sample_id, video, frame in batch
        ]
        assert stateless.execute(batch)[0] == expected
        assert resident.execute(batch)[0] == expected
    stateless.synchronize()
    resident.synchronize()
    for session in (stateless, resident):
        metrics = session.metrics_snapshot()
        assert metrics["pending_calls"] == 0
        assert metrics["transient_buffer_reserved_bytes"] == 0
        assert (
            0
            < metrics["transient_buffer_peak_bytes"]
            <= metrics["transient_buffer_capacity_bytes"]
        )
    assert vclasp.VClaspChunk(str(chunk)).record_count() == 128


def test_small_transient_budget_fails_without_stranding_session(media):
    chunk, _ = media
    limited = reader(chunk, max_inflight_bytes=1, resident_cursor_capacity=0)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="transient bytes"):
            limited.execute([(0, "a", 0)])
    limited.synchronize()


def test_malformed_header_is_rejected_before_catalog_access(media, tmp_path):
    chunk, _ = media
    data = bytearray(chunk.read_bytes())
    data[4:8] = b"\xff" * 4
    malformed = tmp_path / "malformed.vclasp"
    malformed.write_bytes(data)
    with pytest.raises((OSError, ValueError)):
        vclasp.VClaspChunk(str(malformed))


@pytest.mark.parametrize("resident_bytes", [0, 8 << 20])
def test_stage_profiling_preserves_output_and_reports_codec_work(media, resident_bytes):
    chunk, reference = media
    session = reader(
        chunk, resident_encoded_bytes=resident_bytes, resident_cursor_capacity=2
    )
    session.set_stage_profiling(True)
    has_physical_trace = hasattr(session, "start_execution_trace")
    if has_physical_trace:
        session.start_execution_trace(10000, 10000)
    batches = [[(0, "a", 1), (1, "a", 1)], [(2, "a", 3)], [(3, "a", 5)]]
    outputs, _, _, stats, _, _ = session.execute_window(batches)
    for batch, frames in zip(batches, outputs, strict=True):
        assert frames == [
            (sample, reference[video][frame * 12288 : (frame + 1) * 12288], 64, 64)
            for sample, video, frame in batch
        ]
    session.synchronize()
    metrics = session.metrics_snapshot()
    state = session.resident_state_snapshot()
    assert state and all(
        row["scope"] == "last_completed_execution_per_lane" for row in state
    )
    assert all(row["resident_payload_bytes"] <= row["budget_bytes"] for row in state)
    assert all(row["live_cursors"] <= row["cursor_capacity"] for row in state)
    assert metrics["codec_profile_jobs"] > 0
    assert metrics["codec_accepted_packets"] == metrics["submitted_access_units"]
    assert metrics["codec_received_frames"] >= 3
    assert metrics["codec_rgb_copy_bytes"] >= 3 * 12288
    assert stats["python_output_bytes"] == 4 * 12288
    jobs, dropped = session.take_decode_job_trace()
    assert jobs and dropped == 0
    assert {sample for job in jobs for sample in job["consumer_sample_ids"]} == {
        0,
        1,
        2,
        3,
    }
    assert all(job["decoder_context_address"] is not None for job in jobs)
    assert all(
        job["execution_path"] in {"stateless_decoder", "resident_cursor"}
        for job in jobs
    )
    assert (
        sum(len(job["record_ids"]) for job in jobs) == metrics["submitted_access_units"]
    )
    assert all(
        0
        < job["submitted_ns"]
        <= job["worker_started_ns"]
        <= job["resource_acquired_ns"]
        <= job["completed_ns"]
        for job in jobs
    )
    if has_physical_trace:
        ranges, gets, dropped_ranges, decodes, aus, dropped_decodes = (
            session.take_execution_trace()
        )
        assert gets == metrics["physical_ranges"]
        assert aus == metrics["submitted_access_units"]
        assert len(decodes) == aus and dropped_ranges == dropped_decodes == 0
        assert sum(event["physical_requests"] for event in ranges) == gets
    session.set_stage_profiling(False)
    before = session.metrics_snapshot()["codec_profile_jobs"]
    session.execute([(4, "b", 63)])
    session.synchronize()
    assert session.metrics_snapshot()["codec_profile_jobs"] == before
    assert session.take_decode_job_trace() == ([], 0)
