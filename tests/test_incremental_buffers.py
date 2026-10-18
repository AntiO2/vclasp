"""Public-reader regression using generated media and temporary local storage."""

import shutil
import subprocess
from threading import Thread

import pytest
import vclasp


@pytest.mark.parametrize("max_callers", [1, 4])
def test_incremental_buffer_lifetime_preserves_batches_and_duplicates(
    tmp_path, max_callers
):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg CLI is required to generate the temporary fixture")
    source = tmp_path / "source.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
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
    vclasp.build_vclasp_chunk(
        videos=[("video", "fixture", str(source))],
        output_path=str(chunk),
        gop_size=16,
        max_frames=64,
        width=64,
        height=64,
        fps=25,
        crf=23,
        preset="veryfast",
        workers=1,
        dependency_policy="hierarchical_b",
        max_b_frames=7,
        b_pyramid="strict",
        b_adapt=0,
        rc_lookahead=0,
        reference_frames=1,
    )
    model = {
        "request_latency_ns": 0.0,
        "bandwidth_bytes_per_ns": 1.0,
        "io_concurrency": 2.0,
        "selection_tolerance_ns": 0.0,
        "decode_fixed_ns": 0.0,
        "decode_access_unit_ns": 1.0,
        "fetch_decode_overlap": 0.0,
        "runtime_feedback_enabled": 0.0,
    }

    def session():
        return vclasp.VClaspSession.local(
            str(chunk),
            model,
            [],
            max_callers=max_callers,
            decoder_threads=1,
            global_decode_concurrency=2,
            resident_encoded_bytes=0,
            resident_cursor_capacity=0,
            max_merge_gap_bytes=0,
            max_range_bytes=512,
        )

    reader, reference = session(), session()
    batches = [
        [(10, "video", 63), (11, "video", 0), (12, "video", 63)],
        [(20, "video", 17), (21, "video", 8), (22, "video", 33)],
    ]
    expected = [reference.execute(batch)[0] for batch in batches]
    outputs, ready, ordered, stats, _, _ = reader.execute_window(batches)
    assert outputs == expected
    assert [[row[0] for row in batch] for batch in outputs] == [
        [10, 11, 12],
        [20, 21, 22],
    ]
    assert outputs[0][0][1:] == outputs[0][2][1:]
    assert all(
        isinstance(rgb, bytes) and len(rgb) == width * height * 3
        for batch in outputs
        for _, rgb, width, height in batch
    )
    assert len(ready) == len(ordered) == len(batches)
    assert ordered == sorted(ordered)
    assert 0 < stats["incremental_range_buffer_peak_bytes"] <= stats["fetched_bytes"]
    assert stats["incremental_range_buffer_received_bytes"] == stats["fetched_bytes"]
    assert (
        stats["incremental_range_buffer_released_bytes"]
        == stats["incremental_range_buffer_received_bytes"]
    )
    assert stats["incremental_decode_timing_samples"] == stats["decode_groups"]
    assert stats["incremental_decoded_rgb_bytes"] == 5 * 64 * 64 * 3
    assert (
        stats["incremental_range_completion_blocked_sends"] <= stats["physical_ranges"]
    )
    reader.synchronize()
    assert (
        reader.metrics_snapshot()["incremental_range_buffer_peak_bytes"]
        == stats["incremental_range_buffer_peak_bytes"]
    )
    for name in (
        "incremental_range_buffer_received_bytes",
        "incremental_range_buffer_released_bytes",
        "incremental_range_completion_blocked_sends",
        "incremental_range_completion_block_ns",
        "incremental_decode_queue_ns",
        "incremental_decode_resource_wait_ns",
        "incremental_decode_service_ns",
        "incremental_decode_timing_samples",
        "incremental_decoded_rgb_bytes",
    ):
        assert stats[name] >= 0
        assert reader.metrics_snapshot()[name] == stats[name]

    # An invalid request must not poison the persistent reader or strand workers.
    with pytest.raises(RuntimeError):
        reader.execute([(99, "missing-video", 0)])
    assert reader.execute(batches[0])[0] == expected[0]
    reader.synchronize()

    pipeline = reader.pipeline(max_outstanding_batches=2, max_outstanding_targets=6)
    errors = []

    def produce():
        try:
            for batch in batches * 4:
                pipeline.submit(batch)
        except Exception as error:
            errors.append(error)
        finally:
            pipeline.close()

    producer = Thread(target=produce)
    producer.start()
    delivered = []
    while (item := pipeline.take()) is not None:
        sequence, frames, *_ = item
        assert sequence == len(delivered)
        delivered.append(frames)
    producer.join(timeout=10)
    assert not producer.is_alive()
    assert not errors
    assert delivered == expected * 4
    reader.synchronize()
