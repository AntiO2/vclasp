"""Regression for AU reuse across streaming windows without live cursors."""

import json
import os
from pathlib import Path

import pytest
import vclasp

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("budget", [0, 1, 32 << 10, 8 << 20])
def test_incremental_windows_reuse_encoded_aus(budget):
    chunk = os.environ.get("VCLASP_CACHE_TEST_CHUNK")
    video = os.environ.get("VCLASP_CACHE_TEST_VIDEO")
    model_path = os.environ.get("VCLASP_CACHE_TEST_MODEL")
    if not all((chunk, video, model_path)):
        pytest.skip("configure VCLASP_CACHE_TEST_CHUNK/VIDEO/MODEL")
    model = json.loads(Path(model_path).read_text())

    def reader(capacity):
        return vclasp.VClaspSession.local(
            chunk,
            model,
            [],
            max_callers=1,
            decoder_threads=1,
            global_decode_concurrency=2,
            resident_encoded_bytes=capacity,
            resident_cursor_capacity=0,
        )

    cached, baseline = reader(budget), reader(0)
    windows = [list(range(16)), list(range(8, 24)), list(range(8, 24)), [63, 63, 0]]
    snapshots = []
    for frames in windows:
        targets = [[(i, video, frame) for i, frame in enumerate(frames)]]
        cached.start_execution_trace(10000, 10000)
        outputs, _, _, stats, mode, _ = cached.execute_window(targets)
        expected = baseline.execute_window(targets)[0]
        cached.synchronize()
        trace = cached.take_execution_trace()
        assert outputs == expected  # IDs, RGB bytes, dimensions, duplicates and order.
        assert mode.startswith("window_")
        assert len(trace[0]) == trace[1] == stats["physical_ranges"]
        assert len(trace[3]) == trace[4] == stats["submitted_access_units"]
        assert trace[2] == trace[5] == 0
        assert stats["resident_encoded_bytes"] <= budget
        snapshots.append(stats)
    if budget >= 8 << 20:
        assert snapshots[0]["resident_encoded_hits"] == 0
        assert snapshots[1]["resident_encoded_hits"] > 0
        assert snapshots[1]["fetched_bytes"] < snapshots[0]["fetched_bytes"]
        assert snapshots[2]["resident_encoded_misses"] == 0
        assert snapshots[2]["physical_ranges"] == snapshots[2]["fetched_bytes"] == 0
        assert snapshots[2]["submitted_access_units"] > 0
    elif budget <= 1:
        assert all(stats["resident_encoded_hits"] == 0 for stats in snapshots)
        assert all(stats["physical_ranges"] > 0 for stats in snapshots)

    metrics = cached.metrics_snapshot()
    assert metrics["resident_encoded_evictions"] == sum(
        stats["resident_encoded_evictions"] for stats in snapshots
    )
    if budget == 32 << 10:
        assert metrics["resident_encoded_evictions"] > 0
    else:
        assert metrics["resident_encoded_evictions"] == 0
