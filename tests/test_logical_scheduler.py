#!/usr/bin/env python3
"""Tests for LogicalBatchDecoder (auto-configured from chunk)."""
import sys, os, time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__) + '/..')
sys.path.insert(0, '.')
import vclasp

fixture = os.environ.get("VCLASP_TEST_CHUNK")
if not fixture:
    pytest.skip("VCLASP_TEST_CHUNK is not configured", allow_module_level=True)
SMOKE_CHUNK = Path(fixture)
VID = 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'

def test_auto_config():
    c = vclasp.VClaspChunk(str(SMOKE_CHUNK))
    dec = c.create_logical_scheduler()
    assert dec.gop_size() == 8, f"Expected gop_size=8"
    assert dec.tier1_stride() == 4, f"Expected stride=4"
    print(f"  PASS test_auto_config")

def test_read_record_at_deterministic():
    """read_record_at(idx) should return deterministic records."""
    c = vclasp.VClaspChunk(str(SMOKE_CHUNK))
    r0 = c.read_record_at(VID, 2, 0)
    r1 = c.read_record_at(VID, 2, 1)
    r0b = c.read_record_at(VID, 2, 0)
    assert r0 == r0b, "read_record_at(0) should be deterministic"
    assert r0 != r1, "read_record_at(0) != read_record_at(1)"
    print(f"  PASS test_read_record_at_deterministic (idx0={len(r0)}B, idx1={len(r1)}B)")

def test_decode_gop_record_at():
    """decode_gop_record_at(idx) should decode deterministic GOP records."""
    c = vclasp.VClaspChunk(str(SMOKE_CHUNK))
    frames0 = c.decode_gop_record_at(VID, 2, 0)
    frames1 = c.decode_gop_record_at(VID, 2, 1)
    assert len(frames0) == 8, f"FullGop8 record should have 8 frames, got {len(frames0)}"
    assert len(frames1) == 8
    print(f"  PASS test_decode_gop_record_at")

def test_record_count_for():
    c = vclasp.VClaspChunk(str(SMOKE_CHUNK))
    n = c.record_count_for(VID, 2)
    assert n > 0, f"record_count_for returned {n}"
    print(f"  PASS test_record_count_for: {VID} tier2 has {n} records")

def test_batch_decoder():
    c = vclasp.VClaspChunk(str(SMOKE_CHUNK))
    dec = c.create_logical_scheduler()
    available = c.record_count_for(VID, 2)
    assert available > 0
    n = min(50, available)
    requests = [(i, VID, 2, i * dec.gop_size()) for i in range(n)]
    t0 = time.perf_counter()
    results = dec.schedule(requests)
    elapsed = (time.perf_counter() - t0) * 1000
    assert len(results) == n
    sids = [sid for sid, *_ in results]
    assert sids == list(range(n)), f"Order violation"
    print(f"  PASS test_batch_decoder: {n} req in {elapsed:.0f}ms, order OK")

if __name__ == '__main__':
    if not SMOKE_CHUNK.exists():
        print("SKIP: smoke chunks not found")
        sys.exit(0)
    test_auto_config()
    test_read_record_at_deterministic()
    test_decode_gop_record_at()
    test_record_count_for()
    test_batch_decoder()
    print("\nAll tests PASSED")
