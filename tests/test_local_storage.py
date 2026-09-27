"""Unit tests using small byte fixtures in pytest's temporary directory."""

import struct

import pytest

import vclasp


@pytest.mark.parametrize("gap, expected_ranges", [(None, 3), (0, 2), (2, 1)])
def test_planned_ranges_extract_only_requested_records(tmp_path, gap, expected_ranges):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abcXXdefghi")
    # Input order differs from physical order, and one record is requested twice.
    records = [(30, 8, 3), (10, 0, 3), (20, 5, 3), (10, 0, 3)]
    plans = vclasp.plan_byte_ranges(records, merge_threshold_bytes=gap)
    assert len(plans) == expected_ranges

    extracted = {}
    with payload.open("rb") as source:
        for offset, length, members in plans:
            source.seek(offset)
            fetched = source.read(length)
            assert len(fetched) == length
            for record_id, relative_offset, record_length in members:
                assert record_id not in extracted
                extracted[record_id] = fetched[
                    relative_offset : relative_offset + record_length
                ]

    assert extracted == {10: b"abc", 20: b"def", 30: b"ghi"}
    assert [extracted[record_id] for record_id, _, _ in records] == [
        b"ghi",
        b"abc",
        b"def",
        b"abc",
    ]


def test_merge_respects_maximum_range_size():
    plans = vclasp.plan_byte_ranges(
        [(1, 0, 3), (2, 5, 3)], merge_threshold_bytes=2, max_range_bytes=7
    )
    assert [(offset, length) for offset, length, _ in plans] == [(0, 3), (5, 3)]


@pytest.mark.parametrize(
    "records",
    [
        [(1, 0, 0)],
        [(1, (1 << 64) - 1, 2)],
        [(1, 0, 3), (1, 5, 3)],
    ],
)
def test_invalid_record_ranges_raise_value_error(records):
    with pytest.raises(ValueError):
        vclasp.plan_byte_ranges(records)


@pytest.mark.parametrize("data", [b"x", struct.pack("<I", 32) + b"short"])
def test_chunk_rejects_truncated_local_files(tmp_path, data):
    chunk = tmp_path / "truncated.vclasp"
    chunk.write_bytes(data)
    with pytest.raises(OSError, match="smaller than"):
        vclasp.VClaspChunk(str(chunk))


def test_chunk_reports_missing_local_file(tmp_path):
    with pytest.raises(OSError):
        vclasp.VClaspChunk(str(tmp_path / "missing.vclasp"))
