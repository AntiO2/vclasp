use std::path::Path;
use vclasp::chunk::ChunkReader;

fn test_chunk_path() -> Option<String> {
    std::env::var("VCLASP_TEST_CHUNK").ok()
}

fn open_test_chunk() -> Option<ChunkReader> {
    let path = test_chunk_path()?;
    ChunkReader::open(Path::new(&path)).ok()
}

#[test]
#[ignore = "requires VCLASP_TEST_CHUNK"]
fn test_record_count_for_tier0() {
    let reader = open_test_chunk().expect("VCLASP_TEST_CHUNK not set");
    let count = reader.record_count_for("ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi", 0);
    assert_eq!(count, 1, "every video should have exactly 1 tier0 record");
}

#[test]
#[ignore = "requires VCLASP_TEST_CHUNK"]
fn test_record_count_for_tier2_greater_than_tier1() {
    let reader = open_test_chunk().expect("chunk not found");
    let vid = "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi";
    let t1 = reader.record_count_for(vid, 1);
    let t2 = reader.record_count_for(vid, 2);
    assert!(
        t2 >= t1,
        "tier2 records ({}) should be >= tier1 records ({})",
        t2,
        t1
    );
    assert!(t2 > 0, "tier2 should have at least 1 record");
}

#[test]
#[ignore = "requires VCLASP_TEST_CHUNK"]
fn test_record_count_for_unknown_video_returns_zero() {
    let reader = open_test_chunk().expect("chunk not found");
    let count = reader.record_count_for("nonexistent/video.avi", 0);
    assert_eq!(count, 0, "unknown video should return 0 records");
}

#[test]
#[ignore = "requires VCLASP_TEST_CHUNK"]
fn test_read_records_range_single() {
    let mut reader = open_test_chunk().expect("chunk not found");
    let records = reader
        .read_records_range("ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi", 2, 0, 1)
        .expect("read_records_range failed");
    assert_eq!(records.len(), 1, "should read exactly 1 record");
    assert!(!records[0].is_empty(), "record should not be empty");
}

#[test]
#[ignore = "requires VCLASP_TEST_CHUNK"]
fn test_read_records_range_multiple() {
    let mut reader = open_test_chunk().expect("chunk not found");
    let records = reader
        .read_records_range("ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi", 2, 0, 3)
        .expect("read_records_range failed");
    assert_eq!(records.len(), 3, "should read exactly 3 records");
    for (i, rec) in records.iter().enumerate() {
        assert!(!rec.is_empty(), "record {} should not be empty", i);
    }
}

#[test]
#[ignore = "requires VCLASP_TEST_CHUNK"]
fn test_read_records_range_count_exceeds_available() {
    let mut reader = open_test_chunk().expect("chunk not found");
    let vid = "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi";
    let total = reader.record_count_for(vid, 2);
    // Request more than available; should return only what exists
    let records = reader
        .read_records_range(vid, 2, 0, total + 10)
        .expect("read_records_range failed");
    assert_eq!(
        records.len(),
        total,
        "should return at most {:?} records, got {:?}",
        total,
        records.len()
    );
}

#[test]
#[ignore = "requires VCLASP_TEST_CHUNK"]
fn test_read_records_range_start_out_of_bounds() {
    let mut reader = open_test_chunk().expect("chunk not found");
    let result =
        reader.read_records_range("ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi", 0, 999, 1);
    assert!(
        result.is_err(),
        "start_idx out of bounds should be an error"
    );
}

#[test]
#[ignore = "requires VCLASP_TEST_CHUNK"]
fn test_read_records_range_consistency() {
    let mut reader = open_test_chunk().expect("chunk not found");
    let vid = "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi";

    // read_records_range(0, 1) should match read_all_records()[0]
    let first = reader
        .read_records_range(vid, 2, 0, 1)
        .expect("range failed");
    let all = reader.read_all_records(vid, 2).expect("all failed");

    assert_eq!(
        first[0], all[0],
        "first record from range should match first record from all"
    );
}

#[test]
#[ignore = "requires VCLASP_TEST_CHUNK"]
fn test_read_records_range_subset_matches_all_slice() {
    let mut reader = open_test_chunk().expect("chunk not found");
    let vid = "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi";
    let total = reader.record_count_for(vid, 2);
    if total < 3 {
        return; // skip if not enough records
    }

    // read_records_range(1, 2) should match all[1..3]
    let subset = reader
        .read_records_range(vid, 2, 1, 2)
        .expect("range failed");
    let all = reader.read_all_records(vid, 2).expect("all failed");

    assert_eq!(subset.len(), 2);
    assert_eq!(subset[0], all[1]);
    assert_eq!(subset[1], all[2]);
}
