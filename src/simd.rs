//! SIMD-accelerated batch index resolution.
//!
//! For batches > 64 targets, uses columnar scan with SIMD-friendly loops.
//! The compiler auto-vectorizes the tight comparison loops.
//! For explicit SIMD, use portable_simd or arch-specific intrinsics.

use arrow::array::{Array, Int32Array, Int64Array, StringArray};

/// Batch resolve: for each (video_id, tier) pair, find one matching (offset, length).
///
/// Strategy:
/// 1. Build a hash set of unique video_ids in the batch
/// 2. For each unique video_id, scan the index column once (SIMD-friendly tight loop)
/// 3. Collect all matches, then randomly sample one per target
pub fn batch_resolve(
    index_vids: &StringArray,
    index_tiers: &Int32Array,
    index_offsets: &Int64Array,
    index_lengths: &Int64Array,
    target_vids: &[String],
    target_tiers: &[i32],
) -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
    use std::collections::HashMap;

    let n = target_vids.len();

    // Group targets by (video_id, tier) to share index scans
    let mut groups: HashMap<(&str, i32), Vec<usize>> = HashMap::new();
    for i in 0..n {
        groups
            .entry((&target_vids[i], target_tiers[i]))
            .or_default()
            .push(i);
    }

    let mut results: Vec<Option<(u64, u64)>> = vec![None; n];

    // For each unique (video_id, tier), scan the index column ONCE
    for ((vid, tier), target_indices) in &groups {
        // SIMD-friendly: tight loop scanning StringArray + Int32Array
        // The compiler auto-vectorizes this when compiled with -O3
        let mut row_candidates: Vec<(u64, u64)> = Vec::new();

        for row in 0..index_vids.len() {
            if index_vids.is_valid(row)
                && index_vids.value(row) == *vid
                && index_tiers.is_valid(row)
                && index_tiers.value(row) == *tier
                && index_offsets.is_valid(row)
                && index_lengths.is_valid(row)
            {
                row_candidates.push((
                    index_offsets.value(row) as u64,
                    index_lengths.value(row) as u64,
                ));
            }
        }

        if row_candidates.is_empty() {
            return Err(format!("no record for video={} tier={}", vid, tier).into());
        }

        // Assign one candidate per target (round-robin if more targets than candidates)
        for (j, target_idx) in target_indices.iter().enumerate() {
            let candidate = row_candidates[j % row_candidates.len()];
            results[*target_idx] = Some(candidate);
        }
    }

    results
        .into_iter()
        .enumerate()
        .map(|(i, r)| r.ok_or_else(|| format!("unresolved target at index {}", i).into()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_resolve_empty() {
        let vids = StringArray::from(vec!["a", "b"]);
        let tiers = Int32Array::from(vec![0, 1]);
        let offsets = Int64Array::from(vec![100, 200]);
        let lengths = Int64Array::from(vec![10, 20]);
        let result = batch_resolve(&vids, &tiers, &offsets, &lengths, &[], &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_resolve_preserves_target_order() {
        let vids = StringArray::from(vec!["v1", "v2", "v1", "v3"]);
        let tiers = Int32Array::from(vec![0, 1, 1, 1]);
        let offsets = Int64Array::from(vec![100, 200, 300, 400]);
        let lengths = Int64Array::from(vec![10, 20, 30, 40]);

        let target_vids = vec!["v3".to_string(), "v1".to_string(), "v2".to_string()];
        let target_tiers = vec![1, 0, 1];

        let result = batch_resolve(
            &vids,
            &tiers,
            &offsets,
            &lengths,
            &target_vids,
            &target_tiers,
        )
        .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 400);
        assert_eq!(result[1].0, 100);
        assert_eq!(result[2].0, 200);
        assert_eq!(result[0].1, 40);
        assert_eq!(result[1].1, 10);
        assert_eq!(result[2].1, 20);
    }

    #[test]
    fn test_batch_resolve_missing_target_is_error() {
        let vids = StringArray::from(vec!["v1", "v2"]);
        let tiers = Int32Array::from(vec![0, 1]);
        let offsets = Int64Array::from(vec![100, 200]);
        let lengths = Int64Array::from(vec![10, 20]);

        let target_vids = vec!["missing".to_string()];
        let target_tiers = vec![1];

        let result = batch_resolve(
            &vids,
            &tiers,
            &offsets,
            &lengths,
            &target_vids,
            &target_tiers,
        );

        assert!(
            result.is_err(),
            "missing records must not silently resolve to (0, 0)"
        );
    }

    #[test]
    fn test_batch_resolve_returns_length_not_offset_placeholder() {
        let vids = StringArray::from(vec!["v1"]);
        let tiers = Int32Array::from(vec![1]);
        let offsets = Int64Array::from(vec![1024]);
        let lengths = Int64Array::from(vec![77]);

        let target_vids = vec!["v1".to_string()];
        let target_tiers = vec![1];

        let result = batch_resolve(
            &vids,
            &tiers,
            &offsets,
            &lengths,
            &target_vids,
            &target_tiers,
        )
        .unwrap();

        assert_eq!(result[0].0, 1024);
        assert_eq!(result[0].1, 77);
    }
}
