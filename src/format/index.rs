//! Columnar index reader: Parquet-backed, SIMD-friendly batch lookup.

use crate::simd;
use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;

/// Columnar index for fast record lookup.
pub struct IndexReader {
    /// video_id column (StringArray) for hash lookup.
    video_ids: StringArray,
    /// tier column (Int32Array).
    tiers: Int32Array,
    /// record_offset column (Int64Array).
    offsets: Int64Array,
    /// record_length column (Int64Array).
    lengths: Int64Array,
    /// frame_idx column (Int32Array) — absolute frame position of this record.
    frame_idxs: Int32Array,
    /// dependency_kind column (StringArray) — "idr", "gop", or "anchor_p".
    dependency_kinds: StringArray,
    /// video_id -> list of row indices (scalar hash for O(1) per-video lookup).
    video_index: HashMap<String, Vec<usize>>,
    /// Number of records.
    record_count: usize,
}

impl IndexReader {
    /// Parse Parquet index bytes into columnar arrays.
    pub fn from_parquet(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = Bytes::copy_from_slice(data);
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;

        let mut batches = Vec::new();
        for batch in reader {
            batches.push(batch?);
        }

        // Concatenate all batches
        let schema = batches
            .first()
            .ok_or("index has no record batches")?
            .schema();
        let merged = arrow::compute::concat_batches(&schema, &batches)?;

        macro_rules! column {
            ($name:literal) => {
                merged
                    .column_by_name($name)
                    .ok_or(concat!("missing index column: ", $name))?
            };
        }

        let video_ids = column!("video_id")
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("video_id column not StringArray")?
            .clone();
        let tiers = column!("tier")
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or("tier column not Int32Array")?
            .clone();
        let offsets = column!("record_offset")
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or("record_offset column not Int64Array")?
            .clone();
        let lengths = column!("record_length")
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or("record_length column not Int64Array")?
            .clone();
        let frame_idxs = column!("frame_idx")
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or("frame_idx column not Int32Array")?
            .clone();
        let dependency_kinds = column!("dependency_kind")
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("dependency_kind column not StringArray")?
            .clone();

        let n = video_ids.len();
        for row in 0..n {
            if video_ids.is_null(row)
                || tiers.is_null(row)
                || offsets.is_null(row)
                || lengths.is_null(row)
                || offsets.value(row) < 0
                || lengths.value(row) <= 0
            {
                return Err(format!("invalid index row {row}").into());
            }
        }

        // Build hash index: video_id -> row indices
        let mut video_index: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..n {
            if video_ids.is_valid(i) {
                video_index
                    .entry(video_ids.value(i).to_string())
                    .or_default()
                    .push(i);
            }
        }

        Ok(IndexReader {
            video_ids,
            tiers,
            offsets,
            lengths,
            frame_idxs,
            dependency_kinds,
            video_index,
            record_count: n,
        })
    }

    pub fn record_count(&self) -> usize {
        self.record_count
    }

    /// Lookup one random record matching (video_id, tier).
    pub fn lookup_one(
        &self,
        video_id: &str,
        tier: i32,
        rng: &mut fastrand::Rng,
    ) -> Result<(u64, u64), Box<dyn std::error::Error>> {
        let indices = self
            .video_index
            .get(video_id)
            .ok_or_else(|| format!("video_id not found: {}", video_id))?;

        // Filter by tier
        let matches: Vec<usize> = indices
            .iter()
            .filter(|&&i| self.tiers.is_valid(i) && self.tiers.value(i) == tier)
            .copied()
            .collect();

        if matches.is_empty() {
            return Err(format!("no record for video={} tier={}", video_id, tier).into());
        }

        let idx = matches[rng.usize(0..matches.len())];
        Ok((
            self.offsets.value(idx) as u64,
            self.lengths.value(idx) as u64,
        ))
    }

    /// Lookup ALL records matching (video_id, tier) in one scan.
    /// Returns one (offset, length) pair per matching record.
    pub fn lookup_all(
        &self,
        video_id: &str,
        tier: i32,
    ) -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
        let indices = self
            .video_index
            .get(video_id)
            .ok_or_else(|| format!("video_id not found: {}", video_id))?;

        let results: Vec<(u64, u64)> = indices
            .iter()
            .filter(|&&i| self.tiers.is_valid(i) && self.tiers.value(i) == tier)
            .map(|&i| (self.offsets.value(i) as u64, self.lengths.value(i) as u64))
            .collect();

        if results.is_empty() {
            return Err(format!("no record for video={} tier={}", video_id, tier).into());
        }
        Ok(results)
    }

    /// Batch lookup: resolve many (video_id, tier) pairs.
    /// Uses SIMD-accelerated columnar scan for batches > 64.
    pub fn lookup_batch(
        &self,
        video_ids: &[String],
        tiers: &[i32],
    ) -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
        if video_ids.len() != tiers.len() {
            return Err("video and tier counts differ".into());
        }
        let n = video_ids.len();

        // Small batch: use scalar hash lookup (faster for < 64 items)
        if n < 64 {
            let mut results = Vec::with_capacity(n);
            let mut positions = HashMap::<(&str, i32), usize>::new();
            for i in 0..n {
                let candidates = self.lookup_all(&video_ids[i], tiers[i])?;
                let position = positions.entry((&video_ids[i], tiers[i])).or_default();
                results.push(candidates[*position % candidates.len()]);
                *position += 1;
            }
            return Ok(results);
        }

        // Large batch: SIMD columnar scan
        // For each target, scan video_ids column to find matching rows
        simd::batch_resolve(
            &self.video_ids,
            &self.tiers,
            &self.offsets,
            &self.lengths,
            video_ids,
            tiers,
        )
    }

    /// Infer GOP size from frame_idx deltas across all videos' tier2 records.
    pub fn infer_gop_size(&self) -> usize {
        let mut deltas = HashMap::new();
        for indices in self.video_index.values() {
            let mut frames: Vec<i32> = indices
                .iter()
                .filter(|&&i| self.tiers.is_valid(i) && self.tiers.value(i) == 2)
                .map(|&i| self.frame_idxs.value(i))
                .collect();
            frames.sort();
            for w in frames.windows(2) {
                let d = w[1] - w[0];
                if d > 0 {
                    *deltas.entry(d).or_insert(0) += 1;
                }
            }
        }
        deltas
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(d, _)| d as usize)
            .unwrap_or(8)
    }

    /// Infer tier1 stride from frame_idx deltas across tier1 records.
    pub fn infer_tier1_stride(&self, gop_size: usize) -> usize {
        let gs = gop_size.max(1) as i32;
        let mut strides = HashMap::new();
        for indices in self.video_index.values() {
            let mut frames: Vec<i32> = indices
                .iter()
                .filter(|&&i| self.tiers.is_valid(i) && self.tiers.value(i) == 1)
                .map(|&i| self.frame_idxs.value(i))
                .collect();
            frames.sort();
            for w in frames.windows(2) {
                let d = w[1] - w[0];
                if d > 0 && d % gs == 0 {
                    *strides.entry((d / gs) as usize).or_insert(0) += 1;
                }
            }
        }
        strides
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(s, _)| s)
            .unwrap_or(4)
    }

    /// Return the dependency_kind for the first matching record of (video_id, tier).
    pub fn dependency_kind_for(&self, video_id: &str, tier: i32) -> Option<&str> {
        let indices = self.video_index.get(video_id)?;
        for &i in indices {
            if self.tiers.is_valid(i) && self.tiers.value(i) == tier {
                return Some(self.dependency_kinds.value(i));
            }
        }
        None
    }

    /// Expose video_index for external inference (e.g., LogicalScheduler).
    pub fn video_entries(&self) -> &HashMap<String, Vec<usize>> {
        &self.video_index
    }

    /// Deterministic lookup: return the N-th matching record for (video_id, tier).
    pub fn lookup_at(&self, video_id: &str, tier: i32, rec_idx: usize) -> Option<RecordLocation> {
        let indices = self.video_index.get(video_id)?;
        let mut count = 0usize;
        for &i in indices {
            if self.tiers.is_valid(i) && self.tiers.value(i) == tier {
                if count == rec_idx {
                    return Some(RecordLocation {
                        offset: self.offsets.value(i) as u64,
                        length: self.lengths.value(i) as u64,
                        frame_idx: self.frame_idxs.value(i),
                        dependency_kind: self.dependency_kinds.value(i).to_string(),
                    });
                }
                count += 1;
            }
        }
        None
    }
}

/// Physical location of a single record in the chunk.
#[derive(Debug, Clone)]
pub struct RecordLocation {
    pub offset: u64,
    pub length: u64,
    pub frame_idx: i32,
    pub dependency_kind: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_lookup_does_not_change_selection_at_simd_threshold() {
        let index = IndexReader {
            video_ids: StringArray::from(vec!["video", "video"]),
            tiers: Int32Array::from(vec![2, 2]),
            offsets: Int64Array::from(vec![10, 20]),
            lengths: Int64Array::from(vec![3, 4]),
            frame_idxs: Int32Array::from(vec![0, 1]),
            dependency_kinds: StringArray::from(vec!["gop", "gop"]),
            video_index: HashMap::from([("video".into(), vec![0, 1])]),
            record_count: 2,
        };
        for count in [1, 63, 64, 65] {
            assert_eq!(
                index
                    .lookup_batch(&vec!["video".into(); count], &vec![2; count])
                    .unwrap(),
                (0..count)
                    .map(|index| if index % 2 == 0 { (10, 3) } else { (20, 4) })
                    .collect::<Vec<_>>()
            );
        }
        assert!(index.lookup_batch(&["video".into()], &[]).is_err());
    }

    #[test]
    fn empty_parquet_is_an_error_not_an_indexing_panic() {
        let schema = std::sync::Arc::new(arrow::datatypes::Schema::empty());
        let mut bytes = Vec::new();
        parquet::arrow::ArrowWriter::try_new(&mut bytes, schema, None)
            .unwrap()
            .close()
            .unwrap();
        assert!(IndexReader::from_parquet(&bytes).is_err());
    }
}
