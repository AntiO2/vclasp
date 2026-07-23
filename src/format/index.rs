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
        let schema = batches[0].schema();
        let merged = arrow::compute::concat_batches(&schema, &batches)?;

        let video_ids = merged
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("video_id column not StringArray")?
            .clone();
        let tiers = merged
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or("tier column not Int32Array")?
            .clone();
        let offsets = merged
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or("record_offset column not Int64Array")?
            .clone();
        let lengths = merged
            .column(4)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or("record_length column not Int64Array")?
            .clone();
        let frame_idxs = merged
            .column(5)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or("frame_idx column not Int32Array")?
            .clone();
        let dependency_kinds = merged
            .column(7)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("dependency_kind column not StringArray")?
            .clone();

        let n = video_ids.len();

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
        assert_eq!(video_ids.len(), tiers.len());
        let n = video_ids.len();

        // Small batch: use scalar hash lookup (faster for < 64 items)
        if n < 64 {
            let mut rng = fastrand::Rng::new();
            let mut results = Vec::with_capacity(n);
            for i in 0..n {
                results.push(self.lookup_one(&video_ids[i], tiers[i], &mut rng)?);
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
