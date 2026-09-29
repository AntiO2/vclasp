//! Logical batch decoder: maps (video_id, tier, frame_idx) → batch-decoded frames.
//!
//! Auto-configures from the chunk index (infers gop_size and tier1_stride),
//! so the caller never needs to pass codec-level parameters.
//!
//! This is NOT a full scheduler; it only groups requests by (vid, tier) and
//! batch-decodes. A real LookaheadScheduler would operate on an epoch trace,
//! reorder physical I/O by chunk/offset, coalesce ranges, and maintain
//! logical sample order.

#[cfg(feature = "ffmpeg")]
use std::collections::{BTreeMap, HashMap};
#[cfg(feature = "ffmpeg")]
use std::time::Instant;

#[cfg(feature = "ffmpeg")]
use pyo3::prelude::*;
#[cfg(feature = "ffmpeg")]
use pyo3::types::PyBytes;

#[cfg(feature = "ffmpeg")]
use crate::chunk;
#[cfg(feature = "ffmpeg")]
use crate::decoder;

/// A request: which frame to decode from which video and tier.
#[cfg(feature = "ffmpeg")]
#[derive(Debug, Clone)]
pub struct LogicalRequest {
    pub sample_id: i64,
    pub video_id: String,
    pub tier: i32,
    pub frame_idx: i32,
}

/// Result of a batched decode: one decoded RGB24 frame.
#[cfg(feature = "ffmpeg")]
#[derive(Debug)]
pub struct ScheduledFrame {
    pub sample_id: i64,
    pub rgb_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// LogicalBatchDecoder: groups requests by (video_id, tier), batch-decodes
/// only the required record range, extracts target frames, and returns
/// results in the original request order.
///
/// Auto-configured from the chunk index — infers gop_size, tier1_stride,
/// and frames-per-record from the index columns.
#[cfg(feature = "ffmpeg")]
pub struct LogicalBatchDecoder {
    inner: chunk::ChunkReader,
    decoder_pool: decoder::DecoderPool,
    cached_sps: Option<Vec<u8>>,
    gop_size: usize,
    tier1_stride: usize,
    frames_per_record_tier2: usize,
    path: String,
}

#[cfg(feature = "ffmpeg")]
impl LogicalBatchDecoder {
    /// Create a LogicalBatchDecoder from an already-opened ChunkReader.
    /// `file_path` is the chunk path, stored for later scheduler re-creation.
    pub fn new(
        mut reader: chunk::ChunkReader,
        file_path: String,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let gop_size = reader.index.infer_gop_size();
        let tier1_stride = reader.index.infer_tier1_stride(gop_size);

        // Determine frames_per_record for tier2 from dependency_kind
        let dep = reader.read_sps_pps().ok().and_then(|_| {
            reader
                .index
                .video_entries()
                .keys()
                .next()
                .and_then(|vid| reader.index.dependency_kind_for(vid, 2))
        });
        let frames_per_record_tier2 = match dep {
            Some("idr") | None => 1,
            _ => gop_size, // "gop" or "anchor_p"
        };

        Ok(LogicalBatchDecoder {
            inner: reader,
            decoder_pool: decoder::DecoderPool::new(decoder::DecoderConfig::default()),
            cached_sps: None,
            gop_size,
            tier1_stride,
            frames_per_record_tier2,
            path: file_path,
        })
    }

    fn get_sps_pps(&mut self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if self.cached_sps.is_none() {
            let data = self.inner.read_sps_pps()?;
            self.cached_sps = Some(data);
        }
        Ok(self.cached_sps.as_ref().unwrap().clone())
    }

    /// Schedule a batch of logical requests. Groups by (video_id, tier),
    /// batch-decodes only the required record range, extracts the target
    /// frame for each request, and returns results in the original sample_id
    /// order. Missing or inconsistent records are hard errors.
    pub fn schedule(
        &mut self,
        requests: &[LogicalRequest],
    ) -> Result<Vec<ScheduledFrame>, Box<dyn std::error::Error>> {
        let n = requests.len();

        // Group by (video_id, tier), preserving original positions
        let mut groups: HashMap<(String, i32), Vec<(usize, i64, i32)>> = HashMap::new();
        for (pos, req) in requests.iter().enumerate() {
            groups
                .entry((req.video_id.clone(), req.tier))
                .or_default()
                .push((pos, req.sample_id, req.frame_idx));
        }

        let mut results: BTreeMap<usize, ScheduledFrame> = BTreeMap::new();

        for ((vid, tier), group_reqs) in &groups {
            let tier_val = *tier;
            let frames_per_record: usize = if tier_val <= 1 {
                1
            } else {
                self.frames_per_record_tier2
            };
            let total_recs = self.inner.record_count_for(vid, tier_val);

            if total_recs == 0 {
                return Err(format!("no records for video={vid} tier={tier_val}").into());
            }

            let record_idxs: Result<Vec<usize>, Box<dyn std::error::Error>> = group_reqs
                .iter()
                .map(|(_, sample_id, frame_idx)| {
                    if *frame_idx < 0 {
                        return Err(format!(
                            "negative frame_idx={frame_idx} for sample_id={sample_id}"
                        )
                        .into());
                    }
                    let gop_idx = (*frame_idx as usize) / self.gop_size;
                    let idx = match tier_val {
                        0 => 0,
                        1 => gop_idx / self.tier1_stride,
                        _ => gop_idx,
                    };
                    if idx >= total_recs {
                        return Err(format!(
                            "record index {idx} out of range (records={total_recs}) for video={vid} tier={tier_val} sample_id={sample_id} frame_idx={frame_idx}"
                        )
                        .into());
                    }
                    Ok(idx)
                })
                .collect();
            let record_idxs = record_idxs?;

            let min_rec = *record_idxs.iter().min().unwrap_or(&0);
            let max_rec = *record_idxs.iter().max().unwrap_or(&0);
            let count = max_rec - min_rec + 1;

            let sps_pps = self.get_sps_pps()?;
            let records = self
                .inner
                .read_records_range(vid, tier_val, min_rec, count)?;
            let frames =
                decoder::decode_gop_rgb24_batch(&sps_pps, &records, &mut self.decoder_pool)?;

            for (ri, &(pos, sample_id, frame_idx)) in group_reqs.iter().enumerate() {
                let rec_idx = record_idxs[ri];
                let rel_idx = rec_idx - min_rec;
                let intra_offset = (frame_idx as usize) % self.gop_size;
                let flat_idx =
                    rel_idx * frames_per_record + (intra_offset % frames_per_record.max(1));

                let f = frames.get(flat_idx).ok_or_else(|| {
                    format!(
                        "decoder returned {} frames, missing flat index {flat_idx} for video={vid} tier={tier_val} sample_id={sample_id}",
                        frames.len()
                    )
                })?;
                results.insert(
                    pos,
                    ScheduledFrame {
                        sample_id,
                        rgb_bytes: f.data.clone(),
                        width: f.width,
                        height: f.height,
                    },
                );
            }
        }

        // Return in original input order
        let ordered: Result<Vec<ScheduledFrame>, Box<dyn std::error::Error>> = (0..n)
            .map(|pos| {
                results.remove(&pos).ok_or_else(|| {
                    format!("missing scheduler output at request position {pos}").into()
                })
            })
            .collect();

        ordered
    }

    pub fn gop_size(&self) -> usize {
        self.gop_size
    }
    pub fn tier1_stride(&self) -> usize {
        self.tier1_stride
    }
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// LogicalScheduler: cost-based range planner operating on an epoch trace.
///
/// Resolves logical requests to byte ranges via the chunk index, then uses a
/// cost model to decide whether adjacent records should be fetched in one
/// batch decode or split into separate calls. The cost model is:
///
///   merge iff gap_bytes < merge_threshold
///   where gap = next_offset - (current_offset + current_length)
///
/// For object stores, set threshold ≈ (per-GET-latency / per-byte-cost).
/// For local SSD, threshold ≈ 0 (never pay overfetch).
#[cfg(feature = "ffmpeg")]
pub struct LogicalScheduler {
    decoder: LogicalBatchDecoder,
    merge_threshold: Option<usize>,
    backend: Option<Box<dyn crate::backend::StorageBackend + Send>>,
    completion_driven: bool,
    decode_microbatch_records: usize,
    range_priority: RangePriority,
}

#[cfg(feature = "ffmpeg")]
#[derive(Clone, Copy)]
enum RangePriority {
    Offset,
    Smallest,
    Largest,
    Fanout,
    FanoutPerByte,
}

#[cfg(feature = "ffmpeg")]
impl RangePriority {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
             "offset" => Ok(Self::Offset),
             "smallest" => Ok(Self::Smallest),
             "largest" => Ok(Self::Largest),
             "fanout" => Ok(Self::Fanout),
             "fanout_per_byte" => Ok(Self::FanoutPerByte),
             _ => Err(format!(
                 "unknown range priority {value:?}; expected offset, smallest, largest, fanout, or fanout_per_byte"
             )),
         }
    }
}

/// Statistics collected during an epoch execution.
#[cfg(feature = "ffmpeg")]
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_requests: usize,
    pub unique_records: usize,
    pub physical_ranges: usize,
    pub fetched_bytes: u64,
    pub useful_bytes: u64,
    pub overfetch_bytes: u64,
    pub records_decoded: usize,
    pub decode_batches: usize,
    pub total_frames: usize,
    pub max_fetch_records: usize,
    // legacy compatibility
    pub coalesced_gets: usize,
    pub uncoalesced_gets: usize,
    pub bytes_read: u64,
    pub uncoalesced_bytes: u64,
    pub records_read: usize,
    pub max_batch_size: usize,
    pub resolve_ns: u64,
    pub plan_ns: u64,
    pub fetch_wall_ns: u64,
    pub fetch_service_ns_sum: u64,
    pub range_queue_ns_sum: u64,
    pub extract_ns: u64,
    pub decode_ns: u64,
    pub fetch_decode_overlap_ns: u64,
    pub reorder_ns: u64,
    pub total_ns: u64,
    pub time_to_first_ready_target_ns: u64,
}

#[cfg(feature = "ffmpeg")]
struct GlobalDecodeRecord {
    video_id: String,
    tier: i32,
    bytes: Vec<u8>,
    consumers: Vec<(usize, i64, i32)>,
}

#[cfg(feature = "ffmpeg")]
struct GlobalDecodeGroup {
    video_id: String,
    tier: i32,
    frames_per_record: usize,
    records: Vec<GlobalDecodeRecord>,
}

#[cfg(feature = "ffmpeg")]
fn decode_global_groups(
    decoder_state: &mut LogicalBatchDecoder,
    groups: &[GlobalDecodeGroup],
    results: &mut BTreeMap<usize, ScheduledFrame>,
    stats: &mut SchedulerStats,
) -> Result<(), Box<dyn std::error::Error>> {
    for group in groups {
        let record_bytes: Vec<Vec<u8>> = group
            .records
            .iter()
            .map(|record| record.bytes.clone())
            .collect();
        stats.decode_batches += 1;
        stats.records_decoded += record_bytes.len();

        let sps_pps = decoder_state.get_sps_pps()?;
        let frames = decoder::decode_gop_rgb24_batch(
            &sps_pps,
            &record_bytes,
            &mut decoder_state.decoder_pool,
        )?;
        stats.total_frames += frames.len();

        for (record_index, record) in group.records.iter().enumerate() {
            for &(position, sample_id, frame_index) in &record.consumers {
                let intra_offset = (frame_index as usize) % decoder_state.gop_size;
                let flat_index = if group.frames_per_record == 1 {
                    record_index
                } else {
                    record_index * group.frames_per_record
                        + (intra_offset % group.frames_per_record)
                };
                let frame = frames.get(flat_index).ok_or_else(|| {
                    format!(
                        "decoded frame index {} out of bounds for video={} tier={} record={}",
                        flat_index, record.video_id, record.tier, record_index
                    )
                })?;
                results.insert(
                    position,
                    ScheduledFrame {
                        sample_id,
                        rgb_bytes: frame.data.clone(),
                        width: frame.width,
                        height: frame.height,
                    },
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "ffmpeg")]
fn merge_global_groups(pending: &mut Vec<GlobalDecodeGroup>, incoming: Vec<GlobalDecodeGroup>) {
    for mut incoming_group in incoming {
        if let Some(group) = pending.iter_mut().find(|group| {
            group.video_id == incoming_group.video_id
                && group.tier == incoming_group.tier
                && group.frames_per_record == incoming_group.frames_per_record
        }) {
            group.records.append(&mut incoming_group.records);
        } else {
            pending.push(incoming_group);
        }
    }
}

#[cfg(feature = "ffmpeg")]
fn global_group_record_count(groups: &[GlobalDecodeGroup]) -> usize {
    groups.iter().map(|group| group.records.len()).sum()
}

#[cfg(feature = "ffmpeg")]
fn drain_global_microbatch(
    pending: &mut Vec<GlobalDecodeGroup>,
    max_records: usize,
) -> Vec<GlobalDecodeGroup> {
    let mut remaining = max_records.max(1);
    let mut drained = Vec::new();
    let mut index = 0usize;
    while index < pending.len() && remaining > 0 {
        let take = remaining.min(pending[index].records.len());
        let records: Vec<_> = pending[index].records.drain(..take).collect();
        drained.push(GlobalDecodeGroup {
            video_id: pending[index].video_id.clone(),
            tier: pending[index].tier,
            frames_per_record: pending[index].frames_per_record,
            records,
        });
        remaining -= take;
        if pending[index].records.is_empty() {
            pending.remove(index);
        } else {
            index += 1;
        }
    }
    drained
}

#[cfg(feature = "ffmpeg")]
impl LogicalScheduler {
    /// `merge_threshold`: byte gap threshold for merging adjacent records.
    /// None = always decode full [min_rec..max_rec] (batch decoder behavior).
    pub fn new(decoder: LogicalBatchDecoder, merge_threshold: Option<usize>) -> Self {
        LogicalScheduler {
            decoder,
            merge_threshold,
            backend: None,
            completion_driven: false,
            decode_microbatch_records: 1,
            range_priority: RangePriority::Offset,
        }
    }

    /// Attach a storage backend for remote payload reads.
    pub fn with_backend(mut self, backend: Box<dyn crate::backend::StorageBackend + Send>) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn with_completion_driven(mut self, enabled: bool) -> Self {
        self.completion_driven = enabled;
        self
    }

    pub fn with_decode_microbatch_records(mut self, records: usize) -> Self {
        self.decode_microbatch_records = records.max(1);
        self
    }

    pub fn with_range_priority(mut self, priority: &str) -> Result<Self, String> {
        self.range_priority = RangePriority::parse(priority)?;
        Ok(self)
    }

    /// Execute an epoch trace with cost-based range planning.
    pub fn execute(
        &mut self,
        trace: &[LogicalRequest],
    ) -> Result<(Vec<ScheduledFrame>, SchedulerStats), Box<dyn std::error::Error>> {
        let n = trace.len();

        // Group by (video_id, tier), record original position + frame_idx
        let mut groups: HashMap<(String, i32), Vec<(usize, i64, i32, usize)>> = HashMap::new();
        for (pos, req) in trace.iter().enumerate() {
            let gop_idx = (req.frame_idx as usize) / self.decoder.gop_size;
            let rec_idx = match req.tier {
                0 => 0,
                1 => gop_idx / self.decoder.tier1_stride,
                _ => gop_idx,
            };
            groups
                .entry((req.video_id.clone(), req.tier))
                .or_default()
                .push((pos, req.sample_id, req.frame_idx, rec_idx));
        }

        let mut results: BTreeMap<usize, ScheduledFrame> = BTreeMap::new();
        let mut stats = SchedulerStats {
            total_requests: n,
            unique_records: 0,
            physical_ranges: 0,
            fetched_bytes: 0,
            useful_bytes: 0,
            overfetch_bytes: 0,
            records_decoded: 0,
            decode_batches: 0,
            total_frames: 0,
            max_fetch_records: 0,
            coalesced_gets: 0,
            uncoalesced_gets: 0,
            bytes_read: 0,
            uncoalesced_bytes: 0,
            records_read: 0,
            max_batch_size: 0,
            resolve_ns: 0,
            plan_ns: 0,
            fetch_wall_ns: 0,
            fetch_service_ns_sum: 0,
            range_queue_ns_sum: 0,
            extract_ns: 0,
            decode_ns: 0,
            reorder_ns: 0,
            total_ns: 0,
            fetch_decode_overlap_ns: 0,
            time_to_first_ready_target_ns: 0,
        };

        for ((vid, tier), mut group_reqs) in groups {
            let tier_val = tier;
            let frames_per_record: usize = if tier_val <= 1 {
                1
            } else {
                self.decoder.frames_per_record_tier2
            };
            let total_recs = self.decoder.inner.record_count_for(&vid, tier_val);

            // Clamp record indices
            for (_, _, _, rec_idx) in group_reqs.iter_mut() {
                *rec_idx = (*rec_idx).min(total_recs.saturating_sub(1));
            }

            if total_recs == 0 {
                return Err(format!("no records for video={} tier={}", vid, tier_val).into());
            }

            // Sort by record_idx for gap analysis
            group_reqs.sort_by_key(|(_, _, _, r)| *r);

            // Resolve record offsets for cost-based clustering
            let all_ranges = self.decoder.inner.index.lookup_all(&vid, tier_val)?;

            // Split into clusters using byte-gap cost model
            let clusters = if let Some(thresh) = self.merge_threshold {
                cluster_by_byte_gap(&group_reqs, &all_ranges, thresh)
            } else {
                // None = one big cluster (batch decoder behavior)
                if group_reqs.is_empty() {
                    vec![]
                } else {
                    vec![(0, group_reqs.len())]
                }
            };

            for (start, end) in clusters {
                let cluster_slice = &group_reqs[start..end];
                let min_rec = cluster_slice[0].3;
                let max_rec = cluster_slice[cluster_slice.len() - 1].3;
                let count = max_rec - min_rec + 1;

                let sps_pps = self.decoder.get_sps_pps()?;
                let records = self
                    .decoder
                    .inner
                    .read_records_range(&vid, tier_val, min_rec, count)?;
                let frames = decoder::decode_gop_rgb24_batch(
                    &sps_pps,
                    &records,
                    &mut self.decoder.decoder_pool,
                )?;

                stats.decode_batches += 1;
                stats.records_read += count;
                stats.max_batch_size = stats.max_batch_size.max(count);
                stats.total_frames += frames.len();

                for &(pos, sample_id, frame_idx, rec_idx) in cluster_slice {
                    let rel_idx = rec_idx - min_rec;
                    let intra_offset = (frame_idx as usize) % self.decoder.gop_size;
                    let flat_idx =
                        rel_idx * frames_per_record + (intra_offset % frames_per_record.max(1));

                    if flat_idx < frames.len() {
                        let f = &frames[flat_idx];
                        results.insert(
                            pos,
                            ScheduledFrame {
                                sample_id,
                                rgb_bytes: f.data.clone(),
                                width: f.width,
                                height: f.height,
                            },
                        );
                    } else {
                        return Err(format!(
                            "decoded frame index {} out of bounds for video={} tier={} record={}",
                            flat_idx, vid, tier_val, rec_idx
                        )
                        .into());
                    }
                }
            }
        }

        let ordered: Vec<ScheduledFrame> = (0..n)
            .map(|pos| {
                results.remove(&pos).ok_or_else(|| {
                    format!("scheduler produced no result for input position {}", pos)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((ordered, stats))
    }

    pub fn gop_size(&self) -> usize {
        self.decoder.gop_size
    }
    pub fn tier1_stride(&self) -> usize {
        self.decoder.tier1_stride
    }

    /// Global offset-based execution: resolves ALL requests to byte ranges,
    /// sorts by chunk offset, coalesces adjacent ranges where gap < threshold,
    /// then batch-decodes each coalesced range. This crosses (video_id, tier)
    /// boundaries — ideal for object store where per-GET cost dominates.
    pub fn execute_global(
        &mut self,
        trace: &[LogicalRequest],
    ) -> Result<(Vec<ScheduledFrame>, SchedulerStats), Box<dyn std::error::Error>> {
        let total_started = Instant::now();
        let resolve_started = Instant::now();
        let n = trace.len();
        let mut results: BTreeMap<usize, ScheduledFrame> = BTreeMap::new();

        // Step 1: Resolve trace → unique physical records
        let mut dedup: HashMap<
            (String, i32, usize),
            (
                u64,
                u64,
                String,
                i32,
                usize,
                i32,
                usize,
                String,
                Vec<(usize, i64, i32)>,
            ),
        > = HashMap::new();

        for (pos, req) in trace.iter().enumerate() {
            let gop_idx = (req.frame_idx as usize) / self.decoder.gop_size;
            let rec_idx = match req.tier {
                0 => 0,
                1 => gop_idx / self.decoder.tier1_stride,
                _ => gop_idx,
            };
            let total = self.decoder.inner.record_count_for(&req.video_id, req.tier);
            let rec_idx = rec_idx.min(total.saturating_sub(1));

            if total == 0 {
                return Err(format!(
                    "no records for video={} tier={} sample_id={}",
                    req.video_id, req.tier, req.sample_id
                )
                .into());
            }

            let key = (req.video_id.clone(), req.tier, rec_idx);
            if let Some(entry) = dedup.get_mut(&key) {
                entry.8.push((pos, req.sample_id, req.frame_idx));
                continue;
            }

            if let Some(loc) = self
                .decoder
                .inner
                .index
                .lookup_at(&req.video_id, req.tier, rec_idx)
            {
                let fpr = if req.tier <= 1 {
                    1
                } else if loc.dependency_kind == "idr" {
                    1
                } else {
                    self.decoder.gop_size
                };
                dedup.insert(
                    key,
                    (
                        loc.offset,
                        loc.length,
                        req.video_id.clone(),
                        req.tier,
                        rec_idx,
                        req.frame_idx,
                        fpr,
                        loc.dependency_kind,
                        vec![(pos, req.sample_id, req.frame_idx)],
                    ),
                );
            } else {
                return Err(format!(
                    "index lookup failed for video={} tier={} record={} sample_id={}",
                    req.video_id, req.tier, rec_idx, req.sample_id
                )
                .into());
            }
        }

        let unique_records = dedup.len();
        let useful_bytes: u64 = dedup.values().map(|r| r.1).sum();

        // Step 2: Collect, sort by offset
        let mut unique: Vec<(
            u64,
            u64,
            String,
            i32,
            usize,
            i32,
            usize,
            String,
            Vec<(usize, i64, i32)>,
        )> = dedup.into_iter().map(|(_, v)| v).collect();
        unique.sort_by_key(|r| r.0);
        let resolve_ns = resolve_started.elapsed().as_nanos() as u64;

        // Step 3: Build fetch plans with byte-gap coalescing
        let plan_started = Instant::now();
        let mut plans: Vec<(u64, u64, Vec<usize>)> = vec![];
        if let Some(thresh) = self.merge_threshold {
            let mut i = 0usize;
            while i < unique.len() {
                let mut lo = unique[i].0;
                let mut hi = lo + unique[i].1;
                let mut indices = vec![i];
                let mut j = i + 1;
                while j < unique.len() {
                    let gap = unique[j].0.saturating_sub(hi) as usize;
                    if gap > thresh {
                        break;
                    }
                    hi = (unique[j].0 + unique[j].1).max(hi);
                    indices.push(j);
                    j += 1;
                }
                plans.push((lo, hi - lo, indices));
                i = j;
            }
        } else {
            // None = no coalescing: each record is its own plan
            for i in 0..unique.len() {
                plans.push((unique[i].0, unique[i].1, vec![i]));
            }
        }
        let plan_fanout = |plan: &(u64, u64, Vec<usize>)| -> usize {
            plan.2.iter().map(|&index| unique[index].8.len()).sum()
        };
        match self.range_priority {
            RangePriority::Offset => {}
            RangePriority::Smallest => plans.sort_by_key(|plan| plan.1),
            RangePriority::Largest => {
                plans.sort_by(|left, right| right.1.cmp(&left.1));
            }
            RangePriority::Fanout => {
                plans.sort_by(|left, right| {
                    plan_fanout(right)
                        .cmp(&plan_fanout(left))
                        .then_with(|| left.0.cmp(&right.0))
                });
            }
            RangePriority::FanoutPerByte => {
                plans.sort_by(|left, right| {
                    let left_score = plan_fanout(left) as u128 * right.1 as u128;
                    let right_score = plan_fanout(right) as u128 * left.1 as u128;
                    right_score
                        .cmp(&left_score)
                        .then_with(|| left.0.cmp(&right.0))
                });
            }
        }
        let plan_ns = plan_started.elapsed().as_nanos() as u64;

        // Step 4: Execute plans — read byte range, extract only requested records
        let mut stats = SchedulerStats {
            total_requests: n,
            unique_records,
            physical_ranges: plans.len(),
            fetched_bytes: 0,
            useful_bytes,
            overfetch_bytes: 0,
            records_decoded: 0,
            decode_batches: 0,
            total_frames: 0,
            max_fetch_records: 0,
            coalesced_gets: plans.len(),
            uncoalesced_gets: unique_records,
            bytes_read: 0,
            uncoalesced_bytes: useful_bytes,
            records_read: unique_records,
            max_batch_size: 0,
            resolve_ns,
            plan_ns,
            fetch_wall_ns: 0,
            fetch_service_ns_sum: 0,
            range_queue_ns_sum: 0,
            extract_ns: 0,
            decode_ns: 0,
            reorder_ns: 0,
            total_ns: 0,
            fetch_decode_overlap_ns: 0,
            time_to_first_ready_target_ns: 0,
        };

        let extract_plan = |plan_index: usize,
                            buffer: &[u8],
                            stats: &mut SchedulerStats|
         -> Result<Vec<GlobalDecodeGroup>, Box<dyn std::error::Error>> {
            let (plan_offset, plan_length, indices) = plans
                .get(plan_index)
                .ok_or_else(|| format!("backend returned unknown plan index {plan_index}"))?;
            if buffer.len() != *plan_length as usize {
                return Err(format!(
                    "short plan buffer for index {}: {} != {}",
                    plan_index,
                    buffer.len(),
                    plan_length
                )
                .into());
            }
            stats.fetched_bytes += *plan_length;
            stats.max_fetch_records = stats.max_fetch_records.max(indices.len());
            let mut groups: Vec<GlobalDecodeGroup> = Vec::new();
            for &index in indices {
                let offset = unique[index].0;
                let length = unique[index].1;
                let video_id = &unique[index].2;
                let tier = unique[index].3;
                let frames_per_record = unique[index].6;
                let consumers = unique[index].8.clone();
                let start = offset.saturating_sub(*plan_offset) as usize;
                if start + length as usize > buffer.len() {
                    return Err("record slice out of fetch buffer".into());
                }
                let record = GlobalDecodeRecord {
                    video_id: video_id.clone(),
                    tier,
                    bytes: buffer[start..start + length as usize].to_vec(),
                    consumers,
                };
                if let Some(group) = groups.iter_mut().find(|group| {
                    group.video_id == *video_id
                        && group.tier == tier
                        && group.frames_per_record == frames_per_record
                }) {
                    group.records.push(record);
                } else {
                    groups.push(GlobalDecodeGroup {
                        video_id: video_id.clone(),
                        tier,
                        frames_per_record,
                        records: vec![record],
                    });
                }
            }
            Ok(groups)
        };

        let ranges: Vec<_> = plans
            .iter()
            .map(|(offset, length, _)| (*offset, *length))
            .collect();
        let fetch_started = Instant::now();
        if self.completion_driven && self.backend.is_some() {
            let backend = self.backend.as_ref().expect("checked above");
            let decoder_state = &mut self.decoder;
            let microbatch_records = self.decode_microbatch_records;
            let mut pending_groups: Vec<GlobalDecodeGroup> = Vec::new();
            backend.for_each_byte_range(&ranges, &mut |completed| {
                stats.fetch_wall_ns = stats.fetch_wall_ns.max(completed.completed_ns);
                stats.fetch_service_ns_sum +=
                    completed.completed_ns.saturating_sub(completed.started_ns);
                stats.range_queue_ns_sum += completed.started_ns;

                let extract_started = Instant::now();
                let groups = extract_plan(completed.index, &completed.bytes, &mut stats)?;
                stats.extract_ns += extract_started.elapsed().as_nanos() as u64;

                merge_global_groups(&mut pending_groups, groups);
                while global_group_record_count(&pending_groups) >= microbatch_records {
                    let decode_groups =
                        drain_global_microbatch(&mut pending_groups, microbatch_records);
                    let decode_started = Instant::now();
                    decode_global_groups(decoder_state, &decode_groups, &mut results, &mut stats)?;
                    stats.decode_ns += decode_started.elapsed().as_nanos() as u64;
                    if stats.time_to_first_ready_target_ns == 0 && !results.is_empty() {
                        stats.time_to_first_ready_target_ns =
                            total_started.elapsed().as_nanos() as u64;
                    }
                }
                Ok(())
            })?;
            while !pending_groups.is_empty() {
                let decode_groups =
                    drain_global_microbatch(&mut pending_groups, microbatch_records);
                let decode_started = Instant::now();
                decode_global_groups(decoder_state, &decode_groups, &mut results, &mut stats)?;
                stats.decode_ns += decode_started.elapsed().as_nanos() as u64;
                if stats.time_to_first_ready_target_ns == 0 && !results.is_empty() {
                    stats.time_to_first_ready_target_ns = total_started.elapsed().as_nanos() as u64;
                }
            }
        } else {
            let mut completed_ranges: Vec<Option<crate::backend::CompletedRange>> =
                std::iter::repeat_with(|| None).take(ranges.len()).collect();
            if let Some(ref backend) = self.backend {
                backend.for_each_byte_range(&ranges, &mut |completed| {
                    stats.fetch_wall_ns = stats.fetch_wall_ns.max(completed.completed_ns);
                    stats.fetch_service_ns_sum +=
                        completed.completed_ns.saturating_sub(completed.started_ns);
                    stats.range_queue_ns_sum += completed.started_ns;
                    let index = completed.index;
                    completed_ranges[index] = Some(completed);
                    Ok(())
                })?;
            } else {
                for (index, &(offset, length)) in ranges.iter().enumerate() {
                    let range_started = fetch_started.elapsed().as_nanos() as u64;
                    let bytes = self.decoder.inner.read_byte_range(offset, length)?;
                    let completed_ns = fetch_started.elapsed().as_nanos() as u64;
                    stats.fetch_service_ns_sum += completed_ns.saturating_sub(range_started);
                    completed_ranges[index] = Some(crate::backend::CompletedRange {
                        index,
                        physical_request_id: index as u64 + 1,
                        physical_object_offset: offset,
                        physical_object_length: length,
                        physical_requests: 1,
                        physical_fetched_bytes: bytes.len() as u64,
                        bytes,
                        started_ns: range_started,
                        first_byte_ns: completed_ns,
                        completed_ns,
                    });
                }
                stats.fetch_wall_ns = fetch_started.elapsed().as_nanos() as u64;
            }

            let extract_started = Instant::now();
            let mut all_groups = Vec::new();
            for (index, completed) in completed_ranges.into_iter().enumerate() {
                let completed =
                    completed.ok_or_else(|| format!("backend omitted plan index {index}"))?;
                merge_global_groups(
                    &mut all_groups,
                    extract_plan(index, &completed.bytes, &mut stats)?,
                );
            }
            stats.extract_ns = extract_started.elapsed().as_nanos() as u64;

            let decode_started = Instant::now();
            decode_global_groups(&mut self.decoder, &all_groups, &mut results, &mut stats)?;
            stats.decode_ns = decode_started.elapsed().as_nanos() as u64;
            if !results.is_empty() {
                stats.time_to_first_ready_target_ns = total_started.elapsed().as_nanos() as u64;
            }
        }
        stats.overfetch_bytes = stats.fetched_bytes.saturating_sub(useful_bytes);

        // Step 6: Return in original order
        let reorder_started = Instant::now();
        let ordered: Vec<ScheduledFrame> = (0..n)
            .map(|pos| {
                results.remove(&pos).ok_or_else(|| {
                    format!("scheduler produced no result for input position {}", pos)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        stats.reorder_ns = reorder_started.elapsed().as_nanos() as u64;
        // Sync legacy stats fields
        stats.bytes_read = stats.fetched_bytes;
        stats.uncoalesced_bytes = stats.useful_bytes;
        stats.records_read = stats.records_decoded;
        stats.total_ns = total_started.elapsed().as_nanos() as u64;
        let measured_pipeline_ns = stats
            .total_ns
            .saturating_sub(stats.resolve_ns)
            .saturating_sub(stats.plan_ns)
            .saturating_sub(stats.reorder_ns);
        stats.fetch_decode_overlap_ns = stats
            .fetch_wall_ns
            .saturating_add(stats.extract_ns)
            .saturating_add(stats.decode_ns)
            .saturating_sub(measured_pipeline_ns);

        Ok((ordered, stats))
    }
}

/// Split sorted requests (by record_idx) into clusters using byte-gap cost model.
/// `ranges[i]` = (offset, length) for record at index i.
/// Two consecutive records are merged if the byte gap between them < threshold.
#[cfg(feature = "ffmpeg")]
fn cluster_by_byte_gap(
    reqs: &[(usize, i64, i32, usize)],
    ranges: &[(u64, u64)],
    threshold: usize,
) -> Vec<(usize, usize)> {
    if reqs.is_empty() {
        return vec![];
    }
    let mut clusters = vec![];
    let mut start = 0;
    for i in 1..reqs.len() {
        let prev_idx = reqs[i - 1].3;
        let curr_idx = reqs[i].3;
        let prev_end = ranges[prev_idx].0 as usize + ranges[prev_idx].1 as usize;
        let curr_start = ranges[curr_idx].0 as usize;
        let gap = curr_start.saturating_sub(prev_end);
        if gap > threshold {
            clusters.push((start, i));
            start = i;
        }
    }
    clusters.push((start, reqs.len()));
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{CompletedRange, StorageBackend};
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    struct ReverseBackend {
        payload: Vec<u8>,
    }

    impl StorageBackend for ReverseBackend {
        fn read_byte_range(
            &self,
            offset: u64,
            length: u64,
        ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let start = offset as usize;
            let end = start + length as usize;
            Ok(self.payload[start..end].to_vec())
        }

        fn for_each_byte_range(
            &self,
            ranges: &[(u64, u64)],
            callback: &mut dyn FnMut(CompletedRange) -> Result<(), Box<dyn std::error::Error>>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let started = Instant::now();
            for index in (0..ranges.len()).rev() {
                let range_started = started.elapsed().as_nanos() as u64;
                let (offset, length) = ranges[index];
                let bytes = self.read_byte_range(offset, length)?;
                callback(CompletedRange {
                    index,
                    physical_request_id: index as u64 + 1,
                    physical_object_offset: offset,
                    physical_object_length: length,
                    physical_requests: 1,
                    physical_fetched_bytes: bytes.len() as u64,
                    bytes,
                    started_ns: range_started,
                    first_byte_ns: started.elapsed().as_nanos() as u64,
                    completed_ns: started.elapsed().as_nanos() as u64,
                })?;
            }
            Ok(())
        }
    }

    fn open_smoke_chunk() -> Option<chunk::ChunkReader> {
        let path = std::env::var("VCLASP_TEST_CHUNK").ok()?;
        chunk::ChunkReader::open(Path::new(&path)).ok()
    }

    #[test]
    fn test_infer_gop_size_idr_only() {
        let reader = open_smoke_chunk();
        if reader.is_none() {
            return;
        }
        let gop = reader.unwrap().index.infer_gop_size();
        assert_eq!(gop, 8, "IdrOnly chunk has GOP8 frame_idx deltas");
    }

    #[test]
    fn test_infer_tier1_stride() {
        let reader = open_smoke_chunk();
        if reader.is_none() {
            return;
        }
        let stride = reader.unwrap().index.infer_tier1_stride(8);
        assert_eq!(stride, 4, "tier1_stride should be 4");
    }

    #[test]
    fn test_scheduler_creation() {
        let reader = open_smoke_chunk();
        if reader.is_none() {
            return;
        }
        let sched = LogicalBatchDecoder::new(reader.unwrap(), "test.chunk".into());
        assert!(sched.is_ok());
        let s = sched.unwrap();
        assert_eq!(s.gop_size(), 8);
        assert_eq!(s.tier1_stride(), 4);
    }

    #[test]
    fn test_scheduler_rejects_missing_records_instead_of_returning_black_frame() {
        let reader = open_smoke_chunk();
        if reader.is_none() {
            return;
        }
        let decoder = LogicalBatchDecoder::new(reader.unwrap(), "test.chunk".into()).unwrap();
        let mut scheduler = LogicalScheduler::new(decoder, Some(0));
        let request = LogicalRequest {
            sample_id: 7,
            video_id: "missing/video.mp4".to_string(),
            tier: 2,
            frame_idx: 8,
        };
        let error = scheduler
            .execute_global(&[request])
            .unwrap_err()
            .to_string();
        assert!(error.contains("no records"), "unexpected error: {}", error);
    }

    #[test]
    fn test_legacy_schedule_rejects_missing_records() {
        let reader = open_smoke_chunk();
        if reader.is_none() {
            return;
        }
        let mut decoder = LogicalBatchDecoder::new(reader.unwrap(), "test.chunk".into()).unwrap();
        let request = LogicalRequest {
            sample_id: 7,
            video_id: "missing/video.mp4".to_string(),
            tier: 2,
            frame_idx: 8,
        };
        let error = decoder.schedule(&[request]).unwrap_err().to_string();
        assert!(error.contains("no records"), "unexpected error: {}", error);
    }

    #[test]
    fn test_legacy_schedule_rejects_out_of_range_frame() {
        let reader = open_smoke_chunk();
        if reader.is_none() {
            return;
        }
        let mut decoder = LogicalBatchDecoder::new(reader.unwrap(), "test.chunk".into()).unwrap();
        let video_id = decoder
            .inner
            .index
            .video_entries()
            .keys()
            .next()
            .cloned()
            .expect("smoke chunk has no videos");
        let request = LogicalRequest {
            sample_id: 8,
            video_id,
            tier: 2,
            frame_idx: i32::MAX,
        };
        let error = decoder.schedule(&[request]).unwrap_err().to_string();
        assert!(
            error.contains("out of range"),
            "unexpected error: {}",
            error
        );
    }

    #[test]
    fn test_completion_driven_matches_barrier_with_out_of_order_ranges() {
        let Some(path_value) = std::env::var("VCLASP_TEST_CHUNK").ok() else {
            return;
        };
        let path = Path::new(&path_value);
        let first_reader = chunk::ChunkReader::open(path).unwrap();
        let video_ids: Vec<_> = first_reader
            .index
            .video_entries()
            .keys()
            .take(8)
            .cloned()
            .collect();
        let bytes = fs::read(path).unwrap();
        let payload = bytes[first_reader.layout.payload_start..].to_vec();
        let trace: Vec<_> = video_ids
            .iter()
            .enumerate()
            .map(|(index, video_id)| LogicalRequest {
                sample_id: index as i64,
                video_id: video_id.clone(),
                tier: 0,
                frame_idx: 0,
            })
            .collect();

        let barrier_decoder =
            LogicalBatchDecoder::new(first_reader, path.display().to_string()).unwrap();
        let mut barrier = LogicalScheduler::new(barrier_decoder, Some(0)).with_backend(Box::new(
            ReverseBackend {
                payload: payload.clone(),
            },
        ));
        let (barrier_frames, barrier_stats) = barrier.execute_global(&trace).unwrap();

        let completion_reader = chunk::ChunkReader::open(path).unwrap();
        let completion_decoder =
            LogicalBatchDecoder::new(completion_reader, path.display().to_string()).unwrap();
        let mut completion = LogicalScheduler::new(completion_decoder, Some(0))
            .with_backend(Box::new(ReverseBackend { payload }))
            .with_completion_driven(true);
        let (completion_frames, completion_stats) = completion.execute_global(&trace).unwrap();

        assert_eq!(
            barrier_frames
                .iter()
                .map(|frame| frame.sample_id)
                .collect::<Vec<_>>(),
            completion_frames
                .iter()
                .map(|frame| frame.sample_id)
                .collect::<Vec<_>>()
        );
        for (left, right) in barrier_frames.iter().zip(&completion_frames) {
            assert_eq!(left.rgb_bytes, right.rgb_bytes);
            assert_eq!((left.width, left.height), (right.width, right.height));
        }
        assert_eq!(barrier_stats.useful_bytes, completion_stats.useful_bytes);
        assert_eq!(barrier_stats.fetched_bytes, completion_stats.fetched_bytes);
        assert_eq!(
            barrier_stats.physical_ranges,
            completion_stats.physical_ranges
        );
        assert!(completion_stats.time_to_first_ready_target_ns > 0);
        assert!(completion_stats.total_ns >= completion_stats.time_to_first_ready_target_ns);
    }

    #[test]
    fn test_decode_microbatch_is_independent_of_fetch_group_size() {
        let records = (0..10)
            .map(|index| GlobalDecodeRecord {
                video_id: "video".to_string(),
                tier: 2,
                bytes: vec![index],
                consumers: vec![(index as usize, index as i64, index as i32)],
            })
            .collect();
        let mut pending = vec![GlobalDecodeGroup {
            video_id: "video".to_string(),
            tier: 2,
            frames_per_record: 2,
            records,
        }];
        let first = drain_global_microbatch(&mut pending, 4);
        assert_eq!(global_group_record_count(&first), 4);
        assert_eq!(global_group_record_count(&pending), 6);
        let second = drain_global_microbatch(&mut pending, 4);
        assert_eq!(global_group_record_count(&second), 4);
        assert_eq!(global_group_record_count(&pending), 2);
    }
}
