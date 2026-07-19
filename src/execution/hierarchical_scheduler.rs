use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use crate::backend::StorageBackend;
use crate::decoder::{self, DecodedRgbFrame, DecoderConfig, DecoderPool, SharedDecoderSlots};
use crate::hierarchical_ingest::{
    mp4_sample_to_annex_b, HierarchicalCatalog, HierarchicalRecordMeta,
};
use crate::hierarchical_layout::{
    DependencyLookaheadDecision, HierarchicalCostModel, HierarchicalLayoutIndex,
    HierarchicalReadMode,
};
use crate::planner::{self, RecordRange};

#[derive(Debug, Clone)]
pub struct LogicalTarget {
    pub sample_id: u64,
    pub video_id: String,
    pub frame_idx: i32,
}

#[derive(Clone)]
pub struct HierarchicalOutput {
    pub sample_id: u64,
    pub frame: DecodedRgbFrame,
}

pub struct HierarchicalIncrementalWindow {
    pub batches: Vec<Vec<HierarchicalOutput>>,
    pub batch_ready_ns: Vec<u64>,
    pub ordered_delivery_ns: Vec<u64>,
    pub stats: HierarchicalBatchStats,
}

#[derive(Debug, Clone)]
pub struct HierarchicalBatchStats {
    pub logical_targets: usize,
    pub unique_targets: usize,
    pub physical_ranges: usize,
    pub client_requests: usize,
    pub useful_bytes: u64,
    pub fetched_bytes: u64,
    pub submitted_access_units: usize,
    pub decode_groups: usize,
    pub streaming_cache_hits: usize,
    pub streaming_cache_misses: usize,
    pub streaming_cache_resident_bytes: u64,
    pub streaming_cache_budget_bytes: u64,
    pub decoder_state_resets: usize,
    pub plan_ns: u64,
    pub fetch_ns: u64,
    pub assemble_ns: u64,
    pub decode_ns: u64,
    pub total_ns: u64,
    pub predicted_total_ns: f64,
    pub mode: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub enum HierarchicalAction {
    Adaptive,
    ExactClosure,
    FixedGapClosure(u64),
    RegionSelective,
    RegionAll,
}

struct ResolvedPlan {
    ranges: Vec<(u64, u64)>,
    selected_ids: HashSet<u64>,
    useful_bytes: u64,
    predicted_total_ns: f64,
    mode: &'static str,
}

pub struct HierarchicalBatchExecutor {
    catalog: HierarchicalCatalog,
    layout: HierarchicalLayoutIndex,
    backend: Box<dyn StorageBackend>,
    codec_config: Vec<u8>,
    model: HierarchicalCostModel,
    max_merge_gap_bytes: Option<u64>,
    max_range_bytes: Option<u64>,
    decoder_pool: DecoderPool,
    incremental_decoder_slots: SharedDecoderSlots,
    incremental_batch_deadline_fences: bool,
    streaming_states: HashMap<(String, u64), StreamingGopState>,
    streaming_lru: VecDeque<(String, u64)>,
    streaming_cache_bytes: usize,
    streaming_resident_bytes: usize,
    streaming_min_contiguous_targets: usize,
}

struct StreamingGopState {
    record: Vec<u8>,
    cursor: decoder::PrefixCursor,
}

struct IncrementalDecodeJob {
    batch_index: usize,
    requested: Vec<(u64, usize)>,
    codec_config: Vec<u8>,
    vcl_record: Vec<u8>,
    access_units: usize,
}

struct IncrementalDecodedJob {
    frames: Vec<(u64, DecodedRgbFrame)>,
    completed_ns: u64,
    access_units: usize,
}

impl HierarchicalBatchExecutor {
    pub fn plan_dependency_sampler(
        &self,
        base_order: &[u64],
        batch_size: usize,
        lookahead_samples: usize,
    ) -> Result<crate::dependency_sampler::DependencySamplerPlan, String> {
        crate::dependency_sampler::plan_bounded_dependency_batches(
            &self.layout,
            base_order,
            batch_size,
            lookahead_samples,
            self.max_merge_gap_bytes,
            self.max_range_bytes,
            &self.model,
        )
    }

    pub fn plan_label_preserving_dependency_sampler(
        &self,
        base_order: &[crate::dependency_sampler::LabeledDependencySample],
        batch_size: usize,
        lookahead_batches: usize,
    ) -> Result<crate::dependency_sampler::DependencySamplerPlan, String> {
        crate::dependency_sampler::plan_label_preserving_dependency_batches(
            &self.layout,
            base_order,
            batch_size,
            lookahead_batches,
            self.max_merge_gap_bytes,
            self.max_range_bytes,
            &self.model,
        )
    }

    pub fn plan_label_preserving_logical_sampler(
        &self,
        base_order: &[(u64, i64, Vec<(String, i32)>)],
        batch_size: usize,
        lookahead_batches: usize,
    ) -> Result<crate::dependency_sampler::DependencySamplerPlan, String> {
        let samples = base_order
            .iter()
            .map(|(logical_id, class_id, targets)| {
                let target_ids = targets
                    .iter()
                    .map(|(video_id, frame_idx)| {
                        self.catalog
                            .target(video_id, *frame_idx)
                            .map(|record| record.record_id)
                            .ok_or_else(|| {
                                format!(
                                    "logical sample {logical_id} references unknown target ({video_id}, {frame_idx})"
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(crate::dependency_sampler::LabeledDependencySample {
                    logical_id: *logical_id,
                    class_id: *class_id,
                    target_ids,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        self.plan_label_preserving_dependency_sampler(&samples, batch_size, lookahead_batches)
    }

    pub fn choose_lookahead(
        &self,
        batches: &[Vec<LogicalTarget>],
        candidates: &[usize],
        first_batch_slo_ns: f64,
    ) -> Result<DependencyLookaheadDecision, String> {
        let record_batches = batches
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|target| {
                        self.catalog
                            .target(&target.video_id, target.frame_idx)
                            .map(|record| record.record_id)
                            .ok_or_else(|| {
                                format!(
                                    "unknown hierarchical target ({}, {})",
                                    target.video_id, target.frame_idx
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.layout.choose_lookahead(
            &record_batches,
            candidates,
            first_batch_slo_ns,
            self.max_merge_gap_bytes,
            self.max_range_bytes,
            &self.model,
        )
    }

    fn decode_incremental_jobs(
        decoder_slots: &SharedDecoderSlots,
        mut jobs: Vec<IncrementalDecodeJob>,
        total_started: &Instant,
    ) -> Result<Vec<IncrementalDecodedJob>, String> {
        jobs.sort_unstable_by_key(|job| job.batch_index);
        let mut partitions: Vec<Vec<IncrementalDecodeJob>> =
            (0..decoder_slots.len()).map(|_| Vec::new()).collect();
        for (index, job) in jobs.into_iter().enumerate() {
            partitions[index % decoder_slots.len()].push(job);
        }
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (slot, jobs) in decoder_slots.iter().zip(partitions) {
                if jobs.is_empty() {
                    continue;
                }
                handles.push(scope.spawn(move || {
                    let mut pool = slot
                        .lock()
                        .map_err(|_| "hierarchical decoder slot lock poisoned".to_string())?;
                    let mut decoded = Vec::with_capacity(jobs.len());
                    for job in jobs {
                        let ordinals = job
                            .requested
                            .iter()
                            .map(|(_, ordinal)| *ordinal)
                            .collect::<Vec<_>>();
                        let frames = decoder::decode_full_gop_selected_rgb24(
                            &job.codec_config,
                            &job.vcl_record,
                            &mut pool,
                            &ordinals,
                        )
                        .map_err(|error| error.to_string())?;
                        if frames.len() != job.requested.len() {
                            return Err(format!(
                                "decoded {} hierarchical targets from {} requests",
                                frames.len(),
                                job.requested.len()
                            ));
                        }
                        decoded.push(IncrementalDecodedJob {
                            frames: job
                                .requested
                                .into_iter()
                                .zip(frames)
                                .map(|((record_id, _), frame)| (record_id, frame))
                                .collect(),
                            completed_ns: total_started.elapsed().as_nanos() as u64,
                            access_units: job.access_units,
                        });
                    }
                    Ok::<_, String>(decoded)
                }));
            }
            let mut decoded = Vec::new();
            for handle in handles {
                decoded.extend(
                    handle
                        .join()
                        .map_err(|_| "hierarchical decoder worker panicked".to_string())??,
                );
            }
            Ok::<_, String>(decoded)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        catalog: HierarchicalCatalog,
        backend: Box<dyn StorageBackend>,
        codec_config: Vec<u8>,
        model: HierarchicalCostModel,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        decoder_threads: usize,
        incremental_decode_slots: usize,
        incremental_batch_deadline_fences: bool,
        streaming_cache_bytes: usize,
        streaming_min_contiguous_targets: usize,
    ) -> Result<Self, String> {
        if codec_config.is_empty() {
            return Err("hierarchical executor requires codec configuration".to_string());
        }
        let layout = catalog.to_layout_index()?;
        model.validate()?;
        if streaming_cache_bytes > 0 && streaming_min_contiguous_targets == 0 {
            return Err("streaming target threshold must be positive".to_string());
        }
        if incremental_decode_slots == 0 {
            return Err("incremental decoder slots must be positive".to_string());
        }
        Ok(Self {
            catalog,
            layout,
            backend,
            codec_config,
            model,
            max_merge_gap_bytes,
            max_range_bytes,
            decoder_pool: DecoderPool::new(DecoderConfig {
                num_threads: decoder_threads,
            }),
            incremental_decoder_slots: decoder::shared_decoder_slots(incremental_decode_slots),
            incremental_batch_deadline_fences,
            streaming_states: HashMap::new(),
            streaming_lru: VecDeque::new(),
            streaming_cache_bytes,
            streaming_resident_bytes: 0,
            streaming_min_contiguous_targets,
        })
    }

    fn streaming_key(&self, target_records: &[HierarchicalRecordMeta]) -> Option<(String, u64)> {
        if self.streaming_cache_bytes == 0
            || target_records.len() < self.streaming_min_contiguous_targets
        {
            return None;
        }
        let first = target_records.first()?;
        if target_records
            .iter()
            .any(|record| record.video_id != first.video_id || record.gop_id != first.gop_id)
        {
            return None;
        }
        let mut frames = target_records
            .iter()
            .map(|record| record.frame_idx)
            .collect::<Vec<_>>();
        frames.sort_unstable();
        frames.dedup();
        (frames.len() == target_records.len()
            && frames.windows(2).all(|pair| pair[1] == pair[0] + 1))
        .then(|| (first.video_id.clone(), first.gop_id))
    }

    fn execute_mixed_streaming(
        &mut self,
        targets: &[LogicalTarget],
        target_records: &[HierarchicalRecordMeta],
        total_started: Instant,
    ) -> Result<Option<(Vec<HierarchicalOutput>, HierarchicalBatchStats)>, String> {
        let mut order = Vec::<(String, u64)>::new();
        let mut groups =
            HashMap::<(String, u64), (Vec<LogicalTarget>, Vec<HierarchicalRecordMeta>)>::new();
        for (target, record) in targets.iter().zip(target_records) {
            let key = (record.video_id.clone(), record.gop_id);
            if !groups.contains_key(&key) {
                order.push(key.clone());
            }
            let group = groups.entry(key).or_default();
            group.0.push(target.clone());
            group.1.push(record.clone());
        }
        // Splitting an all-miss multi-GOP batch destroys cross-GOP range
        // coalescing. Use mixed streaming only when it can reuse at least one
        // resident decoder cursor; otherwise let the unified span planner
        // fetch adjacent GOP regions together.
        if order.len() < 2
            || !order.iter().any(|key| {
                self.streaming_states.contains_key(key)
                    && self.streaming_key(&groups[key].1).is_some()
            })
        {
            return Ok(None);
        }

        let mut decoded = HashMap::<u64, HierarchicalOutput>::new();
        let mut stats = HierarchicalBatchStats {
            logical_targets: 0,
            unique_targets: 0,
            physical_ranges: 0,
            client_requests: 0,
            useful_bytes: 0,
            fetched_bytes: 0,
            submitted_access_units: 0,
            decode_groups: 0,
            streaming_cache_hits: 0,
            streaming_cache_misses: 0,
            streaming_cache_resident_bytes: 0,
            streaming_cache_budget_bytes: self.streaming_cache_bytes as u64,
            decoder_state_resets: 0,
            plan_ns: 0,
            fetch_ns: 0,
            assemble_ns: 0,
            decode_ns: 0,
            total_ns: 0,
            predicted_total_ns: 0.0,
            mode: "mixed_streaming_adaptive",
        };
        let mut prediction_available = true;
        for key in order {
            let (group_targets, group_records) = groups.remove(&key).unwrap();
            let group_started = Instant::now();
            let (outputs, part) = if let Some(streaming_key) = self.streaming_key(&group_records) {
                self.execute_streaming(
                    &group_targets,
                    &group_records,
                    streaming_key,
                    group_started,
                )?
            } else {
                self.execute_action(&group_targets, HierarchicalAction::Adaptive)?
            };
            for output in outputs {
                decoded.insert(output.sample_id, output);
            }
            stats.logical_targets += part.logical_targets;
            stats.unique_targets += part.unique_targets;
            stats.physical_ranges += part.physical_ranges;
            stats.client_requests += part.client_requests;
            stats.useful_bytes += part.useful_bytes;
            stats.fetched_bytes += part.fetched_bytes;
            stats.submitted_access_units += part.submitted_access_units;
            stats.decode_groups += part.decode_groups;
            stats.streaming_cache_hits += part.streaming_cache_hits;
            stats.streaming_cache_misses += part.streaming_cache_misses;
            stats.decoder_state_resets += part.decoder_state_resets;
            stats.plan_ns += part.plan_ns;
            stats.fetch_ns += part.fetch_ns;
            stats.assemble_ns += part.assemble_ns;
            stats.decode_ns += part.decode_ns;
            if part.predicted_total_ns < 0.0 {
                prediction_available = false;
            } else {
                stats.predicted_total_ns += part.predicted_total_ns;
            }
        }
        stats.streaming_cache_resident_bytes = self.streaming_resident_bytes as u64;
        stats.total_ns = total_started.elapsed().as_nanos() as u64;
        if !prediction_available {
            stats.predicted_total_ns = -1.0;
        }
        let outputs = targets
            .iter()
            .map(|target| {
                decoded
                    .get(&target.sample_id)
                    .cloned()
                    .ok_or_else(|| format!("mixed streaming missed target {}", target.sample_id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some((outputs, stats)))
    }

    fn touch_streaming_key(&mut self, key: &(String, u64)) {
        self.streaming_lru.retain(|value| value != key);
        self.streaming_lru.push_back(key.clone());
    }

    fn execute_streaming(
        &mut self,
        targets: &[LogicalTarget],
        target_records: &[HierarchicalRecordMeta],
        key: (String, u64),
        total_started: Instant,
    ) -> Result<(Vec<HierarchicalOutput>, HierarchicalBatchStats), String> {
        let plan_started = Instant::now();
        let record_ids = self
            .catalog
            .records_for_gop(&key.0, key.1)
            .ok_or_else(|| format!("missing streaming GOP ({}, {})", key.0, key.1))?
            .to_vec();
        let records = record_ids
            .iter()
            .map(|record_id| self.catalog.record(*record_id).unwrap().clone())
            .collect::<Vec<_>>();
        let offset = records.iter().map(|record| record.offset).min().unwrap();
        let end = records
            .iter()
            .map(|record| record.offset + record.length)
            .max()
            .unwrap();
        let plan_ns = plan_started.elapsed().as_nanos() as u64;
        let useful_ids = target_records
            .iter()
            .flat_map(|record| record.closure_record_ids.iter().copied())
            .collect::<HashSet<_>>();
        let useful_bytes = useful_ids
            .iter()
            .map(|record_id| self.catalog.record(*record_id).unwrap().length)
            .sum();
        let mut fetch_ns = 0u64;
        let mut assemble_ns = 0u64;
        let mut fetched_bytes = 0u64;
        let mut physical_ranges = 0usize;
        let mut client_requests = 0usize;
        let cache_hit = self.streaming_states.contains_key(&key);
        if !cache_hit {
            let ranges = [(offset, end - offset)];
            let fetch_started = Instant::now();
            let buffers = self
                .backend
                .read_byte_ranges(&ranges)
                .map_err(|error| error.to_string())?;
            fetch_ns = fetch_started.elapsed().as_nanos() as u64;
            fetched_bytes = buffers.iter().map(|buffer| buffer.len() as u64).sum();
            physical_ranges = self.backend.physical_ranges_for_ranges(&ranges);
            client_requests = self.backend.client_requests_for_ranges(&ranges);
            let assemble_started = Instant::now();
            let mut annex_b = Vec::new();
            for record in &records {
                let sample = Self::extract_record(record, &ranges, &buffers)?;
                annex_b.extend_from_slice(&mp4_sample_to_annex_b(&sample, record.nal_length_size)?);
            }
            let mut self_contained = Vec::with_capacity(self.codec_config.len() + annex_b.len());
            self_contained.extend_from_slice(&self.codec_config);
            self_contained.extend_from_slice(&annex_b);
            let (codec_config, vcl_record, _) =
                decoder::extract_closed_record_parts(&self_contained)
                    .map_err(|error| error.to_string())?;
            if codec_config != self.codec_config {
                return Err("streaming GOP codec configuration mismatch".to_string());
            }
            if vcl_record.len() > self.streaming_cache_bytes {
                return Err(format!(
                    "streaming GOP requires {} bytes, exceeding cache budget {}",
                    vcl_record.len(),
                    self.streaming_cache_bytes
                ));
            }
            while self.streaming_resident_bytes + vcl_record.len() > self.streaming_cache_bytes {
                let evicted = self
                    .streaming_lru
                    .pop_front()
                    .ok_or("streaming cache accounting underflow")?;
                if let Some(state) = self.streaming_states.remove(&evicted) {
                    self.streaming_resident_bytes -= state.record.len();
                }
            }
            let cursor = decoder::PrefixCursor::new(
                &self.codec_config,
                &vcl_record,
                DecoderConfig { num_threads: 1 },
            )
            .map_err(|error| error.to_string())?;
            self.streaming_resident_bytes += vcl_record.len();
            self.streaming_states.insert(
                key.clone(),
                StreamingGopState {
                    record: vcl_record,
                    cursor,
                },
            );
            assemble_ns = assemble_started.elapsed().as_nanos() as u64;
        }
        self.touch_streaming_key(&key);

        let mut display_records = records.iter().collect::<Vec<_>>();
        display_records.sort_unstable_by_key(|record| record.frame_idx);
        let ordinal_by_id = display_records
            .iter()
            .enumerate()
            .map(|(ordinal, record)| (record.record_id, ordinal))
            .collect::<HashMap<_, _>>();
        let mut unique_target_ids = Vec::new();
        let mut seen = HashSet::new();
        for record in target_records {
            if seen.insert(record.record_id) {
                unique_target_ids.push(record.record_id);
            }
        }
        let ordinals = unique_target_ids
            .iter()
            .map(|record_id| ordinal_by_id[record_id])
            .collect::<Vec<_>>();
        let decode_started = Instant::now();
        let state = self.streaming_states.get_mut(&key).unwrap();
        let (submitted_access_units, expected_reset) = state.cursor.preview_ordinals(&ordinals);
        let (frames, decoder_reset) = state
            .cursor
            .decode_selected(&self.codec_config, &state.record, &ordinals)
            .map_err(|error| error.to_string())?;
        debug_assert_eq!(decoder_reset, expected_reset);
        let decode_ns = decode_started.elapsed().as_nanos() as u64;
        let decoded = unique_target_ids
            .into_iter()
            .zip(frames)
            .collect::<HashMap<_, _>>();
        let outputs = targets
            .iter()
            .zip(target_records)
            .map(|(target, record)| HierarchicalOutput {
                sample_id: target.sample_id,
                frame: decoded[&record.record_id].clone(),
            })
            .collect();
        Ok((
            outputs,
            HierarchicalBatchStats {
                logical_targets: targets.len(),
                unique_targets: seen.len(),
                physical_ranges,
                client_requests,
                useful_bytes,
                fetched_bytes,
                submitted_access_units,
                decode_groups: 1,
                streaming_cache_hits: usize::from(cache_hit),
                streaming_cache_misses: usize::from(!cache_hit),
                streaming_cache_resident_bytes: self.streaming_resident_bytes as u64,
                streaming_cache_budget_bytes: self.streaming_cache_bytes as u64,
                decoder_state_resets: usize::from(decoder_reset),
                plan_ns,
                fetch_ns,
                assemble_ns,
                decode_ns,
                total_ns: total_started.elapsed().as_nanos() as u64,
                predicted_total_ns: -1.0,
                mode: if cache_hit {
                    "streaming_cursor_hit"
                } else {
                    "streaming_cursor_fill"
                },
            },
        ))
    }

    fn extract_record(
        record: &HierarchicalRecordMeta,
        ranges: &[(u64, u64)],
        buffers: &[Vec<u8>],
    ) -> Result<Vec<u8>, String> {
        let record_end = record
            .offset
            .checked_add(record.length)
            .ok_or_else(|| format!("record {} range overflow", record.record_id))?;
        let mut output = vec![0u8; record.length as usize];
        let mut covered = 0u64;
        for (&(range_offset, range_length), buffer) in ranges.iter().zip(buffers) {
            if buffer.len() as u64 != range_length {
                return Err("backend returned an incorrect range length".to_string());
            }
            let range_end = range_offset
                .checked_add(range_length)
                .ok_or("fetched range overflow")?;
            let begin = record.offset.max(range_offset);
            let end = record_end.min(range_end);
            if begin >= end {
                continue;
            }
            let source_begin = (begin - range_offset) as usize;
            let target_begin = (begin - record.offset) as usize;
            let length = (end - begin) as usize;
            output[target_begin..target_begin + length]
                .copy_from_slice(&buffer[source_begin..source_begin + length]);
            covered += length as u64;
        }
        if covered != record.length {
            return Err(format!(
                "range plan covered {covered}/{} bytes for record {}",
                record.length, record.record_id
            ));
        }
        Ok(output)
    }

    fn range_indices_covering_record(
        record: &HierarchicalRecordMeta,
        ranges: &[(u64, u64)],
    ) -> Result<Vec<usize>, String> {
        let record_end = record
            .offset
            .checked_add(record.length)
            .ok_or_else(|| format!("record {} range overflow", record.record_id))?;
        let mut covering = Vec::new();
        let mut intervals = Vec::new();
        for (index, &(offset, length)) in ranges.iter().enumerate() {
            let end = offset.checked_add(length).ok_or("planned range overflow")?;
            let begin = record.offset.max(offset);
            let overlap_end = record_end.min(end);
            if begin < overlap_end {
                covering.push(index);
                intervals.push((begin, overlap_end));
            }
        }
        intervals.sort_unstable();
        let mut cursor = record.offset;
        for (begin, end) in intervals {
            if begin > cursor {
                return Err(format!(
                    "range plan leaves a gap before byte {begin} in record {}",
                    record.record_id
                ));
            }
            cursor = cursor.max(end);
        }
        if cursor != record_end {
            return Err(format!(
                "range plan covers record {} only through byte {cursor}, expected {record_end}",
                record.record_id
            ));
        }
        Ok(covering)
    }

    /// Resolve a future request window once, but release its original batches
    /// as soon as their range and decode dependencies complete.
    ///
    /// The range list is ordered by the earliest logical batch that consumes
    /// it. The S3 backend preserves bounded global concurrency while yielding
    /// ranges in completion order, so later-batch I/O can overlap without
    /// forcing the first batch to wait for the complete lookahead window.
    pub fn execute_incremental_window(
        &mut self,
        batches: &[Vec<LogicalTarget>],
    ) -> Result<HierarchicalIncrementalWindow, String> {
        if batches.is_empty() || batches.iter().any(Vec::is_empty) {
            return Err("incremental hierarchical window requires non-empty batches".to_string());
        }
        let total_started = Instant::now();
        let batch_records = batches
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|target| {
                        self.catalog
                            .target(&target.video_id, target.frame_idx)
                            .cloned()
                            .ok_or_else(|| {
                                format!(
                                    "unknown hierarchical target ({}, {})",
                                    target.video_id, target.frame_idx
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let target_records = batch_records.iter().flatten().cloned().collect::<Vec<_>>();

        let plan_started = Instant::now();
        let plan = self.resolve_plan(&target_records, HierarchicalAction::Adaptive)?;
        let plan_ns = plan_started.elapsed().as_nanos() as u64;
        let ResolvedPlan {
            mut ranges,
            selected_ids,
            useful_bytes,
            predicted_total_ns,
            mode,
        } = plan;

        let mut selected_groups = HashMap::<(String, u64), Vec<HierarchicalRecordMeta>>::new();
        for record_id in &selected_ids {
            let record = self
                .catalog
                .record(*record_id)
                .cloned()
                .ok_or_else(|| format!("missing selected record {record_id}"))?;
            selected_groups
                .entry((record.video_id.clone(), record.gop_id))
                .or_default()
                .push(record);
        }
        let closure_mode = mode == "adaptive_closure";
        let mut batch_groups = HashMap::<(usize, String, u64), Vec<HierarchicalRecordMeta>>::new();
        for (batch_index, records) in batch_records.iter().enumerate() {
            for record in records {
                batch_groups
                    .entry((batch_index, record.video_id.clone(), record.gop_id))
                    .or_default()
                    .push(record.clone());
            }
        }
        let mut record_earliest_batch = HashMap::<u64, usize>::new();
        for ((batch_index, video_id, gop_id), targets) in &batch_groups {
            let required_ids = if closure_mode {
                targets
                    .iter()
                    .flat_map(|record| record.closure_record_ids.iter().copied())
                    .collect::<HashSet<_>>()
            } else {
                selected_groups[&(video_id.clone(), *gop_id)]
                    .iter()
                    .map(|record| record.record_id)
                    .collect::<HashSet<_>>()
            };
            for record_id in required_ids {
                if !selected_ids.contains(&record_id) {
                    return Err(format!(
                        "batch {batch_index} requires record {record_id} outside the selected plan"
                    ));
                }
                record_earliest_batch
                    .entry(record_id)
                    .and_modify(|value| *value = (*value).min(*batch_index))
                    .or_insert(*batch_index);
            }
        }

        // Prioritize every physical span by its earliest consuming batch.
        // This changes only request order; closure resolution and byte extents
        // continue to come from the registered catalog and planner.
        let earliest_batch_for_range = |offset: u64, length: u64| {
            let end = offset.saturating_add(length);
            selected_groups
                .values()
                .flatten()
                .filter(|record| {
                    record.offset < end && offset < record.offset.saturating_add(record.length)
                })
                .filter_map(|record| record_earliest_batch.get(&record.record_id).copied())
                .min()
                .unwrap_or(usize::MAX)
        };
        ranges.sort_by_key(|&(offset, length)| (earliest_batch_for_range(offset, length), offset));
        let mut record_ranges = HashMap::<u64, Vec<usize>>::new();
        for records in selected_groups.values() {
            for record in records {
                let covering = Self::range_indices_covering_record(record, &ranges)?;
                record_ranges.insert(record.record_id, covering);
            }
        }

        struct IncrementalDecodeTask {
            batch_index: usize,
            records: Vec<HierarchicalRecordMeta>,
            target_ids: HashSet<u64>,
            range_indices: HashSet<usize>,
        }
        let mut decode_tasks = Vec::with_capacity(batch_groups.len());
        for ((batch_index, video_id, gop_id), targets) in batch_groups {
            let required_ids = if closure_mode {
                targets
                    .iter()
                    .flat_map(|record| record.closure_record_ids.iter().copied())
                    .collect::<HashSet<_>>()
            } else {
                selected_groups[&(video_id, gop_id)]
                    .iter()
                    .map(|record| record.record_id)
                    .collect::<HashSet<_>>()
            };
            let mut task_records = required_ids
                .iter()
                .map(|record_id| {
                    self.catalog
                        .record(*record_id)
                        .cloned()
                        .ok_or_else(|| format!("missing decode-task record {record_id}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            task_records.sort_unstable_by_key(|record| record.decode_ordinal);
            let range_indices = required_ids
                .iter()
                .flat_map(|record_id| record_ranges[record_id].iter().copied())
                .collect::<HashSet<_>>();
            decode_tasks.push(IncrementalDecodeTask {
                batch_index,
                records: task_records,
                target_ids: targets.iter().map(|record| record.record_id).collect(),
                range_indices,
            });
        }

        let mut completed_buffers: Vec<Option<Vec<u8>>> =
            std::iter::repeat_with(|| None).take(ranges.len()).collect();
        let mut pending_tasks = (0..decode_tasks.len()).collect::<HashSet<_>>();
        let mut decoded = HashMap::<u64, DecodedRgbFrame>::new();
        let mut decoded_ready_ns = HashMap::<u64, u64>::new();
        let batch_target_ids = batch_records
            .iter()
            .map(|records| {
                records
                    .iter()
                    .map(|record| record.record_id)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut batch_ready = vec![None; batches.len()];
        let mut fetched_bytes = 0u64;
        let mut fetch_ns = 0u64;
        let mut assemble_ns = 0u64;
        let mut decode_ns = 0u64;
        let mut submitted_access_units = 0usize;
        let mut decode_groups = 0usize;

        let backend = &self.backend;
        let codec_config = &self.codec_config;
        let decoder_slots = &self.incremental_decoder_slots;
        let phases = if self.incremental_batch_deadline_fences {
            let mut phases = Vec::new();
            let mut phase_start = 0;
            while phase_start < ranges.len() {
                let (offset, length) = ranges[phase_start];
                let earliest = earliest_batch_for_range(offset, length);
                let mut phase_end = phase_start + 1;
                while phase_end < ranges.len() {
                    let (next_offset, next_length) = ranges[phase_end];
                    if earliest_batch_for_range(next_offset, next_length) != earliest {
                        break;
                    }
                    phase_end += 1;
                }
                phases.push((phase_start, phase_end));
                phase_start = phase_end;
            }
            phases
        } else {
            vec![(0, ranges.len())]
        };
        for (phase_start, phase_end) in phases {
            let phase_started_ns = total_started.elapsed().as_nanos() as u64;
            backend
                .for_each_byte_range(&ranges[phase_start..phase_end], &mut |mut completed| {
                    completed.index += phase_start;
                    completed.started_ns = completed.started_ns.saturating_add(phase_started_ns);
                    completed.first_byte_ns =
                        completed.first_byte_ns.saturating_add(phase_started_ns);
                    completed.completed_ns =
                        completed.completed_ns.saturating_add(phase_started_ns);
                    if completed.index >= completed_buffers.len() {
                        return Err(format!(
                            "backend returned invalid range index {}",
                            completed.index
                        )
                        .into());
                    }
                    fetched_bytes += completed.bytes.len() as u64;
                    fetch_ns = fetch_ns.max(completed.completed_ns);
                    completed_buffers[completed.index] = Some(completed.bytes);

                    let mut ready_tasks = pending_tasks
                        .iter()
                        .filter(|task_index| {
                            let task = &decode_tasks[**task_index];
                            task.target_ids
                                .iter()
                                .all(|record_id| decoded.contains_key(record_id))
                                || task
                                    .range_indices
                                    .iter()
                                    .all(|index| completed_buffers[*index].is_some())
                        })
                        .copied()
                        .collect::<Vec<_>>();
                    ready_tasks
                        .sort_unstable_by_key(|task_index| decode_tasks[*task_index].batch_index);
                    let mut claimed = decoded.keys().copied().collect::<HashSet<_>>();
                    let mut jobs = Vec::new();
                    for task_index in ready_tasks {
                        pending_tasks.remove(&task_index);
                        let task = &decode_tasks[task_index];
                        let missing_target_ids = task
                            .target_ids
                            .iter()
                            .filter(|record_id| !claimed.contains(record_id))
                            .copied()
                            .collect::<HashSet<_>>();
                        if missing_target_ids.is_empty() {
                            continue;
                        }
                        let records = task.records.iter().collect::<Vec<_>>();

                        let assemble_started = Instant::now();
                        let mut annex_b = Vec::new();
                        for record in &records {
                            let indices = &record_ranges[&record.record_id];
                            let record_fetch_ranges = indices
                                .iter()
                                .map(|index| ranges[*index])
                                .collect::<Vec<_>>();
                            let record_buffers = indices
                                .iter()
                                .map(|index| completed_buffers[*index].as_ref().unwrap().clone())
                                .collect::<Vec<_>>();
                            let sample = Self::extract_record(
                                record,
                                &record_fetch_ranges,
                                &record_buffers,
                            )?;
                            annex_b.extend_from_slice(&mp4_sample_to_annex_b(
                                &sample,
                                record.nal_length_size,
                            )?);
                        }
                        let mut self_contained =
                            Vec::with_capacity(codec_config.len() + annex_b.len());
                        self_contained.extend_from_slice(codec_config);
                        self_contained.extend_from_slice(&annex_b);
                        let (parsed_config, vcl_record, _) =
                            decoder::extract_closed_record_parts(&self_contained)
                                .map_err(|error| error.to_string())?;
                        assemble_ns += assemble_started.elapsed().as_nanos() as u64;

                        let mut display_records = records.clone();
                        display_records.sort_unstable_by_key(|record| record.frame_idx);
                        let requested = display_records
                            .iter()
                            .enumerate()
                            .filter_map(|(ordinal, record)| {
                                missing_target_ids
                                    .contains(&record.record_id)
                                    .then_some((record.record_id, ordinal))
                            })
                            .collect::<Vec<_>>();
                        claimed.extend(missing_target_ids);
                        jobs.push(IncrementalDecodeJob {
                            batch_index: task.batch_index,
                            requested,
                            codec_config: parsed_config,
                            vcl_record,
                            access_units: records.len(),
                        });
                    }
                    if !jobs.is_empty() {
                        let decode_started = Instant::now();
                        let decoded_jobs =
                            Self::decode_incremental_jobs(decoder_slots, jobs, &total_started)?;
                        decode_ns += decode_started.elapsed().as_nanos() as u64;
                        decode_groups += decoded_jobs.len();
                        for job in decoded_jobs {
                            submitted_access_units += job.access_units;
                            for (record_id, frame) in job.frames {
                                decoded.insert(record_id, frame);
                                decoded_ready_ns.insert(record_id, job.completed_ns);
                            }
                        }
                    }
                    for (batch_index, target_ids) in batch_target_ids.iter().enumerate() {
                        if batch_ready[batch_index].is_none()
                            && target_ids
                                .iter()
                                .all(|record_id| decoded.contains_key(record_id))
                        {
                            batch_ready[batch_index] = target_ids
                                .iter()
                                .filter_map(|record_id| decoded_ready_ns.get(record_id).copied())
                                .max();
                        }
                    }
                    Ok(())
                })
                .map_err(|error| error.to_string())?;
        }
        if !pending_tasks.is_empty() {
            return Err(format!(
                "incremental execution left {} decode tasks pending",
                pending_tasks.len()
            ));
        }
        let batch_ready_ns = batch_ready
            .into_iter()
            .enumerate()
            .map(|(index, value)| value.ok_or_else(|| format!("batch {index} never became ready")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ordered_delivery_ns = Vec::with_capacity(batch_ready_ns.len());
        let mut previous = 0u64;
        for &ready in &batch_ready_ns {
            previous = previous.max(ready);
            ordered_delivery_ns.push(previous);
        }
        let outputs = batches
            .iter()
            .zip(&batch_records)
            .map(|(batch, records)| {
                batch
                    .iter()
                    .zip(records)
                    .map(|(target, record)| {
                        Ok(HierarchicalOutput {
                            sample_id: target.sample_id,
                            frame: decoded.get(&record.record_id).cloned().ok_or_else(|| {
                                format!("target {} was not decoded", target.sample_id)
                            })?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let total_ns = total_started.elapsed().as_nanos() as u64;
        let target_record_ids = target_records
            .iter()
            .map(|record| record.record_id)
            .collect::<HashSet<_>>();
        let mode = match mode {
            "adaptive_closure" => "incremental_adaptive_closure",
            "adaptive_region_all" => "incremental_adaptive_region_all",
            _ => "incremental_adaptive",
        };
        Ok(HierarchicalIncrementalWindow {
            batches: outputs,
            batch_ready_ns,
            ordered_delivery_ns,
            stats: HierarchicalBatchStats {
                logical_targets: target_records.len(),
                unique_targets: target_record_ids.len(),
                physical_ranges: backend.physical_ranges_for_ranges(&ranges),
                client_requests: backend.client_requests_for_ranges(&ranges),
                useful_bytes,
                fetched_bytes,
                submitted_access_units,
                decode_groups,
                streaming_cache_hits: 0,
                streaming_cache_misses: 0,
                streaming_cache_resident_bytes: self.streaming_resident_bytes as u64,
                streaming_cache_budget_bytes: self.streaming_cache_bytes as u64,
                decoder_state_resets: 0,
                plan_ns,
                fetch_ns,
                assemble_ns,
                decode_ns,
                total_ns,
                predicted_total_ns,
                mode,
            },
        })
    }

    fn resolve_plan(
        &self,
        target_records: &[HierarchicalRecordMeta],
        action: HierarchicalAction,
    ) -> Result<ResolvedPlan, String> {
        let closure_ids = target_records
            .iter()
            .flat_map(|record| record.closure_record_ids.iter().copied())
            .collect::<HashSet<_>>();
        let closure_records = closure_ids
            .iter()
            .map(|record_id| {
                let record = self
                    .catalog
                    .record(*record_id)
                    .ok_or_else(|| format!("missing closure record {record_id}"))?;
                Ok(RecordRange {
                    record_id: *record_id,
                    offset: record.offset,
                    length: record.length,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let closure_useful_bytes = planner::unique_covered_bytes(&closure_records)?;
        let touched = target_records
            .iter()
            .map(|record| (record.video_id.clone(), record.gop_id))
            .collect::<HashSet<_>>();
        let mut region_ids = HashSet::new();
        for (video_id, gop_id) in &touched {
            region_ids.extend(
                self.catalog
                    .records_for_gop(video_id, *gop_id)
                    .ok_or_else(|| format!("missing GOP region ({video_id}, {gop_id})"))?
                    .iter()
                    .copied(),
            );
        }
        let mut grouped = HashMap::<(String, u64), Vec<&HierarchicalRecordMeta>>::new();
        for record_id in &region_ids {
            let record = self.catalog.record(*record_id).unwrap();
            grouped
                .entry((record.video_id.clone(), record.gop_id))
                .or_default()
                .push(record);
        }
        let region_records = grouped
            .into_values()
            .enumerate()
            .map(|(index, records)| {
                let offset = records.iter().map(|record| record.offset).min().unwrap();
                let end = records
                    .iter()
                    .map(|record| record.offset + record.length)
                    .max()
                    .unwrap();
                RecordRange {
                    record_id: u64::MAX - index as u64,
                    offset,
                    length: end - offset,
                }
            })
            .collect::<Vec<_>>();

        let (range_plans, selected_ids, useful_bytes, predicted_total_ns, mode) = match action {
            HierarchicalAction::Adaptive => {
                let decision = self.layout.choose_plan(
                    &target_records
                        .iter()
                        .map(|record| record.record_id)
                        .collect::<Vec<_>>(),
                    self.max_merge_gap_bytes,
                    self.max_range_bytes,
                    &self.model,
                )?;
                let selected = decision.selected;
                let selected_ids = match selected.mode {
                    HierarchicalReadMode::SparseClosure => closure_ids,
                    HierarchicalReadMode::ContiguousRegion => region_ids,
                };
                let mode = match selected.mode {
                    HierarchicalReadMode::SparseClosure => "adaptive_closure",
                    HierarchicalReadMode::ContiguousRegion => "adaptive_region_all",
                };
                (
                    selected.ranges,
                    selected_ids,
                    selected.useful_bytes,
                    selected.total_ns,
                    mode,
                )
            }
            HierarchicalAction::ExactClosure => (
                planner::plan_byte_ranges(&closure_records, None, self.max_range_bytes)?,
                closure_ids,
                closure_useful_bytes,
                -1.0,
                "exact_closure",
            ),
            HierarchicalAction::FixedGapClosure(gap) => (
                planner::plan_byte_ranges(&closure_records, Some(gap), self.max_range_bytes)?,
                closure_ids,
                closure_useful_bytes,
                -1.0,
                "fixed_gap_closure",
            ),
            HierarchicalAction::RegionSelective => (
                planner::plan_byte_ranges(&region_records, None, self.max_range_bytes)?,
                closure_ids,
                closure_useful_bytes,
                -1.0,
                "region_selective_decode",
            ),
            HierarchicalAction::RegionAll => (
                planner::plan_byte_ranges(&region_records, None, self.max_range_bytes)?,
                region_ids,
                planner::unique_covered_bytes(&region_records)?,
                -1.0,
                "region_all_decode",
            ),
        };
        Ok(ResolvedPlan {
            ranges: range_plans
                .into_iter()
                .map(|range| (range.offset, range.length))
                .collect(),
            selected_ids,
            useful_bytes,
            predicted_total_ns,
            mode,
        })
    }

    pub fn execute(
        &mut self,
        targets: &[LogicalTarget],
    ) -> Result<(Vec<HierarchicalOutput>, HierarchicalBatchStats), String> {
        self.execute_action(targets, HierarchicalAction::Adaptive)
    }

    pub fn execute_action(
        &mut self,
        targets: &[LogicalTarget],
        action: HierarchicalAction,
    ) -> Result<(Vec<HierarchicalOutput>, HierarchicalBatchStats), String> {
        if targets.is_empty() {
            return Err("hierarchical batch is empty".to_string());
        }
        let total_started = Instant::now();
        let target_records = targets
            .iter()
            .map(|target| {
                self.catalog
                    .target(&target.video_id, target.frame_idx)
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "unknown hierarchical target ({}, {})",
                            target.video_id, target.frame_idx
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if matches!(action, HierarchicalAction::Adaptive) {
            if let Some(key) = self.streaming_key(&target_records) {
                return self.execute_streaming(targets, &target_records, key, total_started);
            }
            if let Some(result) =
                self.execute_mixed_streaming(targets, &target_records, total_started)?
            {
                return Ok(result);
            }
        }
        let target_record_ids = target_records
            .iter()
            .map(|record| record.record_id)
            .collect::<Vec<_>>();

        let plan_started = Instant::now();
        let plan = self.resolve_plan(&target_records, action)?;
        let plan_ns = plan_started.elapsed().as_nanos() as u64;
        let ranges = plan.ranges;
        let selected_ids = plan.selected_ids;

        let fetch_started = Instant::now();
        let buffers = self
            .backend
            .read_byte_ranges(&ranges)
            .map_err(|error| error.to_string())?;
        let fetch_ns = fetch_started.elapsed().as_nanos() as u64;

        let assemble_started = Instant::now();
        let mut encoded_records = HashMap::with_capacity(selected_ids.len());
        for record_id in &selected_ids {
            let meta = self
                .catalog
                .record(*record_id)
                .ok_or_else(|| format!("missing selected record {record_id}"))?;
            let sample = Self::extract_record(meta, &ranges, &buffers)?;
            encoded_records.insert(
                *record_id,
                mp4_sample_to_annex_b(&sample, meta.nal_length_size)?,
            );
        }
        let assemble_ns = assemble_started.elapsed().as_nanos() as u64;

        let decode_started = Instant::now();
        let mut groups = HashMap::<(String, u64), Vec<&HierarchicalRecordMeta>>::new();
        for record_id in &selected_ids {
            let record = self.catalog.record(*record_id).unwrap();
            groups
                .entry((record.video_id.clone(), record.gop_id))
                .or_default()
                .push(record);
        }
        let mut decoded = HashMap::<u64, DecodedRgbFrame>::new();
        for ((video_id, gop_id), mut records) in groups {
            records.sort_unstable_by_key(|record| record.decode_ordinal);
            let mut annex_b = Vec::new();
            for record in &records {
                annex_b.extend_from_slice(&encoded_records[&record.record_id]);
            }
            let mut self_contained = Vec::with_capacity(self.codec_config.len() + annex_b.len());
            self_contained.extend_from_slice(&self.codec_config);
            self_contained.extend_from_slice(&annex_b);
            let (codec_config, vcl_record, _) =
                decoder::extract_closed_record_parts(&self_contained)
                    .map_err(|error| error.to_string())?;
            let mut display_records = records.clone();
            display_records.sort_unstable_by_key(|record| record.frame_idx);
            let group_targets = target_records
                .iter()
                .filter(|target| target.video_id == video_id && target.gop_id == gop_id)
                .map(|target| target.record_id)
                .collect::<HashSet<_>>();
            let requested = display_records
                .iter()
                .enumerate()
                .filter_map(|(ordinal, record)| {
                    group_targets
                        .contains(&record.record_id)
                        .then_some((record.record_id, ordinal))
                })
                .collect::<Vec<_>>();
            let ordinals = requested
                .iter()
                .map(|(_, ordinal)| *ordinal)
                .collect::<Vec<_>>();
            let frames = decoder::decode_full_gop_selected_rgb24(
                &codec_config,
                &vcl_record,
                &mut self.decoder_pool,
                &ordinals,
            )
            .map_err(|error| error.to_string())?;
            for ((record_id, _), frame) in requested.into_iter().zip(frames) {
                decoded.insert(record_id, frame);
            }
        }
        let decode_ns = decode_started.elapsed().as_nanos() as u64;
        let outputs = targets
            .iter()
            .zip(&target_records)
            .map(|(target, record)| {
                Ok(HierarchicalOutput {
                    sample_id: target.sample_id,
                    frame: decoded
                        .get(&record.record_id)
                        .cloned()
                        .ok_or_else(|| format!("target {} was not decoded", target.sample_id))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let fetched_bytes = buffers.iter().map(|buffer| buffer.len() as u64).sum();
        Ok((
            outputs,
            HierarchicalBatchStats {
                logical_targets: targets.len(),
                unique_targets: target_record_ids.iter().collect::<HashSet<_>>().len(),
                physical_ranges: self.backend.physical_ranges_for_ranges(&ranges),
                client_requests: self.backend.client_requests_for_ranges(&ranges),
                useful_bytes: plan.useful_bytes,
                fetched_bytes,
                submitted_access_units: selected_ids.len(),
                decode_groups: decoded
                    .keys()
                    .filter_map(|record_id| self.catalog.record(*record_id))
                    .map(|record| (record.video_id.as_str(), record.gop_id))
                    .collect::<HashSet<_>>()
                    .len(),
                streaming_cache_hits: 0,
                streaming_cache_misses: 0,
                streaming_cache_resident_bytes: self.streaming_resident_bytes as u64,
                streaming_cache_budget_bytes: self.streaming_cache_bytes as u64,
                decoder_state_resets: 0,
                plan_ns,
                fetch_ns,
                assemble_ns,
                decode_ns,
                total_ns: total_started.elapsed().as_nanos() as u64,
                predicted_total_ns: plan.predicted_total_ns,
                mode: plan.mode,
            },
        ))
    }
}
