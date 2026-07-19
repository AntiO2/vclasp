use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Instant;

use crate::adaptive_planner::{PlanCandidate, PlanMode};
use crate::backend::{CompletedRange, StorageBackend};
use crate::decoder;
use crate::planner::{ByteCache, PlannedRecord, RangePlan, RecordRange};
use crate::representation::{
    BatchFeatures, BatchStats, DependencyClosure, DependencyKind, LogicalRequest, OuterPlanner,
    PhysicalRecord, Representation, SampleRepresentations,
};

#[derive(Debug, Clone)]
pub struct ClosedRecordDescriptor {
    pub sample_id: u64,
    pub video_id: u64,
    pub offset: u64,
    pub length: u64,
    pub target_ordinal: usize,
}

#[derive(Debug)]
pub struct ClosedRecordFrame {
    pub sample_id: u64,
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

struct PrefixCursorCache {
    entries: HashMap<u64, decoder::PrefixCursor>,
    lru: VecDeque<u64>,
    capacity: usize,
}

impl PrefixCursorCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            capacity,
        }
    }

    fn contains(&self, video_id: u64) -> bool {
        self.entries.contains_key(&video_id)
    }

    fn insert(&mut self, video_id: u64, cursor: decoder::PrefixCursor) {
        while self.entries.len() >= self.capacity {
            if let Some(evicted) = self.lru.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        self.entries.insert(video_id, cursor);
        self.touch(video_id);
    }

    fn get_mut(&mut self, video_id: u64) -> Option<&mut decoder::PrefixCursor> {
        if self.entries.contains_key(&video_id) {
            self.touch(video_id);
        }
        self.entries.get_mut(&video_id)
    }

    fn get(&self, video_id: u64) -> Option<&decoder::PrefixCursor> {
        self.entries.get(&video_id)
    }

    fn touch(&mut self, video_id: u64) {
        if let Some(position) = self.lru.iter().position(|value| *value == video_id) {
            self.lru.remove(position);
        }
        self.lru.push_back(video_id);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

pub struct ClosedRecordBatchExecutor {
    representation: Representation,
    outer_planner: OuterPlanner,
    backend: Box<dyn StorageBackend + Send>,
    merge_threshold_bytes: Option<u64>,
    max_range_bytes: Option<u64>,
    encoded_cache: ByteCache,
    decoded_cache: ByteCache,
    decoder_slots: decoder::SharedDecoderSlots,
    prefix_cursor_slots: Vec<Mutex<PrefixCursorCache>>,
    prefix_max_records: HashMap<u64, PhysicalRecord>,
    prefix_streaming: bool,
    width: u32,
    height: u32,
}

impl ClosedRecordBatchExecutor {
    #[cfg(test)]
    pub(crate) fn decoder_slots_handle(&self) -> &decoder::SharedDecoderSlots {
        &self.decoder_slots
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        descriptors: Vec<ClosedRecordDescriptor>,
        representation: Representation,
        backend: Box<dyn StorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        prefix_streaming: bool,
        prefix_cursor_capacity: usize,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        Self::new_with_decoder_slots(
            descriptors,
            representation,
            backend,
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decoder::shared_decoder_slots(decode_concurrency),
            prefix_streaming,
            prefix_cursor_capacity,
            width,
            height,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_decoder_slots(
        descriptors: Vec<ClosedRecordDescriptor>,
        representation: Representation,
        backend: Box<dyn StorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decoder_slots: decoder::SharedDecoderSlots,
        prefix_streaming: bool,
        prefix_cursor_capacity: usize,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        if decoder_slots.is_empty() {
            return Err("shared decoder slots must be non-empty".to_string());
        }
        let decode_concurrency = decoder_slots.len();
        if !matches!(
            representation,
            Representation::Prefix | Representation::Pair
        ) {
            return Err("closed-record executor supports only Prefix or Pair".to_string());
        }
        if prefix_streaming && representation != Representation::Prefix {
            return Err("prefix_streaming requires Prefix representation".to_string());
        }
        if prefix_streaming && prefix_cursor_capacity < decode_concurrency.max(1) {
            return Err(
                "prefix_cursor_capacity must be at least decode_concurrency in streaming mode"
                    .to_string(),
            );
        }
        let mut prefix_max_by_ordinal = HashMap::new();
        if representation == Representation::Prefix {
            for descriptor in &descriptors {
                let record = PhysicalRecord {
                    record_id: descriptor.sample_id,
                    kind: DependencyKind::Prefix,
                    offset: descriptor.offset,
                    length: descriptor.length,
                };
                prefix_max_by_ordinal
                    .entry(descriptor.video_id)
                    .and_modify(|selected: &mut (usize, PhysicalRecord)| {
                        if descriptor.target_ordinal > selected.0 {
                            *selected = (descriptor.target_ordinal, record.clone());
                        }
                    })
                    .or_insert((descriptor.target_ordinal, record));
            }
        }
        let prefix_max_records = prefix_max_by_ordinal
            .into_iter()
            .map(|(video_id, (_, record))| (video_id, record))
            .collect();
        let samples = descriptors
            .into_iter()
            .map(|descriptor| {
                let closure = DependencyClosure {
                    sample_id: descriptor.sample_id,
                    video_id: descriptor.video_id,
                    representation,
                    records: vec![PhysicalRecord {
                        record_id: descriptor.sample_id,
                        kind: match representation {
                            Representation::Prefix => DependencyKind::Prefix,
                            Representation::Pair => DependencyKind::Pair,
                            Representation::Normalized => unreachable!(),
                        },
                        offset: descriptor.offset,
                        length: descriptor.length,
                    }],
                    target_ordinal: descriptor.target_ordinal,
                };
                SampleRepresentations {
                    sample_id: descriptor.sample_id,
                    prefix: (representation == Representation::Prefix).then_some(closure.clone()),
                    normalized: None,
                    pair: (representation == Representation::Pair).then_some(closure),
                }
            })
            .collect();
        Ok(Self {
            representation,
            outer_planner: OuterPlanner::new(samples)?,
            backend,
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache: ByteCache::new(encoded_cache_bytes),
            decoded_cache: ByteCache::new(decoded_cache_bytes),
            decoder_slots,
            prefix_cursor_slots: (0..decode_concurrency.max(1))
                .map(|slot| {
                    let slots = decode_concurrency.max(1);
                    let capacity = (prefix_cursor_capacity + slots - 1 - slot) / slots;
                    Mutex::new(PrefixCursorCache::new(capacity))
                })
                .collect(),
            prefix_max_records,
            prefix_streaming,
            width,
            height,
        })
    }

    fn extract_plan(
        plan: &RangePlan,
        buffer: &[u8],
        records: &mut HashMap<u64, Vec<u8>>,
    ) -> Result<(), String> {
        if buffer.len() != plan.length as usize {
            return Err(format!(
                "short closed-record range: {} != {}",
                buffer.len(),
                plan.length
            ));
        }
        for PlannedRecord {
            record_id,
            relative_offset,
            length,
        } in &plan.records
        {
            let start = *relative_offset as usize;
            let end = start + *length as usize;
            if end > buffer.len() {
                return Err(format!("closed record {record_id} exceeds fetched range"));
            }
            records.insert(*record_id, buffer[start..end].to_vec());
        }
        Ok(())
    }

    fn decode_pending(
        &self,
        closures: &[DependencyClosure],
        records: &HashMap<u64, Vec<u8>>,
    ) -> Result<(HashMap<u64, Vec<u8>>, u64, u64, usize, usize, usize), String> {
        let mut by_video: HashMap<u64, Vec<&DependencyClosure>> = HashMap::new();
        for closure in closures {
            by_video.entry(closure.video_id).or_default().push(closure);
        }
        let mut partitions: Vec<Vec<(u64, Vec<&DependencyClosure>)>> =
            (0..self.decoder_slots.len()).map(|_| Vec::new()).collect();
        for (video_id, targets) in by_video {
            let slot = video_id as usize % self.decoder_slots.len();
            partitions[slot].push((video_id, targets));
        }

        let representation = self.representation;
        let prefix_streaming = self.prefix_streaming;
        let prefix_max_records = &self.prefix_max_records;
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (slot_index, groups) in partitions.into_iter().enumerate() {
                if groups.is_empty() {
                    continue;
                }
                let slot = &self.decoder_slots[slot_index];
                let cursor_slot = &self.prefix_cursor_slots[slot_index];
                handles.push(scope.spawn(move || {
                    let mut pool = slot
                        .lock()
                        .map_err(|_| "closed-record decoder slot lock poisoned".to_string())?;
                    let mut cursors = cursor_slot
                        .lock()
                        .map_err(|_| "Prefix cursor slot lock poisoned".to_string())?;
                    let mut output = HashMap::new();
                    let mut assemble_ns = 0;
                    let mut decode_ns = 0;
                    let mut state_hits = 0;
                    let mut state_misses = 0;
                    let mut state_resets = 0;
                    for (video_id, targets) in groups {
                        let assemble_started = Instant::now();
                        let (codec_config, closed_records, expected_frames) =
                            if representation == Representation::Pair {
                                let mut codec_config: Option<Vec<u8>> = None;
                                let mut closed_records = Vec::with_capacity(targets.len());
                                let mut expected_frames = Vec::with_capacity(targets.len());
                                for target in &targets {
                                    let record_id = target.records[0].record_id;
                                    let payload = records.get(&record_id).ok_or_else(|| {
                                        format!("missing closed record {record_id}")
                                    })?;
                                    let (config, record, frame_count) =
                                        decoder::extract_closed_record_parts(payload)
                                            .map_err(|error| error.to_string())?;
                                    if frame_count != target.target_ordinal + 1 {
                                        return Err(format!(
                                            "closed-record sample {} has {frame_count} frames, target ordinal is {}",
                                            target.sample_id, target.target_ordinal
                                        ));
                                    }
                                    if let Some(existing) = &codec_config {
                                        if existing != &config {
                                            return Err(format!(
                                                "closed-record video {video_id} has inconsistent SPS/PPS"
                                            ));
                                        }
                                    } else {
                                        codec_config = Some(config);
                                    }
                                    expected_frames.push(frame_count);
                                    closed_records.push(record);
                                }
                                (codec_config, closed_records, Some(expected_frames))
                            } else {
                                let longest_target = targets
                                    .iter()
                                    .max_by_key(|target| target.target_ordinal)
                                    .ok_or_else(|| "Prefix group has no target".to_string())?;
                                let requested_longest_record = &longest_target.records[0];
                                let longest_record = if prefix_streaming {
                                    prefix_max_records.get(&video_id).ok_or_else(|| {
                                        format!("Prefix video {video_id} has no stream record")
                                    })?
                                } else {
                                    requested_longest_record
                                };
                                for target in &targets {
                                    let record = &target.records[0];
                                    if record.offset != longest_record.offset
                                        || record.length > longest_record.length
                                    {
                                        return Err(format!(
                                            "Prefix video {video_id} contains non-nested records"
                                        ));
                                    }
                                }
                                let payload = records
                                    .get(&longest_record.record_id)
                                    .ok_or_else(|| {
                                        format!(
                                            "missing longest Prefix record {}",
                                            longest_record.record_id
                                        )
                                    })?;
                                let (config, record, frame_count) =
                                    decoder::extract_closed_record_parts(payload)
                                        .map_err(|error| error.to_string())?;
                                if frame_count < longest_target.target_ordinal + 1 {
                                    return Err(format!(
                                        "Prefix stream for sample {} has only {frame_count} frames, target ordinal is {}",
                                        longest_target.sample_id, longest_target.target_ordinal
                                    ));
                                }
                                (Some(config), vec![record], None)
                            };
                        assemble_ns += assemble_started.elapsed().as_nanos() as u64;
                        let decode_started = Instant::now();
                        let config = codec_config
                            .as_deref()
                            .ok_or_else(|| "closed-record group has no codec config".to_string())?;
                        let frames = if representation == Representation::Pair {
                            decoder::decode_mixed_closed_targets_continuous_rgb24(
                                config,
                                &closed_records,
                                &mut pool,
                                &expected_frames.ok_or_else(|| {
                                    "Pair group has no frame count".to_string()
                                })?,
                            )
                            .map_err(|error| error.to_string())?
                        } else if prefix_streaming {
                            let existed = cursors.contains(video_id);
                            if !existed {
                                cursors.insert(
                                    video_id,
                                    decoder::PrefixCursor::new(
                                        config,
                                        &closed_records[0],
                                        decoder::DecoderConfig { num_threads: 1 },
                                    )
                                    .map_err(|error| error.to_string())?,
                                );
                            }
                            let (frames, reset) = cursors
                                .get_mut(video_id)
                                .expect("Prefix cursor inserted")
                                .decode_selected(
                                    config,
                                    &closed_records[0],
                                    &targets
                                        .iter()
                                        .map(|target| target.target_ordinal)
                                        .collect::<Vec<_>>(),
                                )
                                .map_err(|error| error.to_string())?;
                            if existed && !reset {
                                state_hits += 1;
                            } else {
                                state_misses += 1;
                            }
                            state_resets += usize::from(reset);
                            frames
                        } else {
                            decoder::decode_gop_selected_rgb24(
                                config,
                                &closed_records[0],
                                &mut pool,
                                &targets
                                    .iter()
                                    .map(|target| target.target_ordinal)
                                    .collect::<Vec<_>>(),
                            )
                            .map_err(|error| error.to_string())?
                        };
                        decode_ns += decode_started.elapsed().as_nanos() as u64;
                        if frames.len() != targets.len() {
                            return Err("Pair decoder output count mismatch".to_string());
                        }
                        for (target, frame) in targets.into_iter().zip(frames) {
                            output.insert(target.sample_id, frame.data);
                        }
                    }
                    Ok::<_, String>((
                        output,
                        assemble_ns,
                        decode_ns,
                        state_hits,
                        state_misses,
                        state_resets,
                    ))
                }));
            }
            let mut output = HashMap::new();
            let mut assemble_ns = 0;
            let mut decode_ns = 0;
            let mut state_hits = 0;
            let mut state_misses = 0;
            let mut state_resets = 0;
            for handle in handles {
                let (
                    part,
                    part_assemble_ns,
                    part_decode_ns,
                    part_state_hits,
                    part_state_misses,
                    part_state_resets,
                ) = handle
                    .join()
                    .map_err(|_| "closed-record decoder worker panicked".to_string())??;
                output.extend(part);
                assemble_ns = assemble_ns.max(part_assemble_ns);
                decode_ns = decode_ns.max(part_decode_ns);
                state_hits += part_state_hits;
                state_misses += part_state_misses;
                state_resets += part_state_resets;
            }
            Ok((
                output,
                assemble_ns,
                decode_ns,
                state_hits,
                state_misses,
                state_resets,
            ))
        })
    }

    pub fn execute(
        &mut self,
        batch: &[u64],
    ) -> Result<(Vec<ClosedRecordFrame>, BatchStats), String> {
        let total_started = Instant::now();
        let mut stats = BatchStats {
            logical_samples: batch.len(),
            ..Default::default()
        };
        let resolve_started = Instant::now();
        let requests = batch
            .iter()
            .map(|sample_id| LogicalRequest {
                sample_id: *sample_id,
            })
            .collect::<Vec<_>>();
        let closures = self
            .outer_planner
            .resolve_closures(&requests, self.representation)?;
        stats.unique_targets = closures.len();
        stats.unique_videos = closures
            .iter()
            .map(|closure| closure.video_id)
            .collect::<std::collections::HashSet<_>>()
            .len();
        stats.dependency_records = closures.iter().map(|closure| closure.records.len()).sum();
        stats.target_ordinal_sum = closures.iter().map(|closure| closure.target_ordinal).sum();
        stats.target_ordinal_max = closures
            .iter()
            .map(|closure| closure.target_ordinal)
            .max()
            .unwrap_or(0);
        stats.resolve_ns = resolve_started.elapsed().as_nanos() as u64;

        let cache_started = Instant::now();
        let mut results = HashMap::new();
        let mut pending = Vec::new();
        for closure in closures {
            if let Some(rgb) = self.decoded_cache.get(closure.sample_id) {
                results.insert(closure.sample_id, rgb);
                stats.decoded_cache_hits += 1;
            } else {
                stats.decoded_cache_misses += 1;
                pending.push(closure);
            }
        }
        let mut resolved = HashMap::new();
        let mut misses = Vec::new();
        let physical_records = if self.representation == Representation::Prefix {
            let mut by_video = HashMap::<u64, &DependencyClosure>::new();
            for closure in &pending {
                by_video
                    .entry(closure.video_id)
                    .and_modify(|selected| {
                        if closure.target_ordinal > selected.target_ordinal {
                            *selected = closure;
                        }
                    })
                    .or_insert(closure);
            }
            by_video
                .into_iter()
                .map(|(video_id, closure)| {
                    if self.prefix_streaming {
                        self.prefix_max_records
                            .get(&video_id)
                            .ok_or_else(|| format!("Prefix video {video_id} has no stream record"))
                    } else {
                        Ok(&closure.records[0])
                    }
                })
                .collect::<Result<Vec<_>, String>>()?
        } else {
            pending
                .iter()
                .map(|closure| &closure.records[0])
                .collect::<Vec<_>>()
        };
        for record in physical_records {
            if let Some(payload) = self.encoded_cache.get(record.record_id) {
                stats.encoded_cache_hits += 1;
                resolved.insert(record.record_id, payload);
            } else {
                stats.encoded_cache_misses += 1;
                misses.push(RecordRange {
                    record_id: record.record_id,
                    offset: record.offset,
                    length: record.length,
                });
            }
        }
        stats.unique_records = stats.encoded_cache_hits + stats.encoded_cache_misses;
        stats.cache_lookup_ns = cache_started.elapsed().as_nanos() as u64;

        let plan_started = Instant::now();
        let plans = crate::planner::plan_byte_ranges(
            &misses,
            self.merge_threshold_bytes,
            self.max_range_bytes,
        )?;
        stats.plan_ns = plan_started.elapsed().as_nanos() as u64;
        stats.physical_ranges = plans.len();
        stats.useful_bytes = crate::planner::unique_covered_bytes(&misses)?;
        stats.planned_ranges = plans.len();
        stats.planned_useful_bytes = stats.useful_bytes;
        stats.planned_fetched_bytes = plans.iter().map(|plan| plan.length).sum();
        let ranges = plans
            .iter()
            .map(|plan| (plan.offset, plan.length))
            .collect::<Vec<_>>();
        stats.client_requests = self.backend.client_requests_for_ranges(&ranges);
        stats.server_entries = self.backend.server_entries_for_ranges(&ranges);

        let fetch_started = Instant::now();
        let mut completed = Vec::<CompletedRange>::new();
        self.backend
            .for_each_byte_range(&ranges, &mut |range| {
                stats.fetch_wall_ns = stats.fetch_wall_ns.max(range.completed_ns);
                stats.fetch_service_ns_sum += range.completed_ns.saturating_sub(range.started_ns);
                stats.range_queue_ns_sum += range.started_ns;
                completed.push(range);
                Ok(())
            })
            .map_err(|error| error.to_string())?;
        completed.sort_by_key(|range| range.index);
        let extract_started = Instant::now();
        for range in completed {
            stats.fetched_bytes += range.bytes.len() as u64;
            Self::extract_plan(&plans[range.index], &range.bytes, &mut resolved)?;
        }
        stats.extract_ns = extract_started.elapsed().as_nanos() as u64;
        let fetch_wall = fetch_started.elapsed().as_nanos() as u64;
        stats.fetch_wall_ns = stats.fetch_wall_ns.max(fetch_wall);
        stats.overfetch_bytes = stats.fetched_bytes.saturating_sub(stats.useful_bytes);

        if !pending.is_empty() {
            let (decoded, assemble_ns, decode_ns, state_hits, state_misses, state_resets) =
                self.decode_pending(&pending, &resolved)?;
            stats.assemble_ns = assemble_ns;
            stats.decode_ns = decode_ns;
            stats.decoder_state_hits = state_hits;
            stats.decoder_state_misses = state_misses;
            stats.decoder_state_resets = state_resets;
            stats.decoded_targets = decoded.len();
            stats.decoded_frames = decoded.len();
            results.extend(decoded);
        }
        for plan in &plans {
            for record in &plan.records {
                if let Some(payload) = resolved.get(&record.record_id) {
                    self.encoded_cache.put(record.record_id, payload.clone());
                }
            }
        }
        for closure in &pending {
            if let Some(rgb) = results.get(&closure.sample_id) {
                self.decoded_cache.put(closure.sample_id, rgb.clone());
            }
        }
        let encoded_cache_stats = self.encoded_cache.stats();
        let decoded_cache_stats = self.decoded_cache.stats();
        stats.encoded_cache_resident_bytes = encoded_cache_stats["resident_bytes"];
        stats.decoded_cache_resident_bytes = decoded_cache_stats["resident_bytes"];
        stats.decoder_state_resident = if self.prefix_streaming {
            self.prefix_cursor_slots
                .iter()
                .map(|slot| {
                    slot.lock()
                        .map(|cursors| cursors.len())
                        .map_err(|_| "Prefix cursor slot lock poisoned".to_string())
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_iter()
                .sum()
        } else {
            self.decoder_slots
                .iter()
                .map(|slot| {
                    slot.lock()
                        .map(|pool| pool.len())
                        .map_err(|_| "closed-record decoder slot lock poisoned".to_string())
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_iter()
                .sum()
        };

        let reorder_started = Instant::now();
        let expected_rgb_bytes = self.width as usize * self.height as usize * 3;
        let frames = batch
            .iter()
            .map(|sample_id| {
                let rgb = results
                    .get(sample_id)
                    .ok_or_else(|| format!("missing closed-record result {sample_id}"))?
                    .clone();
                if rgb.len() != expected_rgb_bytes {
                    return Err(format!(
                        "closed-record RGB size mismatch for {sample_id}: {} != {expected_rgb_bytes}",
                        rgb.len()
                    ));
                }
                Ok(ClosedRecordFrame {
                    sample_id: *sample_id,
                    rgb,
                    width: self.width,
                    height: self.height,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        stats.reorder_ns = reorder_started.elapsed().as_nanos() as u64;
        stats.total_ns = total_started.elapsed().as_nanos() as u64;
        stats.time_to_first_ready_ns = stats.total_ns;
        Ok((frames, stats))
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;

    #[test]
    fn pair_plan_features_deduplicate_targets_and_account_for_overfetch() {
        let descriptors = vec![
            ClosedRecordDescriptor {
                sample_id: 1,
                video_id: 0,
                offset: 0,
                length: 10,
                target_ordinal: 1,
            },
            ClosedRecordDescriptor {
                sample_id: 2,
                video_id: 0,
                offset: 20,
                length: 10,
                target_ordinal: 1,
            },
        ];
        let executor = ClosedRecordBatchExecutor::new(
            descriptors,
            Representation::Pair,
            Box::new(crate::backend::NoopBackend),
            Some(10),
            None,
            1024,
            1024,
            1,
            false,
            0,
            320,
            240,
        )
        .unwrap();
        let features = executor.plan_features(&[1, 2, 2]).unwrap();
        assert_eq!(features.logical_requests, 3);
        assert_eq!(features.unique_targets, 2);
        assert_eq!(features.unique_records, 2);
        assert_eq!(features.physical_ranges, 1);
        assert_eq!(features.useful_bytes, 20);
        assert_eq!(features.fetched_bytes, 30);
        assert_eq!(features.overfetch_bytes, 10);
    }

    #[test]
    fn streaming_prefix_plan_uses_the_full_video_record() {
        let descriptors = vec![
            ClosedRecordDescriptor {
                sample_id: 1,
                video_id: 0,
                offset: 0,
                length: 10,
                target_ordinal: 1,
            },
            ClosedRecordDescriptor {
                sample_id: 2,
                video_id: 0,
                offset: 0,
                length: 20,
                target_ordinal: 2,
            },
        ];
        let executor = ClosedRecordBatchExecutor::new(
            descriptors,
            Representation::Prefix,
            Box::new(crate::backend::NoopBackend),
            None,
            None,
            1024,
            1024,
            1,
            true,
            1,
            320,
            240,
        )
        .unwrap();
        let features = executor.plan_features(&[1]).unwrap();
        assert_eq!(features.logical_requests, 1);
        assert_eq!(features.unique_targets, 1);
        assert_eq!(features.unique_records, 1);
        assert_eq!(features.useful_bytes, 20);
    }
}

impl ClosedRecordBatchExecutor {
    /// Resolve a cold-cache physical plan without performing I/O or decode.
    pub fn plan_features(&self, batch: &[u64]) -> Result<BatchFeatures, String> {
        let requests = batch
            .iter()
            .map(|sample_id| LogicalRequest {
                sample_id: *sample_id,
            })
            .collect::<Vec<_>>();
        let closures = self
            .outer_planner
            .resolve_closures(&requests, self.representation)?;
        let physical_records = if self.representation == Representation::Prefix {
            let mut by_video = HashMap::<u64, &DependencyClosure>::new();
            for closure in &closures {
                by_video
                    .entry(closure.video_id)
                    .and_modify(|selected| {
                        if closure.target_ordinal > selected.target_ordinal {
                            *selected = closure;
                        }
                    })
                    .or_insert(closure);
            }
            by_video
                .into_iter()
                .map(|(video_id, closure)| {
                    if self.prefix_streaming {
                        self.prefix_max_records
                            .get(&video_id)
                            .ok_or_else(|| format!("Prefix video {video_id} has no stream record"))
                    } else {
                        Ok(&closure.records[0])
                    }
                })
                .collect::<Result<Vec<_>, String>>()?
        } else {
            closures
                .iter()
                .map(|closure| &closure.records[0])
                .collect::<Vec<_>>()
        };
        let records = physical_records
            .into_iter()
            .map(|record| RecordRange {
                record_id: record.record_id,
                offset: record.offset,
                length: record.length,
            })
            .collect::<Vec<_>>();
        let ranges = crate::planner::plan_byte_ranges(
            &records,
            self.merge_threshold_bytes,
            self.max_range_bytes,
        )?;
        let useful_bytes = crate::planner::unique_covered_bytes(&records)?;
        let fetched_bytes = ranges.iter().try_fold(0u64, |total, range| {
            total
                .checked_add(range.length)
                .ok_or_else(|| "planned fetched bytes overflow u64".to_string())
        })?;
        let (mean_range_bytes, max_range_bytes, address_span_bytes, contiguous_range_pairs) =
            crate::representation::range_geometry(&ranges)?;
        let (unique_anchor_records, anchor_reuse_hits) =
            crate::representation::anchor_metrics(&closures);
        Ok(BatchFeatures {
            logical_requests: requests.len(),
            unique_targets: closures.len(),
            unique_videos: closures
                .iter()
                .map(|closure| closure.video_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            dependency_records: closures.iter().map(|closure| closure.records.len()).sum(),
            unique_records: records
                .iter()
                .map(|record| record.record_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            useful_bytes,
            physical_ranges: ranges.len(),
            fetched_bytes,
            overfetch_bytes: fetched_bytes.saturating_sub(useful_bytes),
            mean_range_bytes,
            max_range_bytes,
            address_span_bytes,
            contiguous_range_pairs,
            unique_anchor_records,
            anchor_reuse_hits,
            target_ordinal_sum: closures.iter().map(|closure| closure.target_ordinal).sum(),
            target_ordinal_max: closures
                .iter()
                .map(|closure| closure.target_ordinal)
                .max()
                .unwrap_or(0),
        })
    }

    /// Build the Prefix candidate from current cache/cursor state without mutation.
    pub fn adaptive_prefix_candidate(&self, batch: &[u64]) -> Result<PlanCandidate, String> {
        if self.representation != Representation::Prefix {
            return Err("adaptive Prefix preview requires Prefix representation".to_string());
        }
        let requests = batch
            .iter()
            .map(|sample_id| LogicalRequest {
                sample_id: *sample_id,
            })
            .collect::<Vec<_>>();
        let closures = self
            .outer_planner
            .resolve_closures(&requests, Representation::Prefix)?
            .into_iter()
            .filter(|closure| !self.decoded_cache.contains(closure.sample_id))
            .collect::<Vec<_>>();
        let mut by_video = HashMap::<u64, Vec<&DependencyClosure>>::new();
        for closure in &closures {
            by_video.entry(closure.video_id).or_default().push(closure);
        }

        let mut misses = Vec::new();
        let mut prefix_frames = 0;
        let mut prefix_resets = 0;
        for (&video_id, targets) in &by_video {
            let requested_record = targets
                .iter()
                .max_by_key(|target| target.target_ordinal)
                .map(|target| &target.records[0])
                .ok_or_else(|| "Prefix preview found an empty video group".to_string())?;
            let record = if self.prefix_streaming {
                self.prefix_max_records
                    .get(&video_id)
                    .ok_or_else(|| format!("Prefix video {video_id} has no stream record"))?
            } else {
                requested_record
            };
            if !self.encoded_cache.contains(record.record_id) {
                misses.push(RecordRange {
                    record_id: record.record_id,
                    offset: record.offset,
                    length: record.length,
                });
            }
            let ordinals = targets
                .iter()
                .map(|target| target.target_ordinal)
                .collect::<Vec<_>>();
            if self.prefix_streaming {
                let slot = video_id as usize % self.prefix_cursor_slots.len();
                let cursors = self.prefix_cursor_slots[slot]
                    .lock()
                    .map_err(|_| "Prefix cursor slot lock poisoned".to_string())?;
                if let Some(cursor) = cursors.get(video_id) {
                    let (frames, reset) = cursor.preview_ordinals(&ordinals);
                    prefix_frames += frames;
                    prefix_resets += usize::from(reset);
                } else {
                    prefix_frames += ordinals.iter().copied().max().unwrap_or(0) + 1;
                    prefix_resets += 1;
                }
            } else {
                prefix_frames += ordinals.iter().copied().max().unwrap_or(0) + 1;
                prefix_resets += 1;
            }
        }
        let useful_bytes = crate::planner::unique_covered_bytes(&misses)?;
        let plans = crate::planner::plan_byte_ranges(
            &misses,
            self.merge_threshold_bytes,
            self.max_range_bytes,
        )?;
        Ok(PlanCandidate {
            mode: PlanMode::Prefix,
            range_lengths: plans.iter().map(|plan| plan.length).collect(),
            useful_bytes,
            anchor_decodes: 0,
            delta_decodes: 0,
            prefix_frames,
            prefix_resets,
        })
    }
}
