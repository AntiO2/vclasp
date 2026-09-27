use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Instant;

use crate::backend::{CompletedRange, StorageBackend};
use crate::controls::adaptive::{self, PlanCandidate, PlanMode};
use crate::decoder;
use crate::hierarchical_layout::{
    AccessUnitRecord, DependencyLookaheadDecision, GopRegion, HierarchicalCostModel,
    HierarchicalLayoutIndex, RegionDecodeMode, TargetClosure as LayoutTargetClosure,
};
use crate::planner::{self, ByteCache, PlannedRecord, RangePlan, RecordRange, SharedByteCache};
use crate::representation::{
    BatchFeatures, BatchStats, DependencyClosure, DependencyKind, LogicalRequest, OuterPlanner,
    PhysicalRecord, Representation, SampleRepresentations,
};

#[derive(Debug, Clone)]
pub struct NormalizedDescriptor {
    pub sample_id: u64,
    pub video_id: u64,
    pub anchor_group_id: u64,
    pub target_ordinal: usize,
    pub anchor_offset: u64,
    pub anchor_length: u64,
    pub delta_offset: u64,
    pub delta_length: u64,
}

#[derive(Debug)]
pub struct NormalizedFrame {
    pub sample_id: u64,
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct NormalizedLookaheadWindow {
    pub frames: Vec<NormalizedFrame>,
    pub stats: NormalizedStats,
    pub decision: DependencyLookaheadDecision,
    pub first_batch_ready_ns: u64,
}

pub type NormalizedStats = BatchStats;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecodeSchedule {
    Repeated,
    Fused,
    Adaptive {
        anchor_work_units: f64,
        delta_work_units: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RangePlanning {
    ByteGap,
    DependencyGroupSpan,
}

impl DecodeSchedule {
    fn validate(self) -> Result<Self, String> {
        if let Self::Adaptive {
            anchor_work_units,
            delta_work_units,
        } = self
        {
            if !anchor_work_units.is_finite() || anchor_work_units <= 0.0 {
                return Err("adaptive Anchor work units must be finite and positive".to_string());
            }
            if !delta_work_units.is_finite() || delta_work_units <= 0.0 {
                return Err("adaptive Delta work units must be finite and positive".to_string());
            }
        }
        Ok(self)
    }

    pub(crate) fn requires_group_metadata(self) -> bool {
        !matches!(self, Self::Repeated)
    }
}

#[derive(Debug, Clone)]
struct TargetClosure {
    sample_id: u64,
    video_id: u64,
    anchor_group_id: u64,
    target_ordinal: usize,
    anchor_id: u64,
    delta_id: Option<u64>,
}

fn decode_group_id(target: &TargetClosure) -> u64 {
    target.anchor_group_id
}

fn should_fuse_groups(
    groups: &HashMap<u64, Vec<TargetClosure>>,
    decoder_slots: usize,
    schedule: DecodeSchedule,
) -> bool {
    match schedule {
        DecodeSchedule::Repeated => false,
        DecodeSchedule::Fused => true,
        DecodeSchedule::Adaptive {
            anchor_work_units,
            delta_work_units,
        } => {
            let slots = decoder_slots.max(1);
            let mut repeated_load = vec![0.0; slots];
            let mut fused_load = vec![0.0; slots];
            for (&group_id, targets) in groups {
                let mut ordered = targets.iter().collect::<Vec<_>>();
                ordered.sort_by_key(|target| (target.target_ordinal, target.sample_id));
                let fused_slot = group_id as usize % slots;
                fused_load[fused_slot] += anchor_work_units
                    + ordered
                        .iter()
                        .filter(|target| target.delta_id.is_some())
                        .count() as f64
                        * delta_work_units;
                for (index, target) in ordered.into_iter().enumerate() {
                    let repeated_slot = (group_id as usize + index) % slots;
                    repeated_load[repeated_slot] += anchor_work_units
                        + if target.delta_id.is_some() {
                            delta_work_units
                        } else {
                            0.0
                        };
                }
            }
            let repeated_critical = repeated_load.into_iter().fold(0.0, f64::max);
            let fused_critical = fused_load.into_iter().fold(0.0, f64::max);
            fused_critical < repeated_critical
        }
    }
}

fn select_ready_groups(
    pending: &HashMap<u64, TargetClosure>,
    resolved: &HashMap<u64, Vec<u8>>,
    microbatch_targets: usize,
    force: bool,
    priority_sample_ids: Option<&HashSet<u64>>,
) -> Vec<(u64, Vec<u64>)> {
    let mut groups: HashMap<u64, Vec<&TargetClosure>> = HashMap::new();
    for target in pending.values().filter(|target| {
        priority_sample_ids.is_none_or(|sample_ids| sample_ids.contains(&target.sample_id))
    }) {
        groups
            .entry(decode_group_id(target))
            .or_default()
            .push(target);
    }

    let mut ready_groups = groups
        .into_iter()
        .filter_map(|(group_id, mut targets)| {
            let all_ready = targets.iter().all(|target| {
                resolved.contains_key(&target.anchor_id)
                    && target
                        .delta_id
                        .is_none_or(|delta_id| resolved.contains_key(&delta_id))
            });
            if !all_ready {
                return None;
            }
            targets.sort_by_key(|target| (target.target_ordinal, target.sample_id));
            Some((
                group_id,
                targets
                    .into_iter()
                    .map(|target| target.sample_id)
                    .collect::<Vec<_>>(),
            ))
        })
        .collect::<Vec<_>>();
    ready_groups.sort_by_key(|(group_id, sample_ids)| {
        (sample_ids.first().copied().unwrap_or(u64::MAX), *group_id)
    });

    let ready_targets = ready_groups
        .iter()
        .map(|(_, sample_ids)| sample_ids.len())
        .sum::<usize>();
    if ready_targets == 0 || (!force && ready_targets < microbatch_targets) {
        return Vec::new();
    }
    if force {
        return ready_groups;
    }

    let mut selected = Vec::new();
    let mut selected_targets = 0;
    for group in ready_groups {
        selected_targets += group.1.len();
        selected.push(group);
        if selected_targets >= microbatch_targets {
            break;
        }
    }
    selected
}

fn should_use_completion(
    requested: bool,
    plans: &[RangePlan],
    pending: &HashMap<u64, TargetClosure>,
    resolved: &HashMap<u64, Vec<u8>>,
    microbatch_targets: usize,
) -> bool {
    if !requested || plans.len() <= 1 {
        return false;
    }
    let max_single_plan_ready = plans
        .iter()
        .map(|plan| {
            let plan_records: HashSet<u64> =
                plan.records.iter().map(|record| record.record_id).collect();
            pending
                .values()
                .filter(|target| {
                    (resolved.contains_key(&target.anchor_id)
                        || plan_records.contains(&target.anchor_id))
                        && target.delta_id.is_none_or(|delta_id| {
                            resolved.contains_key(&delta_id) || plan_records.contains(&delta_id)
                        })
                })
                .count()
        })
        .max()
        .unwrap_or(0);
    plans.len() > microbatch_targets || max_single_plan_ready >= microbatch_targets
}

pub struct NormalizedBatchExecutor {
    descriptors: HashMap<u64, NormalizedDescriptor>,
    dependency_layout: HierarchicalLayoutIndex,
    outer_planner: OuterPlanner,
    backend: Box<dyn StorageBackend + Send>,
    merge_threshold_bytes: Option<u64>,
    range_planning: RangePlanning,
    max_range_bytes: Option<u64>,
    anchor_cache: SharedByteCache,
    delta_cache: SharedByteCache,
    decoded_cache: ByteCache,
    decoder_slots: decoder::SharedDecoderSlots,
    decode_microbatch_targets: usize,
    decode_schedule: DecodeSchedule,
    width: u32,
    height: u32,
}

impl NormalizedBatchExecutor {
    #[cfg(test)]
    pub(crate) fn decoder_slots_handle(&self) -> &decoder::SharedDecoderSlots {
        &self.decoder_slots
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        descriptors: Vec<NormalizedDescriptor>,
        backend: Box<dyn StorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        anchor_cache_bytes: usize,
        delta_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        decode_microbatch_targets: usize,
        fuse_shared_anchors: bool,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let decode_schedule = if fuse_shared_anchors {
            DecodeSchedule::Fused
        } else {
            DecodeSchedule::Repeated
        };
        Self::new_with_decode_schedule(
            descriptors,
            backend,
            merge_threshold_bytes,
            max_range_bytes,
            anchor_cache_bytes,
            delta_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            decode_microbatch_targets,
            decode_schedule,
            width,
            height,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_decode_schedule(
        descriptors: Vec<NormalizedDescriptor>,
        backend: Box<dyn StorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        anchor_cache_bytes: usize,
        delta_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        decode_microbatch_targets: usize,
        decode_schedule: DecodeSchedule,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        Self::new_with_decode_schedule_and_slots(
            descriptors,
            backend,
            merge_threshold_bytes,
            max_range_bytes,
            anchor_cache_bytes,
            delta_cache_bytes,
            decoded_cache_bytes,
            decode_microbatch_targets,
            decode_schedule,
            decoder::shared_decoder_slots(decode_concurrency),
            width,
            height,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_decode_schedule_and_slots(
        descriptors: Vec<NormalizedDescriptor>,
        backend: Box<dyn StorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        anchor_cache_bytes: usize,
        delta_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_microbatch_targets: usize,
        decode_schedule: DecodeSchedule,
        decoder_slots: decoder::SharedDecoderSlots,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        Self::new_with_decode_schedule_and_slots_and_anchor_cache(
            descriptors,
            backend,
            merge_threshold_bytes,
            max_range_bytes,
            planner::shared_byte_cache(anchor_cache_bytes),
            delta_cache_bytes,
            decoded_cache_bytes,
            decode_microbatch_targets,
            decode_schedule,
            decoder_slots,
            width,
            height,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_decode_schedule_and_slots_and_anchor_cache(
        descriptors: Vec<NormalizedDescriptor>,
        backend: Box<dyn StorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        anchor_cache: SharedByteCache,
        delta_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_microbatch_targets: usize,
        decode_schedule: DecodeSchedule,
        decoder_slots: decoder::SharedDecoderSlots,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        if decoder_slots.is_empty() {
            return Err("shared decoder slots must be non-empty".to_string());
        }
        let decode_schedule = decode_schedule.validate()?;
        let dependency_layout = Self::build_dependency_layout(&descriptors)?;
        let mut by_id = HashMap::with_capacity(descriptors.len());
        let mut samples = Vec::with_capacity(descriptors.len());
        let mut anchor_locations = HashMap::new();
        let mut group_ordinals = HashSet::new();
        for descriptor in descriptors {
            let anchor_id = Self::anchor_id(descriptor.anchor_group_id);
            if let Some(previous) = anchor_locations.insert(
                descriptor.anchor_group_id,
                (descriptor.anchor_offset, descriptor.anchor_length),
            ) {
                if previous != (descriptor.anchor_offset, descriptor.anchor_length) {
                    return Err(format!(
                        "anchor group {} has conflicting physical locations",
                        descriptor.anchor_group_id
                    ));
                }
            }
            if decode_schedule.requires_group_metadata() {
                if !group_ordinals.insert((descriptor.anchor_group_id, descriptor.target_ordinal)) {
                    return Err(format!(
                        "anchor group {} has duplicate target ordinal {}",
                        descriptor.anchor_group_id, descriptor.target_ordinal
                    ));
                }
            }
            let mut records = vec![PhysicalRecord {
                record_id: anchor_id,
                kind: DependencyKind::Anchor,
                offset: descriptor.anchor_offset,
                length: descriptor.anchor_length,
            }];
            if descriptor.target_ordinal > 0 {
                if descriptor.delta_length == 0 {
                    return Err(format!(
                        "normalized Delta sample {} has zero length",
                        descriptor.sample_id
                    ));
                }
                records.push(PhysicalRecord {
                    record_id: descriptor.sample_id,
                    kind: DependencyKind::Delta,
                    offset: descriptor.delta_offset,
                    length: descriptor.delta_length,
                });
            } else if descriptor.delta_length != 0 {
                return Err(format!(
                    "normalized Anchor sample {} must not carry a Delta",
                    descriptor.sample_id
                ));
            }
            samples.push(SampleRepresentations {
                sample_id: descriptor.sample_id,
                prefix: None,
                normalized: Some(DependencyClosure {
                    sample_id: descriptor.sample_id,
                    video_id: descriptor.video_id,
                    representation: Representation::Normalized,
                    records,
                    target_ordinal: descriptor.target_ordinal,
                }),
                pair: None,
            });
            if by_id.insert(descriptor.sample_id, descriptor).is_some() {
                return Err("duplicate normalized sample_id".to_string());
            }
        }
        let outer_planner = OuterPlanner::new(samples)?;
        Ok(Self {
            descriptors: by_id,
            dependency_layout,
            outer_planner,
            backend,
            merge_threshold_bytes,
            range_planning: RangePlanning::ByteGap,
            max_range_bytes,
            anchor_cache,
            delta_cache: planner::shared_byte_cache(delta_cache_bytes),
            decoded_cache: ByteCache::new(decoded_cache_bytes),
            decoder_slots,
            decode_microbatch_targets: decode_microbatch_targets.max(1),
            decode_schedule,
            width,
            height,
        })
    }

    pub(crate) fn set_shared_delta_cache(&mut self, cache: SharedByteCache) {
        self.delta_cache = cache;
    }

    fn anchor_id(video_id: u64) -> u64 {
        (1u64 << 63) | video_id
    }

    fn build_dependency_layout(
        descriptors: &[NormalizedDescriptor],
    ) -> Result<HierarchicalLayoutIndex, String> {
        let mut records = HashMap::<u64, AccessUnitRecord>::new();
        let mut closures = Vec::with_capacity(descriptors.len());
        let mut group_records = HashMap::<(u64, u64), Vec<(usize, u64)>>::new();
        for descriptor in descriptors {
            let group_key = (descriptor.video_id, descriptor.anchor_group_id);
            let anchor_id = Self::anchor_id(descriptor.anchor_group_id);
            let anchor = AccessUnitRecord {
                record_id: anchor_id,
                video_id: descriptor.video_id,
                gop_id: descriptor.anchor_group_id,
                offset: descriptor.anchor_offset,
                length: descriptor.anchor_length,
                decode_ordinal: 0,
            };
            if let Some(previous) = records.insert(anchor_id, anchor.clone()) {
                if previous != anchor {
                    return Err(format!(
                        "normalized Anchor group {} has inconsistent layout metadata",
                        descriptor.anchor_group_id
                    ));
                }
            }
            let group = group_records.entry(group_key).or_default();
            if !group.iter().any(|(_, record_id)| *record_id == anchor_id) {
                group.push((0, anchor_id));
            }
            let mut closure_ids = vec![anchor_id];
            let target_record_id = if descriptor.target_ordinal == 0 {
                anchor_id
            } else {
                let delta_id = descriptor.sample_id;
                let delta = AccessUnitRecord {
                    record_id: delta_id,
                    video_id: descriptor.video_id,
                    gop_id: descriptor.anchor_group_id,
                    offset: descriptor.delta_offset,
                    length: descriptor.delta_length,
                    decode_ordinal: descriptor.target_ordinal,
                };
                if records.insert(delta_id, delta).is_some() {
                    return Err(format!("duplicate normalized Delta record {delta_id}"));
                }
                group.push((descriptor.target_ordinal, delta_id));
                closure_ids.push(delta_id);
                delta_id
            };
            closures.push(LayoutTargetClosure {
                sample_id: descriptor.sample_id,
                video_id: descriptor.video_id,
                gop_id: descriptor.anchor_group_id,
                target_record_id,
                record_ids: closure_ids,
            });
        }
        let mut regions = Vec::with_capacity(group_records.len());
        for ((video_id, gop_id), mut ordered) in group_records {
            ordered.sort_unstable();
            let record_ids = ordered
                .into_iter()
                .map(|(_, record_id)| record_id)
                .collect::<Vec<_>>();
            let offset = record_ids
                .iter()
                .map(|record_id| records[record_id].offset)
                .min()
                .unwrap();
            let end = record_ids
                .iter()
                .map(|record_id| records[record_id].offset + records[record_id].length)
                .max()
                .unwrap();
            regions.push(GopRegion {
                video_id,
                gop_id,
                offset,
                length: end - offset,
                record_ids,
            });
        }
        HierarchicalLayoutIndex::new_with_region_decode_mode(
            records.into_values().collect(),
            closures,
            regions,
            RegionDecodeMode::ClosureOnly,
        )
    }

    pub fn choose_lookahead(
        &self,
        batches: &[Vec<u64>],
        candidates: &[usize],
        first_batch_slo_ns: f64,
        model: &HierarchicalCostModel,
    ) -> Result<DependencyLookaheadDecision, String> {
        let max_lookahead = candidates.iter().copied().max().unwrap_or(0);
        if max_lookahead == 0 || max_lookahead > batches.len() {
            return Err("lookahead candidate falls outside the available batch window".to_string());
        }
        let relevant_descriptors = batches[..max_lookahead]
            .iter()
            .flatten()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|sample_id| {
                self.descriptors
                    .get(&sample_id)
                    .ok_or_else(|| format!("unknown normalized sample {sample_id}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let anchor_cache = self
            .anchor_cache
            .lock()
            .map_err(|_| "shared Anchor cache lock poisoned".to_string())?;
        let delta_cache = self
            .delta_cache
            .lock()
            .map_err(|_| "shared Delta cache lock poisoned".to_string())?;
        let resident_record_ids = relevant_descriptors
            .iter()
            .flat_map(|descriptor| {
                let anchor_id = Self::anchor_id(descriptor.anchor_group_id);
                let delta_id = (descriptor.target_ordinal > 0).then_some(descriptor.sample_id);
                [
                    anchor_cache.contains(anchor_id).then_some(anchor_id),
                    delta_id.filter(|record_id| delta_cache.contains(*record_id)),
                ]
                .into_iter()
                .flatten()
            })
            .collect::<HashSet<_>>();
        self.dependency_layout.choose_lookahead_with_resident(
            batches,
            candidates,
            first_batch_slo_ns,
            &resident_record_ids,
            self.merge_threshold_bytes,
            self.max_range_bytes,
            model,
        )
    }

    fn extract_plan(
        plan: &RangePlan,
        buffer: &[u8],
        resolved: &mut HashMap<u64, Vec<u8>>,
    ) -> Result<(), String> {
        if buffer.len() != plan.length as usize {
            return Err(format!(
                "short normalized range: {} != {}",
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
                return Err("normalized record exceeds fetched range".to_string());
            }
            let value = buffer[start..end].to_vec();
            resolved.insert(*record_id, value);
        }
        Ok(())
    }

    fn bind_explicit_ranges(
        ranges: &[(u64, u64)],
        missing: &[RecordRange],
    ) -> Result<Vec<RangePlan>, String> {
        let (plans, fallback_records) =
            Self::bind_explicit_ranges_internal(ranges, missing, false)?;
        debug_assert_eq!(fallback_records, 0);
        Ok(plans)
    }

    fn bind_explicit_ranges_with_cache_fallback(
        ranges: &[(u64, u64)],
        missing: &[RecordRange],
    ) -> Result<(Vec<RangePlan>, usize), String> {
        Self::bind_explicit_ranges_internal(ranges, missing, true)
    }

    fn bind_explicit_ranges_internal(
        ranges: &[(u64, u64)],
        missing: &[RecordRange],
        add_cache_race_fallback: bool,
    ) -> Result<(Vec<RangePlan>, usize), String> {
        if ranges.is_empty() {
            return if missing.is_empty() {
                Ok((Vec::new(), 0))
            } else if add_cache_race_fallback {
                Ok((
                    missing
                        .iter()
                        .map(|record| RangePlan {
                            offset: record.offset,
                            length: record.length,
                            records: vec![PlannedRecord {
                                record_id: record.record_id,
                                relative_offset: 0,
                                length: record.length,
                            }],
                        })
                        .collect(),
                    missing.len(),
                ))
            } else {
                Err("explicit normalized plan contains no ranges".to_string())
            };
        }
        let mut covered = HashSet::new();
        let mut plans = Vec::with_capacity(ranges.len());
        for &(offset, length) in ranges {
            let end = offset
                .checked_add(length)
                .ok_or_else(|| "explicit normalized range overflows u64".to_string())?;
            if length == 0 {
                return Err("explicit normalized range has zero length".to_string());
            }
            let records = missing
                .iter()
                .filter(|record| {
                    record.offset >= offset && record.offset.saturating_add(record.length) <= end
                })
                .map(|record| {
                    if !covered.insert(record.record_id) {
                        return Err(format!(
                            "explicit normalized ranges cover record {} more than once",
                            record.record_id
                        ));
                    }
                    Ok(PlannedRecord {
                        record_id: record.record_id,
                        relative_offset: record.offset - offset,
                        length: record.length,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            if records.is_empty() {
                // Another executor can populate a shared cache after selection but
                // before this executor binds the selected spans. Drop stale ranges.
                continue;
            }
            plans.push(RangePlan {
                offset,
                length,
                records,
            });
        }
        let mut omitted = missing
            .iter()
            .filter(|record| !covered.contains(&record.record_id))
            .collect::<Vec<_>>();
        omitted.sort_unstable_by_key(|record| (record.offset, record.record_id));
        if !omitted.is_empty() && !add_cache_race_fallback {
            return Err(format!(
                "explicit normalized plan omits {} records: {:?}",
                omitted.len(),
                omitted
                    .iter()
                    .map(|record| record.record_id)
                    .collect::<Vec<_>>()
            ));
        }
        for record in &omitted {
            plans.push(RangePlan {
                offset: record.offset,
                length: record.length,
                records: vec![PlannedRecord {
                    record_id: record.record_id,
                    relative_offset: 0,
                    length: record.length,
                }],
            });
        }
        Ok((plans, omitted.len()))
    }

    fn mark_first_batch_ready(
        results: &HashMap<u64, Vec<u8>>,
        first_batch_ids: &HashSet<u64>,
        first_batch_ready_ns: &mut Option<u64>,
        total_started: &Instant,
    ) {
        if first_batch_ready_ns.is_none()
            && first_batch_ids
                .iter()
                .all(|sample_id| results.contains_key(sample_id))
        {
            *first_batch_ready_ns = Some(total_started.elapsed().as_nanos() as u64);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_ready(
        pending: &mut HashMap<u64, TargetClosure>,
        resolved: &HashMap<u64, Vec<u8>>,
        results: &mut HashMap<u64, Vec<u8>>,
        newly_decoded: &mut HashSet<u64>,
        decoder_slots: &[Mutex<decoder::DecoderPool>],
        decode_microbatch_targets: usize,
        decode_schedule: DecodeSchedule,
        force: bool,
        stats: &mut NormalizedStats,
        total_started: &Instant,
        first_batch_ids: &HashSet<u64>,
        first_batch_ready_ns: &mut Option<u64>,
    ) -> Result<(), String> {
        Self::mark_first_batch_ready(
            results,
            first_batch_ids,
            first_batch_ready_ns,
            total_started,
        );
        let priority_sample_ids = first_batch_ready_ns.is_none().then_some(first_batch_ids);
        let selected_groups = select_ready_groups(
            pending,
            resolved,
            decode_microbatch_targets,
            force,
            priority_sample_ids,
        );
        if selected_groups.is_empty() {
            return Ok(());
        }

        let mut by_group: HashMap<u64, Vec<TargetClosure>> = HashMap::new();
        for (group_id, sample_ids) in selected_groups {
            for sample_id in sample_ids {
                let target = pending
                    .remove(&sample_id)
                    .ok_or_else(|| "ready target disappeared".to_string())?;
                by_group.entry(group_id).or_default().push(target);
            }
        }
        let fuse_shared_anchors =
            should_fuse_groups(&by_group, decoder_slots.len(), decode_schedule);
        stats.decode_groups += by_group.len();
        stats.anchor_decode_invocations += if fuse_shared_anchors {
            by_group.len()
        } else {
            by_group.values().map(Vec::len).sum()
        };
        if fuse_shared_anchors {
            stats.fused_decode_groups += by_group.len();
        } else {
            stats.repeated_decode_groups += by_group.len();
        }

        let mut partitions: Vec<Vec<(u64, Vec<TargetClosure>)>> =
            (0..decoder_slots.len()).map(|_| Vec::new()).collect();
        for (group_id, mut targets) in by_group {
            targets.sort_by_key(|target| target.target_ordinal);
            if fuse_shared_anchors {
                let slot = group_id as usize % decoder_slots.len();
                partitions[slot].push((group_id, targets));
            } else {
                for (index, target) in targets.into_iter().enumerate() {
                    let slot = (group_id as usize + index) % decoder_slots.len();
                    if let Some((_, slot_targets)) = partitions[slot]
                        .iter_mut()
                        .find(|(existing_group, _)| *existing_group == group_id)
                    {
                        slot_targets.push(target);
                    } else {
                        partitions[slot].push((group_id, vec![target]));
                    }
                }
            }
        }
        let decoded = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (slot, groups) in decoder_slots.iter().zip(partitions) {
                if groups.is_empty() {
                    continue;
                }
                handles.push(scope.spawn(move || {
                    let mut thread_assemble_ns = 0;
                    let mut thread_decode_ns = 0;
                    let mut pool = slot
                        .lock()
                        .map_err(|_| "normalized decoder slot lock poisoned".to_string())?;
                    let mut output = Vec::new();
                    for (_group_id, targets) in groups {
                        let assemble_started = Instant::now();
                        let anchor = resolved
                            .get(&targets[0].anchor_id)
                            .ok_or_else(|| "ready target lost anchor".to_string())?;
                        let include_anchor = targets
                            .first()
                            .is_some_and(|target| target.target_ordinal == 0);
                        let deltas: Vec<Vec<u8>> = targets
                            .iter()
                            .filter_map(|target| {
                                let delta_id = target.delta_id?;
                                Some(
                                    resolved
                                        .get(&delta_id)
                                        .expect("ready target lost delta")
                                        .clone(),
                                )
                            })
                            .collect();
                        let repeated_parts = if fuse_shared_anchors {
                            None
                        } else {
                            let (codec_config, idr) = decoder::extract_anchor_parts(anchor)
                                .map_err(|error| error.to_string())?;
                            let records = deltas
                                .iter()
                                .map(|delta| [&idr[..], &delta[..]].concat())
                                .collect::<Vec<_>>();
                            Some((codec_config, records))
                        };
                        thread_assemble_ns += assemble_started.elapsed().as_nanos() as u64;
                        let decode_started = Instant::now();
                        let frames = if let Some((codec_config, records)) = repeated_parts {
                            let mut frames = if include_anchor {
                                decoder::decode_shared_anchor_group_rgb24(
                                    anchor,
                                    &[],
                                    true,
                                    &mut pool,
                                )
                                .map_err(|error| error.to_string())?
                            } else {
                                Vec::new()
                            };
                            let delta_frames = decoder::decode_closed_targets_continuous_rgb24(
                                &codec_config,
                                &records,
                                &mut pool,
                                2,
                            )
                            .map_err(|error| error.to_string())?;
                            frames.extend(delta_frames);
                            Ok(frames)
                        } else {
                            decoder::decode_shared_anchor_group_rgb24(
                                anchor,
                                &deltas,
                                include_anchor,
                                &mut pool,
                            )
                        }
                        .map_err(|error| error.to_string())?;
                        thread_decode_ns += decode_started.elapsed().as_nanos() as u64;
                        if frames.len() != targets.len() {
                            return Err(format!(
                                "decoded {} normalized targets from {} closures",
                                frames.len(),
                                targets.len()
                            ));
                        }
                        output.extend(
                            targets
                                .into_iter()
                                .zip(frames)
                                .map(|(target, frame)| (target.sample_id, frame.data)),
                        );
                    }
                    Ok::<_, String>((output, thread_assemble_ns, thread_decode_ns))
                }));
            }
            let mut output = Vec::new();
            let mut assemble_ns = 0;
            let mut decode_ns = 0;
            for handle in handles {
                let (frames, thread_assemble_ns, thread_decode_ns) = handle
                    .join()
                    .map_err(|_| "normalized decoder worker panicked".to_string())??;
                output.extend(frames);
                assemble_ns = assemble_ns.max(thread_assemble_ns);
                decode_ns = decode_ns.max(thread_decode_ns);
            }
            Ok::<_, String>((output, assemble_ns, decode_ns))
        })?;
        let (decoded, assemble_ns, decode_ns) = decoded;
        for (sample_id, rgb) in decoded {
            results.insert(sample_id, rgb);
            newly_decoded.insert(sample_id);
            stats.decoded_targets += 1;
            stats.decoded_frames += 1;
        }
        stats.assemble_ns += assemble_ns;
        stats.decode_ns += decode_ns;
        if stats.time_to_first_ready_ns == 0 && !results.is_empty() {
            stats.time_to_first_ready_ns = total_started.elapsed().as_nanos() as u64;
        }
        Self::mark_first_batch_ready(
            results,
            first_batch_ids,
            first_batch_ready_ns,
            total_started,
        );
        Ok(())
    }

    fn execute_internal(
        &mut self,
        batch: &[u64],
        completion_driven: bool,
        explicit_ranges: Option<&[(u64, u64)]>,
        first_batch: &[u64],
    ) -> Result<(Vec<NormalizedFrame>, NormalizedStats, u64), String> {
        let total_started = Instant::now();
        if first_batch.is_empty() {
            return Err("normalized first batch must be non-empty".to_string());
        }
        let first_batch_ids = first_batch.iter().copied().collect::<HashSet<_>>();
        let mut first_batch_ready_ns = None;
        let mut stats = NormalizedStats {
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
        let resolved_closures = self
            .outer_planner
            .resolve_closures(&requests, Representation::Normalized)?;
        stats.unique_targets = resolved_closures.len();
        stats.unique_videos = resolved_closures
            .iter()
            .map(|closure| closure.video_id)
            .collect::<HashSet<_>>()
            .len();
        stats.dependency_records = resolved_closures
            .iter()
            .map(|closure| closure.records.len())
            .sum();
        stats.target_ordinal_sum = resolved_closures
            .iter()
            .map(|closure| closure.target_ordinal)
            .sum();
        stats.target_ordinal_max = resolved_closures
            .iter()
            .map(|closure| closure.target_ordinal)
            .max()
            .unwrap_or(0);
        let unique = resolved_closures
            .iter()
            .map(|closure| closure.sample_id)
            .collect::<Vec<_>>();
        let closures = resolved_closures
            .into_iter()
            .map(|closure| {
                let descriptor = self
                    .descriptors
                    .get(&closure.sample_id)
                    .ok_or_else(|| "normalized descriptor disappeared".to_string())?;
                let anchor_id = closure
                    .records
                    .iter()
                    .find(|record| record.kind == DependencyKind::Anchor)
                    .ok_or_else(|| "normalized closure lost Anchor".to_string())?
                    .record_id;
                let delta_id = closure
                    .records
                    .iter()
                    .find(|record| record.kind == DependencyKind::Delta)
                    .map(|record| record.record_id);
                if descriptor.target_ordinal > 0 && delta_id.is_none() {
                    return Err("normalized Delta closure lost Delta".to_string());
                }
                if descriptor.target_ordinal == 0 && delta_id.is_some() {
                    return Err("normalized Anchor closure unexpectedly contains Delta".to_string());
                }
                Ok(TargetClosure {
                    sample_id: closure.sample_id,
                    video_id: closure.video_id,
                    anchor_group_id: descriptor.anchor_group_id,
                    target_ordinal: descriptor.target_ordinal,
                    anchor_id,
                    delta_id,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        stats.resolve_ns = resolve_started.elapsed().as_nanos() as u64;

        let cache_started = Instant::now();
        let mut results = HashMap::new();
        let mut newly_decoded = HashSet::new();
        let mut pending = HashMap::new();
        for closure in closures {
            if let Some(rgb) = self.decoded_cache.get(closure.sample_id) {
                results.insert(closure.sample_id, rgb);
                stats.decoded_cache_hits += 1;
            } else {
                stats.decoded_cache_misses += 1;
                pending.insert(closure.sample_id, closure);
            }
        }

        let mut physical: HashMap<u64, (u64, u64)> = HashMap::new();
        let mut record_groups = HashMap::<u64, u64>::new();
        let mut anchor_ids = HashSet::new();
        for closure in pending.values() {
            let descriptor = self
                .descriptors
                .get(&closure.sample_id)
                .expect("resolved descriptor disappeared");
            physical
                .entry(closure.anchor_id)
                .or_insert((descriptor.anchor_offset, descriptor.anchor_length));
            record_groups.insert(closure.anchor_id, closure.anchor_group_id);
            anchor_ids.insert(closure.anchor_id);
            if let Some(delta_id) = closure.delta_id {
                physical.insert(delta_id, (descriptor.delta_offset, descriptor.delta_length));
                record_groups.insert(delta_id, closure.anchor_group_id);
            }
        }
        stats.unique_records = physical.len();
        let mut resolved = HashMap::new();
        let mut misses = Vec::new();
        {
            let mut anchor_cache = self
                .anchor_cache
                .lock()
                .map_err(|_| "shared Anchor cache lock poisoned".to_string())?;
            let mut delta_cache = self
                .delta_cache
                .lock()
                .map_err(|_| "shared Delta cache lock poisoned".to_string())?;
            for (&record_id, &(offset, length)) in &physical {
                let cached = if anchor_ids.contains(&record_id) {
                    let value = anchor_cache.get(record_id);
                    if value.is_some() {
                        stats.anchor_cache_hits += 1;
                    } else {
                        stats.anchor_cache_misses += 1;
                    }
                    value
                } else {
                    let value = delta_cache.get(record_id);
                    if value.is_some() {
                        stats.delta_cache_hits += 1;
                    } else {
                        stats.delta_cache_misses += 1;
                    }
                    value
                };
                if let Some(value) = cached {
                    resolved.insert(record_id, value);
                    stats.encoded_cache_hits += 1;
                } else {
                    stats.encoded_cache_misses += 1;
                    misses.push(RecordRange {
                        record_id,
                        offset,
                        length,
                    });
                }
            }
        }
        stats.cache_lookup_ns = cache_started.elapsed().as_nanos() as u64;

        Self::decode_ready(
            &mut pending,
            &resolved,
            &mut results,
            &mut newly_decoded,
            &self.decoder_slots,
            self.decode_microbatch_targets,
            self.decode_schedule,
            false,
            &mut stats,
            &total_started,
            &first_batch_ids,
            &mut first_batch_ready_ns,
        )?;

        let plan_started = Instant::now();
        let (plans, cache_race_fallback_records) = if let Some(ranges) = explicit_ranges {
            Self::bind_explicit_ranges_with_cache_fallback(ranges, &misses)?
        } else {
            let plans = match self.range_planning {
                RangePlanning::ByteGap => planner::plan_byte_ranges(
                    &misses,
                    self.merge_threshold_bytes,
                    self.max_range_bytes,
                )?,
                RangePlanning::DependencyGroupSpan => {
                    let grouped = misses
                        .iter()
                        .map(|record| {
                            record_groups
                                .get(&record.record_id)
                                .copied()
                                .map(|group_id| (group_id, record.clone()))
                                .ok_or_else(|| {
                                    format!(
                                        "missing dependency group for record {}",
                                        record.record_id
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    planner::plan_group_spans(&grouped, self.max_range_bytes)?
                }
            };
            (plans, 0)
        };
        stats.cache_race_fallback_records = cache_race_fallback_records;
        stats.plan_ns = plan_started.elapsed().as_nanos() as u64;
        stats.physical_ranges = plans.len();
        stats.useful_bytes = plans
            .iter()
            .flat_map(|plan| &plan.records)
            .map(|record| record.length)
            .sum();
        stats.planned_ranges = plans.len();
        stats.planned_useful_bytes = stats.useful_bytes;
        stats.planned_fetched_bytes = plans.iter().map(|plan| plan.length).sum();
        let ranges: Vec<(u64, u64)> = plans
            .iter()
            .map(|plan| (plan.offset, plan.length))
            .collect();
        stats.physical_ranges = self.backend.physical_ranges_for_ranges(&ranges);
        stats.client_requests = self.backend.client_requests_for_ranges(&ranges);
        stats.server_entries = self.backend.server_entries_for_ranges(&ranges);

        let use_completion = self.backend.supports_streaming_range_completion()
            && should_use_completion(
                completion_driven,
                &plans,
                &pending,
                &resolved,
                self.decode_microbatch_targets,
            );
        stats.completion_selected = use_completion;

        let fetch_started = Instant::now();
        if use_completion {
            let backend = &self.backend;
            let decoder_slots = &self.decoder_slots;
            let decode_microbatch_targets = self.decode_microbatch_targets;
            let decode_schedule = self.decode_schedule;
            backend
                .for_each_byte_range(&ranges, &mut |completed: CompletedRange| {
                    stats.fetch_wall_ns = stats.fetch_wall_ns.max(completed.completed_ns);
                    stats.fetch_service_ns_sum +=
                        completed.completed_ns.saturating_sub(completed.started_ns);
                    stats.range_queue_ns_sum += completed.started_ns;
                    stats.fetched_bytes += completed.bytes.len() as u64;
                    let extract_started = Instant::now();
                    let plan = plans
                        .get(completed.index)
                        .ok_or_else(|| "backend returned unknown normalized plan".to_string())?;
                    Self::extract_plan(plan, &completed.bytes, &mut resolved)?;
                    stats.extract_ns += extract_started.elapsed().as_nanos() as u64;
                    Self::decode_ready(
                        &mut pending,
                        &resolved,
                        &mut results,
                        &mut newly_decoded,
                        decoder_slots,
                        decode_microbatch_targets,
                        decode_schedule,
                        false,
                        &mut stats,
                        &total_started,
                        &first_batch_ids,
                        &mut first_batch_ready_ns,
                    )?;
                    Ok(())
                })
                .map_err(|error| error.to_string())?;
        } else {
            let mut completed = Vec::with_capacity(ranges.len());
            self.backend
                .for_each_byte_range(&ranges, &mut |range| {
                    stats.fetch_wall_ns = stats.fetch_wall_ns.max(range.completed_ns);
                    stats.fetch_service_ns_sum +=
                        range.completed_ns.saturating_sub(range.started_ns);
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
            Self::decode_ready(
                &mut pending,
                &resolved,
                &mut results,
                &mut newly_decoded,
                &self.decoder_slots,
                self.decode_microbatch_targets,
                self.decode_schedule,
                true,
                &mut stats,
                &total_started,
                &first_batch_ids,
                &mut first_batch_ready_ns,
            )?;
        }
        while !pending.is_empty() {
            let pending_before = pending.len();
            Self::decode_ready(
                &mut pending,
                &resolved,
                &mut results,
                &mut newly_decoded,
                &self.decoder_slots,
                self.decode_microbatch_targets,
                self.decode_schedule,
                true,
                &mut stats,
                &total_started,
                &first_batch_ids,
                &mut first_batch_ready_ns,
            )?;
            if pending.len() == pending_before {
                break;
            }
        }
        if let Some(physical_bytes) = self.backend.physical_fetched_bytes_for_ranges(&ranges) {
            stats.fetched_bytes = physical_bytes;
        }
        let fetch_decode_wall_ns = fetch_started.elapsed().as_nanos() as u64;
        stats.fetch_decode_overlap_ns = stats
            .fetch_wall_ns
            .saturating_add(stats.assemble_ns)
            .saturating_add(stats.decode_ns)
            .saturating_sub(fetch_decode_wall_ns);
        stats.overfetch_bytes = stats.fetched_bytes.saturating_sub(stats.useful_bytes);
        if !pending.is_empty() {
            return Err(format!(
                "{} normalized targets were not decoded",
                pending.len()
            ));
        }

        // Network completion order must not alter future cache behavior.
        {
            let mut anchor_cache = self
                .anchor_cache
                .lock()
                .map_err(|_| "shared Anchor cache lock poisoned".to_string())?;
            let mut delta_cache = self
                .delta_cache
                .lock()
                .map_err(|_| "shared Delta cache lock poisoned".to_string())?;
            for plan in &plans {
                for record in &plan.records {
                    if let Some(value) = resolved.get(&record.record_id) {
                        if anchor_ids.contains(&record.record_id) {
                            anchor_cache.put(record.record_id, value.clone());
                        } else {
                            delta_cache.put(record.record_id, value.clone());
                        }
                    }
                }
            }
        }
        for &sample_id in &unique {
            if newly_decoded.contains(&sample_id) {
                if let Some(rgb) = results.get(&sample_id) {
                    self.decoded_cache.put(sample_id, rgb.clone());
                }
            }
        }

        let reorder_started = Instant::now();
        let frames = batch
            .iter()
            .map(|&sample_id| {
                let rgb = results
                    .get(&sample_id)
                    .ok_or_else(|| format!("missing normalized result {sample_id}"))?
                    .clone();
                if rgb.len() != self.width as usize * self.height as usize * 3 {
                    return Err(format!(
                        "normalized RGB size mismatch for {sample_id}: {}",
                        rgb.len()
                    ));
                }
                Ok(NormalizedFrame {
                    sample_id,
                    rgb,
                    width: self.width,
                    height: self.height,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        stats.reorder_ns = reorder_started.elapsed().as_nanos() as u64;
        let anchor_cache_stats = self
            .anchor_cache
            .lock()
            .map_err(|_| "shared Anchor cache lock poisoned".to_string())?
            .stats();
        let delta_cache_stats = self
            .delta_cache
            .lock()
            .map_err(|_| "shared Delta cache lock poisoned".to_string())?
            .stats();
        let decoded_cache_stats = self.decoded_cache.stats();
        stats.anchor_cache_resident_bytes = anchor_cache_stats["resident_bytes"];
        stats.delta_cache_resident_bytes = delta_cache_stats["resident_bytes"];
        stats.encoded_cache_resident_bytes = stats
            .anchor_cache_resident_bytes
            .saturating_add(stats.delta_cache_resident_bytes);
        stats.decoded_cache_resident_bytes = decoded_cache_stats["resident_bytes"];
        stats.decoder_state_resident = self
            .decoder_slots
            .iter()
            .map(|slot| {
                slot.lock()
                    .map(|pool| pool.len())
                    .map_err(|_| "normalized decoder slot lock poisoned".to_string())
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .sum();
        stats.total_ns = total_started.elapsed().as_nanos() as u64;
        let first_batch_ready_ns = first_batch_ready_ns
            .ok_or_else(|| "normalized first batch never became ready".to_string())?;
        Ok((frames, stats, first_batch_ready_ns))
    }

    pub fn execute(
        &mut self,
        batch: &[u64],
        completion_driven: bool,
    ) -> Result<(Vec<NormalizedFrame>, NormalizedStats), String> {
        let (frames, stats, _) = self.execute_internal(batch, completion_driven, None, batch)?;
        Ok((frames, stats))
    }

    pub fn execute_planned_lookahead(
        &mut self,
        batches: &[Vec<u64>],
        candidates: &[usize],
        first_batch_slo_ns: f64,
        model: &HierarchicalCostModel,
        completion_driven: bool,
    ) -> Result<NormalizedLookaheadWindow, String> {
        let decision = self.choose_lookahead(batches, candidates, first_batch_slo_ns, model)?;
        let lookahead = decision.selected.lookahead_batches;
        let flattened = batches[..lookahead]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let first_extents = batches[0]
            .iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|sample_id| {
                self.descriptors
                    .get(sample_id)
                    .ok_or_else(|| format!("unknown normalized sample {sample_id}"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flat_map(|descriptor| {
                let mut extents = vec![(descriptor.anchor_offset, descriptor.anchor_length)];
                if descriptor.delta_length > 0 {
                    extents.push((descriptor.delta_offset, descriptor.delta_length));
                }
                extents
            })
            .collect::<Vec<_>>();
        let mut ranges = decision
            .selected
            .plan
            .ranges
            .iter()
            .map(|range| (range.offset, range.length))
            .collect::<Vec<_>>();
        ranges.sort_by_key(|(offset, length)| {
            let end = offset.saturating_add(*length);
            let serves_first_batch = first_extents.iter().any(|(first_offset, first_length)| {
                let first_end = first_offset.saturating_add(*first_length);
                *first_offset < end && *offset < first_end
            });
            (!serves_first_batch, *offset)
        });
        let (frames, stats, first_batch_ready_ns) =
            self.execute_internal(&flattened, completion_driven, Some(&ranges), &batches[0])?;
        Ok(NormalizedLookaheadWindow {
            frames,
            stats,
            decision,
            first_batch_ready_ns,
        })
    }

    pub fn execute_with_merge_threshold(
        &mut self,
        batch: &[u64],
        completion_driven: bool,
        merge_threshold_bytes: Option<u64>,
    ) -> Result<(Vec<NormalizedFrame>, NormalizedStats), String> {
        let previous = self.merge_threshold_bytes;
        self.merge_threshold_bytes = merge_threshold_bytes;
        let result = self.execute(batch, completion_driven);
        self.merge_threshold_bytes = previous;
        result
    }

    pub fn execute_with_dependency_group_spans(
        &mut self,
        batch: &[u64],
        completion_driven: bool,
    ) -> Result<(Vec<NormalizedFrame>, NormalizedStats), String> {
        let previous = self.range_planning;
        self.range_planning = RangePlanning::DependencyGroupSpan;
        let result = self.execute(batch, completion_driven);
        self.range_planning = previous;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(record_id: u64) -> RangePlan {
        RangePlan {
            offset: record_id * 10,
            length: 10,
            records: vec![PlannedRecord {
                record_id,
                relative_offset: 0,
                length: 10,
            }],
        }
    }

    #[test]
    fn completion_requires_multiple_ranges_and_ready_work() {
        let target = TargetClosure {
            sample_id: 1,
            video_id: 0,
            anchor_group_id: 0,
            target_ordinal: 1,
            anchor_id: 100,
            delta_id: Some(1),
        };
        let pending = HashMap::from([(1, target)]);
        let resolved = HashMap::new();
        assert!(!should_use_completion(
            true,
            &[plan(1)],
            &pending,
            &resolved,
            4
        ));
        assert!(!should_use_completion(
            true,
            &[plan(100), plan(1)],
            &pending,
            &resolved,
            4
        ));
    }

    #[test]
    fn completion_accepts_many_ranges_or_one_ready_microbatch() {
        let pending: HashMap<u64, TargetClosure> = (1..=4)
            .map(|sample_id| {
                (
                    sample_id,
                    TargetClosure {
                        sample_id,
                        video_id: 0,
                        anchor_group_id: 0,
                        target_ordinal: sample_id as usize,
                        anchor_id: 100,
                        delta_id: Some(sample_id),
                    },
                )
            })
            .collect();
        let resolved = HashMap::from([(100, vec![1])]);
        let many = vec![plan(1), plan(2), plan(3), plan(4), plan(5)];
        assert!(should_use_completion(true, &many, &pending, &resolved, 4));

        let one_batch = RangePlan {
            offset: 0,
            length: 40,
            records: (1..=4)
                .map(|record_id| PlannedRecord {
                    record_id,
                    relative_offset: (record_id - 1) * 10,
                    length: 10,
                })
                .collect(),
        };
        assert!(should_use_completion(
            true,
            &[one_batch, plan(9)],
            &pending,
            &resolved,
            4
        ));
    }

    #[test]
    fn decode_groups_follow_anchor_identity_not_video_identity() {
        let first = TargetClosure {
            sample_id: 1,
            video_id: 7,
            anchor_group_id: 70,
            target_ordinal: 1,
            anchor_id: 100,
            delta_id: Some(1),
        };
        let second = TargetClosure {
            sample_id: 2,
            video_id: 7,
            anchor_group_id: 71,
            target_ordinal: 1,
            anchor_id: 101,
            delta_id: Some(2),
        };
        assert_eq!(first.video_id, second.video_id);
        assert_ne!(decode_group_id(&first), decode_group_id(&second));
    }

    fn synthetic_group(group_id: u64, targets: usize) -> Vec<TargetClosure> {
        (1..=targets)
            .map(|ordinal| TargetClosure {
                sample_id: group_id * 100 + ordinal as u64,
                video_id: group_id,
                anchor_group_id: group_id,
                target_ordinal: ordinal,
                anchor_id: group_id + 10_000,
                delta_id: Some(group_id * 100 + ordinal as u64),
            })
            .collect()
    }

    #[test]
    fn adaptive_decode_schedule_trades_anchor_reuse_for_parallelism() {
        let one_large_group = HashMap::from([(0, synthetic_group(0, 16))]);
        let calibrated = DecodeSchedule::Adaptive {
            anchor_work_units: 2.3,
            delta_work_units: 1.0,
        };
        assert!(should_fuse_groups(&one_large_group, 1, calibrated));
        assert!(!should_fuse_groups(&one_large_group, 8, calibrated));

        let many_groups = (0..8)
            .map(|group_id| (group_id, synthetic_group(group_id, 4)))
            .collect::<HashMap<_, _>>();
        assert!(should_fuse_groups(&many_groups, 8, calibrated));
        assert!(should_fuse_groups(
            &one_large_group,
            8,
            DecodeSchedule::Fused
        ));
        assert!(!should_fuse_groups(
            &many_groups,
            1,
            DecodeSchedule::Repeated
        ));
    }

    #[test]
    fn adaptive_decode_schedule_rejects_invalid_calibration() {
        assert!(DecodeSchedule::Adaptive {
            anchor_work_units: 0.0,
            delta_work_units: 1.0,
        }
        .validate()
        .is_err());
        assert!(DecodeSchedule::Adaptive {
            anchor_work_units: 1.0,
            delta_work_units: f64::NAN,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn ready_selection_never_splits_an_anchor_group() {
        let pending: HashMap<u64, TargetClosure> = (1..=6)
            .map(|sample_id| {
                (
                    sample_id,
                    TargetClosure {
                        sample_id,
                        video_id: 0,
                        anchor_group_id: 70,
                        target_ordinal: sample_id as usize,
                        anchor_id: 100,
                        delta_id: Some(sample_id),
                    },
                )
            })
            .collect();
        let mut resolved = HashMap::from([(100, vec![1])]);
        for sample_id in 1..=4 {
            resolved.insert(sample_id, vec![1]);
        }
        assert!(select_ready_groups(&pending, &resolved, 4, false, None).is_empty());

        let first_batch = HashSet::from([1, 2, 3, 4]);
        assert_eq!(
            select_ready_groups(&pending, &resolved, 4, false, Some(&first_batch)),
            vec![(70, vec![1, 2, 3, 4])]
        );

        resolved.insert(5, vec![1]);
        resolved.insert(6, vec![1]);
        let selected = select_ready_groups(&pending, &resolved, 4, false, None);
        assert_eq!(selected, vec![(70, vec![1, 2, 3, 4, 5, 6])]);
    }

    #[test]
    fn fused_descriptor_accepts_anchor_only_group_target() {
        let descriptors = vec![
            NormalizedDescriptor {
                sample_id: 0,
                video_id: 0,
                anchor_group_id: 10,
                target_ordinal: 0,
                anchor_offset: 0,
                anchor_length: 100,
                delta_offset: 100,
                delta_length: 0,
            },
            NormalizedDescriptor {
                sample_id: 1,
                video_id: 0,
                anchor_group_id: 10,
                target_ordinal: 1,
                anchor_offset: 0,
                anchor_length: 100,
                delta_offset: 100,
                delta_length: 20,
            },
        ];
        NormalizedBatchExecutor::new(
            descriptors,
            Box::new(crate::backend::NoopBackend),
            None,
            None,
            0,
            0,
            0,
            1,
            2,
            true,
            320,
            240,
        )
        .expect("Anchor-only and Delta targets should share one Normalized group");
    }

    #[test]
    fn plan_features_are_cold_and_deduplicate_shared_anchor() {
        let descriptors = vec![
            NormalizedDescriptor {
                sample_id: 1,
                video_id: 0,
                anchor_group_id: 10,
                target_ordinal: 1,
                anchor_offset: 0,
                anchor_length: 10,
                delta_offset: 10,
                delta_length: 5,
            },
            NormalizedDescriptor {
                sample_id: 2,
                video_id: 0,
                anchor_group_id: 10,
                target_ordinal: 2,
                anchor_offset: 0,
                anchor_length: 10,
                delta_offset: 15,
                delta_length: 5,
            },
        ];
        let executor = NormalizedBatchExecutor::new(
            descriptors,
            Box::new(crate::backend::NoopBackend),
            None,
            None,
            1024,
            0,
            1024,
            1,
            2,
            true,
            320,
            240,
        )
        .unwrap();
        let features = executor.plan_features(&[1, 2, 2]).unwrap();
        assert_eq!(features.logical_requests, 3);
        assert_eq!(features.unique_targets, 2);
        assert_eq!(features.dependency_records, 4);
        assert_eq!(features.unique_records, 3);
        assert_eq!(features.useful_bytes, 20);
    }

    #[test]
    fn group_span_candidate_excludes_a_cached_anchor() {
        let descriptors = vec![
            NormalizedDescriptor {
                sample_id: 1,
                video_id: 0,
                anchor_group_id: 10,
                target_ordinal: 1,
                anchor_offset: 0,
                anchor_length: 10,
                delta_offset: 10,
                delta_length: 5,
            },
            NormalizedDescriptor {
                sample_id: 2,
                video_id: 0,
                anchor_group_id: 10,
                target_ordinal: 2,
                anchor_offset: 0,
                anchor_length: 10,
                delta_offset: 100,
                delta_length: 5,
            },
        ];
        let executor = NormalizedBatchExecutor::new(
            descriptors,
            Box::new(crate::backend::NoopBackend),
            None,
            None,
            1024,
            0,
            0,
            1,
            2,
            true,
            320,
            240,
        )
        .unwrap();
        executor
            .anchor_cache
            .lock()
            .unwrap()
            .put(NormalizedBatchExecutor::anchor_id(10), vec![0; 10]);

        let candidates = executor.adaptive_candidates(&[1, 2]).unwrap();
        let group_span = candidates
            .iter()
            .find(|candidate| candidate.mode == PlanMode::NormalizedGroupSpan)
            .unwrap();
        assert_eq!(group_span.range_lengths, vec![95]);
        assert_eq!(group_span.useful_bytes, 10);
        assert_eq!(group_span.anchor_decodes, 1);
        assert_eq!(group_span.delta_decodes, 2);
    }

    #[test]
    fn normalized_lookahead_uses_physical_extents_and_first_batch_slo() {
        let descriptors = vec![
            NormalizedDescriptor {
                sample_id: 1,
                video_id: 0,
                anchor_group_id: 10,
                target_ordinal: 1,
                anchor_offset: 0,
                anchor_length: 100,
                delta_offset: 100,
                delta_length: 10,
            },
            NormalizedDescriptor {
                sample_id: 2,
                video_id: 0,
                anchor_group_id: 10,
                target_ordinal: 2,
                anchor_offset: 0,
                anchor_length: 100,
                delta_offset: 110,
                delta_length: 10,
            },
        ];
        let executor = NormalizedBatchExecutor::new(
            descriptors,
            Box::new(crate::backend::NoopBackend),
            Some(0),
            None,
            0,
            0,
            0,
            1,
            1,
            true,
            320,
            240,
        )
        .unwrap();
        let model = HierarchicalCostModel {
            request_latency_ns: 1_000.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 1,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 1.0,
            fetch_decode_overlap: 0.0,
        };
        let batches = vec![vec![1], vec![2]];

        let loose = executor
            .choose_lookahead(&batches, &[1, 2], 2_000.0, &model)
            .unwrap();
        assert_eq!(loose.selected.lookahead_batches, 2);
        assert!(loose.selected_under_slo);

        let tight = executor
            .choose_lookahead(&batches, &[1, 2], 1_115.0, &model)
            .unwrap();
        assert_eq!(tight.selected.lookahead_batches, 1);
        assert!(tight.selected_under_slo);
    }

    #[test]
    fn normalized_lookahead_only_consults_candidate_batches() {
        let executor = NormalizedBatchExecutor::new(
            vec![NormalizedDescriptor {
                sample_id: 1,
                video_id: 0,
                anchor_group_id: 10,
                target_ordinal: 1,
                anchor_offset: 0,
                anchor_length: 100,
                delta_offset: 100,
                delta_length: 10,
            }],
            Box::new(crate::backend::NoopBackend),
            None,
            None,
            0,
            0,
            0,
            1,
            1,
            true,
            320,
            240,
        )
        .unwrap();
        let model = HierarchicalCostModel {
            request_latency_ns: 1_000.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 1,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 1.0,
            fetch_decode_overlap: 0.0,
        };

        let decision = executor
            .choose_lookahead(&[vec![1], vec![999]], &[1], 2_000.0, &model)
            .unwrap();
        assert_eq!(decision.selected.lookahead_batches, 1);
    }

    #[test]
    fn normalized_region_fetch_keeps_closure_only_decode_work() {
        let executor = NormalizedBatchExecutor::new(
            vec![
                NormalizedDescriptor {
                    sample_id: 1,
                    video_id: 0,
                    anchor_group_id: 10,
                    target_ordinal: 1,
                    anchor_offset: 0,
                    anchor_length: 100,
                    delta_offset: 100,
                    delta_length: 10,
                },
                NormalizedDescriptor {
                    sample_id: 2,
                    video_id: 0,
                    anchor_group_id: 10,
                    target_ordinal: 2,
                    anchor_offset: 0,
                    anchor_length: 100,
                    delta_offset: 110,
                    delta_length: 10,
                },
            ],
            Box::new(crate::backend::NoopBackend),
            None,
            None,
            0,
            0,
            0,
            1,
            1,
            true,
            320,
            240,
        )
        .unwrap();
        let model = HierarchicalCostModel {
            request_latency_ns: 1_000.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 1,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 1.0,
            fetch_decode_overlap: 0.0,
        };
        let decision = executor
            .dependency_layout
            .choose_plan(&[1], None, None, &model)
            .unwrap();
        let region = decision
            .alternatives
            .iter()
            .find(|candidate| {
                candidate.mode == crate::hierarchical_layout::HierarchicalReadMode::ContiguousRegion
            })
            .unwrap();
        assert_eq!(region.fetched_bytes, 120);
        assert_eq!(region.useful_bytes, 110);
        assert_eq!(region.access_units_submitted, 2);
    }

    #[test]
    fn explicit_ranges_bind_only_requested_records() {
        let missing = vec![
            RecordRange {
                record_id: 1,
                offset: 0,
                length: 10,
            },
            RecordRange {
                record_id: 2,
                offset: 100,
                length: 10,
            },
        ];
        let plans = NormalizedBatchExecutor::bind_explicit_ranges(&[(0, 110)], &missing).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].records.len(), 2);
        assert!(NormalizedBatchExecutor::bind_explicit_ranges(&[(0, 10)], &missing).is_err());

        let raced =
            NormalizedBatchExecutor::bind_explicit_ranges(&[(0, 10), (100, 10)], &missing[1..])
                .unwrap();
        assert_eq!(raced.len(), 1);
        assert_eq!(raced[0].records[0].record_id, 2);
        assert!(
            NormalizedBatchExecutor::bind_explicit_ranges(&[(0, 10)], &[])
                .unwrap()
                .is_empty()
        );

        let (fallback, count) =
            NormalizedBatchExecutor::bind_explicit_ranges_with_cache_fallback(&[(0, 10)], &missing)
                .unwrap();
        assert_eq!(count, 1);
        assert_eq!(fallback.len(), 2);
        assert_eq!(fallback[1].records[0].record_id, 2);
    }

    #[test]
    fn shared_anchor_cache_survives_across_executors_and_delta_churn() {
        let shared = planner::shared_byte_cache(8);
        let slots = decoder::shared_decoder_slots(1);
        let mut first =
            NormalizedBatchExecutor::new_with_decode_schedule_and_slots_and_anchor_cache(
                Vec::new(),
                Box::new(crate::backend::NoopBackend),
                None,
                None,
                shared.clone(),
                4,
                0,
                1,
                DecodeSchedule::Repeated,
                slots.clone(),
                1,
                1,
            )
            .unwrap();
        let mut second =
            NormalizedBatchExecutor::new_with_decode_schedule_and_slots_and_anchor_cache(
                Vec::new(),
                Box::new(crate::backend::NoopBackend),
                None,
                None,
                shared,
                4,
                0,
                1,
                DecodeSchedule::Repeated,
                slots,
                1,
                1,
            )
            .unwrap();

        first.anchor_cache.lock().unwrap().put(99, vec![1; 8]);
        first.delta_cache.lock().unwrap().put(1, vec![2; 4]);
        first.delta_cache.lock().unwrap().put(2, vec![3; 4]);

        assert_eq!(
            second.anchor_cache.lock().unwrap().get(99),
            Some(vec![1; 8])
        );
        assert_eq!(second.delta_cache.lock().unwrap().get(1), None);
    }

    #[test]
    fn configured_shared_delta_cache_survives_executor_recreation() {
        let shared_delta = planner::shared_byte_cache(8);
        let slots = decoder::shared_decoder_slots(1);
        let mut first = NormalizedBatchExecutor::new_with_decode_schedule_and_slots(
            Vec::new(),
            Box::new(crate::backend::NoopBackend),
            None,
            None,
            0,
            0,
            0,
            1,
            DecodeSchedule::Repeated,
            slots.clone(),
            1,
            1,
        )
        .unwrap();
        let mut second = NormalizedBatchExecutor::new_with_decode_schedule_and_slots(
            Vec::new(),
            Box::new(crate::backend::NoopBackend),
            None,
            None,
            0,
            0,
            0,
            1,
            DecodeSchedule::Repeated,
            slots,
            1,
            1,
        )
        .unwrap();
        first.set_shared_delta_cache(shared_delta.clone());
        second.set_shared_delta_cache(shared_delta);
        first.delta_cache.lock().unwrap().put(7, vec![1; 4]);
        assert_eq!(second.delta_cache.lock().unwrap().get(7), Some(vec![1; 4]));
    }
}

impl NormalizedBatchExecutor {
    /// Resolve a cold-cache physical plan without performing I/O or decode.
    pub fn plan_features(&self, batch: &[u64]) -> Result<BatchFeatures, String> {
        let requests = batch
            .iter()
            .map(|sample_id| LogicalRequest {
                sample_id: *sample_id,
            })
            .collect::<Vec<_>>();
        self.outer_planner
            .resolve_batch(
                &requests,
                Representation::Normalized,
                self.merge_threshold_bytes,
                self.max_range_bytes,
            )
            .map(|resolved| resolved.features)
    }

    /// Build cache-aware candidate plans without changing cache or decoder state.
    pub fn adaptive_candidates(&self, batch: &[u64]) -> Result<Vec<PlanCandidate>, String> {
        let requests = batch
            .iter()
            .map(|sample_id| LogicalRequest {
                sample_id: *sample_id,
            })
            .collect::<Vec<_>>();
        let closures = self
            .outer_planner
            .resolve_closures(&requests, Representation::Normalized)?
            .into_iter()
            .filter(|closure| !self.decoded_cache.contains(closure.sample_id))
            .collect::<Vec<_>>();

        let mut targets = HashMap::new();
        let mut records = HashMap::<u64, (u64, RecordRange)>::new();
        let anchor_cache = self
            .anchor_cache
            .lock()
            .map_err(|_| "shared Anchor cache lock poisoned".to_string())?;
        let delta_cache = self
            .delta_cache
            .lock()
            .map_err(|_| "shared Delta cache lock poisoned".to_string())?;
        for closure in &closures {
            let descriptor = self
                .descriptors
                .get(&closure.sample_id)
                .ok_or_else(|| format!("missing normalized descriptor {}", closure.sample_id))?;
            let anchor_id = Self::anchor_id(descriptor.anchor_group_id);
            let delta_id = (descriptor.target_ordinal > 0).then_some(descriptor.sample_id);
            targets.insert(
                closure.sample_id,
                TargetClosure {
                    sample_id: closure.sample_id,
                    video_id: closure.video_id,
                    anchor_group_id: descriptor.anchor_group_id,
                    target_ordinal: descriptor.target_ordinal,
                    anchor_id,
                    delta_id,
                },
            );
            if !anchor_cache.contains(anchor_id) {
                records.entry(anchor_id).or_insert((
                    descriptor.anchor_group_id,
                    RecordRange {
                        record_id: anchor_id,
                        offset: descriptor.anchor_offset,
                        length: descriptor.anchor_length,
                    },
                ));
            }
            if let Some(delta_id) = delta_id {
                if !delta_cache.contains(delta_id) {
                    records.entry(delta_id).or_insert((
                        descriptor.anchor_group_id,
                        RecordRange {
                            record_id: delta_id,
                            offset: descriptor.delta_offset,
                            length: descriptor.delta_length,
                        },
                    ));
                }
            }
        }
        drop(delta_cache);
        drop(anchor_cache);

        let mut groups = HashMap::<u64, Vec<TargetClosure>>::new();
        for target in targets.values().cloned() {
            groups
                .entry(target.anchor_group_id)
                .or_default()
                .push(target);
        }
        let fused = should_fuse_groups(&groups, self.decoder_slots.len(), self.decode_schedule);
        let anchor_decodes = if fused { groups.len() } else { targets.len() };
        let delta_decodes = targets
            .values()
            .filter(|target| target.delta_id.is_some())
            .count();
        let grouped_records = records.into_values().collect::<Vec<_>>();
        let records = grouped_records
            .iter()
            .map(|(_, record)| record.clone())
            .collect::<Vec<_>>();
        let useful_bytes = planner::unique_covered_bytes(&records)?;
        let mut candidates = adaptive::plans_for_thresholds(&records, self.max_range_bytes)?
            .into_iter()
            .map(|(threshold, plans)| {
                Ok(PlanCandidate {
                    mode: PlanMode::Normalized {
                        merge_threshold_bytes: threshold,
                    },
                    range_lengths: plans.iter().map(|plan| plan.length).collect(),
                    useful_bytes,
                    anchor_decodes,
                    delta_decodes,
                    prefix_frames: 0,
                    prefix_resets: 0,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let group_plans = planner::plan_group_spans(&grouped_records, self.max_range_bytes)?;
        candidates.push(PlanCandidate {
            mode: PlanMode::NormalizedGroupSpan,
            range_lengths: group_plans.iter().map(|plan| plan.length).collect(),
            useful_bytes,
            anchor_decodes,
            delta_decodes,
            prefix_frames: 0,
            prefix_resets: 0,
        });
        Ok(candidates)
    }
}
