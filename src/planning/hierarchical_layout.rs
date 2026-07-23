use std::collections::{HashMap, HashSet};

use crate::planner::{self, RangePlan, RecordRange};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessUnitRecord {
    pub record_id: u64,
    pub video_id: u64,
    pub gop_id: u64,
    pub offset: u64,
    pub length: u64,
    pub decode_ordinal: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetClosure {
    pub sample_id: u64,
    pub video_id: u64,
    pub gop_id: u64,
    pub target_record_id: u64,
    /// Sufficient records in decoder submission order. These IDs must come
    /// from encoder or bitstream ground truth; this module never infers codec
    /// dependencies from frame positions.
    pub record_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GopRegion {
    pub video_id: u64,
    pub gop_id: u64,
    pub offset: u64,
    pub length: u64,
    /// Every access unit in decoder submission order.
    pub record_ids: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchicalReadMode {
    SparseClosure,
    ContiguousRegion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionDecodeMode {
    /// Reading a codec region requires submitting every record in that region.
    AllRecords,
    /// Physical coalescing may cover unrelated records, while decode still
    /// consumes only the registered target closures.
    ClosureOnly,
}

#[derive(Debug, Clone)]
pub struct HierarchicalCostModel {
    pub request_latency_ns: f64,
    pub bandwidth_bytes_per_ns: f64,
    pub io_concurrency: usize,
    /// Optional calibrated fixed overhead for a wave containing 1..N Range
    /// GETs. Values exclude transfer time and must be monotone. An empty
    /// vector retains the legacy max-per-range screening model.
    pub wave_request_overhead_ns: Vec<f64>,
    /// Outcome-free uncertainty band for plan selection. Candidates inside
    /// the band are tie-broken by request count, fetched bytes, and decode
    /// work. This is robust to latency-model noise and also lowers object-store
    /// request cost without consulting evaluation outcomes.
    pub selection_tolerance_ns: f64,
    pub decode_fixed_ns: f64,
    pub decode_access_unit_ns: f64,
    pub fetch_decode_overlap: f64,
}

impl HierarchicalCostModel {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("request_latency_ns", self.request_latency_ns),
            ("decode_fixed_ns", self.decode_fixed_ns),
            ("decode_access_unit_ns", self.decode_access_unit_ns),
            ("selection_tolerance_ns", self.selection_tolerance_ns),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and non-negative"));
            }
        }
        if !self.bandwidth_bytes_per_ns.is_finite() || self.bandwidth_bytes_per_ns <= 0.0 {
            return Err("bandwidth_bytes_per_ns must be finite and positive".to_string());
        }
        if self.io_concurrency == 0 {
            return Err("io_concurrency must be positive".to_string());
        }
        if !self.wave_request_overhead_ns.is_empty() {
            if self.wave_request_overhead_ns.len() < self.io_concurrency {
                return Err(
                    "wave_request_overhead_ns must cover every request count up to io_concurrency"
                        .to_string(),
                );
            }
            let mut previous = 0.0;
            for value in &self.wave_request_overhead_ns {
                if !value.is_finite() || *value < previous {
                    return Err(
                        "wave_request_overhead_ns must be finite, non-negative, and monotone"
                            .to_string(),
                    );
                }
                previous = *value;
            }
        }
        if !self.fetch_decode_overlap.is_finite()
            || !(0.0..=1.0).contains(&self.fetch_decode_overlap)
        {
            return Err("fetch_decode_overlap must be in [0, 1]".to_string());
        }
        Ok(())
    }

    fn io_ns(&self, ranges: &[RangePlan]) -> f64 {
        if !self.wave_request_overhead_ns.is_empty() {
            let mut lengths = ranges.iter().map(|range| range.length).collect::<Vec<_>>();
            lengths.sort_unstable_by(|left, right| right.cmp(left));
            return lengths
                .chunks(self.io_concurrency)
                .map(|wave| {
                    self.wave_request_overhead_ns[wave.len() - 1]
                        + wave.iter().sum::<u64>() as f64 / self.bandwidth_bytes_per_ns
                })
                .sum();
        }
        let mut service = ranges
            .iter()
            .map(|range| {
                self.request_latency_ns + range.length as f64 / self.bandwidth_bytes_per_ns
            })
            .collect::<Vec<_>>();
        // This is a screening model, not a queueing model. Longest operations
        // are grouped together to avoid making the estimate depend on hash order.
        service.sort_by(|left, right| right.total_cmp(left));
        service
            .chunks(self.io_concurrency)
            .map(|wave| wave.iter().copied().fold(0.0, f64::max))
            .sum()
    }

    fn empirical_io_ns(&self, range_count: usize, fetched_bytes: u64) -> f64 {
        debug_assert!(!self.wave_request_overhead_ns.is_empty());
        let full_waves = range_count / self.io_concurrency;
        let remainder = range_count % self.io_concurrency;
        let mut overhead =
            full_waves as f64 * self.wave_request_overhead_ns[self.io_concurrency - 1];
        if remainder != 0 {
            overhead += self.wave_request_overhead_ns[remainder - 1];
        }
        overhead + fetched_bytes as f64 / self.bandwidth_bytes_per_ns
    }

    fn estimate(&self, candidate: HierarchicalPlanCandidate) -> HierarchicalPlanEstimate {
        let io_ns = self.io_ns(&candidate.ranges);
        let decode_ns = self.decode_fixed_ns
            + candidate.access_units_submitted as f64 * self.decode_access_unit_ns;
        let total_ns = io_ns + decode_ns - self.fetch_decode_overlap * io_ns.min(decode_ns);
        let fetched_bytes = candidate.ranges.iter().map(|range| range.length).sum();
        HierarchicalPlanEstimate {
            mode: candidate.mode,
            merge_threshold_bytes: candidate.merge_threshold_bytes,
            ranges: candidate.ranges,
            useful_bytes: candidate.useful_bytes,
            fetched_bytes,
            access_units_submitted: candidate.access_units_submitted,
            io_ns,
            decode_ns,
            total_ns,
        }
    }

    /// Score an execution shape that is produced outside the sparse-span
    /// enumerator, such as a monotonic live-decoder suffix. This keeps all
    /// production actions on the same calibrated request/byte/decode model.
    pub fn estimate_execution(
        &self,
        ranges: &[(u64, u64)],
        access_units_submitted: usize,
        include_decode_fixed: bool,
    ) -> f64 {
        let ranges = ranges
            .iter()
            .map(|(offset, length)| RangePlan {
                offset: *offset,
                length: *length,
                records: Vec::new(),
            })
            .collect::<Vec<_>>();
        let io_ns = self.io_ns(&ranges);
        let decode_ns = if access_units_submitted == 0 {
            0.0
        } else {
            f64::from(include_decode_fixed) * self.decode_fixed_ns
                + access_units_submitted as f64 * self.decode_access_unit_ns
        };
        io_ns + decode_ns - self.fetch_decode_overlap * io_ns.min(decode_ns)
    }

    /// Score a request window whose physical ranges are fetched as one union
    /// while independent decode jobs run under a bounded slot budget.
    pub fn estimate_parallel_window_execution(
        &self,
        ranges: &[(u64, u64)],
        decode_jobs: &[usize],
        decode_concurrency: usize,
    ) -> f64 {
        let ranges = ranges
            .iter()
            .map(|(offset, length)| RangePlan {
                offset: *offset,
                length: *length,
                records: Vec::new(),
            })
            .collect::<Vec<_>>();
        let io_ns = self.io_ns(&ranges);
        let mut decode_service = decode_jobs
            .iter()
            .filter(|access_units| **access_units > 0)
            .map(|access_units| {
                self.decode_fixed_ns + *access_units as f64 * self.decode_access_unit_ns
            })
            .collect::<Vec<_>>();
        decode_service.sort_by(|left, right| right.total_cmp(left));
        let decode_ns = decode_service
            .chunks(decode_concurrency.max(1))
            .map(|wave| wave.iter().copied().fold(0.0, f64::max))
            .sum::<f64>();
        io_ns + decode_ns - self.fetch_decode_overlap * io_ns.min(decode_ns)
    }
}

#[derive(Debug, Clone)]
struct HierarchicalPlanCandidate {
    mode: HierarchicalReadMode,
    merge_threshold_bytes: Option<u64>,
    ranges: Vec<RangePlan>,
    useful_bytes: u64,
    access_units_submitted: usize,
}

#[derive(Debug, Clone)]
pub struct HierarchicalPlanEstimate {
    pub mode: HierarchicalReadMode,
    /// `None` means exact record ranges. `Some(gap)` permits fetching gap
    /// bytes but never submits gap access units to the decoder.
    pub merge_threshold_bytes: Option<u64>,
    pub ranges: Vec<RangePlan>,
    pub useful_bytes: u64,
    pub fetched_bytes: u64,
    pub access_units_submitted: usize,
    pub io_ns: f64,
    pub decode_ns: f64,
    pub total_ns: f64,
}

#[derive(Debug, Clone)]
pub struct HierarchicalPlanDecision {
    pub selected: HierarchicalPlanEstimate,
    pub alternatives: Vec<HierarchicalPlanEstimate>,
}

#[derive(Debug, Clone)]
pub struct DependencyLookaheadEstimate {
    pub lookahead_batches: usize,
    pub logical_samples: usize,
    pub first_batch_ranges: usize,
    pub first_batch_fetched_bytes: u64,
    pub first_batch_access_units: usize,
    pub first_batch_io_ns: f64,
    pub first_batch_decode_ns: f64,
    pub first_batch_total_ns: f64,
    pub predicted_total_ns: f64,
    pub predicted_samples_per_second: f64,
    pub plan: HierarchicalPlanEstimate,
}

#[derive(Debug, Clone)]
pub struct DependencyLookaheadDecision {
    pub selected: DependencyLookaheadEstimate,
    pub alternatives: Vec<DependencyLookaheadEstimate>,
    pub first_batch_slo_ns: f64,
    pub selected_under_slo: bool,
}

#[derive(Debug)]
pub struct HierarchicalLayoutIndex {
    records: HashMap<u64, AccessUnitRecord>,
    closures: HashMap<u64, TargetClosure>,
    regions: HashMap<(u64, u64), GopRegion>,
    region_decode_mode: RegionDecodeMode,
}

impl HierarchicalLayoutIndex {
    fn optimal_empirical_sparse_plan(
        records: &[RecordRange],
        useful_bytes: u64,
        access_units_submitted: usize,
        max_merge_gap_bytes: Option<u64>,
        model: &HierarchicalCostModel,
    ) -> Result<HierarchicalPlanEstimate, String> {
        let mut ordered = records.to_vec();
        ordered.sort_unstable_by_key(|record| (record.offset, record.record_id));
        if ordered.is_empty() {
            return Err("sparse planner received no records".to_string());
        }
        let mut gaps = Vec::new();
        let mut mandatory_cuts = HashSet::new();
        for (index, pair) in ordered.windows(2).enumerate() {
            let end = pair[0]
                .offset
                .checked_add(pair[0].length)
                .ok_or("record range overflow")?;
            if pair[1].offset < end {
                return Err("hierarchical AU records overlap physically".to_string());
            }
            let gap = pair[1].offset - end;
            if max_merge_gap_bytes.is_some_and(|maximum| gap > maximum) {
                mandatory_cuts.insert(index);
            } else if gap > 0 {
                gaps.push((gap, index));
            }
        }
        gaps.sort_unstable_by(|left, right| {
            right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1))
        });
        let full_span = ordered.last().unwrap().offset + ordered.last().unwrap().length
            - ordered.first().unwrap().offset;
        let mandatory_gap_bytes = mandatory_cuts
            .iter()
            .map(|index| {
                let end = ordered[*index].offset + ordered[*index].length;
                ordered[*index + 1].offset - end
            })
            .sum::<u64>();
        let base_ranges = mandatory_cuts.len() + 1;
        let mut removed_bytes = mandatory_gap_bytes;
        let decode_ns =
            model.decode_fixed_ns + access_units_submitted as f64 * model.decode_access_unit_ns;
        let mut candidates = Vec::with_capacity(gaps.len() + 1);
        for optional_cuts in 0..=gaps.len() {
            if optional_cuts > 0 {
                removed_bytes += gaps[optional_cuts - 1].0;
            }
            let range_count = base_ranges + optional_cuts;
            let fetched_bytes = full_span - removed_bytes;
            let io_ns = model.empirical_io_ns(range_count, fetched_bytes);
            let total_ns = io_ns + decode_ns - model.fetch_decode_overlap * io_ns.min(decode_ns);
            candidates.push((optional_cuts, range_count, fetched_bytes, io_ns, total_ns));
        }
        let minimum = candidates
            .iter()
            .map(|candidate| candidate.4)
            .fold(f64::INFINITY, f64::min);
        let selected = candidates
            .into_iter()
            .filter(|candidate| candidate.4 <= minimum + model.selection_tolerance_ns)
            .min_by(|left, right| {
                left.1
                    .cmp(&right.1)
                    .then_with(|| left.2.cmp(&right.2))
                    .then_with(|| left.4.total_cmp(&right.4))
            })
            .expect("empirical sparse planner has candidates");
        let mut cuts = mandatory_cuts;
        cuts.extend(gaps.iter().take(selected.0).map(|(_, index)| *index));
        let mut ranges = Vec::with_capacity(selected.1);
        let mut begin = 0usize;
        for index in 0..ordered.len() {
            if index + 1 != ordered.len() && !cuts.contains(&index) {
                continue;
            }
            let segment = &ordered[begin..=index];
            let offset = segment[0].offset;
            let end = segment.last().unwrap().offset + segment.last().unwrap().length;
            ranges.push(RangePlan {
                offset,
                length: end - offset,
                records: segment
                    .iter()
                    .map(|record| planner::PlannedRecord {
                        record_id: record.record_id,
                        relative_offset: record.offset - offset,
                        length: record.length,
                    })
                    .collect(),
            });
            begin = index + 1;
        }
        Ok(HierarchicalPlanEstimate {
            mode: HierarchicalReadMode::SparseClosure,
            merge_threshold_bytes: None,
            ranges,
            useful_bytes,
            fetched_bytes: selected.2,
            access_units_submitted,
            io_ns: selected.3,
            decode_ns,
            total_ns: selected.4,
        })
    }

    pub fn new(
        records: Vec<AccessUnitRecord>,
        closures: Vec<TargetClosure>,
        regions: Vec<GopRegion>,
    ) -> Result<Self, String> {
        Self::new_with_region_decode_mode(records, closures, regions, RegionDecodeMode::AllRecords)
    }

    pub fn new_with_region_decode_mode(
        records: Vec<AccessUnitRecord>,
        closures: Vec<TargetClosure>,
        regions: Vec<GopRegion>,
        region_decode_mode: RegionDecodeMode,
    ) -> Result<Self, String> {
        let mut records_by_id = HashMap::with_capacity(records.len());
        for record in records {
            if record.length == 0 || record.offset.checked_add(record.length).is_none() {
                return Err(format!("invalid access-unit record {}", record.record_id));
            }
            let record_id = record.record_id;
            if records_by_id.insert(record_id, record).is_some() {
                return Err(format!("duplicate access-unit record {record_id}"));
            }
        }

        let mut regions_by_gop = HashMap::with_capacity(regions.len());
        for region in regions {
            if region.length == 0
                || region.record_ids.is_empty()
                || region.offset.checked_add(region.length).is_none()
            {
                return Err(format!(
                    "invalid GOP region ({}, {})",
                    region.video_id, region.gop_id
                ));
            }
            let region_end = region.offset + region.length;
            let mut previous_decode_ordinal = None;
            for record_id in &region.record_ids {
                let record = records_by_id
                    .get(record_id)
                    .ok_or_else(|| format!("GOP region references unknown record {record_id}"))?;
                if record.video_id != region.video_id || record.gop_id != region.gop_id {
                    return Err(format!("record {record_id} belongs to another GOP"));
                }
                if record.offset < region.offset || record.offset + record.length > region_end {
                    return Err(format!("record {record_id} falls outside its GOP region"));
                }
                if previous_decode_ordinal.is_some_and(|value| value >= record.decode_ordinal) {
                    return Err(format!(
                        "GOP ({}, {}) records are not in decode order",
                        region.video_id, region.gop_id
                    ));
                }
                previous_decode_ordinal = Some(record.decode_ordinal);
            }
            let key = (region.video_id, region.gop_id);
            if regions_by_gop.insert(key, region).is_some() {
                return Err(format!("duplicate GOP region ({}, {})", key.0, key.1));
            }
        }

        let mut closures_by_sample = HashMap::with_capacity(closures.len());
        for closure in closures {
            if closure.record_ids.is_empty()
                || !closure.record_ids.contains(&closure.target_record_id)
            {
                return Err(format!("invalid closure for sample {}", closure.sample_id));
            }
            if !regions_by_gop.contains_key(&(closure.video_id, closure.gop_id)) {
                return Err(format!("closure {} has no GOP region", closure.sample_id));
            }
            let mut previous_decode_ordinal = None;
            for record_id in &closure.record_ids {
                let record = records_by_id.get(record_id).ok_or_else(|| {
                    format!(
                        "closure {} references unknown record {record_id}",
                        closure.sample_id
                    )
                })?;
                if record.video_id != closure.video_id || record.gop_id != closure.gop_id {
                    return Err(format!(
                        "closure {} crosses a GOP boundary",
                        closure.sample_id
                    ));
                }
                if previous_decode_ordinal.is_some_and(|value| value >= record.decode_ordinal) {
                    return Err(format!(
                        "closure {} is not in decoder submission order",
                        closure.sample_id
                    ));
                }
                previous_decode_ordinal = Some(record.decode_ordinal);
            }
            let sample_id = closure.sample_id;
            if closures_by_sample.insert(sample_id, closure).is_some() {
                return Err(format!("duplicate target closure {sample_id}"));
            }
        }
        Ok(Self {
            records: records_by_id,
            closures: closures_by_sample,
            regions: regions_by_gop,
            region_decode_mode,
        })
    }

    pub fn choose_plan(
        &self,
        sample_ids: &[u64],
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        model: &HierarchicalCostModel,
    ) -> Result<HierarchicalPlanDecision, String> {
        self.choose_plan_with_resident(
            sample_ids,
            &HashSet::new(),
            max_merge_gap_bytes,
            max_range_bytes,
            model,
        )
    }

    /// Return every distinct sparse-closure span partition already implied by
    /// the current planner's gap thresholds. This inspection path disables the
    /// empirical fast-path only while enumerating candidates; it then scores
    /// the unchanged physical candidates with the caller's frozen model.
    /// It never participates in online plan selection.
    pub fn enumerate_sparse_candidates(
        &self,
        sample_ids: &[u64],
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        model: &HierarchicalCostModel,
    ) -> Result<Vec<HierarchicalPlanEstimate>, String> {
        let mut enumeration_model = model.clone();
        enumeration_model.wave_request_overhead_ns.clear();
        let decision = self.choose_plan(
            sample_ids,
            max_merge_gap_bytes,
            max_range_bytes,
            &enumeration_model,
        )?;
        let mut candidates = decision
            .alternatives
            .into_iter()
            .filter(|candidate| candidate.mode == HierarchicalReadMode::SparseClosure)
            .map(|candidate| {
                model.estimate(HierarchicalPlanCandidate {
                    mode: candidate.mode,
                    merge_threshold_bytes: candidate.merge_threshold_bytes,
                    ranges: candidate.ranges,
                    useful_bytes: candidate.useful_bytes,
                    access_units_submitted: candidate.access_units_submitted,
                })
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            left.ranges
                .len()
                .cmp(&right.ranges.len())
                .then_with(|| left.fetched_bytes.cmp(&right.fetched_bytes))
                .then_with(|| left.total_ns.total_cmp(&right.total_ns))
        });
        Ok(candidates)
    }

    pub fn choose_plan_with_resident(
        &self,
        sample_ids: &[u64],
        resident_record_ids: &HashSet<u64>,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        model: &HierarchicalCostModel,
    ) -> Result<HierarchicalPlanDecision, String> {
        model.validate()?;
        if sample_ids.is_empty() {
            return Err("hierarchical planner requires at least one sample".to_string());
        }
        let mut seen_samples = HashSet::new();
        let closures = sample_ids
            .iter()
            .filter(|sample_id| seen_samples.insert(**sample_id))
            .map(|sample_id| {
                self.closures
                    .get(sample_id)
                    .ok_or_else(|| format!("unknown hierarchical sample {sample_id}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut sparse_ids = HashSet::new();
        let mut sparse_records = Vec::new();
        for closure in &closures {
            for record_id in &closure.record_ids {
                if sparse_ids.insert(*record_id) && !resident_record_ids.contains(record_id) {
                    let record = &self.records[record_id];
                    sparse_records.push(RecordRange {
                        record_id: *record_id,
                        offset: record.offset,
                        length: record.length,
                    });
                }
            }
        }
        let sparse_useful_bytes = planner::unique_covered_bytes(&sparse_records)?;
        let mut ordered_sparse = sparse_records.clone();
        ordered_sparse.sort_by_key(|record| (record.offset, record.record_id));
        let fast_sparse = if sparse_records.is_empty() {
            None
        } else if !model.wave_request_overhead_ns.is_empty() && max_range_bytes.is_none() {
            Some(Self::optimal_empirical_sparse_plan(
                &sparse_records,
                sparse_useful_bytes,
                sparse_ids.len(),
                max_merge_gap_bytes,
                model,
            )?)
        } else {
            None
        };
        let mut thresholds = vec![None, Some(0)];
        thresholds.extend(ordered_sparse.windows(2).filter_map(|pair| {
            let end = pair[0].offset.checked_add(pair[0].length)?;
            let gap = pair[1].offset.saturating_sub(end);
            max_merge_gap_bytes
                .is_none_or(|maximum| gap <= maximum)
                .then_some(Some(gap))
        }));
        thresholds.sort_by_key(|value| value.map_or((0, 0), |gap| (1, gap)));
        thresholds.dedup();
        let mut signatures = HashSet::new();
        let mut alternatives = Vec::new();
        if sparse_records.is_empty() {
            alternatives.push(model.estimate(HierarchicalPlanCandidate {
                mode: HierarchicalReadMode::SparseClosure,
                merge_threshold_bytes: None,
                ranges: Vec::new(),
                useful_bytes: 0,
                access_units_submitted: sparse_ids.len(),
            }));
        } else {
            for threshold in thresholds {
                if fast_sparse.is_some() {
                    break;
                }
                let ranges =
                    planner::plan_byte_ranges(&sparse_records, threshold, max_range_bytes)?;
                let signature = ranges
                    .iter()
                    .map(|range| (range.offset, range.length))
                    .collect::<Vec<_>>();
                if signatures.insert(signature) {
                    alternatives.push(model.estimate(HierarchicalPlanCandidate {
                        mode: HierarchicalReadMode::SparseClosure,
                        merge_threshold_bytes: threshold,
                        ranges,
                        useful_bytes: sparse_useful_bytes,
                        access_units_submitted: sparse_ids.len(),
                    }));
                }
            }
        }
        if let Some(candidate) = fast_sparse {
            alternatives.push(candidate);
        }

        let mut region_keys = closures
            .iter()
            .map(|closure| (closure.video_id, closure.gop_id))
            .collect::<Vec<_>>();
        region_keys.sort_unstable();
        region_keys.dedup();
        let mut region_records = Vec::with_capacity(region_keys.len());
        let mut region_access_units = 0usize;
        for (index, key) in region_keys.into_iter().enumerate() {
            let region = &self.regions[&key];
            region_records.push(RecordRange {
                record_id: u64::MAX - index as u64,
                offset: region.offset,
                length: region.length,
            });
            region_access_units += region.record_ids.len();
        }
        if self.region_decode_mode == RegionDecodeMode::ClosureOnly {
            region_access_units = sparse_ids.len();
        }
        // Regions are already the intended physical reads. Do not bridge gaps
        // between unrelated GOPs with the closure coalescing threshold.
        let region_ranges = planner::plan_byte_ranges(&region_records, None, max_range_bytes)?;
        let region_useful_bytes = match self.region_decode_mode {
            RegionDecodeMode::AllRecords => planner::unique_covered_bytes(&region_records)?,
            RegionDecodeMode::ClosureOnly => sparse_useful_bytes,
        };

        alternatives.push(model.estimate(HierarchicalPlanCandidate {
            mode: HierarchicalReadMode::ContiguousRegion,
            merge_threshold_bytes: None,
            ranges: region_ranges,
            useful_bytes: region_useful_bytes,
            access_units_submitted: region_access_units,
        }));
        alternatives.sort_by(|left, right| {
            left.total_ns
                .total_cmp(&right.total_ns)
                .then_with(|| left.fetched_bytes.cmp(&right.fetched_bytes))
        });
        let minimum_total_ns = alternatives[0].total_ns;
        let selected = alternatives
            .iter()
            .filter(|candidate| {
                candidate.total_ns <= minimum_total_ns + model.selection_tolerance_ns
            })
            .min_by(|left, right| {
                left.ranges
                    .len()
                    .cmp(&right.ranges.len())
                    .then_with(|| left.fetched_bytes.cmp(&right.fetched_bytes))
                    .then_with(|| {
                        left.access_units_submitted
                            .cmp(&right.access_units_submitted)
                    })
                    .then_with(|| left.total_ns.total_cmp(&right.total_ns))
            })
            .expect("hierarchical planner always has an alternative")
            .clone();
        Ok(HierarchicalPlanDecision {
            selected,
            alternatives,
        })
    }

    /// Choose a future-batch visibility horizon from registered dependency
    /// closures and physical extents. The selector never receives a workload
    /// name. Candidate one is required and is the no-cross-batch-lookahead
    /// fallback.
    pub fn choose_lookahead(
        &self,
        batches: &[Vec<u64>],
        candidates: &[usize],
        first_batch_slo_ns: f64,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        model: &HierarchicalCostModel,
    ) -> Result<DependencyLookaheadDecision, String> {
        self.choose_lookahead_with_models(
            batches,
            candidates,
            first_batch_slo_ns,
            max_merge_gap_bytes,
            max_range_bytes,
            model,
            model,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn choose_lookahead_with_models(
        &self,
        batches: &[Vec<u64>],
        candidates: &[usize],
        first_batch_slo_ns: f64,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        throughput_model: &HierarchicalCostModel,
        latency_model: &HierarchicalCostModel,
    ) -> Result<DependencyLookaheadDecision, String> {
        self.choose_lookahead_with_resident_models(
            batches,
            candidates,
            first_batch_slo_ns,
            &HashSet::new(),
            max_merge_gap_bytes,
            max_range_bytes,
            throughput_model,
            latency_model,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn choose_lookahead_with_resident(
        &self,
        batches: &[Vec<u64>],
        candidates: &[usize],
        first_batch_slo_ns: f64,
        resident_record_ids: &HashSet<u64>,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        model: &HierarchicalCostModel,
    ) -> Result<DependencyLookaheadDecision, String> {
        self.choose_lookahead_with_resident_models(
            batches,
            candidates,
            first_batch_slo_ns,
            resident_record_ids,
            max_merge_gap_bytes,
            max_range_bytes,
            model,
            model,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn choose_lookahead_with_resident_models(
        &self,
        batches: &[Vec<u64>],
        candidates: &[usize],
        first_batch_slo_ns: f64,
        resident_record_ids: &HashSet<u64>,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        throughput_model: &HierarchicalCostModel,
        latency_model: &HierarchicalCostModel,
    ) -> Result<DependencyLookaheadDecision, String> {
        throughput_model.validate()?;
        latency_model.validate()?;
        if batches.is_empty() || batches.iter().any(Vec::is_empty) {
            return Err("lookahead selection requires non-empty logical batches".to_string());
        }
        if !first_batch_slo_ns.is_finite() || first_batch_slo_ns <= 0.0 {
            return Err("first_batch_slo_ns must be finite and positive".to_string());
        }
        let mut candidates = candidates.to_vec();
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.first().copied() != Some(1) {
            return Err("lookahead candidates must include 1 as the fallback".to_string());
        }
        if candidates
            .iter()
            .any(|value| *value == 0 || *value > batches.len())
        {
            return Err("lookahead candidate falls outside the available batch window".to_string());
        }

        let first_closures = batches[0]
            .iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|sample_id| {
                self.closures
                    .get(sample_id)
                    .ok_or_else(|| format!("unknown dependency sample {sample_id}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let first_sparse_ids = first_closures
            .iter()
            .flat_map(|closure| closure.record_ids.iter().copied())
            .collect::<HashSet<_>>();
        let first_sparse_records = first_sparse_ids
            .iter()
            .filter(|record_id| !resident_record_ids.contains(record_id))
            .map(|record_id| {
                let record = &self.records[record_id];
                RecordRange {
                    record_id: *record_id,
                    offset: record.offset,
                    length: record.length,
                }
            })
            .collect::<Vec<_>>();
        let first_region_keys = first_closures
            .iter()
            .map(|closure| (closure.video_id, closure.gop_id))
            .collect::<HashSet<_>>();
        let first_region_records = first_region_keys
            .iter()
            .enumerate()
            .map(|(index, key)| {
                let region = &self.regions[key];
                RecordRange {
                    record_id: u64::MAX - index as u64,
                    offset: region.offset,
                    length: region.length,
                }
            })
            .collect::<Vec<_>>();
        let first_region_access_units = match self.region_decode_mode {
            RegionDecodeMode::AllRecords => first_region_keys
                .iter()
                .map(|key| self.regions[key].record_ids.len())
                .sum::<usize>(),
            RegionDecodeMode::ClosureOnly => first_sparse_ids.len(),
        };

        let mut estimates = Vec::with_capacity(candidates.len());
        for lookahead_batches in candidates {
            let sample_ids = batches[..lookahead_batches]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let logical_samples = sample_ids.len();
            let decision = self.choose_plan_with_resident(
                &sample_ids,
                resident_record_ids,
                max_merge_gap_bytes,
                max_range_bytes,
                throughput_model,
            )?;
            let plan = decision.selected;
            let (first_required, first_batch_access_units) = match plan.mode {
                HierarchicalReadMode::SparseClosure => {
                    (&first_sparse_records, first_sparse_ids.len())
                }
                HierarchicalReadMode::ContiguousRegion => {
                    (&first_region_records, first_region_access_units)
                }
            };
            let first_ranges = plan
                .ranges
                .iter()
                .filter(|range| {
                    let range_end = range.offset.saturating_add(range.length);
                    first_required.iter().any(|record| {
                        let record_end = record.offset.saturating_add(record.length);
                        record.offset < range_end && range.offset < record_end
                    })
                })
                .cloned()
                .collect::<Vec<_>>();
            if first_ranges.is_empty() && !first_required.is_empty() {
                return Err("selected lookahead plan does not cover the first batch".to_string());
            }
            let first_batch_fetched_bytes =
                first_ranges.iter().map(|range| range.length).sum::<u64>();
            let first_batch_io_ns = latency_model.io_ns(&first_ranges);
            let first_batch_decode_ns = latency_model.decode_fixed_ns
                + first_batch_access_units as f64 * latency_model.decode_access_unit_ns;
            let first_batch_total_ns = first_batch_io_ns + first_batch_decode_ns
                - latency_model.fetch_decode_overlap * first_batch_io_ns.min(first_batch_decode_ns);
            let predicted_samples_per_second =
                logical_samples as f64 * 1e9 / plan.total_ns.max(f64::EPSILON);
            estimates.push(DependencyLookaheadEstimate {
                lookahead_batches,
                logical_samples,
                first_batch_ranges: first_ranges.len(),
                first_batch_fetched_bytes,
                first_batch_access_units,
                first_batch_io_ns,
                first_batch_decode_ns,
                first_batch_total_ns,
                predicted_total_ns: plan.total_ns,
                predicted_samples_per_second,
                plan,
            });
        }

        let feasible = estimates
            .iter()
            .filter(|estimate| estimate.first_batch_total_ns <= first_batch_slo_ns)
            .collect::<Vec<_>>();
        let selected_under_slo = !feasible.is_empty();
        let selected = if selected_under_slo {
            feasible
                .into_iter()
                .max_by(|left, right| {
                    left.predicted_samples_per_second
                        .total_cmp(&right.predicted_samples_per_second)
                        .then_with(|| right.lookahead_batches.cmp(&left.lookahead_batches))
                })
                .unwrap()
                .clone()
        } else {
            estimates
                .iter()
                .min_by(|left, right| {
                    left.first_batch_total_ns
                        .total_cmp(&right.first_batch_total_ns)
                        .then_with(|| left.lookahead_batches.cmp(&right.lookahead_batches))
                })
                .unwrap()
                .clone()
        };
        Ok(DependencyLookaheadDecision {
            selected,
            alternatives: estimates,
            first_batch_slo_ns,
            selected_under_slo,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> HierarchicalLayoutIndex {
        let records = (0..8)
            .map(|ordinal| AccessUnitRecord {
                record_id: ordinal,
                video_id: 1,
                gop_id: 0,
                offset: ordinal * 4096,
                length: 1024,
                decode_ordinal: ordinal as usize,
            })
            .collect::<Vec<_>>();
        let closures = vec![
            TargetClosure {
                sample_id: 10,
                video_id: 1,
                gop_id: 0,
                target_record_id: 3,
                record_ids: vec![0, 2, 3],
            },
            TargetClosure {
                sample_id: 11,
                video_id: 1,
                gop_id: 0,
                target_record_id: 7,
                record_ids: vec![0, 4, 6, 7],
            },
        ];
        let regions = vec![GopRegion {
            video_id: 1,
            gop_id: 0,
            offset: 0,
            length: 8 * 4096,
            record_ids: (0..8).collect(),
        }];
        HierarchicalLayoutIndex::new(records, closures, regions).unwrap()
    }

    #[test]
    fn high_request_startup_selects_one_closure_span_instead_of_decoding_gaps() {
        let model = HierarchicalCostModel {
            request_latency_ns: 10_000_000.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 1,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 1.0,
            fetch_decode_overlap: 0.0,
        };
        let decision = index().choose_plan(&[10, 11], None, None, &model).unwrap();
        assert_eq!(decision.selected.mode, HierarchicalReadMode::SparseClosure);
        assert_eq!(decision.selected.ranges.len(), 1);
        assert_eq!(decision.selected.access_units_submitted, 6);
        let region = decision
            .alternatives
            .iter()
            .find(|value| value.mode == HierarchicalReadMode::ContiguousRegion)
            .unwrap();
        assert!(decision.selected.total_ns < region.total_ns);
    }

    #[test]
    fn cheap_parallel_requests_select_sparse_closure() {
        let model = HierarchicalCostModel {
            request_latency_ns: 1.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 8,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 1.0,
            fetch_decode_overlap: 0.0,
        };
        let decision = index().choose_plan(&[10], None, None, &model).unwrap();
        assert_eq!(decision.selected.mode, HierarchicalReadMode::SparseClosure);
        assert_eq!(decision.selected.access_units_submitted, 3);
        assert_eq!(decision.selected.fetched_bytes, 3 * 1024);
    }

    #[test]
    fn duplicate_targets_do_not_duplicate_closure_work() {
        let model = HierarchicalCostModel {
            request_latency_ns: 1.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 8,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 1.0,
            fetch_decode_overlap: 0.0,
        };
        let decision = index().choose_plan(&[10, 10], None, None, &model).unwrap();
        let sparse = decision
            .alternatives
            .iter()
            .find(|value| value.mode == HierarchicalReadMode::SparseClosure)
            .unwrap();
        assert_eq!(sparse.access_units_submitted, 3);
    }

    #[test]
    fn resident_dependency_is_removed_from_fetch_but_not_decode() {
        let model = HierarchicalCostModel {
            request_latency_ns: 1.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 8,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 1.0,
            fetch_decode_overlap: 0.0,
        };
        let decision = index()
            .choose_plan_with_resident(&[10], &HashSet::from([0]), None, None, &model)
            .unwrap();
        let sparse = decision
            .alternatives
            .iter()
            .find(|value| value.mode == HierarchicalReadMode::SparseClosure)
            .unwrap();
        assert_eq!(sparse.access_units_submitted, 3);
        assert_eq!(sparse.useful_bytes, 2 * 1024);

        let cached = index()
            .choose_plan_with_resident(&[10], &HashSet::from([0, 2, 3]), None, None, &model)
            .unwrap();
        let cached_sparse = cached
            .alternatives
            .iter()
            .find(|value| value.mode == HierarchicalReadMode::SparseClosure)
            .unwrap();
        assert!(cached_sparse.ranges.is_empty());
        assert_eq!(cached_sparse.access_units_submitted, 3);
    }

    #[test]
    fn empirical_wave_model_charges_request_count_and_total_bytes() {
        let model = HierarchicalCostModel {
            request_latency_ns: 0.0,
            bandwidth_bytes_per_ns: 2.0,
            io_concurrency: 4,
            wave_request_overhead_ns: vec![100.0, 150.0, 180.0, 200.0],
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 0.0,
            fetch_decode_overlap: 0.0,
        };
        let ranges = vec![
            RangePlan {
                offset: 0,
                length: 10,
                records: Vec::new(),
            },
            RangePlan {
                offset: 100,
                length: 20,
                records: Vec::new(),
            },
        ];
        assert_eq!(model.io_ns(&ranges), 165.0);
    }

    #[test]
    fn empirical_wave_model_rejects_non_monotone_calibration() {
        let model = HierarchicalCostModel {
            request_latency_ns: 0.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 2,
            wave_request_overhead_ns: vec![100.0, 90.0],
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 0.0,
            fetch_decode_overlap: 0.0,
        };
        assert!(model.validate().is_err());
    }

    #[test]
    fn uncertainty_band_prefers_fewer_object_store_requests() {
        let model = HierarchicalCostModel {
            request_latency_ns: 0.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 8,
            wave_request_overhead_ns: vec![
                10_000.0, 15_000.0, 18_500.0, 19_000.0, 19_500.0, 20_000.0, 20_500.0, 21_000.0,
            ],
            selection_tolerance_ns: 500.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 0.0,
            fetch_decode_overlap: 0.0,
        };
        let decision = index().choose_plan(&[10], None, None, &model).unwrap();
        assert_eq!(decision.selected.ranges.len(), 2);
        assert_eq!(decision.selected.fetched_bytes, 6 * 1024);
    }

    #[test]
    fn rejects_closure_that_crosses_gop_boundary() {
        let records = vec![
            AccessUnitRecord {
                record_id: 0,
                video_id: 1,
                gop_id: 0,
                offset: 0,
                length: 10,
                decode_ordinal: 0,
            },
            AccessUnitRecord {
                record_id: 1,
                video_id: 1,
                gop_id: 1,
                offset: 10,
                length: 10,
                decode_ordinal: 0,
            },
        ];
        let error = HierarchicalLayoutIndex::new(
            records,
            vec![TargetClosure {
                sample_id: 1,
                video_id: 1,
                gop_id: 0,
                target_record_id: 0,
                record_ids: vec![0, 1],
            }],
            vec![GopRegion {
                video_id: 1,
                gop_id: 0,
                offset: 0,
                length: 10,
                record_ids: vec![0],
            }],
        )
        .unwrap_err();
        assert!(error.contains("crosses a GOP boundary"));
    }

    #[test]
    fn lookahead_selector_uses_one_as_the_no_future_fallback() {
        let model = HierarchicalCostModel {
            request_latency_ns: 10_000.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 1,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 0.0,
            fetch_decode_overlap: 0.0,
        };
        let batches = vec![vec![10], vec![11]];
        let one = index()
            .choose_lookahead(&batches, &[1, 2], 30_000.0, None, None, &model)
            .unwrap();
        assert_eq!(one.selected.lookahead_batches, 1);
        assert!(one.selected_under_slo);

        let future = index()
            .choose_lookahead(&batches, &[1, 2], 1_000_000.0, None, None, &model)
            .unwrap();
        assert_eq!(future.selected.lookahead_batches, 2);
        assert!(
            future.selected.predicted_samples_per_second
                > one.alternatives[0].predicted_samples_per_second
        );
    }

    #[test]
    fn lookahead_selector_rejects_candidates_without_one() {
        let model = HierarchicalCostModel {
            request_latency_ns: 1.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 1,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 0.0,
            fetch_decode_overlap: 0.0,
        };
        let error = index()
            .choose_lookahead(&[vec![10], vec![11]], &[2], 1_000.0, None, None, &model)
            .unwrap_err();
        assert!(error.contains("include 1"));
    }

    #[test]
    fn lookahead_uses_tail_model_only_for_latency_admission() {
        let expected = HierarchicalCostModel {
            request_latency_ns: 10_000.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 1,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 0.0,
            fetch_decode_overlap: 0.0,
        };
        let mut tail = expected.clone();
        tail.request_latency_ns *= 100.0;
        tail.bandwidth_bytes_per_ns /= 100.0;
        let batches = vec![vec![10], vec![11]];

        let decision = index()
            .choose_lookahead_with_models(
                &batches,
                &[1, 2],
                1_000_000.0,
                None,
                None,
                &expected,
                &tail,
            )
            .unwrap();

        assert_eq!(decision.selected.lookahead_batches, 1);
        assert!(!decision.selected_under_slo);
        assert!(
            decision.alternatives[1].predicted_samples_per_second
                > decision.alternatives[0].predicted_samples_per_second
        );
    }
}
