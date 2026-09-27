use std::collections::HashSet;

use crate::controls::closed_record::{
    ClosedRecordBatchExecutor, ClosedRecordDescriptor, ClosedRecordFrame,
};
use crate::controls::normalized::NormalizedDescriptor;
use crate::controls::normalized::{NormalizedBatchExecutor, NormalizedFrame};
use crate::planner::{self, RangePlan, RecordRange};
use crate::representation::BatchStats;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanMode {
    Normalized { merge_threshold_bytes: Option<u64> },
    NormalizedGroupSpan,
    NormalizedSpan,
    Prefix,
}

/// Derive exact contiguous spans from one corrected Normalized physical layout.
///
/// The builder writes each Anchor group as `[config + I, D1, D2, ...]`.
/// A Prefix for target `k` is therefore the contiguous span from the Anchor
/// offset through `Dk`.  Validation here is intentionally strict: the SDK must
/// not infer decodability from merely nearby offsets.
pub fn derive_normalized_span_descriptors(
    descriptors: &[NormalizedDescriptor],
) -> Result<Vec<ClosedRecordDescriptor>, String> {
    let mut groups = std::collections::HashMap::<u64, Vec<&NormalizedDescriptor>>::new();
    for descriptor in descriptors {
        groups
            .entry(descriptor.anchor_group_id)
            .or_default()
            .push(descriptor);
    }
    let mut prefixes = Vec::with_capacity(descriptors.len());
    for (group_id, mut group) in groups {
        group.sort_by_key(|descriptor| descriptor.target_ordinal);
        let first = group
            .first()
            .ok_or_else(|| format!("Anchor group {group_id} is empty"))?;
        let video_id = first.video_id;
        let anchor_offset = first.anchor_offset;
        let anchor_length = first.anchor_length;
        let anchor_end = anchor_offset
            .checked_add(anchor_length)
            .ok_or_else(|| format!("Anchor group {group_id} end overflows u64"))?;
        let mut cursor = anchor_end;
        let mut previous_ordinal = None;
        for descriptor in group {
            if descriptor.video_id != video_id
                || descriptor.anchor_offset != anchor_offset
                || descriptor.anchor_length != anchor_length
            {
                return Err(format!(
                    "Anchor group {group_id} has inconsistent video or Anchor extent"
                ));
            }
            if let Some(previous) = previous_ordinal {
                if descriptor.target_ordinal != previous + 1 {
                    return Err(format!(
                        "Anchor group {group_id} has non-contiguous target ordinals"
                    ));
                }
            } else if descriptor.target_ordinal > 1 {
                return Err(format!(
                    "Anchor group {group_id} starts at target ordinal {}",
                    descriptor.target_ordinal
                ));
            }
            let prefix_end = if descriptor.target_ordinal == 0 {
                if descriptor.delta_length != 0 || descriptor.delta_offset != anchor_end {
                    return Err(format!(
                        "Anchor target in group {group_id} has a physical Delta"
                    ));
                }
                anchor_end
            } else {
                if descriptor.delta_length == 0 || descriptor.delta_offset != cursor {
                    return Err(format!(
                        "Anchor group {group_id} Delta {} is not contiguous",
                        descriptor.target_ordinal
                    ));
                }
                descriptor
                    .delta_offset
                    .checked_add(descriptor.delta_length)
                    .ok_or_else(|| format!("Anchor group {group_id} Delta end overflows u64"))?
            };
            prefixes.push(ClosedRecordDescriptor {
                sample_id: descriptor.sample_id,
                // Prefix decoder state is scoped to one independently decodable
                // Anchor group, not to the source video's global ID.
                video_id: group_id,
                offset: anchor_offset,
                length: prefix_end - anchor_offset,
                target_ordinal: descriptor.target_ordinal,
            });
            cursor = prefix_end;
            previous_ordinal = Some(descriptor.target_ordinal);
        }
    }
    prefixes.sort_by_key(|descriptor| descriptor.sample_id);
    Ok(prefixes)
}

impl PlanMode {
    pub fn id(self) -> String {
        match self {
            Self::Normalized {
                merge_threshold_bytes: None,
            } => "normalized_exact".to_string(),
            Self::Normalized {
                merge_threshold_bytes: Some(value),
            } => format!("normalized_gap_{value}"),
            Self::NormalizedGroupSpan => "normalized_group_span".to_string(),
            Self::NormalizedSpan => "normalized_contiguous_span".to_string(),
            Self::Prefix => "prefix_stream".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MechanisticModel {
    pub request_latency_ns: f64,
    pub bandwidth_bytes_per_ns: f64,
    pub io_concurrency: usize,
    pub anchor_decode_ns: f64,
    pub delta_decode_ns: f64,
    pub prefix_frame_decode_ns: f64,
    pub prefix_reset_ns: f64,
    pub normalized_fixed_ns: f64,
    pub prefix_fixed_ns: f64,
    pub normalized_overlap: f64,
    pub prefix_overlap: f64,
}

impl MechanisticModel {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("request_latency_ns", self.request_latency_ns),
            ("bandwidth_bytes_per_ns", self.bandwidth_bytes_per_ns),
            ("anchor_decode_ns", self.anchor_decode_ns),
            ("delta_decode_ns", self.delta_decode_ns),
            ("prefix_frame_decode_ns", self.prefix_frame_decode_ns),
            ("prefix_reset_ns", self.prefix_reset_ns),
            ("normalized_fixed_ns", self.normalized_fixed_ns),
            ("prefix_fixed_ns", self.prefix_fixed_ns),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and non-negative"));
            }
        }
        if self.bandwidth_bytes_per_ns == 0.0 {
            return Err("bandwidth_bytes_per_ns must be positive".to_string());
        }
        if self.io_concurrency == 0 {
            return Err("io_concurrency must be positive".to_string());
        }
        for (name, value) in [
            ("normalized_overlap", self.normalized_overlap),
            ("prefix_overlap", self.prefix_overlap),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be in [0, 1]"));
            }
        }
        Ok(())
    }

    fn io_ns(&self, range_lengths: &[u64]) -> f64 {
        range_lengths
            .chunks(self.io_concurrency)
            .map(|wave| {
                wave.iter()
                    .map(|length| {
                        self.request_latency_ns + *length as f64 / self.bandwidth_bytes_per_ns
                    })
                    .fold(0.0, f64::max)
            })
            .sum()
    }

    pub fn estimate(&self, candidate: &PlanCandidate) -> PlanEstimate {
        let io_ns = self.io_ns(&candidate.range_lengths);
        let (decode_ns, fixed_ns, overlap) = match candidate.mode {
            PlanMode::Normalized { .. } | PlanMode::NormalizedGroupSpan => (
                candidate.anchor_decodes as f64 * self.anchor_decode_ns
                    + candidate.delta_decodes as f64 * self.delta_decode_ns,
                self.normalized_fixed_ns,
                self.normalized_overlap,
            ),
            PlanMode::NormalizedSpan | PlanMode::Prefix => (
                candidate.prefix_frames as f64 * self.prefix_frame_decode_ns
                    + candidate.prefix_resets as f64 * self.prefix_reset_ns,
                self.prefix_fixed_ns,
                self.prefix_overlap,
            ),
        };
        let total_ns = fixed_ns + io_ns + decode_ns - overlap * io_ns.min(decode_ns);
        PlanEstimate {
            mode: candidate.mode,
            total_ns,
            io_ns,
            decode_ns,
            physical_ranges: candidate.range_lengths.len(),
            useful_bytes: candidate.useful_bytes,
            fetched_bytes: candidate.range_lengths.iter().sum(),
            overfetch_bytes: candidate
                .range_lengths
                .iter()
                .sum::<u64>()
                .saturating_sub(candidate.useful_bytes),
            anchor_decodes: candidate.anchor_decodes,
            delta_decodes: candidate.delta_decodes,
            prefix_frames: candidate.prefix_frames,
            prefix_resets: candidate.prefix_resets,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanCandidate {
    pub mode: PlanMode,
    pub range_lengths: Vec<u64>,
    pub useful_bytes: u64,
    pub anchor_decodes: usize,
    pub delta_decodes: usize,
    pub prefix_frames: usize,
    pub prefix_resets: usize,
}

#[derive(Debug, Clone)]
pub struct PlanEstimate {
    pub mode: PlanMode,
    pub total_ns: f64,
    pub io_ns: f64,
    pub decode_ns: f64,
    pub physical_ranges: usize,
    pub useful_bytes: u64,
    pub fetched_bytes: u64,
    pub overfetch_bytes: u64,
    pub anchor_decodes: usize,
    pub delta_decodes: usize,
    pub prefix_frames: usize,
    pub prefix_resets: usize,
}

#[derive(Debug, Clone)]
pub struct PlanDecision {
    pub selected: PlanEstimate,
    pub alternatives: Vec<PlanEstimate>,
    pub selector_ns: u64,
}

pub fn choose(
    model: &MechanisticModel,
    candidates: Vec<PlanCandidate>,
) -> Result<PlanDecision, String> {
    model.validate()?;
    if candidates.is_empty() {
        return Err("adaptive planner requires at least one candidate".to_string());
    }
    let started = std::time::Instant::now();
    let mut alternatives = candidates
        .iter()
        .map(|candidate| model.estimate(candidate))
        .collect::<Vec<_>>();
    alternatives.sort_by(|left, right| {
        left.total_ns
            .total_cmp(&right.total_ns)
            .then_with(|| left.mode.id().cmp(&right.mode.id()))
    });
    Ok(PlanDecision {
        selected: alternatives[0].clone(),
        alternatives,
        selector_ns: started.elapsed().as_nanos() as u64,
    })
}

pub fn plans_for_thresholds(
    records: &[RecordRange],
    max_range_bytes: Option<u64>,
) -> Result<Vec<(Option<u64>, Vec<RangePlan>)>, String> {
    let mut ordered = records.to_vec();
    ordered.sort_by_key(|record| (record.offset, record.record_id));
    let mut thresholds = vec![None, Some(0)];
    thresholds.extend(ordered.windows(2).filter_map(|pair| {
        let end = pair[0].offset.checked_add(pair[0].length)?;
        Some(Some(pair[1].offset.saturating_sub(end)))
    }));
    thresholds.sort_by_key(|value| value.map_or((0, 0), |threshold| (1, threshold)));
    thresholds.dedup();

    let mut signatures = HashSet::new();
    let mut result = Vec::new();
    for threshold in thresholds {
        let plans = planner::plan_byte_ranges(records, threshold, max_range_bytes)?;
        let signature = plans
            .iter()
            .map(|plan| (plan.offset, plan.length))
            .collect::<Vec<_>>();
        if signatures.insert(signature) {
            result.push((threshold, plans));
        }
    }
    Ok(result)
}

#[derive(Debug)]
pub struct AdaptiveFrame {
    pub sample_id: u64,
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl From<NormalizedFrame> for AdaptiveFrame {
    fn from(value: NormalizedFrame) -> Self {
        Self {
            sample_id: value.sample_id,
            rgb: value.rgb,
            width: value.width,
            height: value.height,
        }
    }
}

impl From<ClosedRecordFrame> for AdaptiveFrame {
    fn from(value: ClosedRecordFrame) -> Self {
        Self {
            sample_id: value.sample_id,
            rgb: value.rgb,
            width: value.width,
            height: value.height,
        }
    }
}

pub struct AdaptiveBatchExecutor {
    normalized: NormalizedBatchExecutor,
    prefix: ClosedRecordBatchExecutor,
    model: MechanisticModel,
    completion_driven: bool,
    allow_cross_encoding_prefix: bool,
    same_session_span: bool,
}

impl AdaptiveBatchExecutor {
    pub fn new(
        normalized: NormalizedBatchExecutor,
        prefix: ClosedRecordBatchExecutor,
        model: MechanisticModel,
        completion_driven: bool,
        allow_cross_encoding_prefix: bool,
        same_session_span: bool,
    ) -> Result<Self, String> {
        model.validate()?;
        Ok(Self {
            normalized,
            prefix,
            model,
            completion_driven,
            allow_cross_encoding_prefix,
            same_session_span,
        })
    }

    pub fn plan(&self, batch: &[u64]) -> Result<PlanDecision, String> {
        let started = std::time::Instant::now();
        let mut candidates = self.normalized.adaptive_candidates(batch)?;
        if self.allow_cross_encoding_prefix {
            candidates.push(self.prefix.adaptive_prefix_candidate(batch)?);
        }
        if self.same_session_span {
            let mut span = self.prefix.adaptive_prefix_candidate(batch)?;
            span.mode = PlanMode::NormalizedSpan;
            candidates.push(span);
        }
        let mut decision = choose(&self.model, candidates)?;
        decision.selector_ns = started.elapsed().as_nanos() as u64;
        Ok(decision)
    }

    pub fn execute(
        &mut self,
        batch: &[u64],
    ) -> Result<(Vec<AdaptiveFrame>, PlanDecision, BatchStats), String> {
        let decision = self.plan(batch)?;
        let (frames, mut stats) = self.execute_forced(batch, decision.selected.mode)?;
        stats.plan_ns = stats.plan_ns.saturating_add(decision.selector_ns);
        Ok((frames, decision, stats))
    }

    pub fn execute_forced(
        &mut self,
        batch: &[u64],
        mode: PlanMode,
    ) -> Result<(Vec<AdaptiveFrame>, BatchStats), String> {
        match mode {
            PlanMode::Normalized {
                merge_threshold_bytes,
            } => self
                .normalized
                .execute_with_merge_threshold(batch, self.completion_driven, merge_threshold_bytes)
                .map(|(frames, stats)| {
                    (frames.into_iter().map(AdaptiveFrame::from).collect(), stats)
                }),
            PlanMode::NormalizedGroupSpan => self
                .normalized
                .execute_with_dependency_group_spans(batch, self.completion_driven)
                .map(|(frames, stats)| {
                    (frames.into_iter().map(AdaptiveFrame::from).collect(), stats)
                }),
            PlanMode::NormalizedSpan | PlanMode::Prefix => {
                self.prefix.execute(batch).map(|(frames, stats)| {
                    (frames.into_iter().map(AdaptiveFrame::from).collect(), stats)
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> MechanisticModel {
        MechanisticModel {
            request_latency_ns: 1_000.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 1,
            anchor_decode_ns: 0.0,
            delta_decode_ns: 0.0,
            prefix_frame_decode_ns: 0.0,
            prefix_reset_ns: 0.0,
            normalized_fixed_ns: 0.0,
            prefix_fixed_ns: 0.0,
            normalized_overlap: 0.0,
            prefix_overlap: 0.0,
        }
    }

    fn normalized(
        sample_id: u64,
        group: u64,
        ordinal: usize,
        delta_offset: u64,
        delta_length: u64,
    ) -> NormalizedDescriptor {
        NormalizedDescriptor {
            sample_id,
            video_id: 7,
            anchor_group_id: group,
            target_ordinal: ordinal,
            anchor_offset: 100,
            anchor_length: 20,
            delta_offset,
            delta_length,
        }
    }

    #[test]
    fn derives_exact_spans_from_contiguous_normalized_group() {
        let prefixes = derive_normalized_span_descriptors(&[
            normalized(1, 9, 0, 120, 0),
            normalized(2, 9, 1, 120, 5),
            normalized(3, 9, 2, 125, 7),
        ])
        .unwrap();
        assert_eq!(prefixes.len(), 3);
        assert_eq!((prefixes[0].offset, prefixes[0].length), (100, 20));
        assert_eq!((prefixes[1].offset, prefixes[1].length), (100, 25));
        assert_eq!((prefixes[2].offset, prefixes[2].length), (100, 32));
        assert!(prefixes.iter().all(|descriptor| descriptor.video_id == 9));
    }

    #[test]
    fn rejects_span_inference_across_a_physical_gap() {
        let error = derive_normalized_span_descriptors(&[
            normalized(1, 9, 1, 120, 5),
            normalized(2, 9, 2, 130, 7),
        ])
        .unwrap_err();
        assert!(error.contains("not contiguous"));
    }

    #[test]
    fn serial_model_merges_gap_below_latency_bandwidth_product() {
        let records = vec![
            RecordRange {
                record_id: 1,
                offset: 0,
                length: 100,
            },
            RecordRange {
                record_id: 2,
                offset: 600,
                length: 100,
            },
        ];
        let candidates = plans_for_thresholds(&records, None)
            .unwrap()
            .into_iter()
            .map(|(threshold, plans)| PlanCandidate {
                mode: PlanMode::Normalized {
                    merge_threshold_bytes: threshold,
                },
                range_lengths: plans.iter().map(|plan| plan.length).collect(),
                useful_bytes: 200,
                anchor_decodes: 0,
                delta_decodes: 0,
                prefix_frames: 0,
                prefix_resets: 0,
            })
            .collect();
        let decision = choose(&model(), candidates).unwrap();
        assert_eq!(decision.selected.physical_ranges, 1);
        assert_eq!(decision.selected.fetched_bytes, 700);
    }

    #[test]
    fn concurrent_wave_keeps_disjoint_ranges_when_both_fit_in_one_wave() {
        let mut concurrent = model();
        concurrent.io_concurrency = 2;
        let records = vec![
            RecordRange {
                record_id: 1,
                offset: 0,
                length: 100,
            },
            RecordRange {
                record_id: 2,
                offset: 600,
                length: 100,
            },
        ];
        let candidates = plans_for_thresholds(&records, None)
            .unwrap()
            .into_iter()
            .map(|(threshold, plans)| PlanCandidate {
                mode: PlanMode::Normalized {
                    merge_threshold_bytes: threshold,
                },
                range_lengths: plans.iter().map(|plan| plan.length).collect(),
                useful_bytes: 200,
                anchor_decodes: 0,
                delta_decodes: 0,
                prefix_frames: 0,
                prefix_resets: 0,
            })
            .collect();
        let decision = choose(&concurrent, candidates).unwrap();
        assert_eq!(
            decision.selected.mode,
            PlanMode::Normalized {
                merge_threshold_bytes: None
            }
        );
    }

    #[test]
    fn serial_model_keeps_gap_above_latency_bandwidth_product_split() {
        let records = vec![
            RecordRange {
                record_id: 1,
                offset: 0,
                length: 100,
            },
            RecordRange {
                record_id: 2,
                offset: 2_100,
                length: 100,
            },
        ];
        let candidates = plans_for_thresholds(&records, None)
            .unwrap()
            .into_iter()
            .map(|(threshold, plans)| PlanCandidate {
                mode: PlanMode::Normalized {
                    merge_threshold_bytes: threshold,
                },
                range_lengths: plans.iter().map(|plan| plan.length).collect(),
                useful_bytes: 200,
                anchor_decodes: 0,
                delta_decodes: 0,
                prefix_frames: 0,
                prefix_resets: 0,
            })
            .collect();
        let decision = choose(&model(), candidates).unwrap();
        assert_eq!(decision.selected.physical_ranges, 2);
        assert_eq!(decision.selected.fetched_bytes, 200);
    }
}
