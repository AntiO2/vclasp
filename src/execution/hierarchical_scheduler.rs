use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::backend::StorageBackend;
use crate::decoder::{
    self, DecodeBudget, DecodedRgbFrame, DecoderConfig, DecoderPool, SharedDecoderSlots,
};
use crate::hierarchical_ingest::{
    mp4_sample_to_annex_b, HierarchicalCatalog, HierarchicalRecordMeta,
};
#[cfg(feature = "experiment-controls")]
use crate::hierarchical_layout::DependencyLookaheadDecision;
use crate::hierarchical_layout::{
    HierarchicalCostModel, HierarchicalLayoutIndex, HierarchicalReadMode,
};
use crate::planner::{self, RecordRange};
use crate::resident_cache::{
    DependencyLivenessLru, ResidentCandidate, ResidentStatePolicy, WindowDependencyLiveness,
};
use crate::runtime_feedback::{
    RuntimeCostFeedback, RuntimeFeedbackConfig, RuntimeFeedbackSnapshot,
};

#[derive(Debug, Clone)]
pub struct LogicalTarget {
    pub sample_id: u64,
    pub video_id: String,
    pub frame_idx: i32,
}

#[cfg(test)]
mod tests {
    use super::{
        cursor_admission_with_existing_state, frame_thread_release_margin,
        monotonic_visible_reuses, range_size_buckets, read_ahead_allowance, single_gop_key,
        HierarchicalBatchExecutor,
    };
    use crate::backend::NoopBackend;
    use crate::hierarchical_ingest::HierarchicalRecordMeta;
    #[cfg(feature = "experiment-controls")]
    use crate::hierarchical_layout::HierarchicalCostModel;
    #[cfg(feature = "experiment-controls")]
    use crate::runtime_feedback::RuntimeFeedbackConfig;

    fn record(video: &str, gop_id: u64, frame_idx: i32) -> HierarchicalRecordMeta {
        HierarchicalRecordMeta {
            record_id: frame_idx as u64,
            video_id: video.to_string(),
            frame_idx,
            gop_id,
            offset: 0,
            length: 1,
            decode_ordinal: 0,
            closure_record_ids: vec![frame_idx as u64],
            target_output_ordinal: 0,
            nal_length_size: 4,
        }
    }

    #[cfg(feature = "experiment-controls")]
    #[test]
    fn lookahead_refreshes_the_shared_runtime_model_before_selection() {
        let mut first = record("video", 0, 0);
        first.offset = 0;
        first.length = 1024;
        first.closure_record_ids = vec![0];
        let mut second = record("video", 0, 1);
        second.record_id = 1;
        second.offset = 4096;
        second.length = 1024;
        second.decode_ordinal = 1;
        second.closure_record_ids = vec![0, 1];
        second.target_output_ordinal = 1;
        let catalog = crate::hierarchical_ingest::HierarchicalCatalog::from_records_for_test(vec![
            first, second,
        ]);
        let bootstrap = HierarchicalCostModel {
            request_latency_ns: 1_000_000.0,
            bandwidth_bytes_per_ns: 0.1,
            io_concurrency: 1,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 100_000.0,
            decode_access_unit_ns: 20_000.0,
            fetch_decode_overlap: 0.0,
        };
        let config = RuntimeFeedbackConfig {
            min_observations: 1,
            activation_ape_threshold: 10.0,
            activation_stable_observations: 1,
            ..RuntimeFeedbackConfig::default()
        };
        let mut executor = HierarchicalBatchExecutor::new(
            catalog,
            Box::new(NoopBackend),
            vec![1],
            bootstrap.clone(),
            None,
            None,
            1,
            1,
            false,
            super::ResidentStateBudget {
                encoded_bytes: 0,
                read_ahead_bytes: 0,
                live_cursors: 0,
            },
        )
        .unwrap()
        .with_runtime_feedback_config(config)
        .unwrap();
        {
            let mut feedback = executor
                .runtime_feedback
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            feedback.observe(1, 1024, 200_000, 1, 20_000);
            assert!(feedback.snapshot().active);
        }
        assert_eq!(
            executor.model.request_latency_ns,
            bootstrap.request_latency_ns
        );

        let decision = executor
            .choose_lookahead(
                &[vec![super::LogicalTarget {
                    sample_id: 10,
                    video_id: "video".to_string(),
                    frame_idx: 1,
                }]],
                &[1],
                10_000_000.0,
            )
            .unwrap();

        assert_ne!(
            executor.model.request_latency_ns,
            bootstrap.request_latency_ns
        );
        assert_eq!(decision.selected.lookahead_batches, 1);
    }

    #[test]
    fn range_size_buckets_cover_each_boundary_once() {
        let ranges = [
            (0, 1),
            (1, 4_096),
            (2, 4_097),
            (3, 16_384),
            (4, 16_385),
            (5, 65_536),
            (6, 65_537),
            (7, 262_144),
            (8, 262_145),
        ];
        assert_eq!(range_size_buckets(&ranges), [2, 2, 2, 2, 1]);
    }

    #[test]
    fn cursor_compatibility_does_not_require_contiguous_targets() {
        let mut records = [record("video", 7, 8), record("video", 7, 9)];
        records[0].length = 800_000;
        records[1].length = 900_000;
        assert_eq!(single_gop_key(&records), Some(("video".to_string(), 7)));

        records[1].frame_idx = 10;
        assert_eq!(single_gop_key(&records), Some(("video".to_string(), 7)));
    }

    #[test]
    fn read_ahead_requires_an_observed_cursor_hit_and_respects_both_limits() {
        assert_eq!(read_ahead_allowance(false, false, 1_000, 100, 400), 0);
        assert_eq!(read_ahead_allowance(false, true, 1_000, 100, 400), 400);
        assert_eq!(read_ahead_allowance(true, false, 1_000, 100, 400), 400);
        assert_eq!(read_ahead_allowance(true, false, 1_000, 900, 400), 100);
    }

    #[test]
    fn frame_thread_release_margin_tracks_only_extra_pipeline_threads() {
        assert_eq!(frame_thread_release_margin(1), 0);
        assert_eq!(frame_thread_release_margin(2), 1);
        assert_eq!(frame_thread_release_margin(4), 3);
        assert_eq!(frame_thread_release_margin(8), 7);
    }

    #[test]
    fn existing_cursor_must_be_forward_servable_even_when_geometry_is_contiguous() {
        assert!(cursor_admission_with_existing_state(true, None));
        assert!(cursor_admission_with_existing_state(false, Some(true)));
        assert!(!cursor_admission_with_existing_state(true, Some(false)));
    }

    #[test]
    fn monotonic_reuse_excludes_duplicate_and_backward_targets() {
        assert_eq!(
            monotonic_visible_reuses(&[7], &[(1, vec![7]), (2, vec![6])]),
            (0, None)
        );
        assert_eq!(
            monotonic_visible_reuses(&[7], &[(1, vec![8, 8]), (2, vec![9])]),
            (2, Some(1))
        );
        assert_eq!(monotonic_visible_reuses(&[7, 8], &[]), (1, None));
    }

    #[test]
    fn window_liveness_uses_closure_consumers_without_a_target_count_threshold() {
        let mut first = record("video", 0, 1);
        first.closure_record_ids = vec![0, 1];
        let mut second = record("video", 0, 2);
        second.closure_record_ids = vec![0, 2];

        let same_batch =
            HierarchicalBatchExecutor::window_liveness(&[vec![first.clone(), second.clone()]]);
        assert_eq!(same_batch.visible_reuses(0, 0), 1);

        let cross_batch =
            HierarchicalBatchExecutor::window_liveness(&[vec![first.clone()], vec![second]]);
        assert_eq!(cross_batch.visible_reuses(0, 0), 1);
        assert_eq!(cross_batch.visible_reuses(0, 1), 0);

        let duplicate_only =
            HierarchicalBatchExecutor::window_liveness(&[vec![first.clone(), first]]);
        assert_eq!(duplicate_only.visible_reuses(0, 0), 0);
    }

    #[test]
    fn production_executor_does_not_branch_on_benchmark_workload_names() {
        let source = include_str!("hierarchical_scheduler.rs").to_ascii_lowercase();
        let forbidden = [
            ["uni", "form"].concat(),
            ["zi", "pf"].concat(),
            ["same", "_video"].concat(),
            ["same", "-video"].concat(),
            ["sequen", "tial"].concat(),
        ];
        for forbidden in forbidden {
            assert!(
                !source.contains(&forbidden),
                "production executor contains benchmark label {forbidden}"
            );
        }
    }
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
    pub decoded_access_units: usize,
    pub planner_candidate_count: usize,
    pub range_le_4k: usize,
    pub range_4k_to_16k: usize,
    pub range_16k_to_64k: usize,
    pub range_64k_to_256k: usize,
    pub range_gt_256k: usize,
    pub decode_groups: usize,
    pub encoded_cache_hits: usize,
    pub encoded_cache_misses: usize,
    pub encoded_cache_resident_bytes: u64,
    pub resident_cursor_hits: usize,
    pub resident_cursor_misses: usize,
    pub resident_read_ahead_bytes: u64,
    pub resident_encoded_budget_bytes: u64,
    pub resident_cursor_entries: usize,
    pub resident_cursor_capacity: usize,
    pub resident_cursor_candidates: usize,
    pub resident_cursor_selected: usize,
    pub resident_cursor_pinned_entries: usize,
    pub resident_cursor_probationary_entries: usize,
    pub cursor_policy_ns: u64,
    pub decoder_state_resets: usize,
    pub plan_ns: u64,
    pub fetch_ns: u64,
    pub assemble_ns: u64,
    pub decode_ns: u64,
    pub total_ns: u64,
    pub predicted_total_ns: f64,
    pub runtime_feedback_io_observations: usize,
    pub runtime_feedback_decode_observations: usize,
    pub runtime_feedback_rejected_observations: usize,
    pub runtime_feedback_active: bool,
    pub runtime_feedback_io_ape_ppm: u64,
    pub runtime_feedback_decode_ape_ppm: u64,
    pub runtime_feedback_io_tail_multiplier_ppm: u64,
    pub runtime_feedback_decode_tail_multiplier_ppm: u64,
    pub mode: &'static str,
}

pub(crate) fn accumulate_batch_stats(
    total: &mut HierarchicalBatchStats,
    part: &HierarchicalBatchStats,
) {
    total.logical_targets += part.logical_targets;
    total.unique_targets += part.unique_targets;
    total.physical_ranges += part.physical_ranges;
    total.client_requests += part.client_requests;
    total.useful_bytes += part.useful_bytes;
    total.fetched_bytes += part.fetched_bytes;
    total.submitted_access_units += part.submitted_access_units;
    total.decoded_access_units += part.decoded_access_units;
    total.planner_candidate_count += part.planner_candidate_count;
    total.range_le_4k += part.range_le_4k;
    total.range_4k_to_16k += part.range_4k_to_16k;
    total.range_16k_to_64k += part.range_16k_to_64k;
    total.range_64k_to_256k += part.range_64k_to_256k;
    total.range_gt_256k += part.range_gt_256k;
    total.decode_groups += part.decode_groups;
    total.encoded_cache_hits += part.encoded_cache_hits;
    total.encoded_cache_misses += part.encoded_cache_misses;
    total.resident_cursor_hits += part.resident_cursor_hits;
    total.resident_cursor_misses += part.resident_cursor_misses;
    total.decoder_state_resets += part.decoder_state_resets;
    total.resident_cursor_candidates += part.resident_cursor_candidates;
    total.resident_cursor_selected += part.resident_cursor_selected;
    total.cursor_policy_ns += part.cursor_policy_ns;
    total.resident_cursor_entries = part.resident_cursor_entries;
    total.resident_cursor_pinned_entries = part.resident_cursor_pinned_entries;
    total.resident_cursor_probationary_entries = part.resident_cursor_probationary_entries;
    total.plan_ns += part.plan_ns;
    total.fetch_ns += part.fetch_ns;
    total.assemble_ns += part.assemble_ns;
    total.decode_ns += part.decode_ns;
    total.runtime_feedback_io_observations = part.runtime_feedback_io_observations;
    total.runtime_feedback_decode_observations = part.runtime_feedback_decode_observations;
    total.runtime_feedback_rejected_observations = part.runtime_feedback_rejected_observations;
    total.runtime_feedback_active = part.runtime_feedback_active;
    total.runtime_feedback_io_ape_ppm = part.runtime_feedback_io_ape_ppm;
    total.runtime_feedback_decode_ape_ppm = part.runtime_feedback_decode_ape_ppm;
    total.runtime_feedback_io_tail_multiplier_ppm = part.runtime_feedback_io_tail_multiplier_ppm;
    total.runtime_feedback_decode_tail_multiplier_ppm =
        part.runtime_feedback_decode_tail_multiplier_ppm;
    if total.predicted_total_ns >= 0.0 {
        if part.predicted_total_ns < 0.0 {
            total.predicted_total_ns = -1.0;
        } else {
            total.predicted_total_ns += part.predicted_total_ns;
        }
    }
}

fn range_size_buckets(ranges: &[(u64, u64)]) -> [usize; 5] {
    let mut buckets = [0usize; 5];
    for (_, length) in ranges {
        let bucket = match *length {
            0..=4_096 => 0,
            4_097..=16_384 => 1,
            16_385..=65_536 => 2,
            65_537..=262_144 => 3,
            _ => 4,
        };
        buckets[bucket] += 1;
    }
    buckets
}

fn single_gop_key(target_records: &[HierarchicalRecordMeta]) -> Option<(String, u64)> {
    let first = target_records.first()?;
    if target_records
        .iter()
        .any(|record| record.video_id != first.video_id || record.gop_id != first.gop_id)
    {
        return None;
    }
    Some((first.video_id.clone(), first.gop_id))
}

fn resident_gop_bytes(records: &[HierarchicalRecordMeta]) -> Option<usize> {
    (!records.is_empty()).then(|| {
        records.iter().try_fold(0usize, |total, record| {
            total.checked_add(usize::try_from(record.length).ok()?)
        })
    })?
}

fn read_ahead_allowance(
    cache_hit: bool,
    eager_fill: bool,
    cache_budget: usize,
    resident_bytes: usize,
    read_ahead_limit: usize,
) -> usize {
    (cache_hit || eager_fill)
        .then(|| {
            cache_budget
                .saturating_sub(resident_bytes)
                .min(read_ahead_limit)
        })
        .unwrap_or(0)
}

fn frame_thread_release_margin(decoder_threads: usize) -> usize {
    decoder_threads.saturating_sub(1)
}

fn cursor_admission_with_existing_state(
    geometry_admitted: bool,
    existing_cursor_can_serve: Option<bool>,
) -> bool {
    existing_cursor_can_serve.unwrap_or(geometry_admitted)
}

fn monotonic_visible_reuses(
    current_ordinals: &[usize],
    future_ordinals: &[(usize, Vec<usize>)],
) -> (usize, Option<usize>) {
    let mut current = current_ordinals.to_vec();
    current.sort_unstable();
    current.dedup();
    let Some(current_max) = current.last().copied() else {
        return (0, None);
    };
    let mut visible_reuses = current.len().saturating_sub(1);
    let mut seen_future = HashSet::new();
    let mut next_use_batch = None;
    for (batch, ordinals) in future_ordinals {
        let forward = ordinals
            .iter()
            .copied()
            .filter(|ordinal| *ordinal > current_max && seen_future.insert(*ordinal))
            .count();
        if forward > 0 {
            next_use_batch.get_or_insert(*batch);
            visible_reuses += forward;
        }
    }
    (visible_reuses, next_use_batch)
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum HierarchicalAction {
    Adaptive,
    /// Decode a GOP prefix through the largest requested decode ordinal.  This
    /// intentionally ignores the sufficient-closure membership relation.
    KeyframePrefix,
    /// Reuse the calibrated planner's sparse-closure candidates while
    /// excluding the whole-GOP region alternative for same-closure ablation.
    CalibratedClosure,
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
    candidate_count: usize,
    mode: &'static str,
}

pub struct HierarchicalBatchExecutor {
    catalog: HierarchicalCatalog,
    layout: HierarchicalLayoutIndex,
    backend: Box<dyn StorageBackend>,
    codec_config: Vec<u8>,
    bootstrap_model: HierarchicalCostModel,
    model: HierarchicalCostModel,
    runtime_feedback: Arc<Mutex<RuntimeCostFeedback>>,
    max_merge_gap_bytes: Option<u64>,
    max_range_bytes: Option<u64>,
    decoder_pool: DecoderPool,
    decoder_threads: usize,
    cursor_decoder_threads: usize,
    decode_budget: Arc<DecodeBudget>,
    incremental_decoder_slots: SharedDecoderSlots,
    incremental_batch_deadline_fences: bool,
    encoded_cache: planner::ByteCache,
    resident_cursors: HashMap<(String, u64), StreamingGopState>,
    resident_state_policy: Box<dyn ResidentStatePolicy<(String, u64)>>,
    resident_cursor_capacity: usize,
    resident_encoded_budget_bytes: usize,
    resident_read_ahead_budget_bytes: usize,
    resident_read_ahead_bytes: usize,
    active_window_liveness: Option<Arc<WindowDependencyLiveness>>,
    active_window_records: Option<Arc<Vec<Vec<HierarchicalRecordMeta>>>>,
    active_window_batch: usize,
    active_cursor_candidates: HashMap<(String, u64), ResidentCandidate>,
    last_cursor_candidate_count: usize,
    last_cursor_selected_count: usize,
    last_cursor_policy_ns: u64,
}

struct StreamingGopState {
    cursor: decoder::MonotonicGopCursor,
    prefetched: VecDeque<(usize, Vec<u8>)>,
    prefetched_bytes: usize,
}

struct ResidentCursorRequest {
    key: (String, u64),
    targets: Vec<LogicalTarget>,
    target_records: Vec<HierarchicalRecordMeta>,
    unique_target_ids: Vec<u64>,
    target_output_ordinals: Vec<usize>,
    start_decode_ordinal: usize,
    access_units: Vec<Vec<u8>>,
    end_of_stream: bool,
    cache_hit: bool,
    stats: HierarchicalBatchStats,
    total_started: Instant,
}

struct ResidentCursorResult {
    key: (String, u64),
    state: StreamingGopState,
    outputs: Vec<HierarchicalOutput>,
    stats: HierarchicalBatchStats,
}

#[derive(Debug, Clone, Copy)]
pub struct ResidentStateBudget {
    /// Total encoded bytes retained across the AU cache and cursor read-ahead.
    pub encoded_bytes: usize,
    /// Maximum speculative suffix bytes retained by live cursors.
    pub read_ahead_bytes: usize,
    /// Maximum number of live decoder/DPB states.
    pub live_cursors: usize,
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
    pub fn resident_metrics(&self) -> (u64, u64, u64, usize, usize, usize, usize) {
        let (pinned, probationary) = self.resident_state_policy.residency_counts();
        (
            self.encoded_cache.resident_bytes() as u64,
            self.resident_read_ahead_bytes as u64,
            self.resident_encoded_budget_bytes as u64,
            self.resident_cursors.len(),
            self.resident_cursor_capacity,
            pinned,
            probationary,
        )
    }

    #[cfg(feature = "experiment-controls")]
    pub fn inspect_sparse_candidates(
        &self,
        targets: &[LogicalTarget],
    ) -> Result<
        (
            Vec<crate::hierarchical_layout::HierarchicalPlanEstimate>,
            crate::hierarchical_layout::HierarchicalPlanEstimate,
        ),
        String,
    > {
        let sample_ids = targets
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
            .collect::<Result<Vec<_>, _>>()?;
        let candidates = self.layout.enumerate_sparse_candidates(
            &sample_ids,
            self.max_merge_gap_bytes,
            self.max_range_bytes,
            &self.model,
        )?;
        let selected = self
            .layout
            .choose_plan(
                &sample_ids,
                self.max_merge_gap_bytes,
                self.max_range_bytes,
                &self.model,
            )?
            .selected;
        Ok((candidates, selected))
    }

    #[cfg(feature = "experiment-controls")]
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

    #[cfg(feature = "experiment-controls")]
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

    #[cfg(feature = "experiment-controls")]
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

    #[cfg(feature = "experiment-controls")]
    pub fn choose_lookahead(
        &mut self,
        batches: &[Vec<LogicalTarget>],
        candidates: &[usize],
        first_batch_slo_ns: f64,
    ) -> Result<DependencyLookaheadDecision, String> {
        // Execution workers share one feedback state. Refresh here as well as
        // on execute so horizon selection never uses a stale bootstrap after
        // another worker has published runtime observations.
        let latency_model = self.refresh_runtime_feedback_model();
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
        self.layout.choose_lookahead_with_models(
            &record_batches,
            candidates,
            first_batch_slo_ns,
            self.max_merge_gap_bytes,
            self.max_range_bytes,
            &self.model,
            &latency_model,
        )
    }

    fn decode_incremental_jobs(
        decoder_slots: &SharedDecoderSlots,
        mut jobs: Vec<IncrementalDecodeJob>,
        total_started: &Instant,
        decode_budget: &Arc<DecodeBudget>,
        decoder_threads: usize,
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
                let decode_budget = Arc::clone(decode_budget);
                handles.push(scope.spawn(move || {
                    let _permit = decode_budget.acquire(decoder_threads)?;
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
        resident_budget: ResidentStateBudget,
    ) -> Result<Self, String> {
        if codec_config.is_empty() {
            return Err("hierarchical executor requires codec configuration".to_string());
        }
        let layout = catalog.to_layout_index()?;
        model.validate()?;
        let runtime_feedback = RuntimeCostFeedback::new(&model, RuntimeFeedbackConfig::default())?;
        if incremental_decode_slots == 0 {
            return Err("incremental decoder slots must be positive".to_string());
        }
        let decode_budget = DecodeBudget::new(
            decoder_threads
                .max(1)
                .saturating_mul(incremental_decode_slots),
        );
        Ok(Self {
            catalog,
            layout,
            backend,
            codec_config,
            bootstrap_model: model.clone(),
            model,
            runtime_feedback: Arc::new(Mutex::new(runtime_feedback)),
            max_merge_gap_bytes,
            max_range_bytes,
            decoder_pool: DecoderPool::new(DecoderConfig {
                num_threads: decoder_threads,
            }),
            decoder_threads,
            cursor_decoder_threads: decoder_threads,
            decode_budget,
            incremental_decoder_slots: decoder::shared_decoder_slots_with_threads(
                incremental_decode_slots,
                decoder_threads,
            ),
            incremental_batch_deadline_fences,
            encoded_cache: planner::ByteCache::new(resident_budget.encoded_bytes),
            resident_cursors: HashMap::new(),
            resident_state_policy: Box::new(DependencyLivenessLru::new()),
            resident_cursor_capacity: resident_budget.live_cursors,
            resident_encoded_budget_bytes: resident_budget.encoded_bytes,
            resident_read_ahead_budget_bytes: resident_budget.read_ahead_bytes,
            resident_read_ahead_bytes: 0,
            active_window_liveness: None,
            active_window_records: None,
            active_window_batch: 0,
            active_cursor_candidates: HashMap::new(),
            last_cursor_candidate_count: 0,
            last_cursor_selected_count: 0,
            last_cursor_policy_ns: 0,
        })
    }

    pub(crate) fn with_runtime_feedback_config(
        mut self,
        config: RuntimeFeedbackConfig,
    ) -> Result<Self, String> {
        self.runtime_feedback = Arc::new(Mutex::new(RuntimeCostFeedback::new(
            &self.bootstrap_model,
            config,
        )?));
        self.model = self.bootstrap_model.clone();
        Ok(self)
    }

    pub(crate) fn with_shared_runtime_feedback(
        mut self,
        runtime_feedback: Arc<Mutex<RuntimeCostFeedback>>,
    ) -> Self {
        self.runtime_feedback = runtime_feedback;
        self
    }

    fn apply_runtime_feedback_snapshot(
        stats: &mut HierarchicalBatchStats,
        snapshot: RuntimeFeedbackSnapshot,
    ) {
        stats.runtime_feedback_io_observations = snapshot.io_observations;
        stats.runtime_feedback_decode_observations = snapshot.decode_observations;
        stats.runtime_feedback_rejected_observations = snapshot.rejected_observations;
        stats.runtime_feedback_active = snapshot.active;
        stats.runtime_feedback_io_ape_ppm = snapshot.io_absolute_percentage_error_ppm;
        stats.runtime_feedback_decode_ape_ppm = snapshot.decode_absolute_percentage_error_ppm;
        stats.runtime_feedback_io_tail_multiplier_ppm = snapshot.io_tail_multiplier_ppm;
        stats.runtime_feedback_decode_tail_multiplier_ppm = snapshot.decode_tail_multiplier_ppm;
    }

    fn observe_runtime_feedback(&mut self, stats: &mut HierarchicalBatchStats) {
        let mut feedback = self
            .runtime_feedback
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        feedback.observe(
            stats.physical_ranges,
            stats.fetched_bytes,
            stats.fetch_ns,
            stats.submitted_access_units,
            stats.decode_ns,
        );
        self.model = feedback.apply_to(&self.bootstrap_model);
        Self::apply_runtime_feedback_snapshot(stats, feedback.snapshot());
    }

    fn refresh_runtime_feedback_model(&mut self) -> HierarchicalCostModel {
        let feedback = self
            .runtime_feedback
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.model = feedback.apply_to(&self.bootstrap_model);
        feedback.apply_tail_to(&self.bootstrap_model)
    }

    pub(crate) fn with_shared_decode_budget(
        mut self,
        decode_budget: Arc<DecodeBudget>,
        cursor_decoder_threads: usize,
    ) -> Result<Self, String> {
        if cursor_decoder_threads == 0 {
            return Err("cursor decoder threads must be positive".to_string());
        }
        // Validate the requested weight before any worker starts. The permit
        // is immediately released and only establishes that the shared budget
        // can admit one cursor operation.
        drop(decode_budget.acquire(cursor_decoder_threads)?);
        self.decode_budget = decode_budget;
        self.cursor_decoder_threads = cursor_decoder_threads;
        Ok(self)
    }

    fn cursor_compatible_gop_key(
        &self,
        target_records: &[HierarchicalRecordMeta],
    ) -> Option<(String, u64)> {
        if self.resident_encoded_budget_bytes == 0
            || self.resident_cursor_capacity == 0
            || target_records.is_empty()
        {
            return None;
        }
        // The byte budget bounds only encoded read-ahead. The live decoder's
        // DPB is accounted separately and the cursor fetches an incremental
        // suffix, so admission must not require the entire GOP payload to fit
        // in the encoded cache.
        single_gop_key(target_records)
    }

    fn display_ordinals(
        &self,
        key: &(String, u64),
        target_records: &[HierarchicalRecordMeta],
    ) -> Option<Vec<usize>> {
        let mut display_records = self
            .catalog
            .records_for_gop(&key.0, key.1)?
            .iter()
            .map(|record_id| self.catalog.record(*record_id).unwrap())
            .collect::<Vec<_>>();
        display_records.sort_unstable_by_key(|record| record.frame_idx);
        let ordinal_by_id = display_records
            .iter()
            .enumerate()
            .map(|(ordinal, record)| (record.record_id, ordinal))
            .collect::<HashMap<_, _>>();
        target_records
            .iter()
            .map(|record| ordinal_by_id.get(&record.record_id).copied())
            .collect()
    }

    fn resident_candidate(
        &self,
        key: &(String, u64),
        target_records: &[HierarchicalRecordMeta],
        window_records: &[Vec<HierarchicalRecordMeta>],
        liveness: &WindowDependencyLiveness,
        batch_index: usize,
    ) -> ResidentCandidate {
        let dependency_ids = target_records
            .iter()
            .flat_map(|record| record.closure_record_ids.iter().copied())
            .collect::<HashSet<_>>();
        let dependency_reuses = dependency_ids
            .iter()
            .map(|record_id| liveness.visible_reuses(*record_id, batch_index))
            .sum::<usize>();
        let current_ordinals = self
            .display_ordinals(key, target_records)
            .unwrap_or_default();
        let future_ordinals = window_records
            .iter()
            .enumerate()
            .skip(batch_index.saturating_add(1))
            .map(|(future_batch, records)| {
                let group = records
                    .iter()
                    .filter(|record| record.video_id == key.0 && record.gop_id == key.1)
                    .cloned()
                    .collect::<Vec<_>>();
                (
                    future_batch,
                    self.display_ordinals(key, &group).unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>();
        let (mut visible_consumers, mut next_use_batch) =
            monotonic_visible_reuses(&current_ordinals, &future_ordinals);
        // Closure reuse is necessary, but a live decoder cursor additionally
        // requires a distinct target that can be served by monotonic progress.
        if dependency_reuses == 0 {
            visible_consumers = 0;
            next_use_batch = None;
        }
        ResidentCandidate {
            visible_consumers,
            next_use_batch,
        }
    }

    /// Compare a live monotonic decoder suffix with the ordinary adaptive
    /// plan over the same visible requests. The decision depends only on
    /// registered closures, physical extents, current resident state, and the
    /// backend calibration; benchmark workload names never enter the core.
    fn resident_cursor_is_cost_effective(
        &self,
        key: &(String, u64),
        window_records: &[Vec<HierarchicalRecordMeta>],
        batch_index: usize,
    ) -> bool {
        let Some(gop_record_ids) = self.catalog.records_for_gop(&key.0, key.1) else {
            return false;
        };
        let mut gop_records = gop_record_ids
            .iter()
            .filter_map(|record_id| self.catalog.record(*record_id))
            .collect::<Vec<_>>();
        gop_records.sort_unstable_by_key(|record| record.decode_ordinal);
        if gop_records.is_empty() {
            return false;
        }
        let local_ordinal = gop_records
            .iter()
            .enumerate()
            .map(|(ordinal, record)| (record.record_id, ordinal))
            .collect::<HashMap<_, _>>();
        let mut display_records = gop_records.clone();
        display_records.sort_unstable_by_key(|record| record.frame_idx);
        let reorder_margin = display_records
            .iter()
            .enumerate()
            .map(|(display_ordinal, record)| {
                local_ordinal[&record.record_id].abs_diff(display_ordinal)
            })
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .saturating_add(frame_thread_release_margin(self.cursor_decoder_threads));
        let existing = self.resident_cursors.get(key);
        let mut decode_cursor = existing
            .map(|state| state.cursor.next_decode_ordinal())
            .unwrap_or(0);
        let mut fetch_cursor = decode_cursor
            + existing
                .map(|state| state.prefetched.len())
                .unwrap_or_default();
        let mut cursor_total_ns = 0.0;
        let resident_ids = self.resident_record_ids();
        let mut decoder_open = existing.is_some();
        let visible_groups = window_records
            .iter()
            .skip(batch_index)
            .map(|records| {
                records
                    .iter()
                    .filter(|record| record.video_id == key.0 && record.gop_id == key.1)
                    .collect::<Vec<_>>()
            })
            .filter(|group| !group.is_empty())
            .collect::<Vec<_>>();
        let all_target_ids = visible_groups
            .iter()
            .flat_map(|group| group.iter().map(|record| record.record_id))
            .collect::<Vec<_>>();
        let Ok(adaptive) = self.layout.choose_plan_with_resident(
            &all_target_ids,
            &resident_ids,
            self.max_merge_gap_bytes,
            self.max_range_bytes,
            &self.model,
        ) else {
            return false;
        };
        let adaptive_decode_jobs = visible_groups
            .iter()
            .map(|group| match adaptive.selected.mode {
                HierarchicalReadMode::SparseClosure => group
                    .iter()
                    .flat_map(|record| record.closure_record_ids.iter().copied())
                    .collect::<HashSet<_>>()
                    .len(),
                HierarchicalReadMode::ContiguousRegion => gop_records.len(),
            })
            .collect::<Vec<_>>();
        let adaptive_ranges = adaptive
            .selected
            .ranges
            .iter()
            .map(|range| (range.offset, range.length))
            .collect::<Vec<_>>();
        let adaptive_total_ns = self.model.estimate_parallel_window_execution(
            &adaptive_ranges,
            &adaptive_decode_jobs,
            self.incremental_decoder_slots.len(),
        );

        for group in visible_groups {
            let Some(closure_max) = group
                .iter()
                .flat_map(|record| record.closure_record_ids.iter())
                .filter_map(|record_id| local_ordinal.get(record_id).copied())
                .max()
            else {
                return false;
            };
            let required_max = closure_max
                .saturating_add(reorder_margin)
                .min(gop_records.len() - 1);
            let access_units = required_max.saturating_add(1).saturating_sub(decode_cursor);
            let ranges = if fetch_cursor <= required_max {
                let offset = gop_records[fetch_cursor].offset;
                let end = gop_records[required_max].offset + gop_records[required_max].length;
                vec![(offset, end - offset)]
            } else {
                Vec::new()
            };
            cursor_total_ns += self
                .model
                .estimate_execution(&ranges, access_units, !decoder_open);
            decoder_open = true;
            decode_cursor = decode_cursor.max(required_max.saturating_add(1));
            fetch_cursor = fetch_cursor.max(required_max.saturating_add(1));
        }

        cursor_total_ns <= adaptive_total_ns + self.model.selection_tolerance_ns
    }

    fn window_liveness(batch_records: &[Vec<HierarchicalRecordMeta>]) -> WindowDependencyLiveness {
        let consumers = batch_records
            .iter()
            .map(|records| {
                let mut by_dependency = HashMap::<u64, usize>::new();
                let mut seen_targets = HashSet::new();
                for record in records {
                    if !seen_targets.insert(record.record_id) {
                        continue;
                    }
                    for dependency_id in &record.closure_record_ids {
                        if *dependency_id != record.record_id {
                            *by_dependency.entry(*dependency_id).or_default() += 1;
                        }
                    }
                }
                by_dependency
            })
            .collect();
        WindowDependencyLiveness::from_batches(consumers)
    }

    fn window_has_cursor_reuse(
        &self,
        batch_records: &[Vec<HierarchicalRecordMeta>],
        liveness: &WindowDependencyLiveness,
    ) -> bool {
        if self.resident_encoded_budget_bytes == 0 || self.resident_cursor_capacity == 0 {
            return false;
        }
        batch_records
            .iter()
            .enumerate()
            .any(|(batch_index, records)| {
                let mut groups = HashMap::<(String, u64), Vec<HierarchicalRecordMeta>>::new();
                for record in records {
                    groups
                        .entry((record.video_id.clone(), record.gop_id))
                        .or_default()
                        .push(record.clone());
                }
                groups.into_iter().any(|(key, records)| {
                    let candidate = self.resident_candidate(
                        &key,
                        &records,
                        batch_records,
                        liveness,
                        batch_index,
                    );
                    candidate.visible_consumers > 0
                        && self.resident_cursor_is_cost_effective(&key, batch_records, batch_index)
                })
            })
    }

    fn resident_cursor_keys_for_batch(
        &mut self,
        target_records: &[HierarchicalRecordMeta],
        window_records: &[Vec<HierarchicalRecordMeta>],
        liveness: &WindowDependencyLiveness,
        batch_index: usize,
    ) -> HashSet<(String, u64)> {
        let started = Instant::now();
        self.active_cursor_candidates.clear();
        self.last_cursor_candidate_count = 0;
        self.last_cursor_selected_count = 0;
        if self.resident_encoded_budget_bytes == 0 || self.resident_cursor_capacity == 0 {
            self.last_cursor_policy_ns = started.elapsed().as_nanos() as u64;
            return HashSet::new();
        }
        let resident_keys = self.resident_cursors.keys().cloned().collect::<Vec<_>>();
        for key in resident_keys {
            let next_use = window_records
                .iter()
                .enumerate()
                .skip(batch_index.saturating_add(1))
                .find_map(|(future_batch, records)| {
                    let group = records
                        .iter()
                        .filter(|record| record.video_id == key.0 && record.gop_id == key.1)
                        .cloned()
                        .collect::<Vec<_>>();
                    let ordinals = self.display_ordinals(&key, &group)?;
                    self.resident_cursors[&key]
                        .cursor
                        .can_serve(&ordinals)
                        .then_some(future_batch)
                });
            self.resident_state_policy
                .set_priority(&key, next_use.is_some(), next_use);
        }

        let mut groups = HashMap::<(String, u64), Vec<HierarchicalRecordMeta>>::new();
        for record in target_records {
            groups
                .entry((record.video_id.clone(), record.gop_id))
                .or_default()
                .push(record.clone());
        }
        let mut keys = HashSet::new();
        self.last_cursor_candidate_count = groups.len();
        for (key, records) in groups {
            let display_ordinals = self.display_ordinals(&key, &records);
            let candidate =
                self.resident_candidate(&key, &records, window_records, liveness, batch_index);
            if self.resident_state_policy.admit(candidate) {
                keys.insert(key.clone());
            }
            let existing_cursor_can_serve = self
                .resident_cursors
                .get(&key)
                .zip(display_ordinals.as_ref())
                .map(|(state, ordinals)| state.cursor.can_serve(ordinals));
            let geometry_admitted = cursor_admission_with_existing_state(
                keys.contains(&key),
                existing_cursor_can_serve,
            );
            if geometry_admitted
                && self.resident_cursor_is_cost_effective(&key, window_records, batch_index)
            {
                self.active_cursor_candidates.insert(key.clone(), candidate);
                keys.insert(key);
            } else {
                keys.remove(&key);
            }
        }
        self.last_cursor_selected_count = keys.len();
        self.last_cursor_policy_ns = started.elapsed().as_nanos() as u64;
        keys
    }

    fn resident_record_ids(&self) -> HashSet<u64> {
        self.encoded_cache.resident_ids()
    }

    fn cached_encoded_record(&mut self, record_id: u64) -> Option<Vec<u8>> {
        self.encoded_cache.get(record_id)
    }

    fn rebalance_encoded_cache(&mut self) {
        self.encoded_cache.resize(
            self.resident_encoded_budget_bytes
                .saturating_sub(self.resident_read_ahead_bytes),
        );
    }

    fn execute_resident_state(
        &mut self,
        targets: &[LogicalTarget],
        target_records: &[HierarchicalRecordMeta],
        resident_cursor_keys: &HashSet<(String, u64)>,
        total_started: Instant,
        cursor_decoder_threads_hint: usize,
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
        if order.len() < 2 || resident_cursor_keys.is_empty() {
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
            decoded_access_units: 0,
            planner_candidate_count: 0,
            range_le_4k: 0,
            range_4k_to_16k: 0,
            range_16k_to_64k: 0,
            range_64k_to_256k: 0,
            range_gt_256k: 0,
            decode_groups: 0,
            encoded_cache_hits: 0,
            encoded_cache_misses: 0,
            encoded_cache_resident_bytes: 0,
            resident_cursor_hits: 0,
            resident_cursor_misses: 0,
            resident_read_ahead_bytes: 0,
            resident_encoded_budget_bytes: self.resident_encoded_budget_bytes as u64,
            resident_cursor_entries: self.resident_cursors.len(),
            resident_cursor_capacity: self.resident_cursor_capacity,
            resident_cursor_candidates: self.last_cursor_candidate_count,
            resident_cursor_selected: self.last_cursor_selected_count,
            resident_cursor_pinned_entries: self.resident_state_policy.residency_counts().0,
            resident_cursor_probationary_entries: self.resident_state_policy.residency_counts().1,
            cursor_policy_ns: self.last_cursor_policy_ns,
            decoder_state_resets: 0,
            plan_ns: 0,
            fetch_ns: 0,
            assemble_ns: 0,
            decode_ns: 0,
            total_ns: 0,
            predicted_total_ns: 0.0,
            runtime_feedback_io_observations: 0,
            runtime_feedback_decode_observations: 0,
            runtime_feedback_rejected_observations: 0,
            runtime_feedback_active: false,
            runtime_feedback_io_ape_ppm: 0,
            runtime_feedback_decode_ape_ppm: 0,
            runtime_feedback_io_tail_multiplier_ppm: 0,
            runtime_feedback_decode_tail_multiplier_ppm: 0,
            mode: "resident_state_mixed",
        };
        let mut fallback_targets = Vec::new();
        for key in order {
            let (group_targets, group_records) = groups.remove(&key).unwrap();
            if resident_cursor_keys.contains(&key)
                && self.cursor_compatible_gop_key(&group_records).is_some()
            {
                let (outputs, part) = self.execute_resident_cursor(
                    &group_targets,
                    &group_records,
                    key,
                    Instant::now(),
                    cursor_decoder_threads_hint,
                )?;
                for output in outputs {
                    decoded.insert(output.sample_id, output);
                }
                accumulate_batch_stats(&mut stats, &part);
            } else {
                fallback_targets.extend(group_targets);
            }
        }
        // Preserve one global closure plan for every target not served by a
        // resident cursor. Partitioning fallback work by GOP would discard the
        // cross-video coalescing and I/O concurrency available to the ordinary
        // adaptive path.
        if !fallback_targets.is_empty() {
            let (outputs, part) = self.execute_action_with_context(
                &fallback_targets,
                HierarchicalAction::Adaptive,
                Some(cursor_decoder_threads_hint),
                false,
            )?;
            for output in outputs {
                decoded.insert(output.sample_id, output);
            }
            accumulate_batch_stats(&mut stats, &part);
        }
        stats.resident_read_ahead_bytes = self.resident_read_ahead_bytes as u64;
        stats.resident_cursor_entries = self.resident_cursors.len();
        stats.encoded_cache_resident_bytes = self.encoded_cache.resident_bytes() as u64;
        stats.total_ns = total_started.elapsed().as_nanos() as u64;
        let outputs = targets
            .iter()
            .map(|target| {
                decoded.get(&target.sample_id).cloned().ok_or_else(|| {
                    format!("resident-state path missed target {}", target.sample_id)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some((outputs, stats)))
    }

    fn prepare_resident_cursor(
        &mut self,
        targets: &[LogicalTarget],
        target_records: &[HierarchicalRecordMeta],
        key: (String, u64),
        total_started: Instant,
        cursor_decoder_threads_hint: usize,
    ) -> Result<ResidentCursorRequest, String> {
        let plan_started = Instant::now();
        let record_ids = self
            .catalog
            .records_for_gop(&key.0, key.1)
            .ok_or_else(|| format!("missing streaming GOP ({}, {})", key.0, key.1))?
            .to_vec();
        let mut records = record_ids
            .iter()
            .map(|record_id| self.catalog.record(*record_id).unwrap().clone())
            .collect::<Vec<_>>();
        records.sort_unstable_by_key(|record| record.decode_ordinal);
        let local_decode_ordinal = records
            .iter()
            .enumerate()
            .map(|(ordinal, record)| (record.record_id, ordinal))
            .collect::<HashMap<_, _>>();
        let useful_ids = target_records
            .iter()
            .flat_map(|record| record.closure_record_ids.iter().copied())
            .collect::<HashSet<_>>();
        let useful_bytes = useful_ids
            .iter()
            .map(|record_id| self.catalog.record(*record_id).unwrap().length)
            .sum();
        let cache_hit = self.resident_cursors.contains_key(&key);
        if !cache_hit {
            while self.resident_cursors.len() >= self.resident_cursor_capacity {
                let evicted = self
                    .resident_state_policy
                    .victim()
                    .ok_or("resident cursor cache accounting underflow")?;
                if let Some(state) = self.resident_cursors.remove(&evicted) {
                    self.resident_read_ahead_bytes = self
                        .resident_read_ahead_bytes
                        .saturating_sub(state.prefetched_bytes);
                }
            }
            self.resident_cursors.insert(
                key.clone(),
                StreamingGopState {
                    cursor: decoder::MonotonicGopCursor::new(DecoderConfig {
                        num_threads: cursor_decoder_threads_hint,
                    })
                    .map_err(|error| error.to_string())?,
                    prefetched: VecDeque::new(),
                    prefetched_bytes: 0,
                },
            );
            self.resident_state_policy.on_insert(key.clone());
        } else {
            self.resident_state_policy.on_hit(&key);
        }
        if let Some(candidate) = self.active_cursor_candidates.get(&key) {
            self.resident_state_policy.set_priority(
                &key,
                candidate.next_use_batch.is_some(),
                candidate.next_use_batch,
            );
        }

        let mut unique_target_ids = Vec::new();
        let mut seen = HashSet::new();
        for record in target_records {
            if seen.insert(record.record_id) {
                unique_target_ids.push(record.record_id);
            }
        }
        let unique_target_records = unique_target_ids
            .iter()
            .map(|record_id| self.catalog.record(*record_id).unwrap().clone())
            .collect::<Vec<_>>();
        let target_output_ordinals = self
            .display_ordinals(&key, &unique_target_records)
            .ok_or_else(|| format!("missing GOP display order for ({}, {})", key.0, key.1))?;
        if !self
            .resident_cursors
            .get(&key)
            .unwrap()
            .cursor
            .can_serve(&target_output_ordinals)
        {
            return Err(format!(
                "resident cursor ({}, {}) cannot serve a backward request",
                key.0, key.1
            ));
        }
        let start_decode_ordinal = self
            .resident_cursors
            .get(&key)
            .unwrap()
            .cursor
            .next_decode_ordinal();
        let closure_max_decode_ordinal = useful_ids
            .iter()
            .filter_map(|record_id| local_decode_ordinal.get(record_id).copied())
            .max()
            .ok_or("resident cursor request has no required access unit")?;
        // libavcodec may retain decoded B frames until subsequent packets make
        // presentation order unambiguous. Derive a conservative release margin
        // from this GOP's registered decode/display permutation instead of
        // hard-coding a codec-specific B-frame count.
        let mut display_records = records.iter().collect::<Vec<_>>();
        display_records.sort_unstable_by_key(|record| record.frame_idx);
        let resident_cursor_threads = self
            .resident_cursors
            .get(&key)
            .unwrap()
            .cursor
            .decoder_threads();
        let reorder_margin = display_records
            .iter()
            .enumerate()
            .map(|(display_ordinal, record)| {
                local_decode_ordinal[&record.record_id].abs_diff(display_ordinal)
            })
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            // FFmpeg frame threading delays output behind packet submission.
            // Keep the cursor live by feeding that bounded pipeline instead
            // of flushing it between requests, which would discard the DPB.
            .saturating_add(frame_thread_release_margin(resident_cursor_threads));
        let last_decode_ordinal = records.len() - 1;
        let required_max_decode_ordinal = closure_max_decode_ordinal
            .saturating_add(reorder_margin)
            .min(last_decode_ordinal);
        let (queue_len, queued_consumed_bytes) = {
            let state = self.resident_cursors.get(&key).unwrap();
            let queued_consumed_bytes = state
                .prefetched
                .iter()
                .take_while(|(ordinal, _)| *ordinal <= required_max_decode_ordinal)
                .map(|(_, bytes)| bytes.len())
                .sum::<usize>();
            (state.prefetched.len(), queued_consumed_bytes)
        };
        let fetch_start_ordinal = start_decode_ordinal + queue_len;
        let base_resident_after_consume = self
            .resident_read_ahead_bytes
            .saturating_sub(queued_consumed_bytes);
        // Establishing a cursor is not itself evidence that the caller will
        // continue through this GOP. Avoid speculative bytes on first touch;
        // once a later request hits the same live cursor, observed forward
        // reuse admits the backend-calibrated read-ahead allowance.
        // A first-touch cursor may eagerly stage its suffix only when the
        // complete encoded GOP is itself within this worker's cache budget.
        // Larger GOPs still keep a live DPB but advance incrementally.
        let eager_fill = !cache_hit
            && resident_gop_bytes(&records)
                .is_some_and(|gop_bytes| gop_bytes <= self.resident_encoded_budget_bytes);
        let mut read_ahead_budget = read_ahead_allowance(
            cache_hit,
            eager_fill,
            self.resident_encoded_budget_bytes,
            base_resident_after_consume,
            self.resident_read_ahead_budget_bytes,
        );
        let mut fetch_records = Vec::<(usize, &HierarchicalRecordMeta)>::new();
        if fetch_start_ordinal <= required_max_decode_ordinal {
            fetch_records.extend(records.iter().enumerate().filter(|(ordinal, _)| {
                *ordinal >= fetch_start_ordinal && *ordinal <= required_max_decode_ordinal
            }));
        }
        let first_read_ahead_ordinal = required_max_decode_ordinal
            .saturating_add(1)
            .max(fetch_start_ordinal);
        for (ordinal, record) in records
            .iter()
            .enumerate()
            .filter(|(ordinal, _)| *ordinal >= first_read_ahead_ordinal)
        {
            let estimated_bytes = usize::try_from(record.length)
                .map_err(|_| "read-ahead record length exceeds usize".to_string())?;
            if estimated_bytes > read_ahead_budget {
                break;
            }
            fetch_records.push((ordinal, record));
            read_ahead_budget -= estimated_bytes;
        }
        if fetch_start_ordinal <= required_max_decode_ordinal {
            let mandatory_count = required_max_decode_ordinal - fetch_start_ordinal + 1;
            if fetch_records.len() < mandatory_count {
                return Err(format!(
                    "GOP ({}, {}) has a non-contiguous decode-ordinal extension",
                    key.0, key.1
                ));
            }
        }
        let ranges = if fetch_records.is_empty() {
            Vec::new()
        } else {
            let offset = fetch_records
                .iter()
                .map(|(_, record)| record.offset)
                .min()
                .unwrap();
            let end = fetch_records
                .iter()
                .map(|(_, record)| record.offset + record.length)
                .max()
                .unwrap();
            vec![(offset, end - offset)]
        };
        let plan_ns = plan_started.elapsed().as_nanos() as u64;
        let fetch_started = Instant::now();
        let buffers = if ranges.is_empty() {
            Vec::new()
        } else {
            self.backend
                .read_byte_ranges(&ranges)
                .map_err(|error| error.to_string())?
        };
        let fetch_ns = fetch_started.elapsed().as_nanos() as u64;
        let fetched_bytes = buffers.iter().map(|buffer| buffer.len() as u64).sum();
        let physical_ranges = self.backend.physical_ranges_for_ranges(&ranges);
        let client_requests = self.backend.client_requests_for_ranges(&ranges);
        let range_buckets = range_size_buckets(&ranges);
        let assemble_started = Instant::now();
        let fetched_access_units = fetch_records
            .iter()
            .map(|(ordinal, record)| {
                let sample = Self::extract_record(record, &ranges, &buffers)?;
                Ok((
                    *ordinal,
                    mp4_sample_to_annex_b(&sample, record.nal_length_size)?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        {
            let state = self.resident_cursors.get_mut(&key).unwrap();
            for (ordinal, access_unit) in fetched_access_units {
                state.prefetched_bytes += access_unit.len();
                self.resident_read_ahead_bytes += access_unit.len();
                state.prefetched.push_back((ordinal, access_unit));
            }
        }
        let mut access_units = Vec::new();
        {
            let state = self.resident_cursors.get_mut(&key).unwrap();
            while state
                .prefetched
                .front()
                .is_some_and(|(ordinal, _)| *ordinal <= required_max_decode_ordinal)
            {
                let (_, access_unit) = state.prefetched.pop_front().unwrap();
                state.prefetched_bytes -= access_unit.len();
                self.resident_read_ahead_bytes -= access_unit.len();
                access_units.push(access_unit);
            }
        }
        self.rebalance_encoded_cache();
        let assemble_ns = assemble_started.elapsed().as_nanos() as u64;
        let end_of_stream = required_max_decode_ordinal == last_decode_ordinal;
        Ok(ResidentCursorRequest {
            key,
            targets: targets.to_vec(),
            target_records: target_records.to_vec(),
            unique_target_ids,
            target_output_ordinals,
            start_decode_ordinal,
            access_units,
            end_of_stream,
            cache_hit,
            total_started,
            stats: HierarchicalBatchStats {
                logical_targets: targets.len(),
                unique_targets: seen.len(),
                physical_ranges,
                client_requests,
                useful_bytes,
                fetched_bytes,
                submitted_access_units: 0,
                decoded_access_units: 0,
                planner_candidate_count: 1,
                range_le_4k: range_buckets[0],
                range_4k_to_16k: range_buckets[1],
                range_16k_to_64k: range_buckets[2],
                range_64k_to_256k: range_buckets[3],
                range_gt_256k: range_buckets[4],
                decode_groups: 1,
                encoded_cache_hits: 0,
                encoded_cache_misses: 0,
                encoded_cache_resident_bytes: self.encoded_cache.resident_bytes() as u64,
                resident_cursor_hits: usize::from(cache_hit),
                resident_cursor_misses: usize::from(!cache_hit),
                resident_read_ahead_bytes: self.resident_read_ahead_bytes as u64,
                resident_encoded_budget_bytes: self.resident_encoded_budget_bytes as u64,
                resident_cursor_entries: self.resident_cursors.len(),
                resident_cursor_capacity: self.resident_cursor_capacity,
                resident_cursor_candidates: self.last_cursor_candidate_count,
                resident_cursor_selected: self.last_cursor_selected_count,
                resident_cursor_pinned_entries: self.resident_state_policy.residency_counts().0,
                resident_cursor_probationary_entries: self
                    .resident_state_policy
                    .residency_counts()
                    .1,
                cursor_policy_ns: self.last_cursor_policy_ns,
                decoder_state_resets: 0,
                plan_ns,
                fetch_ns,
                assemble_ns,
                decode_ns: 0,
                total_ns: 0,
                predicted_total_ns: -1.0,
                runtime_feedback_io_observations: 0,
                runtime_feedback_decode_observations: 0,
                runtime_feedback_rejected_observations: 0,
                runtime_feedback_active: false,
                runtime_feedback_io_ape_ppm: 0,
                runtime_feedback_decode_ape_ppm: 0,
                runtime_feedback_io_tail_multiplier_ppm: 0,
                runtime_feedback_decode_tail_multiplier_ppm: 0,
                mode: "resident_cursor_prepared",
            },
        })
    }

    fn decode_resident_cursor_job(
        mut request: ResidentCursorRequest,
        mut state: StreamingGopState,
        codec_config: &[u8],
        decode_budget: &Arc<DecodeBudget>,
    ) -> Result<ResidentCursorResult, String> {
        let _permit = decode_budget.acquire(state.cursor.decoder_threads())?;
        let decode_started = Instant::now();
        let submitted_access_units = request.access_units.len();
        let frames = state
            .cursor
            .decode_extension(
                codec_config,
                request.start_decode_ordinal,
                &request.access_units,
                &request.target_output_ordinals,
                request.end_of_stream,
            )
            .map_err(|error| error.to_string())?;
        let mut decoded = request
            .unique_target_ids
            .iter()
            .copied()
            .zip(frames)
            .collect::<HashMap<_, _>>();
        let mut remaining_consumers = request.target_records.iter().fold(
            HashMap::<u64, usize>::new(),
            |mut counts, record| {
                *counts.entry(record.record_id).or_default() += 1;
                counts
            },
        );
        let outputs = request
            .targets
            .iter()
            .zip(&request.target_records)
            .map(|(target, record)| {
                let remaining = remaining_consumers
                    .get_mut(&record.record_id)
                    .expect("every target record has a consumer count");
                *remaining -= 1;
                let frame = if *remaining == 0 {
                    decoded
                        .remove(&record.record_id)
                        .expect("decoded cursor frame must exist")
                } else {
                    decoded[&record.record_id].clone()
                };
                HierarchicalOutput {
                    sample_id: target.sample_id,
                    frame,
                }
            })
            .collect();
        request.stats.submitted_access_units = submitted_access_units;
        request.stats.decoded_access_units = submitted_access_units;
        request.stats.decoder_state_resets = 0;
        request.stats.decode_ns = decode_started.elapsed().as_nanos() as u64;
        request.stats.total_ns = request.total_started.elapsed().as_nanos() as u64;
        request.stats.mode = if request.cache_hit {
            "resident_cursor_hit"
        } else {
            "resident_cursor_fill"
        };
        Ok(ResidentCursorResult {
            key: request.key,
            state,
            outputs,
            stats: request.stats,
        })
    }

    fn execute_resident_cursor(
        &mut self,
        targets: &[LogicalTarget],
        target_records: &[HierarchicalRecordMeta],
        key: (String, u64),
        total_started: Instant,
        cursor_decoder_threads_hint: usize,
    ) -> Result<(Vec<HierarchicalOutput>, HierarchicalBatchStats), String> {
        let request = self.prepare_resident_cursor(
            targets,
            target_records,
            key.clone(),
            total_started,
            cursor_decoder_threads_hint,
        )?;
        let state = self.resident_cursors.remove(&key).ok_or_else(|| {
            format!(
                "prepared resident cursor ({}, {}) disappeared",
                key.0, key.1
            )
        })?;
        let state_prefetched_bytes = state.prefetched_bytes;
        let decoded = Self::decode_resident_cursor_job(
            request,
            state,
            &self.codec_config,
            &self.decode_budget,
        );
        let mut result = match decoded {
            Ok(result) => result,
            Err(error) => {
                self.resident_read_ahead_bytes = self
                    .resident_read_ahead_bytes
                    .saturating_sub(state_prefetched_bytes);
                self.resident_state_policy.on_remove(&key);
                self.rebalance_encoded_cache();
                return Err(error);
            }
        };
        self.resident_cursors
            .insert(result.key.clone(), result.state);
        result.stats.resident_read_ahead_bytes = self.resident_read_ahead_bytes as u64;
        result.stats.resident_cursor_entries = self.resident_cursors.len();
        Ok((result.outputs, result.stats))
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

    /// Execute every visible batch through the same adaptive path used by
    /// single-batch calls whenever the window proves dependency reuse. Windows
    /// without reuse retain the completion-driven sparse executor below.
    pub fn execute_window(
        &mut self,
        batches: &[Vec<LogicalTarget>],
    ) -> Result<HierarchicalIncrementalWindow, String> {
        if batches.is_empty() || batches.iter().any(Vec::is_empty) {
            return Err("incremental hierarchical window requires non-empty batches".to_string());
        }
        self.refresh_runtime_feedback_model();
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
        let liveness = Self::window_liveness(&batch_records);
        if !self.window_has_cursor_reuse(&batch_records, &liveness) {
            return self.execute_stateless_incremental_window(batches);
        }

        let total_started = Instant::now();
        self.active_window_liveness = Some(Arc::new(liveness));
        self.active_window_records = Some(Arc::new(batch_records));
        self.active_window_batch = 0;
        let result = (|| {
            let mut outputs = Vec::with_capacity(batches.len());
            let mut batch_ready_ns = Vec::with_capacity(batches.len());
            let mut ordered_delivery_ns = Vec::with_capacity(batches.len());
            let mut total_stats: Option<HierarchicalBatchStats> = None;
            for (batch_index, batch) in batches.iter().enumerate() {
                self.active_window_batch = batch_index;
                let (batch_outputs, part) = self.execute(batch)?;
                let ready = total_started.elapsed().as_nanos() as u64;
                outputs.push(batch_outputs);
                batch_ready_ns.push(ready);
                ordered_delivery_ns.push(ready);
                if let Some(total) = total_stats.as_mut() {
                    accumulate_batch_stats(total, &part);
                    total.encoded_cache_resident_bytes = part.encoded_cache_resident_bytes;
                    total.resident_read_ahead_bytes = part.resident_read_ahead_bytes;
                    total.resident_cursor_entries = part.resident_cursor_entries;
                } else {
                    total_stats = Some(part);
                }
            }
            let mut stats = total_stats.expect("non-empty window produces statistics");
            stats.total_ns = total_started.elapsed().as_nanos() as u64;
            stats.mode = "window_liveness_adaptive";
            Ok(HierarchicalIncrementalWindow {
                batches: outputs,
                batch_ready_ns,
                ordered_delivery_ns,
                stats,
            })
        })();
        self.active_window_liveness = None;
        self.active_window_records = None;
        self.active_window_batch = 0;
        result
    }

    /// Resolve a future request window once, but release its original batches
    /// as soon as their range and decode dependencies complete.
    ///
    /// The range list is ordered by the earliest logical batch that consumes
    /// it. The S3 backend preserves bounded global concurrency while yielding
    /// ranges in completion order, so later-batch I/O can overlap without
    /// forcing the first batch to wait for the complete lookahead window.
    fn execute_stateless_incremental_window(
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
        let plan = self.resolve_plan(
            &target_records,
            HierarchicalAction::Adaptive,
            &HashSet::new(),
        )?;
        let plan_ns = plan_started.elapsed().as_nanos() as u64;
        let ResolvedPlan {
            mut ranges,
            selected_ids,
            useful_bytes,
            predicted_total_ns,
            candidate_count,
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
        let decode_budget = &self.decode_budget;
        let decoder_threads = self.decoder_threads;
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
                        let decoded_jobs = Self::decode_incremental_jobs(
                            decoder_slots,
                            jobs,
                            &total_started,
                            decode_budget,
                            decoder_threads,
                        )?;
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
        let range_buckets = range_size_buckets(&ranges);
        let mut result = HierarchicalIncrementalWindow {
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
                decoded_access_units: submitted_access_units,
                planner_candidate_count: candidate_count,
                range_le_4k: range_buckets[0],
                range_4k_to_16k: range_buckets[1],
                range_16k_to_64k: range_buckets[2],
                range_64k_to_256k: range_buckets[3],
                range_gt_256k: range_buckets[4],
                decode_groups,
                encoded_cache_hits: 0,
                encoded_cache_misses: 0,
                encoded_cache_resident_bytes: self.encoded_cache.resident_bytes() as u64,
                resident_cursor_hits: 0,
                resident_cursor_misses: 0,
                resident_read_ahead_bytes: self.resident_read_ahead_bytes as u64,
                resident_encoded_budget_bytes: self.resident_encoded_budget_bytes as u64,
                resident_cursor_entries: self.resident_cursors.len(),
                resident_cursor_capacity: self.resident_cursor_capacity,
                resident_cursor_candidates: self.last_cursor_candidate_count,
                resident_cursor_selected: self.last_cursor_selected_count,
                resident_cursor_pinned_entries: self.resident_state_policy.residency_counts().0,
                resident_cursor_probationary_entries: self
                    .resident_state_policy
                    .residency_counts()
                    .1,
                cursor_policy_ns: self.last_cursor_policy_ns,
                decoder_state_resets: 0,
                plan_ns,
                fetch_ns,
                assemble_ns,
                decode_ns,
                total_ns,
                predicted_total_ns,
                runtime_feedback_io_observations: 0,
                runtime_feedback_decode_observations: 0,
                runtime_feedback_rejected_observations: 0,
                runtime_feedback_active: false,
                runtime_feedback_io_ape_ppm: 0,
                runtime_feedback_decode_ape_ppm: 0,
                runtime_feedback_io_tail_multiplier_ppm: 0,
                runtime_feedback_decode_tail_multiplier_ppm: 0,
                mode,
            },
        };
        self.observe_runtime_feedback(&mut result.stats);
        Ok(result)
    }

    fn resolve_plan(
        &self,
        target_records: &[HierarchicalRecordMeta],
        action: HierarchicalAction,
        resident_record_ids: &HashSet<u64>,
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
        let mut prefix_ids = HashSet::new();
        for (video_id, gop_id) in &touched {
            let target_max_ordinal = target_records
                .iter()
                .filter(|record| record.video_id == *video_id && record.gop_id == *gop_id)
                .map(|record| record.decode_ordinal)
                .max()
                .ok_or_else(|| format!("missing target for GOP ({video_id}, {gop_id})"))?;
            for record_id in self
                .catalog
                .records_for_gop(video_id, *gop_id)
                .ok_or_else(|| format!("missing GOP region ({video_id}, {gop_id})"))?
            {
                let record = self.catalog.record(*record_id).unwrap();
                if record.decode_ordinal <= target_max_ordinal {
                    prefix_ids.insert(*record_id);
                }
            }
        }
        let prefix_records = prefix_ids
            .iter()
            .map(|record_id| {
                let record = self.catalog.record(*record_id).unwrap();
                RecordRange {
                    record_id: *record_id,
                    offset: record.offset,
                    length: record.length,
                }
            })
            .collect::<Vec<_>>();
        // Scanner-style prefix access is one contiguous read per touched GOP,
        // from the keyframe/AU start through the furthest required decode
        // ordinal.  The selected IDs remain only the prefix AUs, so transfer
        // gaps never become decoder submissions.
        let mut prefix_grouped = HashMap::<(String, u64), Vec<&HierarchicalRecordMeta>>::new();
        for record_id in &prefix_ids {
            let record = self.catalog.record(*record_id).unwrap();
            prefix_grouped
                .entry((record.video_id.clone(), record.gop_id))
                .or_default()
                .push(record);
        }
        let prefix_spans = prefix_grouped
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

        let (range_plans, selected_ids, useful_bytes, predicted_total_ns, candidate_count, mode) =
            match action {
                HierarchicalAction::Adaptive => {
                    let decision = self.layout.choose_plan_with_resident(
                        &target_records
                            .iter()
                            .map(|record| record.record_id)
                            .collect::<Vec<_>>(),
                        resident_record_ids,
                        self.max_merge_gap_bytes,
                        self.max_range_bytes,
                        &self.model,
                    )?;
                    let selected = decision.selected;
                    let candidate_count = decision.alternatives.len();
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
                        candidate_count,
                        mode,
                    )
                }
                HierarchicalAction::CalibratedClosure => {
                    let decision = self.layout.choose_plan(
                        &target_records
                            .iter()
                            .map(|record| record.record_id)
                            .collect::<Vec<_>>(),
                        self.max_merge_gap_bytes,
                        self.max_range_bytes,
                        &self.model,
                    )?;
                    let minimum = decision
                        .alternatives
                        .iter()
                        .filter(|candidate| candidate.mode == HierarchicalReadMode::SparseClosure)
                        .map(|candidate| candidate.total_ns)
                        .fold(f64::INFINITY, f64::min);
                    let candidate_count = decision
                        .alternatives
                        .iter()
                        .filter(|candidate| candidate.mode == HierarchicalReadMode::SparseClosure)
                        .count();
                    let selected = decision
                        .alternatives
                        .iter()
                        .filter(|candidate| {
                            candidate.mode == HierarchicalReadMode::SparseClosure
                                && candidate.total_ns <= minimum + self.model.selection_tolerance_ns
                        })
                        .min_by(|left, right| {
                            left.ranges
                                .len()
                                .cmp(&right.ranges.len())
                                .then_with(|| left.fetched_bytes.cmp(&right.fetched_bytes))
                                .then_with(|| left.total_ns.total_cmp(&right.total_ns))
                        })
                        .ok_or_else(|| {
                            "calibrated planner has no sparse-closure candidate".to_string()
                        })?;
                    (
                        selected.ranges.clone(),
                        closure_ids,
                        selected.useful_bytes,
                        selected.total_ns,
                        candidate_count,
                        "calibrated_closure",
                    )
                }
                HierarchicalAction::KeyframePrefix => (
                    planner::plan_byte_ranges(&prefix_spans, None, self.max_range_bytes)?,
                    prefix_ids,
                    planner::unique_covered_bytes(&prefix_records)?,
                    -1.0,
                    1,
                    "keyframe_prefix",
                ),
                HierarchicalAction::ExactClosure => (
                    planner::plan_byte_ranges(&closure_records, None, self.max_range_bytes)?,
                    closure_ids,
                    closure_useful_bytes,
                    -1.0,
                    1,
                    "exact_closure",
                ),
                HierarchicalAction::FixedGapClosure(gap) => (
                    planner::plan_byte_ranges(&closure_records, Some(gap), self.max_range_bytes)?,
                    closure_ids,
                    closure_useful_bytes,
                    -1.0,
                    1,
                    "fixed_gap_closure",
                ),
                HierarchicalAction::RegionSelective => (
                    planner::plan_byte_ranges(&region_records, None, self.max_range_bytes)?,
                    closure_ids,
                    closure_useful_bytes,
                    -1.0,
                    1,
                    "region_selective_decode",
                ),
                HierarchicalAction::RegionAll => (
                    planner::plan_byte_ranges(&region_records, None, self.max_range_bytes)?,
                    region_ids,
                    planner::unique_covered_bytes(&region_records)?,
                    -1.0,
                    1,
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
            candidate_count,
            mode,
        })
    }

    pub fn execute(
        &mut self,
        targets: &[LogicalTarget],
    ) -> Result<(Vec<HierarchicalOutput>, HierarchicalBatchStats), String> {
        self.refresh_runtime_feedback_model();
        let (outputs, mut stats) =
            self.execute_action_with_context(targets, HierarchicalAction::Adaptive, None, true)?;
        self.observe_runtime_feedback(&mut stats);
        Ok((outputs, stats))
    }

    pub(crate) fn execute_with_cursor_threads(
        &mut self,
        targets: &[LogicalTarget],
        cursor_decoder_threads_hint: Option<usize>,
    ) -> Result<(Vec<HierarchicalOutput>, HierarchicalBatchStats), String> {
        self.refresh_runtime_feedback_model();
        let (outputs, mut stats) = self.execute_action_with_context(
            targets,
            HierarchicalAction::Adaptive,
            cursor_decoder_threads_hint,
            true,
        )?;
        self.observe_runtime_feedback(&mut stats);
        Ok((outputs, stats))
    }

    pub(crate) fn execute_action(
        &mut self,
        targets: &[LogicalTarget],
        action: HierarchicalAction,
    ) -> Result<(Vec<HierarchicalOutput>, HierarchicalBatchStats), String> {
        self.execute_action_with_context(targets, action, None, true)
    }

    fn execute_action_with_context(
        &mut self,
        targets: &[LogicalTarget],
        action: HierarchicalAction,
        cursor_decoder_threads_hint: Option<usize>,
        allow_resident_cursor: bool,
    ) -> Result<(Vec<HierarchicalOutput>, HierarchicalBatchStats), String> {
        if targets.is_empty() {
            return Err("hierarchical batch is empty".to_string());
        }
        let total_started = Instant::now();
        let cursor_decoder_threads_hint = cursor_decoder_threads_hint
            .unwrap_or(self.cursor_decoder_threads)
            .max(1)
            .min(self.cursor_decoder_threads);
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
        if allow_resident_cursor && matches!(action, HierarchicalAction::Adaptive) {
            let (liveness, window_records, batch_index) =
                if let Some(liveness) = &self.active_window_liveness {
                    (
                        Arc::clone(liveness),
                        self.active_window_records
                            .as_ref()
                            .map(Arc::clone)
                            .expect("active window liveness has matching records"),
                        self.active_window_batch,
                    )
                } else {
                    (
                        Arc::new(Self::window_liveness(std::slice::from_ref(&target_records))),
                        Arc::new(vec![target_records.clone()]),
                        0,
                    )
                };
            let resident_cursor_keys = self.resident_cursor_keys_for_batch(
                &target_records,
                window_records.as_ref(),
                liveness.as_ref(),
                batch_index,
            );
            if let Some(key) = self.cursor_compatible_gop_key(&target_records) {
                if resident_cursor_keys.contains(&key) {
                    return self.execute_resident_cursor(
                        targets,
                        &target_records,
                        key,
                        total_started,
                        cursor_decoder_threads_hint,
                    );
                }
            }
            if let Some(result) = self.execute_resident_state(
                targets,
                &target_records,
                &resident_cursor_keys,
                total_started,
                cursor_decoder_threads_hint,
            )? {
                return Ok(result);
            }
        }
        let target_record_ids = target_records
            .iter()
            .map(|record| record.record_id)
            .collect::<Vec<_>>();

        let plan_started = Instant::now();
        let resident_record_ids = if matches!(action, HierarchicalAction::Adaptive) {
            self.resident_record_ids()
        } else {
            HashSet::new()
        };
        let plan = self.resolve_plan(&target_records, action, &resident_record_ids)?;
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
        let mut encoded_cache_hits = 0usize;
        let mut encoded_cache_misses = 0usize;
        if matches!(action, HierarchicalAction::Adaptive) {
            for record_id in selected_ids
                .iter()
                .filter(|record_id| resident_record_ids.contains(record_id))
            {
                let encoded = self.cached_encoded_record(*record_id).ok_or_else(|| {
                    format!("resident record {record_id} disappeared after planning snapshot")
                })?;
                encoded_records.insert(*record_id, encoded);
                encoded_cache_hits += 1;
            }
        }
        for record_id in &selected_ids {
            if encoded_records.contains_key(record_id) {
                continue;
            }
            if matches!(action, HierarchicalAction::Adaptive) {
                encoded_cache_misses += 1;
            }
            let meta = self
                .catalog
                .record(*record_id)
                .ok_or_else(|| format!("missing selected record {record_id}"))?;
            let sample = Self::extract_record(meta, &ranges, &buffers)?;
            let encoded = mp4_sample_to_annex_b(&sample, meta.nal_length_size)?;
            if matches!(action, HierarchicalAction::Adaptive) {
                self.encoded_cache.put(*record_id, encoded.clone());
            }
            encoded_records.insert(*record_id, encoded);
        }
        let assemble_ns = assemble_started.elapsed().as_nanos() as u64;

        let _decode_permit = self.decode_budget.acquire(self.decoder_threads)?;
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
        let range_buckets = range_size_buckets(&ranges);
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
                decoded_access_units: selected_ids.len(),
                planner_candidate_count: plan.candidate_count,
                range_le_4k: range_buckets[0],
                range_4k_to_16k: range_buckets[1],
                range_16k_to_64k: range_buckets[2],
                range_64k_to_256k: range_buckets[3],
                range_gt_256k: range_buckets[4],
                decode_groups: decoded
                    .keys()
                    .filter_map(|record_id| self.catalog.record(*record_id))
                    .map(|record| (record.video_id.as_str(), record.gop_id))
                    .collect::<HashSet<_>>()
                    .len(),
                encoded_cache_hits,
                encoded_cache_misses,
                encoded_cache_resident_bytes: self.encoded_cache.resident_bytes() as u64,
                resident_cursor_hits: 0,
                resident_cursor_misses: 0,
                resident_read_ahead_bytes: self.resident_read_ahead_bytes as u64,
                resident_encoded_budget_bytes: self.resident_encoded_budget_bytes as u64,
                resident_cursor_entries: self.resident_cursors.len(),
                resident_cursor_capacity: self.resident_cursor_capacity,
                resident_cursor_candidates: self.last_cursor_candidate_count,
                resident_cursor_selected: self.last_cursor_selected_count,
                resident_cursor_pinned_entries: self.resident_state_policy.residency_counts().0,
                resident_cursor_probationary_entries: self
                    .resident_state_policy
                    .residency_counts()
                    .1,
                cursor_policy_ns: self.last_cursor_policy_ns,
                decoder_state_resets: 0,
                plan_ns,
                fetch_ns,
                assemble_ns,
                decode_ns,
                total_ns: total_started.elapsed().as_nanos() as u64,
                predicted_total_ns: plan.predicted_total_ns,
                runtime_feedback_io_observations: 0,
                runtime_feedback_decode_observations: 0,
                runtime_feedback_rejected_observations: 0,
                runtime_feedback_active: false,
                runtime_feedback_io_ape_ppm: 0,
                runtime_feedback_decode_ape_ppm: 0,
                runtime_feedback_io_tail_multiplier_ppm: 0,
                runtime_feedback_decode_tail_multiplier_ppm: 0,
                mode: plan.mode,
            },
        ))
    }
}
