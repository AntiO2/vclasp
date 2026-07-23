use std::collections::{HashMap, VecDeque};

use crate::hierarchical_layout::{HierarchicalCostModel, HierarchicalLayoutIndex};

#[derive(Debug, Clone)]
pub struct DependencySamplerPlan {
    pub batches: Vec<Vec<u64>>,
    pub predicted_total_ns: f64,
    pub max_displacement: usize,
    pub mean_absolute_displacement: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledDependencySample {
    pub logical_id: u64,
    pub class_id: i64,
    pub target_ids: Vec<u64>,
}

/// Reorder a seeded epoch permutation within a bounded sample window.
///
/// The first sample of every batch retains FIFO priority. Remaining positions
/// minimize the same dependency-closure cost used by the runtime planner. A
/// deadline prevents any sample from moving later by more than the configured
/// lookahead window. Every input occurrence is emitted exactly once.
#[allow(clippy::too_many_arguments)]
pub fn plan_bounded_dependency_batches(
    layout: &HierarchicalLayoutIndex,
    base_order: &[u64],
    batch_size: usize,
    lookahead_samples: usize,
    max_merge_gap_bytes: Option<u64>,
    max_range_bytes: Option<u64>,
    model: &HierarchicalCostModel,
) -> Result<DependencySamplerPlan, String> {
    if base_order.is_empty() {
        return Err("dependency sampler requires a non-empty base order".to_string());
    }
    if batch_size == 0 {
        return Err("dependency sampler batch size must be positive".to_string());
    }
    if lookahead_samples == 0 {
        return Err("dependency sampler lookahead must be positive".to_string());
    }

    let mut next_input = 0usize;
    let mut pending = VecDeque::<(usize, u64)>::new();
    let fill_pending = |pending: &mut VecDeque<(usize, u64)>, next_input: &mut usize| {
        while pending.len() < lookahead_samples && *next_input < base_order.len() {
            pending.push_back((*next_input, base_order[*next_input]));
            *next_input += 1;
        }
    };
    fill_pending(&mut pending, &mut next_input);

    let mut batches = Vec::with_capacity(base_order.len().div_ceil(batch_size));
    let mut output_rank = HashMap::<usize, usize>::with_capacity(base_order.len());
    let mut output_position = 0usize;
    let mut predicted_total_ns = 0.0;

    while !pending.is_empty() {
        let target_batch_len = batch_size.min(base_order.len() - output_position);
        let mut batch = Vec::with_capacity(target_batch_len);
        while batch.len() < target_batch_len {
            let force_fifo = batch.is_empty()
                || pending.front().is_some_and(|(original_rank, _)| {
                    original_rank.saturating_add(lookahead_samples) <= output_position
                });
            let selected_index = if force_fifo || pending.len() == 1 {
                0
            } else {
                let mut best: Option<(usize, f64, usize)> = None;
                for (index, (original_rank, sample_id)) in pending.iter().enumerate() {
                    let mut candidate_batch = batch.clone();
                    candidate_batch.push(*sample_id);
                    let estimate = layout.choose_plan(
                        &candidate_batch,
                        max_merge_gap_bytes,
                        max_range_bytes,
                        model,
                    )?;
                    let candidate = (index, estimate.selected.total_ns, *original_rank);
                    if best.as_ref().is_none_or(|current| {
                        candidate.1.total_cmp(&current.1).is_lt()
                            || (candidate.1 == current.1 && candidate.2 < current.2)
                    }) {
                        best = Some(candidate);
                    }
                }
                best.expect("non-empty pending sampler window").0
            };
            let (original_rank, sample_id) = pending
                .remove(selected_index)
                .expect("selected dependency sampler item exists");
            output_rank.insert(original_rank, output_position);
            output_position += 1;
            batch.push(sample_id);
            fill_pending(&mut pending, &mut next_input);
        }
        predicted_total_ns += layout
            .choose_plan(&batch, max_merge_gap_bytes, max_range_bytes, model)?
            .selected
            .total_ns;
        batches.push(batch);
    }

    let mut displacement_sum = 0usize;
    let mut max_displacement = 0usize;
    for original_rank in 0..base_order.len() {
        let emitted_rank = output_rank[&original_rank];
        let displacement = emitted_rank.abs_diff(original_rank);
        displacement_sum += displacement;
        max_displacement = max_displacement.max(displacement);
    }
    Ok(DependencySamplerPlan {
        batches,
        predicted_total_ns,
        max_displacement,
        mean_absolute_displacement: displacement_sum as f64 / base_order.len() as f64,
    })
}

/// Preserve the exact label sequence of a seeded RandomSampler while choosing
/// among same-label samples in a bounded number of loader-visible batches.
///
/// A logical training sample may contain multiple video targets (for example,
/// the frames of one clip). Physical cost is evaluated over the union of all
/// target closures in the candidate training batch. Every visible window is
/// drained before the next one is admitted, so sample displacement is bounded
/// by the window rather than by an epoch-wide per-class queue.
#[allow(clippy::too_many_arguments)]
pub fn plan_label_preserving_dependency_batches(
    layout: &HierarchicalLayoutIndex,
    base_order: &[LabeledDependencySample],
    batch_size: usize,
    lookahead_batches: usize,
    max_merge_gap_bytes: Option<u64>,
    max_range_bytes: Option<u64>,
    model: &HierarchicalCostModel,
) -> Result<DependencySamplerPlan, String> {
    if base_order.is_empty() {
        return Err("label-preserving sampler requires a non-empty base order".to_string());
    }
    if batch_size == 0 || lookahead_batches == 0 {
        return Err("batch size and lookahead batches must be positive".to_string());
    }
    if base_order.iter().any(|sample| sample.target_ids.is_empty()) {
        return Err("every logical sample must contain at least one target".to_string());
    }

    let mut batches = Vec::with_capacity(base_order.len().div_ceil(batch_size));
    let mut output_rank = HashMap::<usize, usize>::with_capacity(base_order.len());
    let mut output_position = 0usize;
    let mut predicted_total_ns = 0.0;
    let window_samples = batch_size
        .checked_mul(lookahead_batches)
        .ok_or("label-preserving lookahead window overflow")?;
    for (window_index, window) in base_order.chunks(window_samples).enumerate() {
        let window_start = window_index * window_samples;
        let window_end = window_start + window.len();
        let mut queues = HashMap::<i64, VecDeque<(usize, &LabeledDependencySample)>>::new();
        for (local_rank, sample) in window.iter().enumerate() {
            queues
                .entry(sample.class_id)
                .or_default()
                .push_back((window_start + local_rank, sample));
        }
        while output_position < window_end {
            let target_batch_len = batch_size.min(window_end - output_position);
            let mut batch_logical_ids = Vec::with_capacity(target_batch_len);
            let mut batch_target_ids = Vec::new();
            while batch_logical_ids.len() < target_batch_len {
                let required_class = base_order[output_position].class_id;
                let queue = queues
                    .get_mut(&required_class)
                    .expect("visible-window label queue exists");
                let selected_index =
                    if lookahead_batches == 1 || batch_logical_ids.is_empty() || queue.len() == 1 {
                        0
                    } else {
                        let mut best: Option<(usize, f64, usize)> = None;
                        for (index, (original_rank, sample)) in queue.iter().enumerate() {
                            let mut candidate_targets = batch_target_ids.clone();
                            candidate_targets.extend_from_slice(&sample.target_ids);
                            let estimate = layout.choose_plan(
                                &candidate_targets,
                                max_merge_gap_bytes,
                                max_range_bytes,
                                model,
                            )?;
                            let candidate = (index, estimate.selected.total_ns, *original_rank);
                            if best.as_ref().is_none_or(|current| {
                                candidate.1.total_cmp(&current.1).is_lt()
                                    || (candidate.1 == current.1 && candidate.2 < current.2)
                            }) {
                                best = Some(candidate);
                            }
                        }
                        best.expect("label queue has a candidate").0
                    };
                let (original_rank, sample) = queue
                    .remove(selected_index)
                    .expect("selected labeled sampler item exists");
                output_rank.insert(original_rank, output_position);
                output_position += 1;
                batch_logical_ids.push(sample.logical_id);
                batch_target_ids.extend_from_slice(&sample.target_ids);
            }
            predicted_total_ns += layout
                .choose_plan(
                    &batch_target_ids,
                    max_merge_gap_bytes,
                    max_range_bytes,
                    model,
                )?
                .selected
                .total_ns;
            batches.push(batch_logical_ids);
        }
    }

    let mut displacement_sum = 0usize;
    let mut max_displacement = 0usize;
    for original_rank in 0..base_order.len() {
        let displacement = output_rank[&original_rank].abs_diff(original_rank);
        displacement_sum += displacement;
        max_displacement = max_displacement.max(displacement);
    }
    if max_displacement >= window_samples {
        return Err(format!(
            "label-preserving displacement {max_displacement} exceeds visible window {window_samples}"
        ));
    }
    Ok(DependencySamplerPlan {
        batches,
        predicted_total_ns,
        max_displacement,
        mean_absolute_displacement: displacement_sum as f64 / base_order.len() as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchical_layout::{AccessUnitRecord, GopRegion, TargetClosure};

    fn fixture() -> (HierarchicalLayoutIndex, HierarchicalCostModel) {
        let records = (0..8)
            .map(|record_id| AccessUnitRecord {
                record_id,
                video_id: record_id / 4,
                gop_id: record_id / 4,
                offset: record_id * 100,
                length: 20,
                decode_ordinal: (record_id % 4) as usize,
            })
            .collect::<Vec<_>>();
        let closures = vec![
            TargetClosure {
                sample_id: 0,
                video_id: 0,
                gop_id: 0,
                target_record_id: 0,
                record_ids: vec![0],
            },
            TargetClosure {
                sample_id: 1,
                video_id: 0,
                gop_id: 0,
                target_record_id: 1,
                record_ids: vec![0, 1],
            },
            TargetClosure {
                sample_id: 2,
                video_id: 0,
                gop_id: 0,
                target_record_id: 2,
                record_ids: vec![0, 2],
            },
            TargetClosure {
                sample_id: 3,
                video_id: 0,
                gop_id: 0,
                target_record_id: 3,
                record_ids: vec![0, 3],
            },
            TargetClosure {
                sample_id: 4,
                video_id: 1,
                gop_id: 1,
                target_record_id: 4,
                record_ids: vec![4],
            },
            TargetClosure {
                sample_id: 5,
                video_id: 1,
                gop_id: 1,
                target_record_id: 5,
                record_ids: vec![4, 5],
            },
            TargetClosure {
                sample_id: 6,
                video_id: 1,
                gop_id: 1,
                target_record_id: 6,
                record_ids: vec![4, 6],
            },
            TargetClosure {
                sample_id: 7,
                video_id: 1,
                gop_id: 1,
                target_record_id: 7,
                record_ids: vec![4, 7],
            },
        ];
        let regions = vec![
            GopRegion {
                video_id: 0,
                gop_id: 0,
                offset: 0,
                length: 320,
                record_ids: vec![0, 1, 2, 3],
            },
            GopRegion {
                video_id: 1,
                gop_id: 1,
                offset: 400,
                length: 320,
                record_ids: vec![4, 5, 6, 7],
            },
        ];
        let layout = HierarchicalLayoutIndex::new(records, closures, regions).unwrap();
        let model = HierarchicalCostModel {
            request_latency_ns: 1_000_000.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 1,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 0.0,
            decode_access_unit_ns: 1.0,
            fetch_decode_overlap: 0.0,
        };
        (layout, model)
    }

    #[test]
    fn lookahead_one_is_exact_base_order() {
        let (layout, model) = fixture();
        let base = vec![0, 4, 1, 5, 2, 6, 3, 7];
        let plan =
            plan_bounded_dependency_batches(&layout, &base, 2, 1, None, None, &model).unwrap();
        assert_eq!(plan.batches.concat(), base);
        assert_eq!(plan.max_displacement, 0);
    }

    #[test]
    fn bounded_plan_preserves_occurrences_and_groups_shared_dependencies() {
        let (layout, model) = fixture();
        let base = vec![0, 4, 1, 5, 2, 6, 3, 7];
        let plan =
            plan_bounded_dependency_batches(&layout, &base, 2, 4, None, None, &model).unwrap();
        let mut actual = plan.batches.concat();
        let mut expected = base.clone();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected);
        assert!(plan.max_displacement < 6);
        assert_eq!(plan.batches[0], vec![0, 1]);
        assert!(plan.max_displacement <= 4);
    }

    #[test]
    fn duplicate_occurrences_are_not_dropped() {
        let (layout, model) = fixture();
        let base = vec![0, 4, 0, 4, 1, 5];
        let plan =
            plan_bounded_dependency_batches(&layout, &base, 3, 4, None, None, &model).unwrap();
        let mut actual = plan.batches.concat();
        let mut expected = base;
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn label_preserving_sampler_keeps_every_label_position() {
        let (layout, model) = fixture();
        let base = vec![
            LabeledDependencySample {
                logical_id: 10,
                class_id: 0,
                target_ids: vec![0],
            },
            LabeledDependencySample {
                logical_id: 11,
                class_id: 1,
                target_ids: vec![4],
            },
            LabeledDependencySample {
                logical_id: 12,
                class_id: 0,
                target_ids: vec![1],
            },
            LabeledDependencySample {
                logical_id: 13,
                class_id: 1,
                target_ids: vec![5],
            },
            LabeledDependencySample {
                logical_id: 14,
                class_id: 0,
                target_ids: vec![2],
            },
            LabeledDependencySample {
                logical_id: 15,
                class_id: 1,
                target_ids: vec![6],
            },
        ];
        let by_id = base
            .iter()
            .map(|sample| (sample.logical_id, sample.class_id))
            .collect::<HashMap<_, _>>();
        let plan =
            plan_label_preserving_dependency_batches(&layout, &base, 2, 3, None, None, &model)
                .unwrap();
        let output = plan.batches.concat();
        assert_eq!(
            output.iter().map(|id| by_id[id]).collect::<Vec<_>>(),
            base.iter()
                .map(|sample| sample.class_id)
                .collect::<Vec<_>>()
        );
        let mut actual = output;
        let mut expected = base
            .iter()
            .map(|sample| sample.logical_id)
            .collect::<Vec<_>>();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn label_preserving_lookahead_one_is_exact_order() {
        let (layout, model) = fixture();
        let base = (0..8)
            .map(|logical_id| LabeledDependencySample {
                logical_id: logical_id + 100,
                class_id: (logical_id % 2) as i64,
                target_ids: vec![logical_id],
            })
            .collect::<Vec<_>>();
        let plan =
            plan_label_preserving_dependency_batches(&layout, &base, 4, 1, None, None, &model)
                .unwrap();
        assert_eq!(
            plan.batches.concat(),
            base.iter()
                .map(|sample| sample.logical_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(plan.max_displacement, 0);
    }

    #[test]
    fn label_preserving_sampler_never_crosses_visible_windows() {
        let (layout, model) = fixture();
        let base = (0..12)
            .map(|rank| LabeledDependencySample {
                logical_id: rank + 200,
                class_id: (rank % 2) as i64,
                target_ids: vec![rank % 8],
            })
            .collect::<Vec<_>>();
        let plan =
            plan_label_preserving_dependency_batches(&layout, &base, 2, 2, None, None, &model)
                .unwrap();
        let output = plan.batches.concat();
        for (expected, actual) in base.chunks(4).zip(output.chunks(4)) {
            let mut expected_ids = expected
                .iter()
                .map(|sample| sample.logical_id)
                .collect::<Vec<_>>();
            let mut actual_ids = actual.to_vec();
            expected_ids.sort_unstable();
            actual_ids.sort_unstable();
            assert_eq!(actual_ids, expected_ids);
        }
        assert!(plan.max_displacement < 4);
    }
}
