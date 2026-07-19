use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::backend::StorageBackend;
use crate::decoder;
use crate::normalized_scheduler::{DecodeSchedule, NormalizedBatchExecutor, NormalizedDescriptor};
use crate::pair_scheduler::{ClosedRecordBatchExecutor, ClosedRecordDescriptor};
use crate::representation::{
    BatchStats, DependencyClosure, DependencyKind, LogicalRequest, OuterPlanner, PhysicalRecord,
    Representation, SampleRepresentations,
};

#[derive(Debug)]
pub struct PortfolioFrame {
    pub sample_id: u64,
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct BudgetedPairBatchExecutor {
    outer_planner: OuterPlanner,
    materialized: HashSet<u64>,
    normalized: NormalizedBatchExecutor,
    pair: ClosedRecordBatchExecutor,
    materialization_budget_bytes: u64,
    materialization_used_bytes: u64,
    global_io_concurrency: usize,
    global_decoder_slots: usize,
}

impl BudgetedPairBatchExecutor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        normalized_descriptors: Vec<NormalizedDescriptor>,
        pair_descriptors: Vec<ClosedRecordDescriptor>,
        normalized_backend: Box<dyn StorageBackend + Send>,
        pair_backend: Box<dyn StorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        normalized_encoded_cache_bytes: usize,
        pair_encoded_cache_bytes: usize,
        normalized_decoded_cache_bytes: usize,
        pair_decoded_cache_bytes: usize,
        decode_concurrency: usize,
        decode_microbatch_targets: usize,
        normalized_decode_schedule: DecodeSchedule,
        global_io_concurrency: usize,
        materialization_budget_bytes: u64,
        materialization_used_bytes: u64,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        if materialization_used_bytes > materialization_budget_bytes {
            return Err(format!(
                "materialization exceeds budget: {materialization_used_bytes} > {materialization_budget_bytes}"
            ));
        }
        if global_io_concurrency == 0 {
            return Err("global I/O concurrency must be positive".to_string());
        }
        let normalized_by_id = normalized_descriptors
            .iter()
            .map(|descriptor| (descriptor.sample_id, descriptor.clone()))
            .collect::<HashMap<_, _>>();
        if normalized_by_id.len() != normalized_descriptors.len() {
            return Err("duplicate normalized descriptor".to_string());
        }
        let pair_by_id = pair_descriptors
            .iter()
            .map(|descriptor| (descriptor.sample_id, descriptor.clone()))
            .collect::<HashMap<_, _>>();
        if pair_by_id.len() != pair_descriptors.len() {
            return Err("duplicate materialized Pair descriptor".to_string());
        }
        let materialized = pair_by_id.keys().copied().collect::<HashSet<_>>();
        let descriptor_bytes = pair_descriptors
            .iter()
            .map(|value| value.length)
            .sum::<u64>();
        if descriptor_bytes != materialization_used_bytes {
            return Err(format!(
                "materialized descriptor bytes {descriptor_bytes} != reported used bytes {materialization_used_bytes}"
            ));
        }

        let mut samples = Vec::with_capacity(normalized_descriptors.len());
        for descriptor in &normalized_descriptors {
            let anchor_id = (1u64 << 63) | descriptor.anchor_group_id;
            let mut records = vec![PhysicalRecord {
                record_id: anchor_id,
                kind: DependencyKind::Anchor,
                offset: descriptor.anchor_offset,
                length: descriptor.anchor_length,
            }];
            if descriptor.target_ordinal > 0 {
                records.push(PhysicalRecord {
                    record_id: descriptor.sample_id,
                    kind: DependencyKind::Delta,
                    offset: descriptor.delta_offset,
                    length: descriptor.delta_length,
                });
            }
            let normalized = DependencyClosure {
                sample_id: descriptor.sample_id,
                video_id: descriptor.video_id,
                representation: Representation::Normalized,
                records,
                target_ordinal: descriptor.target_ordinal,
            };
            let pair = pair_by_id
                .get(&descriptor.sample_id)
                .map(|value| DependencyClosure {
                    sample_id: value.sample_id,
                    video_id: value.video_id,
                    representation: Representation::Pair,
                    records: vec![PhysicalRecord {
                        record_id: value.sample_id,
                        kind: DependencyKind::Pair,
                        offset: value.offset,
                        length: value.length,
                    }],
                    target_ordinal: value.target_ordinal,
                });
            samples.push(SampleRepresentations {
                sample_id: descriptor.sample_id,
                prefix: None,
                normalized: Some(normalized),
                pair,
            });
        }
        for sample_id in pair_by_id.keys() {
            if !normalized_by_id.contains_key(sample_id) {
                return Err(format!(
                    "materialized Pair sample {sample_id} has no Normalized fallback"
                ));
            }
        }

        let shared_decoder_slots = decoder::shared_decoder_slots(decode_concurrency);
        let global_decoder_slots = shared_decoder_slots.len();
        let normalized = NormalizedBatchExecutor::new_with_decode_schedule_and_slots(
            normalized_descriptors,
            normalized_backend,
            merge_threshold_bytes,
            max_range_bytes,
            normalized_encoded_cache_bytes,
            0,
            normalized_decoded_cache_bytes,
            decode_microbatch_targets,
            normalized_decode_schedule,
            shared_decoder_slots.clone(),
            width,
            height,
        )?;
        let pair = ClosedRecordBatchExecutor::new_with_decoder_slots(
            pair_descriptors,
            Representation::Pair,
            pair_backend,
            merge_threshold_bytes,
            max_range_bytes,
            pair_encoded_cache_bytes,
            pair_decoded_cache_bytes,
            shared_decoder_slots,
            false,
            0,
            width,
            height,
        )?;
        Ok(Self {
            outer_planner: OuterPlanner::new(samples)?,
            materialized,
            normalized,
            pair,
            materialization_budget_bytes,
            materialization_used_bytes,
            global_io_concurrency,
            global_decoder_slots,
        })
    }

    pub fn execute(
        &mut self,
        batch: &[u64],
        completion_driven: bool,
    ) -> Result<(Vec<PortfolioFrame>, BatchStats), String> {
        let total_started = Instant::now();
        let requests = batch
            .iter()
            .map(|sample_id| LogicalRequest {
                sample_id: *sample_id,
            })
            .collect::<Vec<_>>();
        let unique_videos = self
            .outer_planner
            .resolve_closures(&requests, Representation::Normalized)?
            .iter()
            .map(|closure| closure.video_id)
            .collect::<HashSet<_>>()
            .len();
        let route = self.outer_planner.route_materialized_pair(
            &requests,
            &self.materialized,
            Representation::Normalized,
        )?;
        let pair_ids = route
            .pair_requests
            .iter()
            .map(|request| request.sample_id)
            .collect::<Vec<_>>();
        let fallback_ids = route
            .fallback_requests
            .iter()
            .map(|request| request.sample_id)
            .collect::<Vec<_>>();

        let (pair_frames, pair_stats, fallback_frames, fallback_stats, branch_overlap_ns) =
            if !pair_ids.is_empty() && !fallback_ids.is_empty() {
                let timeline = Instant::now();
                let pair_executor = &mut self.pair;
                let fallback_executor = &mut self.normalized;
                let (pair_result, fallback_result) = std::thread::scope(|scope| {
                    let pair_handle = scope.spawn(|| {
                        let started = timeline.elapsed().as_nanos() as u64;
                        let result = pair_executor.execute(&pair_ids);
                        let completed = timeline.elapsed().as_nanos() as u64;
                        (result, started, completed)
                    });
                    let fallback_handle = scope.spawn(|| {
                        let started = timeline.elapsed().as_nanos() as u64;
                        let result = fallback_executor.execute(&fallback_ids, completion_driven);
                        let completed = timeline.elapsed().as_nanos() as u64;
                        (result, started, completed)
                    });
                    let pair_joined = pair_handle.join();
                    let fallback_joined = fallback_handle.join();
                    let pair =
                        pair_joined.map_err(|_| "Pair portfolio branch panicked".to_string())?;
                    let fallback = fallback_joined
                        .map_err(|_| "Normalized portfolio branch panicked".to_string())?;
                    Ok::<_, String>((pair, fallback))
                })?;
                let (pair_execution, pair_started, pair_completed) = pair_result;
                let (fallback_execution, fallback_started, fallback_completed) = fallback_result;
                let (pair_frames, pair_stats) = pair_execution?;
                let (fallback_frames, fallback_stats) = fallback_execution?;
                let overlap_started = pair_started.max(fallback_started);
                let overlap_completed = pair_completed.min(fallback_completed);
                (
                    pair_frames,
                    pair_stats,
                    fallback_frames,
                    fallback_stats,
                    overlap_completed.saturating_sub(overlap_started),
                )
            } else {
                let (pair_frames, pair_stats) = if pair_ids.is_empty() {
                    (Vec::new(), BatchStats::default())
                } else {
                    self.pair.execute(&pair_ids)?
                };
                let (fallback_frames, fallback_stats) = if fallback_ids.is_empty() {
                    (Vec::new(), BatchStats::default())
                } else {
                    self.normalized.execute(&fallback_ids, completion_driven)?
                };
                (pair_frames, pair_stats, fallback_frames, fallback_stats, 0)
            };

        let mut results = HashMap::with_capacity(pair_frames.len() + fallback_frames.len());
        for frame in pair_frames {
            results.insert(
                frame.sample_id,
                PortfolioFrame {
                    sample_id: frame.sample_id,
                    rgb: frame.rgb,
                    width: frame.width,
                    height: frame.height,
                },
            );
        }
        for frame in fallback_frames {
            results.insert(
                frame.sample_id,
                PortfolioFrame {
                    sample_id: frame.sample_id,
                    rgb: frame.rgb,
                    width: frame.width,
                    height: frame.height,
                },
            );
        }
        let reorder_started = Instant::now();
        let ordered = batch
            .iter()
            .map(|sample_id| {
                let frame = results
                    .get(sample_id)
                    .ok_or_else(|| format!("portfolio omitted sample {sample_id}"))?;
                Ok(PortfolioFrame {
                    sample_id: frame.sample_id,
                    rgb: frame.rgb.clone(),
                    width: frame.width,
                    height: frame.height,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let pair_branch = (
            pair_stats.total_ns,
            pair_stats.fetch_wall_ns,
            pair_stats.decode_ns,
        );
        let fallback_branch = (
            fallback_stats.total_ns,
            fallback_stats.fetch_wall_ns,
            fallback_stats.decode_ns,
        );
        let mut stats = merge_stats(pair_stats, fallback_stats);
        stats.logical_samples = batch.len();
        stats.unique_videos = unique_videos;
        stats.materialized_targets = route.materialized_unique;
        stats.fallback_targets = route.fallback_unique;
        stats.materialization_budget_bytes = self.materialization_budget_bytes;
        stats.materialization_used_bytes = self.materialization_used_bytes;
        stats.portfolio_parallel_branches =
            usize::from(!pair_ids.is_empty()) + usize::from(!fallback_ids.is_empty());
        stats.global_io_concurrency = self.global_io_concurrency;
        stats.global_decoder_slots = self.global_decoder_slots;
        stats.portfolio_branch_overlap_ns = branch_overlap_ns;
        stats.pair_branch_total_ns = pair_branch.0;
        stats.fallback_branch_total_ns = fallback_branch.0;
        stats.pair_branch_fetch_wall_ns = pair_branch.1;
        stats.fallback_branch_fetch_wall_ns = fallback_branch.1;
        stats.pair_branch_decode_ns = pair_branch.2;
        stats.fallback_branch_decode_ns = fallback_branch.2;
        stats.reorder_ns += reorder_started.elapsed().as_nanos() as u64;
        stats.total_ns = total_started.elapsed().as_nanos() as u64;
        Ok((ordered, stats))
    }
}

fn merge_stats(left: BatchStats, right: BatchStats) -> BatchStats {
    BatchStats {
        logical_samples: left.logical_samples + right.logical_samples,
        unique_targets: left.unique_targets + right.unique_targets,
        unique_videos: left.unique_videos + right.unique_videos,
        dependency_records: left.dependency_records + right.dependency_records,
        target_ordinal_sum: left.target_ordinal_sum + right.target_ordinal_sum,
        target_ordinal_max: left.target_ordinal_max.max(right.target_ordinal_max),
        planned_ranges: left.planned_ranges + right.planned_ranges,
        planned_useful_bytes: left.planned_useful_bytes + right.planned_useful_bytes,
        planned_fetched_bytes: left.planned_fetched_bytes + right.planned_fetched_bytes,
        unique_records: left.unique_records + right.unique_records,
        physical_ranges: left.physical_ranges + right.physical_ranges,
        client_requests: left.client_requests + right.client_requests,
        server_entries: left.server_entries + right.server_entries,
        useful_bytes: left.useful_bytes + right.useful_bytes,
        fetched_bytes: left.fetched_bytes + right.fetched_bytes,
        overfetch_bytes: left.overfetch_bytes + right.overfetch_bytes,
        decoded_targets: left.decoded_targets + right.decoded_targets,
        decoded_frames: left.decoded_frames + right.decoded_frames,
        decode_groups: left.decode_groups + right.decode_groups,
        anchor_decode_invocations: left.anchor_decode_invocations + right.anchor_decode_invocations,
        fused_decode_groups: left.fused_decode_groups + right.fused_decode_groups,
        repeated_decode_groups: left.repeated_decode_groups + right.repeated_decode_groups,
        encoded_cache_hits: left.encoded_cache_hits + right.encoded_cache_hits,
        encoded_cache_misses: left.encoded_cache_misses + right.encoded_cache_misses,
        anchor_cache_hits: left.anchor_cache_hits + right.anchor_cache_hits,
        anchor_cache_misses: left.anchor_cache_misses + right.anchor_cache_misses,
        delta_cache_hits: left.delta_cache_hits + right.delta_cache_hits,
        delta_cache_misses: left.delta_cache_misses + right.delta_cache_misses,
        decoded_cache_hits: left.decoded_cache_hits + right.decoded_cache_hits,
        decoded_cache_misses: left.decoded_cache_misses + right.decoded_cache_misses,
        encoded_cache_resident_bytes: left.encoded_cache_resident_bytes
            + right.encoded_cache_resident_bytes,
        anchor_cache_resident_bytes: left.anchor_cache_resident_bytes
            + right.anchor_cache_resident_bytes,
        delta_cache_resident_bytes: left.delta_cache_resident_bytes
            + right.delta_cache_resident_bytes,
        decoded_cache_resident_bytes: left.decoded_cache_resident_bytes
            + right.decoded_cache_resident_bytes,
        decoder_state_hits: left.decoder_state_hits + right.decoder_state_hits,
        decoder_state_misses: left.decoder_state_misses + right.decoder_state_misses,
        decoder_state_resets: left.decoder_state_resets + right.decoder_state_resets,
        // Both branches inspect the same shared decoder-slot set.
        decoder_state_resident: left
            .decoder_state_resident
            .max(right.decoder_state_resident),
        portfolio_parallel_branches: left.portfolio_parallel_branches
            + right.portfolio_parallel_branches,
        global_io_concurrency: left.global_io_concurrency.max(right.global_io_concurrency),
        global_decoder_slots: left.global_decoder_slots.max(right.global_decoder_slots),
        portfolio_branch_overlap_ns: left.portfolio_branch_overlap_ns
            + right.portfolio_branch_overlap_ns,
        pair_branch_total_ns: left.pair_branch_total_ns + right.pair_branch_total_ns,
        fallback_branch_total_ns: left.fallback_branch_total_ns + right.fallback_branch_total_ns,
        pair_branch_fetch_wall_ns: left.pair_branch_fetch_wall_ns + right.pair_branch_fetch_wall_ns,
        fallback_branch_fetch_wall_ns: left.fallback_branch_fetch_wall_ns
            + right.fallback_branch_fetch_wall_ns,
        pair_branch_decode_ns: left.pair_branch_decode_ns + right.pair_branch_decode_ns,
        fallback_branch_decode_ns: left.fallback_branch_decode_ns + right.fallback_branch_decode_ns,
        resolve_ns: left.resolve_ns.max(right.resolve_ns),
        cache_lookup_ns: left.cache_lookup_ns.max(right.cache_lookup_ns),
        plan_ns: left.plan_ns.max(right.plan_ns),
        fetch_wall_ns: left.fetch_wall_ns.max(right.fetch_wall_ns),
        fetch_service_ns_sum: left.fetch_service_ns_sum + right.fetch_service_ns_sum,
        range_queue_ns_sum: left.range_queue_ns_sum + right.range_queue_ns_sum,
        extract_ns: left.extract_ns.max(right.extract_ns),
        assemble_ns: left.assemble_ns.max(right.assemble_ns),
        decode_ns: left.decode_ns.max(right.decode_ns),
        rgb_convert_ns: left.rgb_convert_ns.max(right.rgb_convert_ns),
        fetch_decode_overlap_ns: left.fetch_decode_overlap_ns + right.fetch_decode_overlap_ns,
        reorder_ns: left.reorder_ns + right.reorder_ns,
        total_ns: left.total_ns.max(right.total_ns),
        time_to_first_ready_ns: match (left.time_to_first_ready_ns, right.time_to_first_ready_ns) {
            (0, value) | (value, 0) => value,
            (left, right) => left.min(right),
        },
        completion_selected: left.completion_selected || right.completion_selected,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::NoopBackend;
    use std::sync::Arc;

    #[test]
    fn portfolio_branches_share_decoder_slots() {
        let normalized = vec![
            NormalizedDescriptor {
                sample_id: 1,
                video_id: 7,
                anchor_group_id: 9,
                target_ordinal: 0,
                anchor_offset: 0,
                anchor_length: 10,
                delta_offset: 10,
                delta_length: 0,
            },
            NormalizedDescriptor {
                sample_id: 2,
                video_id: 7,
                anchor_group_id: 9,
                target_ordinal: 1,
                anchor_offset: 0,
                anchor_length: 10,
                delta_offset: 10,
                delta_length: 5,
            },
        ];
        let pair = vec![ClosedRecordDescriptor {
            sample_id: 1,
            video_id: 7,
            offset: 0,
            length: 10,
            target_ordinal: 0,
        }];
        let executor = BudgetedPairBatchExecutor::new(
            normalized,
            pair,
            Box::new(NoopBackend),
            Box::new(NoopBackend),
            None,
            None,
            0,
            0,
            0,
            0,
            2,
            1,
            DecodeSchedule::Repeated,
            2,
            10,
            10,
            320,
            240,
        )
        .expect("portfolio construction must succeed");

        assert!(Arc::ptr_eq(
            executor.normalized.decoder_slots_handle(),
            executor.pair.decoder_slots_handle(),
        ));
        assert_eq!(executor.global_decoder_slots, 2);
        assert_eq!(executor.global_io_concurrency, 2);
    }

    #[test]
    fn merged_stats_do_not_double_count_shared_decoder_residency() {
        let left = BatchStats {
            decoder_state_resident: 2,
            ..Default::default()
        };
        let right = BatchStats {
            decoder_state_resident: 2,
            ..Default::default()
        };

        assert_eq!(merge_stats(left, right).decoder_state_resident, 2);
    }
}
