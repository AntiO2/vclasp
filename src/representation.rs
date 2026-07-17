use std::collections::{HashMap, HashSet};

use crate::planner::{self, RangePlan, RecordRange};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Representation {
    Prefix,
    Normalized,
    Pair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyKind {
    Prefix,
    Anchor,
    Delta,
    Pair,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalRecord {
    pub record_id: u64,
    pub kind: DependencyKind,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyClosure {
    pub sample_id: u64,
    pub video_id: u64,
    pub representation: Representation,
    pub records: Vec<PhysicalRecord>,
    /// Zero-based frame ordinal emitted by decoding this closure.
    pub target_ordinal: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalRequest {
    pub sample_id: u64,
}

#[derive(Debug, Clone)]
pub struct SampleRepresentations {
    pub sample_id: u64,
    pub prefix: Option<DependencyClosure>,
    pub normalized: Option<DependencyClosure>,
    pub pair: Option<DependencyClosure>,
}

impl SampleRepresentations {
    fn get(&self, representation: Representation) -> Option<&DependencyClosure> {
        match representation {
            Representation::Prefix => self.prefix.as_ref(),
            Representation::Normalized => self.normalized.as_ref(),
            Representation::Pair => self.pair.as_ref(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedBatch {
    /// Original request order, including duplicates.
    pub requests: Vec<LogicalRequest>,
    /// Unique closures in first-request order.
    pub closures: Vec<DependencyClosure>,
    pub ranges: Vec<RangePlan>,
    /// Label-free features derived from the requested dependency closures and
    /// the resulting physical plan. A selector may consume these features but
    /// must not branch on a benchmark workload name.
    pub features: BatchFeatures,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchFeatures {
    pub logical_requests: usize,
    pub unique_targets: usize,
    pub unique_videos: usize,
    pub dependency_records: usize,
    pub unique_records: usize,
    pub useful_bytes: u64,
    pub physical_ranges: usize,
    pub fetched_bytes: u64,
    pub overfetch_bytes: u64,
    pub mean_range_bytes: u64,
    pub max_range_bytes: u64,
    pub address_span_bytes: u64,
    pub contiguous_range_pairs: usize,
    pub unique_anchor_records: usize,
    pub anchor_reuse_hits: usize,
    pub target_ordinal_sum: usize,
    pub target_ordinal_max: usize,
}

pub(crate) fn range_geometry(ranges: &[RangePlan]) -> Result<(u64, u64, u64, usize), String> {
    if ranges.is_empty() {
        return Ok((0, 0, 0, 0));
    }
    let mut ordered = ranges
        .iter()
        .map(|range| {
            range
                .offset
                .checked_add(range.length)
                .map(|end| (range.offset, end, range.length))
                .ok_or_else(|| "planned range end overflow u64".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    ordered.sort_unstable_by_key(|value| (value.0, value.1));
    let min_offset = ordered[0].0;
    let max_end = ordered
        .iter()
        .map(|value| value.1)
        .max()
        .unwrap_or(min_offset);
    let max_length = ordered.iter().map(|value| value.2).max().unwrap_or(0);
    let total = ordered.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(value.2)
            .ok_or_else(|| "planned range bytes overflow u64".to_string())
    })?;
    let contiguous_pairs = ordered
        .windows(2)
        .filter(|pair| pair[0].1 == pair[1].0)
        .count();
    Ok((
        total / ordered.len() as u64,
        max_length,
        max_end.saturating_sub(min_offset),
        contiguous_pairs,
    ))
}

pub(crate) fn anchor_metrics(closures: &[DependencyClosure]) -> (usize, usize) {
    let mentions = closures
        .iter()
        .flat_map(|closure| &closure.records)
        .filter(|record| record.kind == DependencyKind::Anchor)
        .map(|record| record.record_id)
        .collect::<Vec<_>>();
    let unique = mentions.iter().copied().collect::<HashSet<_>>().len();
    (unique, mentions.len().saturating_sub(unique))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedPairRoute {
    /// Original requests routed to Pair, preserving duplicates and order.
    pub pair_requests: Vec<LogicalRequest>,
    /// Original requests routed to the fallback representation.
    pub fallback_requests: Vec<LogicalRequest>,
    pub fallback_representation: Representation,
    pub materialized_unique: usize,
    pub fallback_unique: usize,
}

#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub logical_samples: usize,
    pub unique_targets: usize,
    pub unique_videos: usize,
    pub dependency_records: usize,
    pub target_ordinal_sum: usize,
    pub target_ordinal_max: usize,
    pub planned_ranges: usize,
    pub planned_useful_bytes: u64,
    pub planned_fetched_bytes: u64,
    pub cache_race_fallback_records: usize,
    pub unique_records: usize,
    pub physical_ranges: usize,
    pub client_requests: usize,
    pub server_entries: usize,
    pub useful_bytes: u64,
    pub fetched_bytes: u64,
    pub overfetch_bytes: u64,
    pub decoded_targets: usize,
    pub decoded_frames: usize,
    pub decode_groups: usize,
    pub anchor_decode_invocations: usize,
    pub fused_decode_groups: usize,
    pub repeated_decode_groups: usize,
    pub encoded_cache_hits: usize,
    pub encoded_cache_misses: usize,
    pub anchor_cache_hits: usize,
    pub anchor_cache_misses: usize,
    pub delta_cache_hits: usize,
    pub delta_cache_misses: usize,
    pub decoded_cache_hits: usize,
    pub decoded_cache_misses: usize,
    pub encoded_cache_resident_bytes: u64,
    pub anchor_cache_resident_bytes: u64,
    pub delta_cache_resident_bytes: u64,
    pub decoded_cache_resident_bytes: u64,
    pub decoder_state_hits: usize,
    pub decoder_state_misses: usize,
    pub decoder_state_resets: usize,
    pub decoder_state_resident: usize,
    pub materialized_targets: usize,
    pub fallback_targets: usize,
    pub materialization_budget_bytes: u64,
    pub materialization_used_bytes: u64,
    pub portfolio_parallel_branches: usize,
    pub global_io_concurrency: usize,
    pub global_decoder_slots: usize,
    pub portfolio_branch_overlap_ns: u64,
    pub pair_branch_total_ns: u64,
    pub fallback_branch_total_ns: u64,
    pub pair_branch_fetch_wall_ns: u64,
    pub fallback_branch_fetch_wall_ns: u64,
    pub pair_branch_decode_ns: u64,
    pub fallback_branch_decode_ns: u64,
    pub resolve_ns: u64,
    pub cache_lookup_ns: u64,
    pub plan_ns: u64,
    pub fetch_wall_ns: u64,
    pub fetch_service_ns_sum: u64,
    pub range_queue_ns_sum: u64,
    pub extract_ns: u64,
    pub assemble_ns: u64,
    pub decode_ns: u64,
    pub rgb_convert_ns: u64,
    pub fetch_decode_overlap_ns: u64,
    pub reorder_ns: u64,
    pub total_ns: u64,
    pub time_to_first_ready_ns: u64,
    pub completion_selected: bool,
}

#[derive(Debug)]
pub struct OuterPlanner {
    samples: HashMap<u64, SampleRepresentations>,
}

impl OuterPlanner {
    pub fn new(samples: Vec<SampleRepresentations>) -> Result<Self, String> {
        let mut by_id = HashMap::with_capacity(samples.len());
        for sample in samples {
            let sample_id = sample.sample_id;
            for closure in [&sample.prefix, &sample.normalized, &sample.pair]
                .into_iter()
                .flatten()
            {
                validate_closure(sample_id, closure)?;
            }
            if by_id.insert(sample_id, sample).is_some() {
                return Err(format!("duplicate sample_id {sample_id}"));
            }
        }
        Ok(Self { samples: by_id })
    }

    pub fn resolve_batch(
        &self,
        requests: &[LogicalRequest],
        representation: Representation,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
    ) -> Result<ResolvedBatch, String> {
        let closures = self.resolve_closures(requests, representation)?;
        let ranges = Self::plan_closures(&closures, merge_threshold_bytes, max_range_bytes)?;
        let features = Self::derive_batch_features(requests, &closures, &ranges)?;
        Ok(ResolvedBatch {
            requests: requests.to_vec(),
            closures,
            ranges,
            features,
        })
    }

    pub fn resolve_closures(
        &self,
        requests: &[LogicalRequest],
        representation: Representation,
    ) -> Result<Vec<DependencyClosure>, String> {
        let mut seen = HashSet::new();
        let mut closures = Vec::new();
        for request in requests {
            if !seen.insert(request.sample_id) {
                continue;
            }
            let sample = self
                .samples
                .get(&request.sample_id)
                .ok_or_else(|| format!("unknown sample_id {}", request.sample_id))?;
            let closure = sample.get(representation).ok_or_else(|| {
                format!(
                    "sample_id {} has no {:?} representation",
                    request.sample_id, representation
                )
            })?;
            closures.push(closure.clone());
        }
        Ok(closures)
    }

    pub fn route_materialized_pair(
        &self,
        requests: &[LogicalRequest],
        materialized: &HashSet<u64>,
        fallback_representation: Representation,
    ) -> Result<MaterializedPairRoute, String> {
        if fallback_representation == Representation::Pair {
            return Err("materialized Pair fallback cannot also be Pair".to_string());
        }
        let mut pair_requests = Vec::new();
        let mut fallback_requests = Vec::new();
        let mut pair_unique = HashSet::new();
        let mut fallback_unique = HashSet::new();
        for request in requests {
            let sample = self
                .samples
                .get(&request.sample_id)
                .ok_or_else(|| format!("unknown sample_id {}", request.sample_id))?;
            if materialized.contains(&request.sample_id) {
                sample.get(Representation::Pair).ok_or_else(|| {
                    format!(
                        "materialized sample_id {} has no Pair representation",
                        request.sample_id
                    )
                })?;
                pair_requests.push(*request);
                pair_unique.insert(request.sample_id);
            } else {
                sample.get(fallback_representation).ok_or_else(|| {
                    format!(
                        "fallback sample_id {} has no {:?} representation",
                        request.sample_id, fallback_representation
                    )
                })?;
                fallback_requests.push(*request);
                fallback_unique.insert(request.sample_id);
            }
        }
        Ok(MaterializedPairRoute {
            pair_requests,
            fallback_requests,
            fallback_representation,
            materialized_unique: pair_unique.len(),
            fallback_unique: fallback_unique.len(),
        })
    }

    pub fn plan_closures(
        closures: &[DependencyClosure],
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
    ) -> Result<Vec<RangePlan>, String> {
        let ranges = closures
            .iter()
            .flat_map(|closure| &closure.records)
            .map(|record| RecordRange {
                record_id: record.record_id,
                offset: record.offset,
                length: record.length,
            })
            .collect::<Vec<_>>();
        planner::plan_byte_ranges(&ranges, merge_threshold_bytes, max_range_bytes)
    }

    pub fn derive_batch_features(
        requests: &[LogicalRequest],
        closures: &[DependencyClosure],
        ranges: &[RangePlan],
    ) -> Result<BatchFeatures, String> {
        let record_ranges = closures
            .iter()
            .flat_map(|closure| &closure.records)
            .map(|record| RecordRange {
                record_id: record.record_id,
                offset: record.offset,
                length: record.length,
            })
            .collect::<Vec<_>>();
        let useful_bytes = planner::unique_covered_bytes(&record_ranges)?;
        let fetched_bytes = ranges.iter().try_fold(0u64, |total, range| {
            total
                .checked_add(range.length)
                .ok_or_else(|| "planned fetched bytes overflow u64".to_string())
        })?;
        let unique_records = record_ranges
            .iter()
            .map(|record| record.record_id)
            .collect::<HashSet<_>>()
            .len();
        let (mean_range_bytes, max_range_bytes, address_span_bytes, contiguous_range_pairs) =
            range_geometry(ranges)?;
        let (unique_anchor_records, anchor_reuse_hits) = anchor_metrics(closures);
        Ok(BatchFeatures {
            logical_requests: requests.len(),
            unique_targets: closures.len(),
            unique_videos: closures
                .iter()
                .map(|closure| closure.video_id)
                .collect::<HashSet<_>>()
                .len(),
            dependency_records: record_ranges.len(),
            unique_records,
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
}

fn validate_closure(sample_id: u64, closure: &DependencyClosure) -> Result<(), String> {
    if closure.sample_id != sample_id {
        return Err(format!(
            "sample key {sample_id} disagrees with closure sample_id {}",
            closure.sample_id
        ));
    }
    if closure.records.is_empty() {
        return Err(format!("sample_id {sample_id} has an empty closure"));
    }
    let kinds = closure
        .records
        .iter()
        .map(|record| record.kind)
        .collect::<Vec<_>>();
    let valid = match closure.representation {
        Representation::Prefix => kinds.iter().all(|kind| *kind == DependencyKind::Prefix),
        Representation::Normalized => {
            (closure.target_ordinal == 0 && kinds == [DependencyKind::Anchor])
                || (closure.target_ordinal > 0
                    && kinds.len() == 2
                    && kinds[0] == DependencyKind::Anchor
                    && kinds[1] == DependencyKind::Delta)
        }
        Representation::Pair => kinds == [DependencyKind::Pair],
    };
    if !valid {
        return Err(format!(
            "sample_id {sample_id} has invalid {:?} dependency kinds: {:?}",
            closure.representation, kinds
        ));
    }
    for record in &closure.records {
        if record.length == 0 {
            return Err(format!("record {} has zero length", record.record_id));
        }
        record
            .offset
            .checked_add(record.length)
            .ok_or_else(|| format!("record {} range overflows u64", record.record_id))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(record_id: u64, kind: DependencyKind, offset: u64) -> PhysicalRecord {
        PhysicalRecord {
            record_id,
            kind,
            offset,
            length: 10,
        }
    }

    fn closure(
        sample_id: u64,
        representation: Representation,
        records: Vec<PhysicalRecord>,
    ) -> DependencyClosure {
        DependencyClosure {
            sample_id,
            video_id: 7,
            representation,
            records,
            target_ordinal: 1,
        }
    }

    fn sample(sample_id: u64, base: u64) -> SampleRepresentations {
        SampleRepresentations {
            sample_id,
            prefix: Some(closure(
                sample_id,
                Representation::Prefix,
                vec![record(base, DependencyKind::Prefix, base * 100)],
            )),
            normalized: Some(closure(
                sample_id,
                Representation::Normalized,
                vec![
                    record(99, DependencyKind::Anchor, 0),
                    record(base + 1, DependencyKind::Delta, base * 100 + 20),
                ],
            )),
            pair: Some(closure(
                sample_id,
                Representation::Pair,
                vec![record(base + 2, DependencyKind::Pair, base * 100 + 40)],
            )),
        }
    }

    #[test]
    fn one_planner_resolves_all_representations_in_request_order() {
        let planner = OuterPlanner::new(vec![sample(1, 10), sample(2, 20)]).unwrap();
        let requests = [
            LogicalRequest { sample_id: 2 },
            LogicalRequest { sample_id: 1 },
            LogicalRequest { sample_id: 2 },
        ];
        for representation in [
            Representation::Prefix,
            Representation::Normalized,
            Representation::Pair,
        ] {
            let resolved = planner
                .resolve_batch(&requests, representation, None, None)
                .unwrap();
            assert_eq!(resolved.requests, requests);
            assert_eq!(
                resolved
                    .closures
                    .iter()
                    .map(|closure| closure.sample_id)
                    .collect::<Vec<_>>(),
                vec![2, 1]
            );
        }
    }

    #[test]
    fn normalized_shared_anchor_is_deduplicated_in_physical_plan() {
        let planner = OuterPlanner::new(vec![sample(1, 10), sample(2, 20)]).unwrap();
        let resolved = planner
            .resolve_batch(
                &[
                    LogicalRequest { sample_id: 1 },
                    LogicalRequest { sample_id: 2 },
                ],
                Representation::Normalized,
                None,
                None,
            )
            .unwrap();
        assert_eq!(resolved.ranges.len(), 3);
        assert_eq!(
            resolved
                .ranges
                .iter()
                .flat_map(|range| &range.records)
                .filter(|record| record.record_id == 99)
                .count(),
            1
        );
    }

    #[test]
    fn normalized_anchor_target_is_a_one_record_closure() {
        let anchor = DependencyClosure {
            sample_id: 3,
            video_id: 7,
            representation: Representation::Normalized,
            records: vec![record(99, DependencyKind::Anchor, 0)],
            target_ordinal: 0,
        };
        let planner = OuterPlanner::new(vec![SampleRepresentations {
            sample_id: 3,
            prefix: None,
            normalized: Some(anchor),
            pair: None,
        }])
        .unwrap();
        let resolved = planner
            .resolve_batch(
                &[LogicalRequest { sample_id: 3 }],
                Representation::Normalized,
                None,
                None,
            )
            .unwrap();
        assert_eq!(resolved.features.dependency_records, 1);
        assert_eq!(resolved.features.target_ordinal_max, 0);
    }

    #[test]
    fn batch_features_are_derived_from_requests_and_physical_closures() {
        let planner = OuterPlanner::new(vec![sample(1, 10), sample(2, 20)]).unwrap();
        let resolved = planner
            .resolve_batch(
                &[
                    LogicalRequest { sample_id: 1 },
                    LogicalRequest { sample_id: 2 },
                    LogicalRequest { sample_id: 1 },
                ],
                Representation::Normalized,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            resolved.features,
            BatchFeatures {
                logical_requests: 3,
                unique_targets: 2,
                unique_videos: 1,
                dependency_records: 4,
                unique_records: 3,
                useful_bytes: 30,
                physical_ranges: 3,
                fetched_bytes: 30,
                overfetch_bytes: 0,
                mean_range_bytes: 10,
                max_range_bytes: 10,
                address_span_bytes: 2030,
                contiguous_range_pairs: 0,
                unique_anchor_records: 1,
                anchor_reuse_hits: 1,
                target_ordinal_sum: 2,
                target_ordinal_max: 1,
            }
        );
    }

    #[test]
    fn malformed_representation_is_rejected() {
        let mut invalid = sample(1, 10);
        invalid.normalized = Some(closure(
            1,
            Representation::Normalized,
            vec![record(1, DependencyKind::Delta, 0)],
        ));
        assert!(OuterPlanner::new(vec![invalid])
            .unwrap_err()
            .contains("invalid Normalized"));
    }

    #[test]
    fn unavailable_representation_is_an_error() {
        let mut value = sample(1, 10);
        value.pair = None;
        let planner = OuterPlanner::new(vec![value]).unwrap();
        let error = planner
            .resolve_batch(
                &[LogicalRequest { sample_id: 1 }],
                Representation::Pair,
                None,
                None,
            )
            .unwrap_err();
        assert!(error.contains("no Pair representation"));
    }

    #[test]
    fn materialized_pair_route_preserves_branch_order_and_duplicates() {
        let planner = OuterPlanner::new(vec![sample(1, 10), sample(2, 20)]).unwrap();
        let requests = [
            LogicalRequest { sample_id: 2 },
            LogicalRequest { sample_id: 1 },
            LogicalRequest { sample_id: 2 },
        ];
        let route = planner
            .route_materialized_pair(&requests, &HashSet::from([2]), Representation::Normalized)
            .unwrap();
        assert_eq!(
            route.pair_requests,
            vec![
                LogicalRequest { sample_id: 2 },
                LogicalRequest { sample_id: 2 }
            ]
        );
        assert_eq!(
            route.fallback_requests,
            vec![LogicalRequest { sample_id: 1 }]
        );
        assert_eq!(route.materialized_unique, 1);
        assert_eq!(route.fallback_unique, 1);
    }

    #[test]
    fn materialized_pair_route_rejects_missing_branch_representation() {
        let mut value = sample(1, 10);
        value.pair = None;
        let planner = OuterPlanner::new(vec![value]).unwrap();
        let error = planner
            .route_materialized_pair(
                &[LogicalRequest { sample_id: 1 }],
                &HashSet::from([1]),
                Representation::Normalized,
            )
            .unwrap_err();
        assert!(error.contains("no Pair representation"));
    }
}
