use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use crate::backend::{AIStoreObjectRange, MultiObjectStorageBackend};
use crate::decoder;
use crate::planner::{self, ByteCache, RangePlan, RecordRange};
use crate::representation::BatchStats;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentDescriptor {
    pub sample_id: u64,
    pub video_id: u64,
    pub fragment_id: u64,
    pub object_key: String,
    pub offset: u64,
    pub length: u64,
    pub target_ordinal: usize,
    pub codec_config: Vec<u8>,
    pub mp4_length_prefixed: bool,
    pub mp4_container: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalFragmentTarget {
    pub sample_id: u64,
    pub video_id: u64,
    pub target_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GopDescriptor {
    pub video_id: u64,
    pub gop_ordinal: u64,
    pub fragment_id: u64,
    pub object_key: String,
    pub offset: u64,
    pub length: u64,
    pub codec_config: Vec<u8>,
    pub mp4_length_prefixed: bool,
}

enum FragmentIndex {
    Direct(HashMap<u64, FragmentDescriptor>),
    Gop {
        targets: HashMap<u64, LogicalFragmentTarget>,
        gops: HashMap<(u64, u64), GopDescriptor>,
        gop_size: u64,
    },
}

impl FragmentIndex {
    fn resolve(&self, sample_id: u64) -> Result<FragmentDescriptor, String> {
        match self {
            Self::Direct(descriptors) => descriptors
                .get(&sample_id)
                .cloned()
                .ok_or_else(|| format!("unknown fragment sample_id {sample_id}")),
            Self::Gop {
                targets,
                gops,
                gop_size,
            } => {
                let target = targets
                    .get(&sample_id)
                    .ok_or_else(|| format!("unknown fragment sample_id {sample_id}"))?;
                let gop_ordinal = target.target_frame / *gop_size;
                let gop = gops.get(&(target.video_id, gop_ordinal)).ok_or_else(|| {
                    format!(
                        "missing GOP index for sample {sample_id}: video {} GOP {}",
                        target.video_id, gop_ordinal
                    )
                })?;
                Ok(FragmentDescriptor {
                    sample_id,
                    video_id: target.video_id,
                    fragment_id: gop.fragment_id,
                    object_key: gop.object_key.clone(),
                    offset: gop.offset,
                    length: gop.length,
                    target_ordinal: (target.target_frame % *gop_size) as usize,
                    codec_config: gop.codec_config.clone(),
                    mp4_length_prefixed: gop.mp4_length_prefixed,
                    mp4_container: false,
                })
            }
        }
    }
}

#[derive(Debug)]
pub struct FragmentFrame {
    pub sample_id: u64,
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

struct FragmentGroup<'a> {
    fragment_id: u64,
    descriptor: &'a FragmentDescriptor,
    targets: Vec<&'a FragmentDescriptor>,
}

pub struct FragmentBatchExecutor {
    index: FragmentIndex,
    backend: Box<dyn MultiObjectStorageBackend + Send>,
    merge_threshold_bytes: Option<u64>,
    max_range_bytes: Option<u64>,
    encoded_cache: ByteCache,
    decoded_cache: ByteCache,
    decoder_slots: decoder::SharedDecoderSlots,
    width: u32,
    height: u32,
}

impl FragmentBatchExecutor {
    pub(crate) fn with_shared_decoder_slots(
        mut self,
        decoder_slots: decoder::SharedDecoderSlots,
    ) -> Result<Self, String> {
        if decoder_slots.is_empty() {
            return Err("shared decoder slots must be non-empty".to_string());
        }
        self.decoder_slots = decoder_slots;
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        descriptors: Vec<FragmentDescriptor>,
        backend: Box<dyn MultiObjectStorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        if descriptors.is_empty() {
            return Err("fragment executor requires descriptors".to_string());
        }
        if width == 0 || height == 0 {
            return Err("fragment dimensions must be positive".to_string());
        }
        let mut by_sample = HashMap::with_capacity(descriptors.len());
        let mut physical = HashMap::<u64, (&str, u64, u64, &[u8], bool, bool)>::new();
        for descriptor in &descriptors {
            if descriptor.object_key.is_empty()
                || descriptor.length == 0
                || descriptor.offset.checked_add(descriptor.length).is_none()
                || (!descriptor.mp4_container && descriptor.codec_config.is_empty())
            {
                return Err(format!(
                    "invalid fragment descriptor for sample {}",
                    descriptor.sample_id
                ));
            }
            let address = (
                descriptor.object_key.as_str(),
                descriptor.offset,
                descriptor.length,
                descriptor.codec_config.as_slice(),
                descriptor.mp4_length_prefixed,
                descriptor.mp4_container,
            );
            if let Some(existing) = physical.insert(descriptor.fragment_id, address) {
                if existing != address {
                    return Err(format!(
                        "fragment {} has conflicting physical descriptors",
                        descriptor.fragment_id
                    ));
                }
            }
        }
        drop(physical);
        for descriptor in descriptors {
            let sample_id = descriptor.sample_id;
            if by_sample.insert(sample_id, descriptor).is_some() {
                return Err(format!("duplicate fragment sample_id {sample_id}"));
            }
        }
        Self::with_index(
            FragmentIndex::Direct(by_sample),
            backend,
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            width,
            height,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_gop_index(
        targets: Vec<LogicalFragmentTarget>,
        gops: Vec<GopDescriptor>,
        gop_size: u64,
        backend: Box<dyn MultiObjectStorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        if targets.is_empty() || gops.is_empty() {
            return Err("GOP index requires targets and GOP descriptors".to_string());
        }
        if gop_size == 0 {
            return Err("GOP size must be positive".to_string());
        }
        let mut by_target = HashMap::with_capacity(targets.len());
        for target in targets {
            let sample_id = target.sample_id;
            if by_target.insert(sample_id, target).is_some() {
                return Err(format!("duplicate logical sample_id {sample_id}"));
            }
        }
        let mut by_gop = HashMap::with_capacity(gops.len());
        let mut physical = HashMap::<u64, (&str, u64, u64, &[u8], bool)>::new();
        for gop in &gops {
            if gop.object_key.is_empty()
                || gop.length == 0
                || gop.offset.checked_add(gop.length).is_none()
                || gop.codec_config.is_empty()
            {
                return Err(format!(
                    "invalid GOP descriptor for video {} GOP {}",
                    gop.video_id, gop.gop_ordinal
                ));
            }
            let address = (
                gop.object_key.as_str(),
                gop.offset,
                gop.length,
                gop.codec_config.as_slice(),
                gop.mp4_length_prefixed,
            );
            if let Some(existing) = physical.insert(gop.fragment_id, address) {
                if existing != address {
                    return Err(format!(
                        "fragment {} has conflicting physical descriptors",
                        gop.fragment_id
                    ));
                }
            }
        }
        drop(physical);
        for gop in gops {
            let key = (gop.video_id, gop.gop_ordinal);
            if by_gop.insert(key, gop).is_some() {
                return Err(format!(
                    "duplicate GOP descriptor for video {} GOP {}",
                    key.0, key.1
                ));
            }
        }
        for target in by_target.values() {
            let key = (target.video_id, target.target_frame / gop_size);
            if !by_gop.contains_key(&key) {
                return Err(format!(
                    "logical sample {} has no GOP descriptor for video {} GOP {}",
                    target.sample_id, key.0, key.1
                ));
            }
        }
        Self::with_index(
            FragmentIndex::Gop {
                targets: by_target,
                gops: by_gop,
                gop_size,
            },
            backend,
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            width,
            height,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_index(
        index: FragmentIndex,
        backend: Box<dyn MultiObjectStorageBackend + Send>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("fragment dimensions must be positive".to_string());
        }
        Ok(Self {
            index,
            backend,
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache: ByteCache::new(encoded_cache_bytes),
            decoded_cache: ByteCache::new(decoded_cache_bytes),
            decoder_slots: decoder::shared_decoder_slots(decode_concurrency),
            width,
            height,
        })
    }

    fn mp4_sample_range_to_annex_b(input: &[u8]) -> Result<Vec<u8>, String> {
        let mut output = Vec::with_capacity(input.len());
        let mut position = 0usize;
        while position < input.len() {
            if input.len() - position < 4 {
                return Err("truncated MP4 sample length".to_string());
            }
            let length = u32::from_be_bytes(
                input[position..position + 4]
                    .try_into()
                    .expect("four-byte slice"),
            ) as usize;
            position += 4;
            let end = position
                .checked_add(length)
                .ok_or_else(|| "MP4 sample length overflow".to_string())?;
            if length == 0 || end > input.len() {
                return Err("truncated or empty MP4 sample payload".to_string());
            }
            output.extend_from_slice(&[0, 0, 1]);
            output.extend_from_slice(&input[position..end]);
            position = end;
        }
        Ok(output)
    }

    fn group<'a>(descriptors: Vec<&'a FragmentDescriptor>) -> Vec<FragmentGroup<'a>> {
        let mut positions = HashMap::new();
        let mut groups = Vec::<FragmentGroup<'a>>::new();
        for descriptor in descriptors {
            let position = match positions.get(&descriptor.fragment_id) {
                Some(position) => *position,
                None => {
                    let position = groups.len();
                    positions.insert(descriptor.fragment_id, position);
                    groups.push(FragmentGroup {
                        fragment_id: descriptor.fragment_id,
                        descriptor,
                        targets: Vec::new(),
                    });
                    position
                }
            };
            groups[position].targets.push(descriptor);
        }
        groups
    }

    fn decode_groups(
        &self,
        groups: &[FragmentGroup<'_>],
        payloads: &HashMap<u64, Vec<u8>>,
    ) -> Result<(HashMap<u64, Vec<u8>>, u64, u64, u64), String> {
        let mut partitions: Vec<Vec<&FragmentGroup<'_>>> =
            (0..self.decoder_slots.len()).map(|_| Vec::new()).collect();
        let partition_count = partitions.len();
        for (index, group) in groups.iter().enumerate() {
            // GOP fragments are self-contained, so decoder affinity provides
            // no state-reuse benefit. Round-robin assignment avoids hash
            // collisions serializing independent groups on one slot.
            partitions[index % partition_count].push(group);
        }
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (slot_index, groups) in partitions.into_iter().enumerate() {
                if groups.is_empty() {
                    continue;
                }
                let slot = &self.decoder_slots[slot_index];
                handles.push(scope.spawn(move || {
                    let mut pool = slot
                        .lock()
                        .map_err(|_| "fragment decoder slot lock poisoned".to_string())?;
                    let mut outputs = HashMap::new();
                    let mut converted = 0u64;
                    let mut assemble_ns = 0u64;
                    let mut decode_ns = 0u64;
                    for group in groups {
                        let payload = payloads.get(&group.fragment_id).ok_or_else(|| {
                            format!("missing fetched fragment {}", group.fragment_id)
                        })?;
                        let assemble_started = Instant::now();
                        let mut ordinals = Vec::new();
                        let mut ordinal_consumers = HashMap::<usize, Vec<u64>>::new();
                        for target in &group.targets {
                            ordinal_consumers
                                .entry(target.target_ordinal)
                                .or_insert_with(|| {
                                    ordinals.push(target.target_ordinal);
                                    Vec::new()
                                })
                                .push(target.sample_id);
                        }
                        let frames = if group.descriptor.mp4_container {
                            assemble_ns += assemble_started.elapsed().as_nanos() as u64;
                            let decode_started = Instant::now();
                            let frames = decoder::decode_mp4_selected_rgb24(
                                payload,
                                &ordinals,
                                decoder::DecoderConfig { num_threads: 1 },
                            )
                            .map_err(|error| error.to_string())?;
                            decode_ns += decode_started.elapsed().as_nanos() as u64;
                            frames
                        } else {
                            let mut self_contained = Vec::with_capacity(
                                group.descriptor.codec_config.len() + payload.len(),
                            );
                            self_contained.extend_from_slice(&group.descriptor.codec_config);
                            self_contained.extend_from_slice(payload);
                            let (codec_config, vcl_record, frame_count) =
                                decoder::extract_closed_record_parts(&self_contained)
                                    .map_err(|error| error.to_string())?;
                            if let Some(ordinal) = ordinals
                                .iter()
                                .copied()
                                .find(|ordinal| *ordinal >= frame_count)
                            {
                                return Err(format!(
                                    "fragment {} target ordinal {} exceeds frame count {}",
                                    group.fragment_id, ordinal, frame_count
                                ));
                            }
                            assemble_ns += assemble_started.elapsed().as_nanos() as u64;
                            let decode_started = Instant::now();
                            let frames = decoder::decode_full_gop_selected_rgb24(
                                &codec_config,
                                &vcl_record,
                                &mut pool,
                                &ordinals,
                            )
                            .map_err(|error| error.to_string())?;
                            decode_ns += decode_started.elapsed().as_nanos() as u64;
                            frames
                        };
                        if frames.len() != ordinals.len() {
                            return Err(format!(
                                "fragment {} returned {} frames for {} ordinals",
                                group.fragment_id,
                                frames.len(),
                                ordinals.len()
                            ));
                        }
                        for (ordinal, frame) in ordinals.into_iter().zip(frames) {
                            converted += 1;
                            for sample_id in &ordinal_consumers[&ordinal] {
                                outputs.insert(*sample_id, frame.data.clone());
                            }
                        }
                    }
                    Ok::<_, String>((outputs, converted, assemble_ns, decode_ns))
                }));
            }
            let mut outputs = HashMap::new();
            let mut converted = 0;
            let mut assemble_ns = 0;
            let mut decode_ns = 0;
            for handle in handles {
                let (part, part_converted, part_assemble_ns, part_decode_ns) = handle
                    .join()
                    .map_err(|_| "fragment decoder worker panicked".to_string())??;
                outputs.extend(part);
                converted += part_converted;
                assemble_ns = assemble_ns.max(part_assemble_ns);
                decode_ns = decode_ns.max(part_decode_ns);
            }
            Ok((outputs, converted, assemble_ns, decode_ns))
        })
    }

    pub fn execute(&mut self, batch: &[u64]) -> Result<(Vec<FragmentFrame>, BatchStats), String> {
        let total_started = Instant::now();
        let mut stats = BatchStats {
            logical_samples: batch.len(),
            ..Default::default()
        };

        let resolve_started = Instant::now();
        let mut seen = HashSet::new();
        let unique = batch
            .iter()
            .filter_map(|sample_id| seen.insert(*sample_id).then_some(*sample_id))
            .map(|sample_id| self.index.resolve(sample_id))
            .collect::<Result<Vec<_>, _>>()?;
        stats.unique_targets = unique.len();
        stats.unique_videos = unique
            .iter()
            .map(|descriptor| descriptor.video_id)
            .collect::<HashSet<_>>()
            .len();
        stats.dependency_records = unique
            .iter()
            .map(|descriptor| descriptor.fragment_id)
            .collect::<HashSet<_>>()
            .len();
        stats.target_ordinal_sum = unique.iter().map(|value| value.target_ordinal).sum();
        stats.target_ordinal_max = unique
            .iter()
            .map(|value| value.target_ordinal)
            .max()
            .unwrap_or(0);
        stats.resolve_ns = resolve_started.elapsed().as_nanos() as u64;

        let cache_started = Instant::now();
        let mut outputs = HashMap::new();
        let mut pending = Vec::new();
        for descriptor in unique {
            if let Some(rgb) = self.decoded_cache.get(descriptor.sample_id) {
                stats.decoded_cache_hits += 1;
                outputs.insert(descriptor.sample_id, rgb);
            } else {
                stats.decoded_cache_misses += 1;
                pending.push(descriptor);
            }
        }
        let groups = Self::group(pending.iter().collect());
        stats.unique_records = groups.len();
        stats.decode_groups = groups.len();
        stats.anchor_decode_invocations = groups.len();
        let mut payloads = HashMap::new();
        let mut misses = Vec::new();
        for group in &groups {
            if let Some(payload) = self.encoded_cache.get(group.fragment_id) {
                stats.encoded_cache_hits += 1;
                payloads.insert(group.fragment_id, payload);
            } else {
                stats.encoded_cache_misses += 1;
                misses.push(group);
            }
        }
        stats.cache_lookup_ns = cache_started.elapsed().as_nanos() as u64;

        let plan_started = Instant::now();
        let mut records_by_object = BTreeMap::<String, Vec<RecordRange>>::new();
        let miss_by_id = misses
            .iter()
            .map(|group| (group.fragment_id, *group))
            .collect::<HashMap<_, _>>();
        for group in &misses {
            records_by_object
                .entry(group.descriptor.object_key.clone())
                .or_default()
                .push(RecordRange {
                    record_id: group.fragment_id,
                    offset: group.descriptor.offset,
                    length: group.descriptor.length,
                });
        }
        let mut object_plans = Vec::<(String, RangePlan)>::new();
        for (object_key, records) in records_by_object {
            for plan in planner::plan_byte_ranges(
                &records,
                self.merge_threshold_bytes,
                self.max_range_bytes,
            )? {
                object_plans.push((object_key.clone(), plan));
            }
        }
        let ranges = object_plans
            .iter()
            .map(|(object_key, plan)| AIStoreObjectRange {
                object_key: object_key.clone(),
                offset: plan.offset,
                length: plan.length,
            })
            .collect::<Vec<_>>();
        stats.planned_ranges = ranges.len();
        stats.physical_ranges = ranges.len();
        stats.planned_useful_bytes = misses.iter().map(|group| group.descriptor.length).sum();
        stats.planned_fetched_bytes = ranges.iter().map(|range| range.length).sum();
        stats.useful_bytes = stats.planned_useful_bytes;
        stats.client_requests = self.backend.client_requests_for_ranges(&ranges);
        stats.server_entries = self.backend.server_entries_for_ranges(&ranges);
        stats.plan_ns = plan_started.elapsed().as_nanos() as u64;

        let fetched = if ranges.is_empty() {
            Vec::new()
        } else {
            let fetch_started = Instant::now();
            let fetched = self
                .backend
                .fetch_object_ranges(&ranges)
                .map_err(|error| error.to_string())?;
            stats.fetch_wall_ns = fetch_started.elapsed().as_nanos() as u64;
            stats.fetch_service_ns_sum = stats.fetch_wall_ns;
            fetched
        };
        if fetched.len() != object_plans.len() {
            return Err(format!(
                "fragment backend returned {} entries for {} ranges",
                fetched.len(),
                object_plans.len()
            ));
        }
        let extract_started = Instant::now();
        for ((_, plan), fetched_range) in object_plans.into_iter().zip(fetched) {
            if fetched_range.len() != plan.length as usize {
                return Err(format!(
                    "fragment range length mismatch: {} != {}",
                    fetched_range.len(),
                    plan.length
                ));
            }
            stats.fetched_bytes += fetched_range.len() as u64;
            for record in plan.records {
                let group = miss_by_id
                    .get(&record.record_id)
                    .ok_or_else(|| format!("missing fragment plan {}", record.record_id))?;
                let start = record.relative_offset as usize;
                let end = start + record.length as usize;
                if end > fetched_range.len() {
                    return Err(format!(
                        "fragment {} exceeds fetched range",
                        record.record_id
                    ));
                }
                let raw_payload = &fetched_range[start..end];
                let payload = if group.descriptor.mp4_container {
                    raw_payload.to_vec()
                } else if group.descriptor.mp4_length_prefixed {
                    Self::mp4_sample_range_to_annex_b(raw_payload)?
                } else {
                    raw_payload.to_vec()
                };
                self.encoded_cache.put(group.fragment_id, payload.clone());
                payloads.insert(group.fragment_id, payload);
            }
        }
        stats.extract_ns = extract_started.elapsed().as_nanos() as u64;
        stats.overfetch_bytes = stats.fetched_bytes.saturating_sub(stats.useful_bytes);

        if !pending.is_empty() {
            let (decoded, converted, assemble_ns, decode_ns) =
                self.decode_groups(&groups, &payloads)?;
            stats.assemble_ns = assemble_ns;
            stats.decode_ns = decode_ns;
            stats.decoded_targets = pending.len();
            stats.decoded_frames = converted as usize;
            for descriptor in &pending {
                let rgb = decoded.get(&descriptor.sample_id).ok_or_else(|| {
                    format!("missing decoded fragment sample {}", descriptor.sample_id)
                })?;
                self.decoded_cache.put(descriptor.sample_id, rgb.clone());
            }
            outputs.extend(decoded);
        }
        stats.encoded_cache_resident_bytes = self.encoded_cache.stats()["resident_bytes"];
        stats.decoded_cache_resident_bytes = self.decoded_cache.stats()["resident_bytes"];

        let reorder_started = Instant::now();
        let expected_rgb_bytes = self.width as usize * self.height as usize * 3;
        let mut remaining = HashMap::<u64, usize>::new();
        for sample_id in batch {
            *remaining.entry(*sample_id).or_default() += 1;
        }
        let frames = batch
            .iter()
            .map(|sample_id| {
                let count = remaining
                    .get_mut(sample_id)
                    .expect("batch sample missing remaining count");
                *count -= 1;
                let rgb = if *count == 0 {
                    outputs
                        .remove(sample_id)
                        .ok_or_else(|| format!("missing fragment output {sample_id}"))?
                } else {
                    outputs
                        .get(sample_id)
                        .ok_or_else(|| format!("missing fragment output {sample_id}"))?
                        .clone()
                };
                if rgb.len() != expected_rgb_bytes {
                    return Err(format!(
                        "fragment RGB size mismatch for {sample_id}: {} != {expected_rgb_bytes}",
                        rgb.len()
                    ));
                }
                Ok(FragmentFrame {
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
mod tests {
    use super::*;

    struct EmptyBackend;

    impl MultiObjectStorageBackend for EmptyBackend {
        fn fetch_object_ranges(
            &self,
            ranges: &[AIStoreObjectRange],
        ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
            assert!(ranges.is_empty());
            Ok(Vec::new())
        }
    }

    fn descriptor(sample_id: u64, fragment_id: u64, offset: u64) -> FragmentDescriptor {
        FragmentDescriptor {
            sample_id,
            video_id: 1,
            fragment_id,
            object_key: "video.mp4".to_string(),
            offset,
            length: 10,
            target_ordinal: sample_id as usize,
            codec_config: vec![1, 2, 3],
            mp4_length_prefixed: false,
            mp4_container: false,
        }
    }

    #[test]
    fn rejects_conflicting_fragment_addresses() {
        let error = FragmentBatchExecutor::new(
            vec![descriptor(1, 7, 10), descriptor(2, 7, 20)],
            Box::new(EmptyBackend),
            None,
            None,
            0,
            0,
            1,
            320,
            240,
        )
        .err()
        .expect("conflicting descriptors must fail");
        assert!(error.contains("conflicting physical descriptors"));
    }

    #[test]
    fn rejects_duplicate_sample_ids() {
        let error = FragmentBatchExecutor::new(
            vec![descriptor(1, 7, 10), descriptor(1, 7, 10)],
            Box::new(EmptyBackend),
            None,
            None,
            0,
            0,
            1,
            320,
            240,
        )
        .err()
        .expect("duplicate sample IDs must fail");
        assert!(error.contains("duplicate fragment sample_id"));
    }

    #[test]
    fn concurrent_clients_can_share_one_decoder_slot_budget() {
        let shared = decoder::shared_decoder_slots(2);
        let executor = FragmentBatchExecutor::new(
            vec![descriptor(1, 7, 10)],
            Box::new(EmptyBackend),
            None,
            None,
            0,
            0,
            8,
            320,
            240,
        )
        .unwrap()
        .with_shared_decoder_slots(shared.clone())
        .unwrap();
        assert_eq!(executor.decoder_slots.len(), 2);
        assert!(std::sync::Arc::ptr_eq(&executor.decoder_slots, &shared));
    }

    #[test]
    fn mp4_segment_descriptor_does_not_require_annex_b_codec_config() {
        let mut mp4 = descriptor(1, 7, 0);
        mp4.codec_config.clear();
        mp4.mp4_container = true;
        assert!(FragmentBatchExecutor::new(
            vec![mp4],
            Box::new(EmptyBackend),
            None,
            None,
            0,
            0,
            1,
            320,
            240,
        )
        .is_ok());

        let mut closed_h264 = descriptor(1, 7, 0);
        closed_h264.codec_config.clear();
        assert!(FragmentBatchExecutor::new(
            vec![closed_h264],
            Box::new(EmptyBackend),
            None,
            None,
            0,
            0,
            1,
            320,
            240,
        )
        .is_err());
    }

    #[test]
    fn converts_mp4_length_prefixed_samples_to_annex_b() {
        let input = [0, 0, 0, 3, 0x65, 1, 2, 0, 0, 0, 2, 0x41, 3];
        let output = FragmentBatchExecutor::mp4_sample_range_to_annex_b(&input).unwrap();
        assert_eq!(output, vec![0, 0, 1, 0x65, 1, 2, 0, 0, 1, 0x41, 3]);
        assert!(FragmentBatchExecutor::mp4_sample_range_to_annex_b(&input[..5]).is_err());
    }

    #[test]
    fn gop_index_resolves_the_same_physical_descriptor_as_direct_index() {
        let expected = FragmentDescriptor {
            sample_id: 9,
            video_id: 3,
            fragment_id: 17,
            object_key: "packed.bin".to_string(),
            offset: 100,
            length: 20,
            target_ordinal: 7,
            codec_config: vec![1, 2, 3],
            mp4_length_prefixed: true,
            mp4_container: false,
        };
        let direct = FragmentIndex::Direct(HashMap::from([(9, expected.clone())]));
        let gop = FragmentIndex::Gop {
            targets: HashMap::from([(
                9,
                LogicalFragmentTarget {
                    sample_id: 9,
                    video_id: 3,
                    target_frame: 15,
                },
            )]),
            gops: HashMap::from([(
                (3, 1),
                GopDescriptor {
                    video_id: 3,
                    gop_ordinal: 1,
                    fragment_id: 17,
                    object_key: "packed.bin".to_string(),
                    offset: 100,
                    length: 20,
                    codec_config: vec![1, 2, 3],
                    mp4_length_prefixed: true,
                },
            )]),
            gop_size: 8,
        };
        assert_eq!(direct.resolve(9).unwrap(), expected);
        assert_eq!(gop.resolve(9).unwrap(), expected);
    }

    #[test]
    fn gop_index_constructor_rejects_missing_target_closure() {
        let error = FragmentBatchExecutor::new_gop_index(
            vec![LogicalFragmentTarget {
                sample_id: 9,
                video_id: 3,
                target_frame: 15,
            }],
            vec![GopDescriptor {
                video_id: 3,
                gop_ordinal: 0,
                fragment_id: 16,
                object_key: "packed.bin".to_string(),
                offset: 0,
                length: 20,
                codec_config: vec![1, 2, 3],
                mp4_length_prefixed: true,
            }],
            8,
            Box::new(EmptyBackend),
            None,
            None,
            0,
            0,
            1,
            320,
            240,
        )
        .err()
        .expect("missing GOP closure must fail");
        assert!(error.contains("has no GOP descriptor"));
    }
}
