use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordRange {
    pub record_id: u64,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRecord {
    pub record_id: u64,
    pub relative_offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangePlan {
    pub offset: u64,
    pub length: u64,
    pub records: Vec<PlannedRecord>,
}

/// Number of distinct physical bytes covered by the requested records before
/// gap coalescing. Prefix records may overlap, so summing record lengths would
/// double-count bytes and can exceed the actual fetched range.
pub fn unique_covered_bytes(records: &[RecordRange]) -> Result<u64, String> {
    let mut intervals = records
        .iter()
        .map(|record| {
            if record.length == 0 {
                return Err(format!("record {} has zero length", record.record_id));
            }
            let end = record
                .offset
                .checked_add(record.length)
                .ok_or_else(|| format!("record {} range overflows u64", record.record_id))?;
            Ok((record.offset, end))
        })
        .collect::<Result<Vec<_>, String>>()?;
    intervals.sort_unstable();
    let Some((mut start, mut end)) = intervals.first().copied() else {
        return Ok(0);
    };
    let mut covered = 0_u64;
    for (next_start, next_end) in intervals.into_iter().skip(1) {
        if next_start <= end {
            end = end.max(next_end);
        } else {
            covered += end - start;
            start = next_start;
            end = next_end;
        }
    }
    Ok(covered + end - start)
}

pub fn plan_byte_ranges(
    records: &[RecordRange],
    merge_threshold_bytes: Option<u64>,
    max_range_bytes: Option<u64>,
) -> Result<Vec<RangePlan>, String> {
    let mut dedup = HashMap::with_capacity(records.len());
    for record in records {
        if record.length == 0 {
            return Err(format!("record {} has zero length", record.record_id));
        }
        record
            .offset
            .checked_add(record.length)
            .ok_or_else(|| format!("record {} range overflows u64", record.record_id))?;
        if let Some(previous) = dedup.insert(record.record_id, record.clone()) {
            if previous.offset != record.offset || previous.length != record.length {
                return Err(format!(
                    "record {} resolves to conflicting ranges",
                    record.record_id
                ));
            }
        }
    }

    let mut ordered: Vec<RecordRange> = dedup.into_values().collect();
    ordered.sort_by_key(|record| (record.offset, record.record_id));
    plan_ordered_byte_ranges(&ordered, merge_threshold_bytes, max_range_bytes)
}

/// Internal path for already validated, unique, offset-sorted records. Reuse
/// this ordering when evaluating many thresholds for the same closure.
pub(crate) fn plan_ordered_byte_ranges(
    ordered: &[RecordRange],
    merge_threshold_bytes: Option<u64>,
    max_range_bytes: Option<u64>,
) -> Result<Vec<RangePlan>, String> {
    let mut plans: Vec<RangePlan> = Vec::new();

    for record in ordered {
        let record_end = record.offset + record.length;
        let mut merged = false;
        if let (Some(threshold), Some(last)) = (merge_threshold_bytes, plans.last_mut()) {
            let plan_end = last.offset + last.length;
            let merged_end = plan_end.max(record_end);
            let merged_length = merged_end - last.offset;
            let within_limit = max_range_bytes
                .map(|limit| merged_length <= limit)
                .unwrap_or(true);
            if record.offset <= plan_end.saturating_add(threshold) && within_limit {
                last.length = merged_length;
                last.records.push(PlannedRecord {
                    record_id: record.record_id,
                    relative_offset: record.offset - last.offset,
                    length: record.length,
                });
                merged = true;
            }
        }
        if !merged {
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
    }
    Ok(plans)
}

/// Coalesce missing records only within the same codec dependency group.
///
/// This is intentionally different from a global byte-gap threshold. Once an
/// When an Anchor is cached, the remaining Delta records form independent
/// dependency groups. One range per group preserves that reuse without joining
/// physically adjacent records from unrelated groups.
pub fn plan_group_spans(
    grouped_records: &[(u64, RecordRange)],
    max_range_bytes: Option<u64>,
) -> Result<Vec<RangePlan>, String> {
    let mut record_groups = HashMap::<u64, u64>::new();
    let mut groups = HashMap::<u64, Vec<RecordRange>>::new();
    for (group_id, record) in grouped_records {
        if let Some(previous_group) = record_groups.insert(record.record_id, *group_id) {
            if previous_group != *group_id {
                return Err(format!(
                    "record {} belongs to conflicting dependency groups",
                    record.record_id
                ));
            }
        }
        groups.entry(*group_id).or_default().push(record.clone());
    }

    let mut plans = Vec::new();
    for records in groups.into_values() {
        plans.extend(plan_byte_ranges(&records, Some(u64::MAX), max_range_bytes)?);
    }
    plans.sort_by_key(|plan| plan.offset);
    Ok(plans)
}

#[derive(Debug)]
pub struct ByteCache {
    capacity_bytes: usize,
    resident_bytes: usize,
    entries: HashMap<u64, (bytes::Bytes, u64)>,
    lru: std::collections::BTreeSet<(u64, u64)>,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

pub type SharedByteCache = Arc<Mutex<ByteCache>>;

pub fn shared_byte_cache(capacity_bytes: usize) -> SharedByteCache {
    Arc::new(Mutex::new(ByteCache::new(capacity_bytes)))
}

impl ByteCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            resident_bytes: 0,
            entries: HashMap::new(),
            lru: std::collections::BTreeSet::new(),
            clock: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn touch(&mut self, record_id: u64) {
        if self.clock == u64::MAX {
            let order = self.lru.iter().map(|(_, id)| *id).collect::<Vec<_>>();
            self.lru.clear();
            for (stamp, id) in order.into_iter().enumerate() {
                self.entries.get_mut(&id).expect("resident entry").1 = stamp as u64;
                self.lru.insert((stamp as u64, id));
            }
            self.clock = self.entries.len() as u64;
        }
        if let Some((_, stamp)) = self.entries.get_mut(&record_id) {
            self.lru.remove(&(*stamp, record_id));
            *stamp = self.clock;
            self.lru.insert((*stamp, record_id));
            self.clock += 1;
        }
    }

    pub fn get_shared(&mut self, record_id: u64) -> Option<bytes::Bytes> {
        let value = self.entries.get(&record_id).map(|(value, _)| value.clone());
        if value.is_some() {
            self.hits += 1;
            self.touch(record_id);
        } else {
            self.misses += 1;
        }
        value
    }

    pub fn get(&mut self, record_id: u64) -> Option<Vec<u8>> {
        self.get_shared(record_id).map(Vec::from)
    }

    pub fn contains(&self, record_id: u64) -> bool {
        self.entries.contains_key(&record_id)
    }
    pub fn resident_ids(&self) -> HashSet<u64> {
        self.entries.keys().copied().collect()
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    fn remove_shared(&mut self, record_id: u64) -> Option<bytes::Bytes> {
        let (value, stamp) = self.entries.remove(&record_id)?;
        self.resident_bytes -= value.len();
        self.lru.remove(&(stamp, record_id));
        Some(value)
    }

    pub fn remove(&mut self, record_id: u64) -> Option<Vec<u8>> {
        self.remove_shared(record_id).map(Vec::from)
    }

    fn evict_oldest(&mut self) {
        let (_, victim) = *self.lru.first().expect("resident LRU entry");
        self.remove_shared(victim).expect("resident cache entry");
        self.evictions += 1;
    }

    pub fn resize(&mut self, capacity_bytes: usize) {
        self.capacity_bytes = capacity_bytes;
        while self.resident_bytes > self.capacity_bytes {
            self.evict_oldest();
        }
    }

    pub fn put(&mut self, record_id: u64, value: Vec<u8>) -> bool {
        self.put_shared(record_id, bytes::Bytes::from(value))
    }

    /// Retain an independently owned AU; callers must not pass a small slice
    /// backed by an oversized GET buffer, which would evade byte accounting.
    pub fn put_shared(&mut self, record_id: u64, value: bytes::Bytes) -> bool {
        if value.len() > self.capacity_bytes || self.capacity_bytes == 0 {
            return false;
        }
        self.remove_shared(record_id);
        while value.len() > self.capacity_bytes - self.resident_bytes {
            self.evict_oldest();
        }
        self.resident_bytes += value.len();
        self.entries.insert(record_id, (value, self.clock));
        self.touch(record_id);
        true
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.resident_bytes = 0;
        self.clock = 0;
    }

    pub fn stats(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("capacity_bytes".to_string(), self.capacity_bytes as u64),
            ("resident_bytes".to_string(), self.resident_bytes as u64),
            ("entries".to_string(), self.entries.len() as u64),
            ("hits".to_string(), self.hits),
            ("misses".to_string(), self.misses),
            ("evictions".to_string(), self.evictions),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_cache_hits_do_not_copy_au_storage() {
        let mut cache = ByteCache::new(16);
        let value = bytes::Bytes::from(vec![7; 8]);
        let pointer = value.as_ptr();
        cache.put_shared(1, value);
        let hit = cache.get_shared(1).unwrap();
        assert_eq!(pointer, hit.as_ptr());
        cache.clock = u64::MAX;
        cache.get_shared(1).unwrap();
        assert_eq!(cache.lru.len(), 1);
        assert_eq!(cache.resident_bytes(), 8);
        cache.put(2, vec![8; 16]);
        assert!(!cache.contains(1));
        assert_eq!(hit.as_ref(), &[7; 8]);
    }

    #[test]
    fn ordered_planning_matches_public_planning_under_caps() {
        let records = (0..32)
            .map(|id| RecordRange {
                record_id: id,
                offset: id * 13,
                length: 5,
            })
            .collect::<Vec<_>>();
        for gap in [None, Some(0), Some(8), Some(16), Some(u64::MAX)] {
            for cap in [None, Some(5), Some(32), Some(128)] {
                assert_eq!(
                    plan_byte_ranges(&records, gap, cap).unwrap(),
                    plan_ordered_byte_ranges(&records, gap, cap).unwrap()
                );
            }
        }
    }

    #[test]
    fn exact_plan_deduplicates_without_merging() {
        let records = vec![
            RecordRange {
                record_id: 2,
                offset: 200,
                length: 10,
            },
            RecordRange {
                record_id: 1,
                offset: 100,
                length: 20,
            },
            RecordRange {
                record_id: 1,
                offset: 100,
                length: 20,
            },
        ];
        let plans = plan_byte_ranges(&records, None, None).unwrap();
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].offset, 100);
        assert_eq!(plans[1].offset, 200);
    }

    #[test]
    fn group_spans_merge_within_but_not_across_dependency_groups() {
        let records = vec![
            (
                10,
                RecordRange {
                    record_id: 1,
                    offset: 100,
                    length: 10,
                },
            ),
            (
                10,
                RecordRange {
                    record_id: 2,
                    offset: 140,
                    length: 10,
                },
            ),
            (
                20,
                RecordRange {
                    record_id: 3,
                    offset: 160,
                    length: 10,
                },
            ),
        ];
        let plans = plan_group_spans(&records, None).unwrap();
        assert_eq!(plans.len(), 2);
        assert_eq!((plans[0].offset, plans[0].length), (100, 50));
        assert_eq!((plans[1].offset, plans[1].length), (160, 10));
        assert_eq!(plans[0].records.len(), 2);
    }

    #[test]
    fn group_spans_respect_the_maximum_range_size() {
        let records = vec![
            (
                10,
                RecordRange {
                    record_id: 1,
                    offset: 100,
                    length: 10,
                },
            ),
            (
                10,
                RecordRange {
                    record_id: 2,
                    offset: 140,
                    length: 10,
                },
            ),
        ];
        let plans = plan_group_spans(&records, Some(32)).unwrap();
        assert_eq!(plans.len(), 2);
    }

    #[test]
    fn useful_bytes_count_overlapping_prefixes_once() {
        let records = vec![
            RecordRange {
                record_id: 1,
                offset: 100,
                length: 20,
            },
            RecordRange {
                record_id: 2,
                offset: 100,
                length: 50,
            },
            RecordRange {
                record_id: 3,
                offset: 200,
                length: 10,
            },
        ];
        assert_eq!(unique_covered_bytes(&records).unwrap(), 60);
    }

    #[test]
    fn threshold_plan_merges_gaps_but_honors_max_range() {
        let records = vec![
            RecordRange {
                record_id: 1,
                offset: 100,
                length: 20,
            },
            RecordRange {
                record_id: 2,
                offset: 128,
                length: 10,
            },
            RecordRange {
                record_id: 3,
                offset: 150,
                length: 10,
            },
        ];
        let plans = plan_byte_ranges(&records, Some(16), Some(45)).unwrap();
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].length, 38);
        assert_eq!(plans[0].records[1].relative_offset, 28);
        assert_eq!(plans[1].offset, 150);
    }

    #[test]
    fn conflicting_duplicate_is_an_error() {
        let records = vec![
            RecordRange {
                record_id: 1,
                offset: 100,
                length: 20,
            },
            RecordRange {
                record_id: 1,
                offset: 101,
                length: 20,
            },
        ];
        assert!(plan_byte_ranges(&records, None, None).is_err());
    }

    #[test]
    fn byte_cache_is_bounded_and_lru() {
        let mut cache = ByteCache::new(6);
        assert!(cache.put(1, vec![1; 3]));
        assert!(cache.put(2, vec![2; 3]));
        assert_eq!(cache.get(1), Some(vec![1; 3]));
        assert!(cache.put(3, vec![3; 3]));
        assert_eq!(cache.get(2), None);
        assert_eq!(cache.get(1), Some(vec![1; 3]));
        let stats = cache.stats();
        assert_eq!(stats["resident_bytes"], 6);
        assert_eq!(stats["evictions"], 1);
    }

    #[test]
    fn byte_cache_resize_and_remove_preserve_accounting() {
        let mut cache = ByteCache::new(8);
        assert!(cache.put(1, vec![1; 4]));
        assert!(cache.put(2, vec![2; 4]));
        assert_eq!(cache.resident_ids(), HashSet::from([1, 2]));

        cache.resize(4);
        assert_eq!(cache.resident_ids(), HashSet::from([2]));
        assert_eq!(cache.resident_bytes(), 4);

        assert_eq!(cache.remove(2), Some(vec![2; 4]));
        assert_eq!(cache.resident_bytes(), 0);
        assert!(cache.resident_ids().is_empty());
    }
}
