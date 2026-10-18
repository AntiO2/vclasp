use crate::hierarchical_layout::{
    AccessUnitRecord, GopRegion, HierarchicalLayoutIndex, RegionDecodeMode, TargetClosure,
};
use arrow::array::{Array, BinaryArray, Int32Array, Int64Array, StringArray};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HierarchicalRecordMeta {
    pub record_id: u64,
    pub video_id: String,
    pub frame_idx: i32,
    pub gop_id: u64,
    pub pts: i64,
    pub dts: i64,
    pub offset: u64,
    pub length: u64,
    pub decode_ordinal: usize,
    pub closure_record_ids: std::sync::Arc<[u64]>,
    pub target_output_ordinal: usize,
    pub nal_length_size: usize,
}

#[derive(Debug, Clone)]
pub struct HierarchicalCatalog {
    records: HashMap<u64, HierarchicalRecordMeta>,
    targets: HashMap<(String, i32), u64>,
    gop_records: HashMap<(String, u64), Vec<u64>>,
}

impl HierarchicalCatalog {
    #[cfg(test)]
    pub(crate) fn from_records_for_test(records: Vec<HierarchicalRecordMeta>) -> Self {
        let mut by_id = HashMap::with_capacity(records.len());
        let mut targets = HashMap::with_capacity(records.len());
        let mut gop_records = HashMap::<(String, u64), Vec<u64>>::new();
        for record in records {
            targets.insert(
                (record.video_id.clone(), record.frame_idx),
                record.record_id,
            );
            gop_records
                .entry((record.video_id.clone(), record.gop_id))
                .or_default()
                .push(record.record_id);
            by_id.insert(record.record_id, record);
        }
        for record_ids in gop_records.values_mut() {
            record_ids.sort_unstable_by_key(|record_id| by_id[record_id].decode_ordinal);
        }
        Self {
            records: by_id,
            targets,
            gop_records,
        }
    }

    pub fn from_parquet(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let reader =
            ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(data))?.build()?;
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        if batches.is_empty() {
            return Err("hierarchical index has no record batches".into());
        }
        let schema = batches[0].schema();
        let merged = arrow::compute::concat_batches(&schema, &batches)?;
        macro_rules! column {
            ($name:literal, $kind:ty) => {
                merged
                    .column_by_name($name)
                    .ok_or(concat!("hierarchical index is missing ", $name))?
                    .as_any()
                    .downcast_ref::<$kind>()
                    .ok_or(concat!("hierarchical index has invalid type for ", $name))?
            };
        }
        let record_ids = column!("record_id", Int64Array);
        let video_ids = column!("video_id", StringArray);
        let frame_idxs = column!("frame_idx", Int32Array);
        let gop_ids = column!("gop_id", Int32Array);
        let offsets = column!("record_offset", Int64Array);
        let lengths = column!("record_length", Int64Array);
        let packet_indices = column!("packet_index", Int32Array);
        let presentation_timestamps = column!("pts", Int64Array);
        let decode_timestamps = column!("dts", Int64Array);
        let closure_ids = column!("closure_record_ids", BinaryArray);
        let output_ordinals = column!("target_output_ordinal", Int32Array);
        let nal_length_sizes = column!("nal_length_size", Int32Array);
        let mut records = HashMap::with_capacity(merged.num_rows());
        let mut targets = HashMap::with_capacity(merged.num_rows());
        for row in 0..merged.num_rows() {
            if [
                record_ids as &dyn Array,
                video_ids,
                frame_idxs,
                gop_ids,
                offsets,
                lengths,
                packet_indices,
                presentation_timestamps,
                decode_timestamps,
                closure_ids,
                output_ordinals,
                nal_length_sizes,
            ]
            .iter()
            .any(|column| column.is_null(row))
            {
                return Err(format!("null AU catalog value at row {row}").into());
            }
            let parse_non_negative = |value: i64, field: &str| -> Result<u64, String> {
                value
                    .try_into()
                    .map_err(|_| format!("hierarchical {field} is negative at row {row}"))
            };
            let record_id = parse_non_negative(record_ids.value(row), "record_id")?;
            let closure_bytes = closure_ids.value(row);
            if closure_bytes.len() % 8 != 0 {
                return Err(format!("closure bytes are not u64 aligned at row {row}").into());
            }
            let closure_record_ids = closure_bytes
                .chunks_exact(8)
                .map(|value| u64::from_le_bytes(value.try_into().expect("8-byte chunk")))
                .collect::<Vec<_>>();
            if closure_record_ids.is_empty() || !closure_record_ids.contains(&record_id) {
                return Err(format!("invalid closure for record {record_id}").into());
            }
            let output_ordinal: usize = output_ordinals
                .value(row)
                .try_into()
                .map_err(|_| format!("negative output ordinal at row {row}"))?;
            if output_ordinal >= closure_record_ids.len() {
                return Err(format!("output ordinal outside closure at row {row}").into());
            }
            let nal_length_size: usize = nal_length_sizes
                .value(row)
                .try_into()
                .map_err(|_| format!("negative NAL length size at row {row}"))?;
            if !(1..=4).contains(&nal_length_size) {
                return Err(format!("unsupported NAL length size at row {row}").into());
            }
            let meta = HierarchicalRecordMeta {
                record_id,
                video_id: video_ids.value(row).to_string(),
                frame_idx: frame_idxs.value(row),
                gop_id: gop_ids
                    .value(row)
                    .try_into()
                    .map_err(|_| format!("negative GOP id at row {row}"))?,
                pts: presentation_timestamps.value(row),
                dts: decode_timestamps.value(row),
                offset: parse_non_negative(offsets.value(row), "record_offset")?,
                length: parse_non_negative(lengths.value(row), "record_length")?,
                decode_ordinal: packet_indices
                    .value(row)
                    .try_into()
                    .map_err(|_| format!("negative packet index at row {row}"))?,
                closure_record_ids: closure_record_ids.into(),
                target_output_ordinal: output_ordinal,
                nal_length_size,
            };
            if meta.video_id.is_empty()
                || meta.frame_idx < 0
                || meta.length == 0
                || meta.offset.checked_add(meta.length).is_none()
            {
                return Err(format!("invalid AU extent or logical target at row {row}").into());
            }
            if targets
                .insert((meta.video_id.clone(), meta.frame_idx), record_id)
                .is_some()
            {
                return Err(format!(
                    "duplicate hierarchical target ({}, {})",
                    meta.video_id, meta.frame_idx
                )
                .into());
            }
            if records.insert(record_id, meta).is_some() {
                return Err(format!("duplicate hierarchical record {record_id}").into());
            }
        }
        for record in records.values() {
            let mut previous = None;
            for dependency_id in record.closure_record_ids.iter() {
                let dependency = records.get(dependency_id).ok_or_else(|| {
                    format!(
                        "record {} references missing record {dependency_id}",
                        record.record_id
                    )
                })?;
                if dependency.video_id != record.video_id || dependency.gop_id != record.gop_id {
                    return Err(
                        format!("record {} has a cross-GOP closure", record.record_id).into(),
                    );
                }
                if previous.is_some_and(|value| value >= dependency.decode_ordinal) {
                    return Err(format!(
                        "record {} closure is not in decode order",
                        record.record_id
                    )
                    .into());
                }
                previous = Some(dependency.decode_ordinal);
            }
        }
        let mut gop_records = HashMap::<(String, u64), Vec<u64>>::new();
        for record in records.values() {
            gop_records
                .entry((record.video_id.clone(), record.gop_id))
                .or_default()
                .push(record.record_id);
        }
        for record_ids in gop_records.values_mut() {
            record_ids.sort_unstable_by_key(|record_id| records[record_id].decode_ordinal);
            if record_ids
                .windows(2)
                .any(|pair| records[&pair[0]].dts >= records[&pair[1]].dts)
            {
                return Err(
                    "hierarchical GOP decode timestamps are not strictly increasing".into(),
                );
            }
        }
        Ok(Self {
            records,
            targets,
            gop_records,
        })
    }

    pub fn target(&self, video_id: &str, frame_idx: i32) -> Option<&HierarchicalRecordMeta> {
        let record_id = self.targets.get(&(video_id.to_string(), frame_idx))?;
        self.records.get(record_id)
    }

    pub fn record(&self, record_id: u64) -> Option<&HierarchicalRecordMeta> {
        self.records.get(&record_id)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn records(&self) -> impl Iterator<Item = &HierarchicalRecordMeta> {
        self.records.values()
    }

    pub fn records_for_gop(&self, video_id: &str, gop_id: u64) -> Option<&[u64]> {
        self.gop_records
            .get(&(video_id.to_string(), gop_id))
            .map(Vec::as_slice)
    }

    pub fn to_layout_index(&self) -> Result<HierarchicalLayoutIndex, String> {
        let mut video_ids = self
            .records
            .values()
            .map(|record| record.video_id.clone())
            .collect::<Vec<_>>();
        video_ids.sort();
        video_ids.dedup();
        let video_numbers = video_ids
            .into_iter()
            .enumerate()
            .map(|(index, video_id)| (video_id, index as u64))
            .collect::<HashMap<_, _>>();
        let records = self
            .records
            .values()
            .map(|record| AccessUnitRecord {
                record_id: record.record_id,
                video_id: video_numbers[&record.video_id],
                gop_id: record.gop_id,
                offset: record.offset,
                length: record.length,
                decode_ordinal: record.decode_ordinal,
            })
            .collect::<Vec<_>>();
        let closures = self
            .records
            .values()
            .map(|record| TargetClosure {
                sample_id: record.record_id,
                video_id: video_numbers[&record.video_id],
                gop_id: record.gop_id,
                target_record_id: record.record_id,
                record_ids: record.closure_record_ids.clone(),
            })
            .collect::<Vec<_>>();
        let mut grouped = HashMap::<(String, u64), Vec<&HierarchicalRecordMeta>>::new();
        for record in self.records.values() {
            grouped
                .entry((record.video_id.clone(), record.gop_id))
                .or_default()
                .push(record);
        }
        let mut regions = Vec::with_capacity(grouped.len());
        for ((video_id, gop_id), mut group) in grouped {
            group.sort_unstable_by_key(|record| record.decode_ordinal);
            let offset = group.iter().map(|record| record.offset).min().unwrap();
            let end = group
                .iter()
                .map(|record| record.offset + record.length)
                .max()
                .unwrap();
            regions.push(GopRegion {
                video_id: video_numbers[&video_id],
                gop_id,
                offset,
                length: end - offset,
                record_ids: group.iter().map(|record| record.record_id).collect(),
            });
        }
        HierarchicalLayoutIndex::new_with_region_decode_mode(
            records,
            closures,
            regions,
            RegionDecodeMode::ClosureOnly,
        )
    }
}
