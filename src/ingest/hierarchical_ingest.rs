use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::BufReader;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{Array, ArrayRef, BinaryArray, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

use crate::chunk::{self, ChunkWriteConfig};
use crate::encoder::{X264Encoder, MAX_ANCHOR_P_GROUP_FRAMES};
use crate::hierarchical_layout::{
    AccessUnitRecord, GopRegion, HierarchicalLayoutIndex, RegionDecodeMode, TargetClosure,
};

#[derive(Debug, Clone)]
pub struct VideoInput {
    pub video_id: String,
    pub class_name: String,
    pub source_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct HierarchicalBuildOptions {
    pub output_path: PathBuf,
    pub baseline_mp4_dir: Option<PathBuf>,
    pub ffmpeg_path: PathBuf,
    pub ffprobe_path: PathBuf,
    pub gop_size: u32,
    pub max_frames: u32,
    pub width: u16,
    pub height: u16,
    pub fps: u16,
    pub crf: u8,
    pub preset: String,
    pub workers: usize,
    pub dependency_policy: DependencyPolicy,
    pub hierarchical_b: HierarchicalBConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyPolicy {
    HierarchicalB,
    ChainedP,
    SharedAnchor,
}

impl DependencyPolicy {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "hierarchical_b" => Ok(Self::HierarchicalB),
            "chained_p" => Ok(Self::ChainedP),
            "shared_anchor" => Ok(Self::SharedAnchor),
            _ => Err(format!(
                "unknown dependency policy {value}; expected hierarchical_b|chained_p|shared_anchor"
            )),
        }
    }

    fn dependency_kind(self) -> &'static str {
        match self {
            Self::HierarchicalB => "hierarchical_b_closure",
            Self::ChainedP => "chained_p_closure",
            Self::SharedAnchor => "shared_anchor_closure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BFramePyramid {
    None,
    Strict,
}

impl BFramePyramid {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "none" => Ok(Self::None),
            "strict" => Ok(Self::Strict),
            _ => Err(format!(
                "unknown B-frame pyramid {value}; expected none|strict"
            )),
        }
    }

    fn x264_value(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Strict => "strict",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HierarchicalBConfig {
    pub max_b_frames: u8,
    pub pyramid: BFramePyramid,
    pub adaptive: u8,
    pub lookahead: u16,
    pub reference_frames: u8,
}

impl Default for HierarchicalBConfig {
    fn default() -> Self {
        Self {
            max_b_frames: 7,
            pyramid: BFramePyramid::Strict,
            adaptive: 0,
            lookahead: 0,
            reference_frames: 1,
        }
    }
}

impl HierarchicalBConfig {
    fn validate(self, gop_size: u32) -> Result<(), String> {
        if self.max_b_frames > 16 {
            return Err("hierarchical-B max_b_frames must be at most 16".to_string());
        }
        if u32::from(self.max_b_frames) >= gop_size {
            return Err(format!(
                "hierarchical-B max_b_frames {} must be smaller than gop_size {gop_size}",
                self.max_b_frames
            ));
        }
        if self.adaptive > 2 {
            return Err("hierarchical-B adaptive mode must be 0, 1, or 2".to_string());
        }
        if self.adaptive > 0 && self.lookahead < u16::from(self.max_b_frames).saturating_add(1) {
            return Err(format!(
                "hierarchical-B adaptive mode {} requires lookahead >= {}",
                self.adaptive,
                u16::from(self.max_b_frames).saturating_add(1)
            ));
        }
        if self.reference_frames == 0 || self.reference_frames > 16 {
            return Err("hierarchical-B reference_frames must be in 1..=16".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct HierarchicalBuildStats {
    pub workers: usize,
    pub videos: usize,
    pub records: usize,
    pub targets: usize,
    pub payload_bytes: u64,
    pub index_bytes: u64,
    pub chunk_bytes: u64,
    pub max_closure_records: usize,
    pub mean_closure_records: f64,
    pub median_closure_records: f64,
    pub p95_closure_records: f64,
    pub gops: usize,
    pub i_frames: usize,
    pub p_frames: usize,
    pub b_frames: usize,
    pub max_b_run: usize,
    pub total_seconds: f64,
    pub encode_seconds: f64,
    pub au_parse_seconds: f64,
    pub closure_construction_seconds: f64,
    pub closure_validation_seconds: f64,
    pub payload_copy_seconds: f64,
    pub index_serialization_seconds: f64,
    pub chunk_write_seconds: f64,
}

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
    pub closure_record_ids: Vec<u64>,
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
                closure_record_ids,
                target_output_ordinal: output_ordinal,
                nal_length_size,
            };
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
            for dependency_id in &record.closure_record_ids {
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

pub fn mp4_sample_to_annex_b(data: &[u8], nal_length_size: usize) -> Result<Vec<u8>, String> {
    if !(1..=4).contains(&nal_length_size) {
        return Err(format!("unsupported NAL length size {nal_length_size}"));
    }
    let mut cursor = 0usize;
    let mut output = Vec::with_capacity(data.len() + 16);
    while cursor < data.len() {
        if cursor + nal_length_size > data.len() {
            return Err("truncated MP4 sample NAL length".to_string());
        }
        let mut length_bytes = [0u8; 4];
        length_bytes[4 - nal_length_size..]
            .copy_from_slice(&data[cursor..cursor + nal_length_size]);
        cursor += nal_length_size;
        let length = u32::from_be_bytes(length_bytes) as usize;
        if length == 0 || cursor + length > data.len() {
            return Err("invalid MP4 sample NAL length".to_string());
        }
        output.extend_from_slice(&[0, 0, 0, 1]);
        output.extend_from_slice(&data[cursor..cursor + length]);
        cursor += length;
    }
    if output.is_empty() {
        return Err("MP4 sample contains no NAL units".to_string());
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct PacketMeta {
    pts: i64,
    dts: i64,
    pos: u64,
    size: u64,
}

#[derive(Debug, Clone)]
struct CodedPictureMeta {
    pts: i64,
    frame_type: String,
}

#[derive(Debug)]
struct ProbedVideo {
    config: Vec<u8>,
    nal_length_size: usize,
    packets: Vec<PacketMeta>,
    pts_to_ordinal: HashMap<i64, usize>,
    packet_indices: Vec<usize>,
    frame_types: Vec<String>,
}

fn parallel_map<T, F>(items: usize, workers: usize, operation: F) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(usize) -> Result<T, String> + Sync,
{
    let next = AtomicUsize::new(0);
    let results = Mutex::new(
        (0..items)
            .map(|_| None)
            .collect::<Vec<Option<Result<T, String>>>>(),
    );
    std::thread::scope(|scope| {
        for _ in 0..workers.min(items).max(1) {
            let operation = &operation;
            let next = &next;
            let results = &results;
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= items {
                    break;
                }
                results.lock().expect("parallel result mutex poisoned")[index] =
                    Some(operation(index));
            });
        }
    });
    results
        .into_inner()
        .map_err(|_| "parallel result mutex poisoned".to_string())?
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result
                .ok_or_else(|| format!("parallel worker did not produce item {index}"))?
                .map_err(|error| format!("item {index}: {error}"))
        })
        .collect()
}

#[derive(Debug, Clone)]
struct RecordRow {
    tier: i32,
    video_id: String,
    class_name: String,
    record_offset: i64,
    record_length: i64,
    frame_idx: i32,
    codec_config_id: i32,
    dependency_kind: String,
    record_id: i64,
    packet_index: i32,
    gop_id: i32,
    pts: i64,
    dts: i64,
    frame_type: String,
    closure_record_ids: Vec<u8>,
    target_output_ordinal: i32,
    nal_length_size: i32,
}

fn json_i64(value: &serde_json::Value, field: &str) -> Result<i64, String> {
    let value = value
        .get(field)
        .ok_or_else(|| format!("ffprobe row is missing {field}"))?;
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
        .ok_or_else(|| format!("ffprobe field {field} is not an integer"))
}

fn run_json(command: &Path, arguments: &[&str]) -> Result<serde_json::Value, String> {
    let output = Command::new(command)
        .args(arguments)
        .output()
        .map_err(|error| format!("failed to execute {}: {error}", command.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed: {}",
            command.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| format!("invalid ffprobe JSON: {error}"))
}

fn probe_packets(ffprobe: &Path, path: &Path) -> Result<Vec<PacketMeta>, String> {
    let path_text = path.to_string_lossy();
    let value = run_json(
        ffprobe,
        &[
            "-v",
            "error",
            "-nofind_stream_info",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,dts,pos,size",
            "-of",
            "json",
            &path_text,
        ],
    )?;
    value["packets"]
        .as_array()
        .ok_or_else(|| "ffprobe packets output is not an array".to_string())?
        .iter()
        .map(|row| {
            Ok(PacketMeta {
                pts: json_i64(row, "pts")?,
                dts: json_i64(row, "dts")?,
                pos: json_i64(row, "pos")?
                    .try_into()
                    .map_err(|_| "negative packet position".to_string())?,
                size: json_i64(row, "size")?
                    .try_into()
                    .map_err(|_| "negative packet size".to_string())?,
            })
        })
        .collect()
}

#[cfg(feature = "ffmpeg")]
fn parse_coded_pictures(
    path: &Path,
    packets: &[PacketMeta],
    config: &[u8],
    nal_length_size: usize,
) -> Result<Vec<CodedPictureMeta>, String> {
    use ffmpeg_next as ffmpeg;
    use std::ptr;

    struct Parser(*mut ffmpeg::ffi::AVCodecParserContext);
    impl Drop for Parser {
        fn drop(&mut self) {
            unsafe { ffmpeg::ffi::av_parser_close(self.0) };
        }
    }
    ffmpeg::init().map_err(|error| error.to_string())?;
    let parser =
        unsafe { ffmpeg::ffi::av_parser_init(ffmpeg::ffi::AVCodecID::AV_CODEC_ID_H264 as i32) };
    if parser.is_null() {
        return Err("H.264 parser unavailable".to_string());
    }
    let parser = Parser(parser);
    let mut context = ffmpeg::codec::context::Context::new();
    unsafe {
        (*parser.0).flags |= ffmpeg::ffi::PARSER_FLAG_COMPLETE_FRAMES as i32;
        (*context.as_mut_ptr()).codec_id = ffmpeg::ffi::AVCodecID::AV_CODEC_ID_H264;
        (*context.as_mut_ptr()).codec_type = ffmpeg::ffi::AVMediaType::AVMEDIA_TYPE_VIDEO;
    }
    let mut source = File::open(path).map_err(|error| error.to_string())?;
    let file_size = source.metadata().map_err(|error| error.to_string())?.len();
    let mut pictures = Vec::with_capacity(packets.len());
    for (index, packet) in packets.iter().enumerate() {
        if packet
            .pos
            .checked_add(packet.size)
            .is_none_or(|end| end > file_size)
        {
            return Err(format!("packet {index} lies outside the encoded file"));
        }
        let size = usize::try_from(packet.size).map_err(|_| "packet is too large")?;
        i32::try_from(size).map_err(|_| "packet exceeds the parser input limit")?;
        let mut sample = vec![0; size];
        source
            .seek(SeekFrom::Start(packet.pos))
            .map_err(|error| error.to_string())?;
        source
            .read_exact(&mut sample)
            .map_err(|error| error.to_string())?;
        let mut input = if index == 0 {
            config.to_vec()
        } else {
            Vec::new()
        };
        input.extend(mp4_sample_to_annex_b(&sample, nal_length_size)?);
        let input_size = i32::try_from(input.len()).map_err(|_| "parser input is too large")?;
        input.resize(
            input.len() + ffmpeg::ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize,
            0,
        );
        let mut output = ptr::null_mut();
        let mut output_size = 0;
        // The parser reads coded syntax only. No codec is opened and no packet
        // is sent to a decoder; sample identity comes from container timestamps.
        let consumed = unsafe {
            (*parser.0).pict_type = ffmpeg::ffi::AVPictureType::AV_PICTURE_TYPE_NONE as i32;
            ffmpeg::ffi::av_parser_parse2(
                parser.0,
                context.as_mut_ptr(),
                &mut output,
                &mut output_size,
                input.as_ptr(),
                input_size,
                packet.pts,
                packet.dts,
                i64::try_from(packet.pos).map_err(|_| "packet position exceeds i64")?,
            )
        };
        if consumed != input_size || output_size <= 0 || output.is_null() {
            return Err(format!("packet {index} is not one complete coded picture"));
        }
        let picture_type = unsafe { (*parser.0).pict_type };
        let frame_type = match picture_type {
            value if value == ffmpeg::ffi::AVPictureType::AV_PICTURE_TYPE_I as i32 => "I",
            value if value == ffmpeg::ffi::AVPictureType::AV_PICTURE_TYPE_P as i32 => "P",
            value if value == ffmpeg::ffi::AVPictureType::AV_PICTURE_TYPE_B as i32 => "B",
            value => {
                return Err(format!(
                    "unsupported coded picture type {value} in packet {index}"
                ))
            }
        };
        pictures.push(CodedPictureMeta {
            pts: packet.pts,
            frame_type: frame_type.to_string(),
        });
    }
    Ok(pictures)
}

#[cfg(not(feature = "ffmpeg"))]
fn parse_coded_pictures(
    _path: &Path,
    _packets: &[PacketMeta],
    _config: &[u8],
    _nal_length_size: usize,
) -> Result<Vec<CodedPictureMeta>, String> {
    Err("coded-picture parsing requires the ffmpeg build feature".to_string())
}

fn decode_ffprobe_hex_dump(value: &str) -> Result<Vec<u8>, String> {
    let mut result = Vec::new();
    for line in value.lines() {
        let Some((_, rest)) = line.split_once(':') else {
            continue;
        };
        let hex = rest.trim_start().split("  ").next().unwrap_or("");
        for group in hex.split_whitespace() {
            if group.len() % 2 != 0 || !group.bytes().all(|value| value.is_ascii_hexdigit()) {
                return Err("ffprobe extradata contains malformed hex".to_string());
            }
            for index in (0..group.len()).step_by(2) {
                result.push(
                    u8::from_str_radix(&group[index..index + 2], 16)
                        .map_err(|error| format!("invalid ffprobe hex: {error}"))?,
                );
            }
        }
    }
    if result.is_empty() {
        return Err("ffprobe returned empty codec extradata".to_string());
    }
    Ok(result)
}

fn probe_extradata(ffprobe: &Path, path: &Path) -> Result<Vec<u8>, String> {
    let path_text = path.to_string_lossy();
    let value = run_json(
        ffprobe,
        &[
            "-v",
            "error",
            "-nofind_stream_info",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=extradata",
            "-show_data",
            "-of",
            "json",
            &path_text,
        ],
    )?;
    let dump = value["streams"]
        .as_array()
        .and_then(|streams| streams.first())
        .and_then(|stream| stream["extradata"].as_str())
        .ok_or_else(|| "ffprobe did not return codec extradata".to_string())?;
    decode_ffprobe_hex_dump(dump)
}

fn avcc_to_annex_b(data: &[u8]) -> Result<(Vec<u8>, usize), String> {
    if data.len() < 7 || data[0] != 1 {
        return Err("unsupported AVCDecoderConfigurationRecord".to_string());
    }
    let nal_length_size = ((data[4] & 0x03) + 1) as usize;
    let mut cursor = 6usize;
    let mut output = Vec::new();
    let sps_count = (data[5] & 0x1f) as usize;
    for _ in 0..sps_count {
        if cursor + 2 > data.len() {
            return Err("truncated AVCC SPS length".to_string());
        }
        let length = u16::from_be_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;
        if cursor + length > data.len() {
            return Err("truncated AVCC SPS".to_string());
        }
        output.extend_from_slice(&[0, 0, 0, 1]);
        output.extend_from_slice(&data[cursor..cursor + length]);
        cursor += length;
    }
    if cursor >= data.len() {
        return Err("AVCC record has no PPS count".to_string());
    }
    let pps_count = data[cursor] as usize;
    cursor += 1;
    for _ in 0..pps_count {
        if cursor + 2 > data.len() {
            return Err("truncated AVCC PPS length".to_string());
        }
        let length = u16::from_be_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;
        if cursor + length > data.len() {
            return Err("truncated AVCC PPS".to_string());
        }
        output.extend_from_slice(&[0, 0, 0, 1]);
        output.extend_from_slice(&data[cursor..cursor + length]);
        cursor += length;
    }
    if output.is_empty() {
        return Err("AVCC record contains no SPS/PPS".to_string());
    }
    Ok((output, nal_length_size))
}

fn encode_video(
    video: &VideoInput,
    output: &Path,
    options: &HierarchicalBuildOptions,
) -> Result<(), String> {
    let hierarchical_b = options.hierarchical_b;
    let x264_params = format!(
        "b-pyramid={}:open-gop=0:rc-lookahead={}:sync-lookahead=0",
        hierarchical_b.pyramid.x264_value(),
        hierarchical_b.lookahead
    );
    let status = Command::new(&options.ffmpeg_path)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&video.source_path)
        .args(["-an", "-frames:v"])
        .arg(options.max_frames.to_string())
        .args([
            "-vf",
            &format!(
                "scale={}:{},setsar=1,setpts=N/({}*TB)",
                options.width, options.height, options.fps
            ),
        ])
        .args(["-r", &options.fps.to_string(), "-c:v", "libx264", "-preset"])
        .arg(&options.preset)
        .args(["-tune", "zerolatency", "-crf"])
        .arg(options.crf.to_string())
        .args(["-pix_fmt", "yuv420p", "-threads", "1", "-g"])
        .arg(options.gop_size.to_string())
        .args(["-keyint_min"])
        .arg(options.gop_size.to_string())
        .args([
            "-sc_threshold",
            "0",
            "-bf",
            &hierarchical_b.max_b_frames.to_string(),
            "-b_strategy",
            &hierarchical_b.adaptive.to_string(),
            "-refs",
            &hierarchical_b.reference_frames.to_string(),
            "-x264-params",
            &x264_params,
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-movflags",
            "+faststart",
        ])
        .arg(output)
        .status()
        .map_err(|error| format!("failed to execute FFmpeg: {error}"))?;
    if !status.success() {
        return Err(format!("FFmpeg failed for {}", video.source_path.display()));
    }
    Ok(())
}

fn encode_reference_policy_video(
    video: &VideoInput,
    output: &Path,
    options: &HierarchicalBuildOptions,
    shared_anchor: bool,
) -> Result<(), String> {
    let frame_size = options.width as usize * options.height as usize * 3 / 2;
    let mut decode = Command::new(&options.ffmpeg_path);
    decode
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&video.source_path)
        .args(["-an", "-frames:v"])
        .arg(options.max_frames.to_string())
        .args([
            "-vf",
            &format!(
                "scale={}:{},setsar=1,setpts=N/({}*TB)",
                options.width, options.height, options.fps
            ),
            "-r",
            &options.fps.to_string(),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = decode
        .spawn()
        .map_err(|error| format!("failed to start FFmpeg decode: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "FFmpeg raw-video stdout is unavailable".to_string())?;
    let mut source = BufReader::new(stdout);
    let mut encoder = X264Encoder::new_configured(
        options.width as u32,
        options.height as u32,
        options.crf as u32,
        options.gop_size,
        shared_anchor,
        options.fps as u32,
        &options.preset,
    )?;
    let raw_path = output.with_extension("policy.h264");
    let mut raw = File::create(&raw_path)
        .map_err(|error| format!("failed to create {}: {error}", raw_path.display()))?;
    let mut frame = vec![0u8; frame_size];
    let mut frame_index = 0u32;
    loop {
        let mut filled = 0usize;
        while filled < frame_size {
            let count = source
                .read(&mut frame[filled..])
                .map_err(|error| format!("failed to read decoded frame: {error}"))?;
            if count == 0 {
                break;
            }
            filled += count;
        }
        if filled == 0 {
            break;
        }
        if filled != frame_size {
            let _ = std::fs::remove_file(&raw_path);
            return Err(format!(
                "partial decoded frame for {}: {filled}/{frame_size} bytes",
                video.video_id
            ));
        }
        let encoded = encoder
            .try_encode_frame(&frame, frame_index % options.gop_size == 0)
            .map_err(|error| format!("x264 failed at frame {frame_index}: {error}"))?;
        raw.write_all(&encoded)
            .map_err(|error| format!("failed to write raw H.264: {error}"))?;
        frame_index += 1;
    }
    loop {
        let encoded = encoder.try_flush()?;
        if encoded.is_empty() {
            break;
        }
        raw.write_all(&encoded)
            .map_err(|error| format!("failed to flush raw H.264: {error}"))?;
    }
    raw.flush()
        .map_err(|error| format!("failed to flush {}: {error}", raw_path.display()))?;
    let decode_output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for FFmpeg decode: {error}"))?;
    if !decode_output.status.success() {
        let _ = std::fs::remove_file(&raw_path);
        return Err(format!(
            "FFmpeg decode failed for {}: {}",
            video.source_path.display(),
            String::from_utf8_lossy(&decode_output.stderr)
        ));
    }
    if frame_index == 0 {
        let _ = std::fs::remove_file(&raw_path);
        return Err(format!("no source frames decoded for {}", video.video_id));
    }
    let remux = Command::new(&options.ffmpeg_path)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-fflags",
            "+genpts",
            "-r",
        ])
        .arg(options.fps.to_string())
        .args(["-i"])
        .arg(&raw_path)
        .args(["-an", "-c:v", "copy", "-movflags", "+faststart"])
        .arg(output)
        .output()
        .map_err(|error| format!("failed to start FFmpeg remux: {error}"))?;
    let _ = std::fs::remove_file(&raw_path);
    if !remux.status.success() {
        return Err(format!(
            "FFmpeg remux failed for {}: {}",
            video.video_id,
            String::from_utf8_lossy(&remux.stderr)
        ));
    }
    Ok(())
}

fn encode_video_for_policy(
    video: &VideoInput,
    output: &Path,
    options: &HierarchicalBuildOptions,
) -> Result<(), String> {
    match options.dependency_policy {
        DependencyPolicy::HierarchicalB => encode_video(video, output, options),
        DependencyPolicy::ChainedP => encode_reference_policy_video(video, output, options, false),
        DependencyPolicy::SharedAnchor => {
            encode_reference_policy_video(video, output, options, true)
        }
    }
}

fn validate_encoded_frame_count(
    video_id: &str,
    packets: usize,
    frames: usize,
    max_frames: usize,
) -> Result<(), String> {
    if packets == 0 || packets != frames || packets > max_frames {
        return Err(format!(
            "video {video_id} produced packets={packets} frames={frames}, maximum={max_frames}"
        ));
    }
    Ok(())
}

fn probe_video(
    video: &VideoInput,
    encoded_path: &Path,
    options: &HierarchicalBuildOptions,
) -> Result<ProbedVideo, String> {
    let packets = probe_packets(&options.ffprobe_path, encoded_path)?;
    let (config, nal_length_size) =
        avcc_to_annex_b(&probe_extradata(&options.ffprobe_path, encoded_path)?)?;
    let frames = parse_coded_pictures(encoded_path, &packets, &config, nal_length_size)?;
    validate_encoded_frame_count(
        &video.video_id,
        packets.len(),
        frames.len(),
        options.max_frames as usize,
    )?;
    let mut frame_by_pts = HashMap::new();
    for frame in frames {
        if frame_by_pts.insert(frame.pts, frame.frame_type).is_some() {
            return Err(format!("duplicate coded picture PTS {}", frame.pts));
        }
    }
    let mut display_pts = packets.iter().map(|packet| packet.pts).collect::<Vec<_>>();
    display_pts.sort_unstable();
    display_pts.dedup();
    if display_pts.len() != packets.len() {
        return Err(format!("video {} has duplicate packet PTS", video.video_id));
    }
    let pts_to_ordinal = display_pts
        .iter()
        .enumerate()
        .map(|(ordinal, pts)| (*pts, ordinal))
        .collect::<HashMap<_, _>>();
    let mut packet_indices = vec![0usize; packets.len()];
    let mut frame_types = vec![String::new(); packets.len()];
    for (packet_index, packet) in packets.iter().enumerate() {
        let ordinal = pts_to_ordinal[&packet.pts];
        packet_indices[ordinal] = packet_index;
        frame_types[ordinal] = frame_by_pts
            .get(&packet.pts)
            .ok_or_else(|| format!("missing coded picture for PTS {}", packet.pts))?
            .clone();
    }
    Ok(ProbedVideo {
        config,
        nal_length_size,
        packets,
        pts_to_ordinal,
        packet_indices,
        frame_types,
    })
}

fn derive_closures(
    frame_types: &[String],
    packet_indices: &[usize],
) -> Result<(Vec<Vec<usize>>, Vec<usize>), String> {
    let keyframes = frame_types
        .iter()
        .enumerate()
        .filter_map(|(ordinal, value)| (value == "I").then_some(ordinal))
        .collect::<Vec<_>>();
    if keyframes.is_empty() || keyframes[0] != 0 {
        return Err("hierarchical stream does not begin with an I frame".to_string());
    }
    let mut closures = vec![Vec::new(); frame_types.len()];
    let mut gop_ids = vec![0usize; frame_types.len()];
    for (gop_id, start) in keyframes.iter().copied().enumerate() {
        let stop = keyframes
            .get(gop_id + 1)
            .map_or(frame_types.len() - 1, |value| value - 1);
        let references = (start..=stop)
            .filter(|ordinal| matches!(frame_types[*ordinal].as_str(), "I" | "P"))
            .collect::<Vec<_>>();
        if references.first().copied() != Some(start) {
            return Err(format!("GOP {gop_id} has no leading I frame"));
        }
        for target in start..=stop {
            gop_ids[target] = gop_id;
            let mut closure = match frame_types[target].as_str() {
                "I" => vec![target],
                "P" => references
                    .iter()
                    .copied()
                    .filter(|value| *value <= target)
                    .collect(),
                "B" => {
                    let left = references
                        .iter()
                        .copied()
                        .filter(|value| *value < target)
                        .max()
                        .ok_or_else(|| format!("B frame {target} has no left reference"))?;
                    let right = references
                        .iter()
                        .copied()
                        .filter(|value| *value > target)
                        .min()
                        .ok_or_else(|| format!("B frame {target} has no right reference"))?;
                    let reference_b = (left + 1..right)
                        .min_by_key(|ordinal| packet_indices[*ordinal])
                        .ok_or_else(|| format!("B interval {left}:{right} is empty"))?;
                    let mut values = references
                        .iter()
                        .copied()
                        .filter(|value| *value <= right)
                        .collect::<Vec<_>>();
                    if target != reference_b {
                        values.push(reference_b);
                    }
                    values.push(target);
                    values
                }
                value => return Err(format!("unsupported frame type {value}")),
            };
            closure.sort_unstable_by_key(|ordinal| packet_indices[*ordinal]);
            closure.dedup();
            closures[target] = closure;
        }
    }
    Ok((closures, gop_ids))
}

fn derive_shared_anchor_closures(
    frame_types: &[String],
    packet_indices: &[usize],
) -> Result<(Vec<Vec<usize>>, Vec<usize>), String> {
    if frame_types.len() != packet_indices.len() {
        return Err("shared-Anchor closure inputs have different lengths".to_string());
    }
    let anchors = frame_types
        .iter()
        .enumerate()
        .filter_map(|(ordinal, value)| (value == "I").then_some(ordinal))
        .collect::<Vec<_>>();
    if anchors.first().copied() != Some(0) {
        return Err("shared-Anchor stream does not begin with an I frame".to_string());
    }
    if frame_types
        .iter()
        .any(|value| !matches!(value.as_str(), "I" | "P"))
    {
        return Err("shared-Anchor stream must contain only I/P frames".to_string());
    }
    let mut closures = vec![Vec::new(); frame_types.len()];
    let mut gop_ids = vec![0usize; frame_types.len()];
    for (gop_id, start) in anchors.iter().copied().enumerate() {
        let stop = anchors
            .get(gop_id + 1)
            .copied()
            .unwrap_or(frame_types.len());
        for target in start..stop {
            gop_ids[target] = gop_id;
            let mut closure = if target == start {
                vec![start]
            } else {
                vec![start, target]
            };
            closure.sort_unstable_by_key(|ordinal| packet_indices[*ordinal]);
            closures[target] = closure;
        }
    }
    Ok((closures, gop_ids))
}

fn validate_closures(
    closures: &[Vec<usize>],
    gop_ids: &[usize],
    packet_indices: &[usize],
) -> Result<(), String> {
    if closures.len() != gop_ids.len() || closures.len() != packet_indices.len() {
        return Err("closure validation inputs have different lengths".to_string());
    }
    for (target, closure) in closures.iter().enumerate() {
        if closure.is_empty() || !closure.contains(&target) {
            return Err(format!("target {target} is absent from its closure"));
        }
        let mut previous_packet = None;
        for dependency in closure {
            if *dependency >= closures.len() {
                return Err(format!(
                    "target {target} references missing frame {dependency}"
                ));
            }
            if gop_ids[*dependency] != gop_ids[target] {
                return Err(format!("target {target} has a cross-GOP dependency"));
            }
            let packet = packet_indices[*dependency];
            if previous_packet.is_some_and(|value| value >= packet) {
                return Err(format!("target {target} closure is not in decode order"));
            }
            previous_packet = Some(packet);
        }
    }
    Ok(())
}

fn percentile(sorted: &[usize], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[rank.min(sorted.len() - 1)] as f64
}

fn write_index(path: &Path, rows: &[RecordRow]) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("tier", DataType::Int32, false),
        Field::new("video_id", DataType::Utf8, false),
        Field::new("class_name", DataType::Utf8, false),
        Field::new("record_offset", DataType::Int64, false),
        Field::new("record_length", DataType::Int64, false),
        Field::new("frame_idx", DataType::Int32, false),
        Field::new("codec_config_id", DataType::Int32, false),
        Field::new("dependency_kind", DataType::Utf8, false),
        Field::new("record_id", DataType::Int64, false),
        Field::new("packet_index", DataType::Int32, false),
        Field::new("gop_id", DataType::Int32, false),
        Field::new("pts", DataType::Int64, false),
        Field::new("dts", DataType::Int64, false),
        Field::new("frame_type", DataType::Utf8, false),
        Field::new("closure_record_ids", DataType::Binary, false),
        Field::new("target_output_ordinal", DataType::Int32, false),
        Field::new("nal_length_size", DataType::Int32, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.tier),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.video_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.class_name.as_str()),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.record_offset),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.record_length),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.frame_idx),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.codec_config_id),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.dependency_kind.as_str()),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.record_id),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.packet_index),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.gop_id),
        )),
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.pts))),
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.dts))),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.frame_type.as_str()),
        )),
        Arc::new(BinaryArray::from_iter_values(
            rows.iter().map(|row| row.closure_record_ids.as_slice()),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.target_output_ordinal),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.nal_length_size),
        )),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub fn build_vclasp_chunk_internal(
    videos: &[VideoInput],
    options: &HierarchicalBuildOptions,
) -> Result<HierarchicalBuildStats, Box<dyn std::error::Error>> {
    let total_started = Instant::now();
    if videos.is_empty() {
        return Err("video list is empty".into());
    }
    if options.gop_size < 2 || options.max_frames == 0 || options.fps == 0 || options.workers == 0 {
        return Err(
            "gop_size must be at least 2 and frame/fps/worker limits must be positive".into(),
        );
    }
    if options.dependency_policy == DependencyPolicy::HierarchicalB {
        options.hierarchical_b.validate(options.gop_size)?;
    } else if options.hierarchical_b != HierarchicalBConfig::default() {
        return Err(
            "hierarchical-B encoder parameters are only valid for dependency_policy=hierarchical_b"
                .into(),
        );
    }
    if options.dependency_policy == DependencyPolicy::SharedAnchor
        && options.gop_size > MAX_ANCHOR_P_GROUP_FRAMES
    {
        return Err(format!(
            "shared_anchor gop_size {} exceeds x264 reference limit {}",
            options.gop_size, MAX_ANCHOR_P_GROUP_FRAMES
        )
        .into());
    }
    let parent = options
        .output_path
        .parent()
        .ok_or("output path has no parent")?;
    std::fs::create_dir_all(parent)?;
    if let Some(directory) = &options.baseline_mp4_dir {
        std::fs::create_dir_all(directory)?;
    }
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let prefix = format!(".vclasp-{}-{nonce}", std::process::id());
    let payload_path = parent.join(format!("{prefix}.payload"));
    let index_path = parent.join(format!("{prefix}.parquet"));
    let encoded_paths = (0..videos.len())
        .map(|video_index| match &options.baseline_mp4_dir {
            Some(directory) => directory.join(format!("{video_index:06}.mp4")),
            None => parent.join(format!("{prefix}-{video_index}.mp4")),
        })
        .collect::<Vec<_>>();
    let mut payload = File::create(&payload_path)?;
    let mut rows = Vec::new();
    let mut canonical_config = None;
    let mut canonical_nal_length_size = None;
    let mut payload_cursor = 0u64;
    let mut max_closure_records = 0usize;
    let mut closure_lengths = Vec::new();
    let mut gops = 0usize;
    let mut i_frames = 0usize;
    let mut p_frames = 0usize;
    let mut b_frames = 0usize;
    let mut max_b_run = 0usize;
    let mut encode_seconds = 0.0;
    let mut au_parse_seconds = 0.0;
    let mut closure_construction_seconds = 0.0;
    let mut closure_validation_seconds = 0.0;
    let mut payload_copy_seconds = 0.0;
    let mut index_serialization_seconds = 0.0;
    let mut chunk_write_seconds = 0.0;

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let stage_started = Instant::now();
        parallel_map(videos.len(), options.workers, |video_index| {
            encode_video_for_policy(&videos[video_index], &encoded_paths[video_index], options)
        })?;
        encode_seconds = stage_started.elapsed().as_secs_f64();

        let stage_started = Instant::now();
        let probed = parallel_map(videos.len(), options.workers, |video_index| {
            probe_video(&videos[video_index], &encoded_paths[video_index], options)
        })?;
        au_parse_seconds = stage_started.elapsed().as_secs_f64();

        for (video_index, (video, probed)) in videos.iter().zip(probed).enumerate() {
            let encoded_path = &encoded_paths[video_index];
            let ProbedVideo {
                config,
                nal_length_size,
                packets,
                pts_to_ordinal,
                packet_indices,
                frame_types,
            } = probed;
            if canonical_config
                .as_ref()
                .is_some_and(|value| value != &config)
                || canonical_nal_length_size.is_some_and(|value| value != nal_length_size)
            {
                return Err(format!(
                    "encoded video {} does not share the canonical codec configuration",
                    video.video_id
                )
                .into());
            }
            canonical_config.get_or_insert(config);
            canonical_nal_length_size.get_or_insert(nal_length_size);
            let mut current_b_run = 0usize;
            for frame_type in &frame_types {
                match frame_type.as_str() {
                    "I" => {
                        i_frames += 1;
                        current_b_run = 0;
                    }
                    "P" => {
                        p_frames += 1;
                        current_b_run = 0;
                    }
                    "B" => {
                        b_frames += 1;
                        current_b_run += 1;
                        max_b_run = max_b_run.max(current_b_run);
                    }
                    _ => {}
                }
            }
            let stage_started = Instant::now();
            let (closures, gop_ids) = if options.dependency_policy == DependencyPolicy::SharedAnchor
            {
                derive_shared_anchor_closures(&frame_types, &packet_indices)?
            } else {
                derive_closures(&frame_types, &packet_indices)?
            };
            closure_construction_seconds += stage_started.elapsed().as_secs_f64();
            let stage_started = Instant::now();
            validate_closures(&closures, &gop_ids, &packet_indices)?;
            closure_validation_seconds += stage_started.elapsed().as_secs_f64();
            gops += gop_ids.iter().max().map_or(0, |value| value + 1);

            let stage_started = Instant::now();
            let mut source = OpenOptions::new().read(true).open(&encoded_path)?;
            let mut global_record_ids = vec![0i64; packets.len()];
            let base_record_id = rows.len() as i64;
            for ordinal in 0..packets.len() {
                global_record_ids[ordinal] = base_record_id + packet_indices[ordinal] as i64;
            }
            for (packet_index, packet) in packets.iter().enumerate() {
                let ordinal = pts_to_ordinal[&packet.pts];
                source.seek(SeekFrom::Start(packet.pos))?;
                let mut bytes = vec![0u8; packet.size as usize];
                source.read_exact(&mut bytes)?;
                payload.write_all(&bytes)?;
                let closure = &closures[ordinal];
                max_closure_records = max_closure_records.max(closure.len());
                closure_lengths.push(closure.len());
                let mut closure_bytes = Vec::with_capacity(closure.len() * 8);
                for dependency in closure {
                    closure_bytes.extend_from_slice(&global_record_ids[*dependency].to_le_bytes());
                }
                let target_output_ordinal = {
                    let mut display = closure.clone();
                    display.sort_unstable();
                    display
                        .iter()
                        .position(|value| *value == ordinal)
                        .ok_or("target is absent from its closure")? as i32
                };
                rows.push(RecordRow {
                    tier: 2,
                    video_id: video.video_id.clone(),
                    class_name: video.class_name.clone(),
                    record_offset: payload_cursor as i64,
                    record_length: packet.size as i64,
                    frame_idx: ordinal as i32,
                    codec_config_id: 0,
                    dependency_kind: options.dependency_policy.dependency_kind().to_string(),
                    record_id: base_record_id + packet_index as i64,
                    packet_index: packet_index as i32,
                    gop_id: gop_ids[ordinal] as i32,
                    pts: packet.pts,
                    dts: packet.dts,
                    frame_type: frame_types[ordinal].clone(),
                    closure_record_ids: closure_bytes,
                    target_output_ordinal,
                    nal_length_size: nal_length_size as i32,
                });
                payload_cursor += packet.size;
            }
            payload_copy_seconds += stage_started.elapsed().as_secs_f64();
            if options.baseline_mp4_dir.is_none() {
                std::fs::remove_file(encoded_path)?;
            }
        }
        payload.flush()?;
        let stage_started = Instant::now();
        write_index(&index_path, &rows)?;
        index_serialization_seconds += stage_started.elapsed().as_secs_f64();
        let index = std::fs::read(&index_path)?;
        let mut payload_reader = File::open(&payload_path)?;
        let created_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let config = ChunkWriteConfig {
            width: options.width,
            height: options.height,
            fps_num: options.fps,
            fps_den: 1,
            created_at,
            ..ChunkWriteConfig::default()
        };
        let stage_started = Instant::now();
        chunk::write_chunk_from_parts(
            &options.output_path,
            canonical_config
                .as_deref()
                .ok_or("missing codec configuration")?,
            &mut payload_reader,
            payload_cursor,
            &index,
            &config,
        )?;
        chunk_write_seconds += stage_started.elapsed().as_secs_f64();
        Ok(())
    })();

    let index_bytes = std::fs::metadata(&index_path)
        .map(|value| value.len())
        .unwrap_or(0);
    let _ = std::fs::remove_file(&payload_path);
    let _ = std::fs::remove_file(&index_path);
    if options.baseline_mp4_dir.is_none() {
        for encoded_path in &encoded_paths {
            let _ = std::fs::remove_file(encoded_path);
        }
    }
    result?;
    closure_lengths.sort_unstable();
    let mean_closure_records =
        closure_lengths.iter().sum::<usize>() as f64 / closure_lengths.len().max(1) as f64;
    Ok(HierarchicalBuildStats {
        workers: options.workers,
        videos: videos.len(),
        records: rows.len(),
        targets: rows.len(),
        payload_bytes: payload_cursor,
        index_bytes,
        chunk_bytes: std::fs::metadata(&options.output_path)?.len(),
        max_closure_records,
        mean_closure_records,
        median_closure_records: percentile(&closure_lengths, 0.5),
        p95_closure_records: percentile(&closure_lengths, 0.95),
        gops,
        i_frames,
        p_frames,
        b_frames,
        max_b_run,
        total_seconds: total_started.elapsed().as_secs_f64(),
        encode_seconds,
        au_parse_seconds,
        closure_construction_seconds,
        closure_validation_seconds,
        payload_copy_seconds,
        index_serialization_seconds,
        chunk_write_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn coded_picture_parser_matches_separate_decoded_metadata_reference() {
        let directory = tempfile::tempdir().unwrap();
        for b_frames in [0, 7] {
            let path = directory.path().join(format!("b{b_frames}.mp4"));
            let status = Command::new("ffmpeg")
                .args([
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=25",
                    "-frames:v",
                    "35",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-g",
                    "16",
                    "-keyint_min",
                    "16",
                    "-sc_threshold",
                    "0",
                    "-bf",
                ])
                .arg(b_frames.to_string())
                .args([
                    "-refs",
                    "1",
                    "-x264-params",
                    "b-pyramid=strict:b-adapt=0:rc-lookahead=0:open-gop=0",
                ])
                .arg(&path)
                .status()
                .unwrap();
            assert!(status.success());
            let packets = probe_packets(Path::new("ffprobe"), &path).unwrap();
            let (config, length_size) =
                avcc_to_annex_b(&probe_extradata(Path::new("ffprobe"), &path).unwrap()).unwrap();
            let coded = parse_coded_pictures(&path, &packets, &config, length_size).unwrap();
            assert_eq!(coded.len(), 35);
            assert!(coded.iter().any(|picture| picture.frame_type == "I"));
            if b_frames > 0 {
                assert!(coded.iter().any(|picture| picture.frame_type == "B"));
                assert!(packets.windows(2).any(|pair| pair[0].pts > pair[1].pts));
            }
            // This is a separate test-only decoding reference, not ingestion.
            let decoded = run_json(
                Path::new("ffprobe"),
                &[
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_frames",
                    "-show_entries",
                    "frame=pts,pict_type",
                    "-of",
                    "json",
                    path.to_str().unwrap(),
                ],
            )
            .unwrap();
            let reference = decoded["frames"]
                .as_array()
                .unwrap()
                .iter()
                .map(|frame| {
                    (
                        json_i64(frame, "pts").unwrap(),
                        frame["pict_type"].as_str().unwrap().to_string(),
                    )
                })
                .collect::<HashMap<_, _>>();
            assert_eq!(
                coded
                    .into_iter()
                    .map(|picture| (picture.pts, picture.frame_type))
                    .collect::<HashMap<_, _>>(),
                reference
            );
            let mut outside = packets.clone();
            outside[0].pos = u64::MAX;
            assert!(parse_coded_pictures(&path, &outside, &config, length_size)
                .unwrap_err()
                .contains("outside the encoded file"));
        }
    }

    #[test]
    fn parallel_map_preserves_input_order_and_worker_bound() {
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let values = parallel_map(24, 4, |index| {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(current, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis((index % 3) as u64));
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(index * 2)
        })
        .unwrap();
        assert_eq!(values, (0..24).map(|index| index * 2).collect::<Vec<_>>());
        assert!(peak.load(Ordering::SeqCst) <= 4);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn parses_avcc_configuration() {
        let avcc = [
            1, 100, 0, 13, 0xff, 0xe1, 0, 3, 0x67, 0x64, 0, 1, 0, 2, 0x68, 0xef,
        ];
        let (annex_b, length_size) = avcc_to_annex_b(&avcc).unwrap();
        assert_eq!(length_size, 4);
        assert_eq!(annex_b, [0, 0, 0, 1, 0x67, 0x64, 0, 0, 0, 0, 1, 0x68, 0xef]);
    }

    #[test]
    fn derives_hierarchical_closure_in_decode_order() {
        let types = ["I", "B", "B", "B", "P"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let packet_indices = vec![0, 3, 2, 4, 1];
        let (closures, gops) = derive_closures(&types, &packet_indices).unwrap();
        assert_eq!(closures[1], vec![0, 4, 2, 1]);
        assert_eq!(closures[2], vec![0, 4, 2]);
        assert_eq!(gops, vec![0; 5]);
    }

    #[test]
    fn parses_registered_dependency_policies() {
        assert_eq!(
            DependencyPolicy::parse("hierarchical_b").unwrap(),
            DependencyPolicy::HierarchicalB
        );
        assert_eq!(
            DependencyPolicy::parse("chained_p").unwrap(),
            DependencyPolicy::ChainedP
        );
        assert_eq!(
            DependencyPolicy::parse("shared_anchor").unwrap(),
            DependencyPolicy::SharedAnchor
        );
        assert!(DependencyPolicy::parse("uniform").is_err());
    }

    #[test]
    fn validates_hierarchical_b_parameters_before_encoding() {
        HierarchicalBConfig::default().validate(16).unwrap();
        HierarchicalBConfig {
            max_b_frames: 3,
            pyramid: BFramePyramid::Strict,
            adaptive: 2,
            lookahead: 16,
            reference_frames: 3,
        }
        .validate(16)
        .unwrap();
        assert!(HierarchicalBConfig {
            max_b_frames: 16,
            ..HierarchicalBConfig::default()
        }
        .validate(16)
        .is_err());
        assert!(HierarchicalBConfig {
            max_b_frames: 7,
            adaptive: 2,
            lookahead: 7,
            ..HierarchicalBConfig::default()
        }
        .validate(16)
        .is_err());
        assert!(HierarchicalBConfig {
            reference_frames: 0,
            ..HierarchicalBConfig::default()
        }
        .validate(16)
        .is_err());
    }

    #[test]
    fn shared_anchor_closure_contains_only_anchor_and_target() {
        let types = ["I", "P", "P", "P", "I", "P"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let packet_indices = vec![0, 1, 2, 3, 4, 5];
        let (closures, gops) = derive_shared_anchor_closures(&types, &packet_indices).unwrap();
        assert_eq!(
            closures,
            vec![
                vec![0],
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![4],
                vec![4, 5]
            ]
        );
        assert_eq!(gops, vec![0, 0, 0, 0, 1, 1]);
        validate_closures(&closures, &gops, &packet_indices).unwrap();
    }

    #[test]
    fn shared_anchor_closure_rejects_b_frames() {
        let types = ["I", "B", "P"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let error = derive_shared_anchor_closures(&types, &[0, 2, 1]).unwrap_err();
        assert!(error.contains("only I/P"));
    }

    #[test]
    fn validates_closure_membership_gop_and_decode_order() {
        let closures = vec![vec![0], vec![0, 1], vec![0, 1, 2]];
        let gops = vec![0, 0, 0];
        let packets = vec![0, 1, 2];
        validate_closures(&closures, &gops, &packets).unwrap();
        assert!(validate_closures(&[vec![0], vec![0]], &gops[..2], &packets[..2]).is_err());
        assert!(validate_closures(&[vec![0], vec![0, 1]], &[0, 1], &packets[..2]).is_err());
        assert!(validate_closures(&[vec![0], vec![1, 0]], &gops[..2], &packets[..2]).is_err());
    }

    #[test]
    fn converts_mp4_sample_to_annex_b() {
        let sample = [0, 0, 0, 2, 0x65, 0xaa, 0, 0, 0, 1, 0x41];
        assert_eq!(
            mp4_sample_to_annex_b(&sample, 4).unwrap(),
            [0, 0, 0, 1, 0x65, 0xaa, 0, 0, 0, 1, 0x41]
        );
    }

    #[test]
    fn max_frames_is_an_upper_bound_not_an_exact_length() {
        validate_encoded_frame_count("short", 165, 165, 557).unwrap();
        validate_encoded_frame_count("full", 557, 557, 557).unwrap();
        assert!(validate_encoded_frame_count("empty", 0, 0, 557).is_err());
        assert!(validate_encoded_frame_count("mismatch", 165, 164, 557).is_err());
        assert!(validate_encoded_frame_count("overflow", 558, 558, 557).is_err());
    }
}
