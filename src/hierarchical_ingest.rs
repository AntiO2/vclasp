use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{Array, ArrayRef, BinaryArray, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

use crate::builder::VideoInput;
use crate::chunk::{self, ChunkWriteConfig};
use crate::hierarchical_layout::{
    AccessUnitRecord, GopRegion, HierarchicalLayoutIndex, TargetClosure,
};

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
}

#[derive(Debug, Clone)]
pub struct HierarchicalBuildStats {
    pub videos: usize,
    pub records: usize,
    pub targets: usize,
    pub payload_bytes: u64,
    pub index_bytes: u64,
    pub chunk_bytes: u64,
    pub max_closure_records: usize,
}

#[derive(Debug, Clone)]
pub struct HierarchicalRecordMeta {
    pub record_id: u64,
    pub video_id: String,
    pub frame_idx: i32,
    pub gop_id: u64,
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
        HierarchicalLayoutIndex::new(records, closures, regions)
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
struct FrameMeta {
    pts: i64,
    frame_type: String,
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

fn probe_frames(ffprobe: &Path, path: &Path) -> Result<Vec<FrameMeta>, String> {
    let path_text = path.to_string_lossy();
    let value = run_json(
        ffprobe,
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
            &path_text,
        ],
    )?;
    value["frames"]
        .as_array()
        .ok_or_else(|| "ffprobe frames output is not an array".to_string())?
        .iter()
        .map(|row| {
            let frame_type = row["pict_type"]
                .as_str()
                .ok_or_else(|| "ffprobe frame is missing pict_type".to_string())?;
            if !matches!(frame_type, "I" | "P" | "B") {
                return Err(format!("unsupported decoded frame type {frame_type}"));
            }
            Ok(FrameMeta {
                pts: json_i64(row, "pts")?,
                frame_type: frame_type.to_string(),
            })
        })
        .collect()
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
    let x264_params = "b-pyramid=strict:open-gop=0:rc-lookahead=0:sync-lookahead=0";
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
            "7",
            "-b_strategy",
            "0",
            "-refs",
            "1",
            "-x264-params",
            x264_params,
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

pub fn build_hierarchical_chunk(
    videos: &[VideoInput],
    options: &HierarchicalBuildOptions,
) -> Result<HierarchicalBuildStats, Box<dyn std::error::Error>> {
    if videos.is_empty() {
        return Err("video list is empty".into());
    }
    if options.gop_size < 8 || options.max_frames == 0 || options.fps == 0 {
        return Err("gop_size must be at least 8 and frame/fps limits must be positive".into());
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
    let prefix = format!(".hierarchical-{}-{nonce}", std::process::id());
    let payload_path = parent.join(format!("{prefix}.payload"));
    let index_path = parent.join(format!("{prefix}.parquet"));
    let mut payload = File::create(&payload_path)?;
    let mut rows = Vec::new();
    let mut canonical_config = None;
    let mut canonical_nal_length_size = None;
    let mut payload_cursor = 0u64;
    let mut max_closure_records = 0usize;

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        for (video_index, video) in videos.iter().enumerate() {
            let encoded_path = match &options.baseline_mp4_dir {
                Some(directory) => directory.join(format!("{video_index:06}.mp4")),
                None => parent.join(format!("{prefix}-{video_index}.mp4")),
            };
            encode_video(video, &encoded_path, options)?;
            let packets = probe_packets(&options.ffprobe_path, &encoded_path)?;
            let frames = probe_frames(&options.ffprobe_path, &encoded_path)?;
            validate_encoded_frame_count(
                &video.video_id,
                packets.len(),
                frames.len(),
                options.max_frames as usize,
            )?;
            let (config, nal_length_size) =
                avcc_to_annex_b(&probe_extradata(&options.ffprobe_path, &encoded_path)?)?;
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

            let mut frame_by_pts = HashMap::new();
            for frame in frames {
                if frame_by_pts.insert(frame.pts, frame.frame_type).is_some() {
                    return Err(format!("duplicate decoded PTS {}", frame.pts).into());
                }
            }
            let mut display_pts = packets.iter().map(|packet| packet.pts).collect::<Vec<_>>();
            display_pts.sort_unstable();
            display_pts.dedup();
            if display_pts.len() != packets.len() {
                return Err(format!("video {} has duplicate packet PTS", video.video_id).into());
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
                    .ok_or_else(|| format!("missing decoded frame for PTS {}", packet.pts))?
                    .clone();
            }
            let (closures, gop_ids) = derive_closures(&frame_types, &packet_indices)?;

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
                    dependency_kind: "hierarchical_b".to_string(),
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
            if options.baseline_mp4_dir.is_none() {
                std::fs::remove_file(&encoded_path)?;
            }
        }
        payload.flush()?;
        write_index(&index_path, &rows)?;
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
        Ok(())
    })();

    let index_bytes = std::fs::metadata(&index_path)
        .map(|value| value.len())
        .unwrap_or(0);
    let _ = std::fs::remove_file(&payload_path);
    let _ = std::fs::remove_file(&index_path);
    result?;
    Ok(HierarchicalBuildStats {
        videos: videos.len(),
        records: rows.len(),
        targets: rows.len(),
        payload_bytes: payload_cursor,
        index_bytes,
        chunk_bytes: std::fs::metadata(&options.output_path)?.len(),
        max_closure_records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
