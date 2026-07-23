use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::encoder::{X264Encoder, MAX_ANCHOR_P_GROUP_FRAMES};

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use crate::chunk::{
    self, ChunkWriteConfig, DEFAULT_CODEC, DEFAULT_FPS_DEN, DEFAULT_FPS_NUM, DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
};

#[derive(Debug, Clone)]
pub struct VideoInput {
    pub video_id: String,
    pub class_name: String,
    pub source_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ChunkBuildOptions {
    pub output_path: PathBuf,
    pub ffmpeg_path: PathBuf,
    /// Per-tier storage policy. Key = tier index (0, 1, 2, ...).
    /// Tier 0 is always forced to IdrOnly regardless of this HashMap.
    pub tier_policies: HashMap<i32, StoragePolicy>,
    pub tier1_stride_idrs: usize,
    pub width: u16,
    pub height: u16,
    pub crf: u8,
    pub preset: String,
    /// Optional per-video frame limit applied before encoding.
    pub max_frames: Option<u32>,
    pub progress_every: usize,
}

#[derive(Debug, Clone)]
pub struct ChunkBuildStats {
    pub videos: usize,
    pub records: usize,
    pub tier0_records: usize,
    pub tier1_records: usize,
    pub tier2_records: usize,
    pub payload_bytes: u64,
    pub index_bytes: u64,
    pub chunk_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct FusedNormalizedBuildStats {
    pub videos: usize,
    pub anchor_groups: usize,
    pub samples: usize,
    pub data_bytes: u64,
    pub index_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct TwoLevelPageBuildStats {
    pub videos: usize,
    pub groups: usize,
    pub pages: usize,
    pub samples: usize,
    pub verified_samples: usize,
    pub data_bytes: u64,
    pub index_bytes: u64,
    pub root_bytes: u64,
    pub checkpoint_bytes: u64,
    pub target_delta_bytes: u64,
    pub max_closure_records: usize,
}

#[derive(Debug, Clone)]
struct FusedNormalizedRow {
    sample_id: i64,
    video_id: i64,
    frame_idx: i32,
    anchor_group_id: i64,
    target_ordinal: i32,
    anchor_offset: i64,
    anchor_length: i64,
    delta_offset: i64,
    delta_length: i64,
}

#[derive(Debug, Clone)]
struct TwoLevelPageRow {
    sample_id: i64,
    video_id: String,
    class_name: String,
    frame_idx: i32,
    group_id: i64,
    page_id: i64,
    target_ordinal: i32,
    root_offset: i64,
    root_length: i64,
    checkpoint_offset: i64,
    checkpoint_length: i64,
    target_offset: i64,
    target_length: i64,
    page_offset: i64,
    page_length: i64,
    closure_records: i32,
}

#[derive(Debug, Clone, Copy)]
enum X264ReferenceMode {
    AnchorP,
    TwoLevel { page_size: u32 },
}

#[derive(Debug, Clone)]
struct IndexRow {
    tier: i32,
    video_id: String,
    class_name: String,
    record_offset: i64,
    record_length: i64,
    frame_idx: i32,
    codec_config_id: i32,
    dependency_kind: String,
}

#[derive(Debug, Clone, Copy)]
enum Section {
    Tier0,
    Tier1,
    Tier2,
}

/// Storage policy: how records are structured bytes-wise.
///
/// Orthogonal to Tier — any Tier can use any policy (except Tier 0 forced IdrOnly).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StoragePolicy {
    /// Each record = one independently decodable IDR frame NAL.
    IdrOnly,
    /// Each record = complete GOP [IDR, P₁, P₂, ..., Pₙ₋₁].
    /// Pₖ references the immediate predecessor (Pₖ₋₁ or IDR).
    FullGop { gop_size: u32 },
    /// Each record = [IDR, P₁, P₂, ..., Pₙ₋₁] with anchor-P dependencies.
    /// ALL Pₖ reference only the IDR anchor (not each other).
    AnchorP { gop_size: u32 },
}

pub fn parse_storage_policy(name: &str, gop_size: u32) -> Result<StoragePolicy, String> {
    match name {
        "idr_only" => Ok(StoragePolicy::IdrOnly),
        "full_gop" => {
            if gop_size == 0 {
                return Err("full_gop requires a positive gop_size".to_string());
            }
            Ok(StoragePolicy::FullGop { gop_size })
        }
        "anchor_p" => {
            validate_anchor_p_group_size(gop_size).map_err(|error| error.to_string())?;
            Ok(StoragePolicy::AnchorP { gop_size })
        }
        _ => Err(format!(
            "unknown storage policy {name}, expected idr_only|full_gop|anchor_p"
        )),
    }
}

impl ChunkBuildOptions {
    pub fn default_for_output(output_path: PathBuf) -> Self {
        let mut tier_policies = HashMap::new();
        tier_policies.insert(0, StoragePolicy::IdrOnly);
        tier_policies.insert(1, StoragePolicy::IdrOnly);
        tier_policies.insert(2, StoragePolicy::IdrOnly);
        Self {
            output_path,
            ffmpeg_path: PathBuf::from("ffmpeg"),
            tier_policies,
            tier1_stride_idrs: 4,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            crf: 23,
            preset: "veryfast".to_string(),
            max_frames: None,
            progress_every: 100,
        }
    }
}

pub fn build_chunk_from_videos(
    videos: &[VideoInput],
    options: &ChunkBuildOptions,
) -> Result<ChunkBuildStats, Box<dyn std::error::Error>> {
    if videos.is_empty() {
        return Err("video list is empty".into());
    }
    if options.tier1_stride_idrs == 0 {
        return Err("tier1_stride_idrs must be positive".into());
    }
    if options.max_frames == Some(0) {
        return Err("max_frames must be positive when specified".into());
    }
    for (tier, policy) in &options.tier_policies {
        match policy {
            StoragePolicy::FullGop { gop_size } | StoragePolicy::AnchorP { gop_size } => {
                if *gop_size == 0 {
                    return Err(format!("tier {} policy has gop_size==0", tier).into());
                }
            }
            _ => {}
        }
        if let StoragePolicy::AnchorP { gop_size } = policy {
            validate_anchor_p_group_size(*gop_size)?;
        }
    }

    let out_dir = options
        .output_path
        .parent()
        .ok_or("output path must have a parent directory")?;
    std::fs::create_dir_all(out_dir)?;

    let tier0_path = options.output_path.with_extension("tier0.tmp");
    let tier1_path = options.output_path.with_extension("tier1.tmp");
    let tier2_path = options.output_path.with_extension("tier2.tmp");
    let payload_path = options.output_path.with_extension("payload.tmp");

    let mut tier0_file = File::create(&tier0_path)?;
    let mut tier1_file = File::create(&tier1_path)?;
    let mut tier2_file = File::create(&tier2_path)?;

    let mut tier0_len = 0u64;
    let mut tier1_len = 0u64;
    let mut tier2_len = 0u64;
    let mut rows: Vec<(Section, IndexRow)> = Vec::new();
    let mut canonical_sps_pps: Option<Vec<u8>> = None;

    let tier1_policy = options
        .tier_policies
        .get(&1)
        .copied()
        .unwrap_or(StoragePolicy::IdrOnly);
    let tier2_policy = options
        .tier_policies
        .get(&2)
        .copied()
        .unwrap_or(StoragePolicy::IdrOnly);
    let (encode_gop, encode_anchor_p) = derive_encode_params(&options.tier_policies);

    for (i, video) in videos.iter().enumerate() {
        let raw = encode_to_annex_b(video, options, encode_gop, encode_anchor_p)?;
        if canonical_sps_pps.is_none() {
            let (sps_pps, _) = extract_sps_pps_and_idrs(&raw)?;
            canonical_sps_pps = Some(sps_pps);
        }
        let (_, gops_t0) = extract_gops(&raw, StoragePolicy::IdrOnly)?;
        let rep_idx = gops_t0.len() / 2;
        let rep_bytes = gops_t0[rep_idx].concat();
        tier0_file.write_all(&rep_bytes)?;
        rows.push((
            Section::Tier0,
            IndexRow {
                tier: 0,
                video_id: video.video_id.clone(),
                class_name: video.class_name.clone(),
                record_offset: tier0_len as i64,
                record_length: rep_bytes.len() as i64,
                frame_idx: (rep_idx as u32 * encode_gop) as i32,
                codec_config_id: 0,
                dependency_kind: "idr".to_string(),
            },
        ));
        tier0_len += rep_bytes.len() as u64;

        let (_, gops_t1) = extract_gops(&raw, tier1_policy)?;
        for (gop_idx, gop) in gops_t1.iter().enumerate() {
            if gop_idx % options.tier1_stride_idrs == 0 {
                let bytes = gop.concat();
                tier1_file.write_all(&bytes)?;
                rows.push((
                    Section::Tier1,
                    IndexRow {
                        tier: 1,
                        video_id: video.video_id.clone(),
                        class_name: video.class_name.clone(),
                        record_offset: tier1_len as i64,
                        record_length: bytes.len() as i64,
                        frame_idx: (gop_idx as u32 * encode_gop) as i32,
                        codec_config_id: 0,
                        dependency_kind: policy_dep_kind(tier1_policy).to_string(),
                    },
                ));
                tier1_len += bytes.len() as u64;
            }
        }

        let (_, gops_t2) = extract_gops(&raw, tier2_policy)?;
        for (gop_idx, gop) in gops_t2.iter().enumerate() {
            let bytes = gop.concat();
            tier2_file.write_all(&bytes)?;
            rows.push((
                Section::Tier2,
                IndexRow {
                    tier: 2,
                    video_id: video.video_id.clone(),
                    class_name: video.class_name.clone(),
                    record_offset: tier2_len as i64,
                    record_length: bytes.len() as i64,
                    frame_idx: (gop_idx as u32 * encode_gop) as i32,
                    codec_config_id: 0,
                    dependency_kind: policy_dep_kind(tier2_policy).to_string(),
                },
            ));
            tier2_len += bytes.len() as u64;
        }

        if options.progress_every > 0
            && ((i + 1) % options.progress_every == 0 || i + 1 == videos.len())
        {
            eprintln!(
                "encoded {}/{} videos, tier0={}MB tier1={}MB tier2={}MB records={}",
                i + 1,
                videos.len(),
                tier0_len / 1024 / 1024,
                tier1_len / 1024 / 1024,
                tier2_len / 1024 / 1024,
                rows.len()
            );
        }
    }

    tier0_file.flush()?;
    tier1_file.flush()?;
    tier2_file.flush()?;
    drop(tier0_file);
    drop(tier1_file);
    drop(tier2_file);

    adjust_offsets(&mut rows, tier0_len, tier1_len);
    rows.sort_by_key(|(_, row)| (row.tier, row.record_offset));
    let final_rows: Vec<IndexRow> = rows.into_iter().map(|(_, row)| row).collect();
    let index_bytes = write_index_to_bytes(&final_rows)?;

    let mut payload_file = File::create(&payload_path)?;
    append_file(&tier0_path, &mut payload_file)?;
    append_file(&tier1_path, &mut payload_file)?;
    append_file(&tier2_path, &mut payload_file)?;
    payload_file.flush()?;
    drop(payload_file);

    let mut payload_reader = File::open(&payload_path)?;
    let payload_len = payload_reader.metadata()?.len();
    payload_reader.seek(SeekFrom::Start(0))?;
    let created_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let config = ChunkWriteConfig {
        codec: DEFAULT_CODEC.to_string(),
        width: options.width,
        height: options.height,
        fps_num: DEFAULT_FPS_NUM,
        fps_den: DEFAULT_FPS_DEN,
        created_at,
    };
    let sps_pps = canonical_sps_pps.ok_or("no SPS/PPS extracted")?;
    chunk::write_chunk_from_parts(
        &options.output_path,
        &sps_pps,
        &mut payload_reader,
        payload_len,
        &index_bytes,
        &config,
    )?;

    let chunk_bytes = options.output_path.metadata()?.len();
    cleanup_tmp(&[tier0_path, tier1_path, tier2_path, payload_path]);

    Ok(ChunkBuildStats {
        videos: videos.len(),
        records: final_rows.len(),
        tier0_records: final_rows.iter().filter(|r| r.tier == 0).count(),
        tier1_records: final_rows.iter().filter(|r| r.tier == 1).count(),
        tier2_records: final_rows.iter().filter(|r| r.tier == 2).count(),
        payload_bytes: payload_len,
        index_bytes: index_bytes.len() as u64,
        chunk_bytes,
    })
}

/// Build an experimental Anchor/Delta layout whose Deltas are produced by one
/// corrected Anchor-P encoder stream. Optional group-Anchor rows make GOP
/// boundaries addressable without storing a zero-length physical Delta.
pub fn build_fused_normalized_layout(
    videos: &[VideoInput],
    data_path: &Path,
    index_path: &Path,
    mut options: ChunkBuildOptions,
    gop_size: u32,
    include_group_anchor_targets: bool,
) -> Result<FusedNormalizedBuildStats, Box<dyn std::error::Error>> {
    if videos.is_empty() {
        return Err("video list is empty".into());
    }
    if gop_size < 2 {
        return Err("fused Normalized gop_size must be at least two".into());
    }
    validate_anchor_p_group_size(gop_size)?;
    if data_path == index_path {
        return Err("data_path and index_path must differ".into());
    }
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    options.output_path = data_path.to_path_buf();

    let mut output = File::create(data_path)?;
    let mut rows = Vec::new();
    let mut data_bytes = 0u64;
    let mut sample_id = 0i64;
    let mut anchor_group_id = 0i64;
    for (video_id, video) in videos.iter().enumerate() {
        let raw = encode_to_annex_b(video, &options, gop_size, true)?;
        let (codec_config, gops) = extract_gops(&raw, StoragePolicy::AnchorP { gop_size })?;
        for (gop_index, gop) in gops.into_iter().enumerate() {
            if gop.is_empty() {
                continue;
            }
            let anchor_offset = data_bytes;
            output.write_all(&codec_config)?;
            output.write_all(&gop[0])?;
            let anchor_length = (codec_config.len() + gop[0].len()) as u64;
            data_bytes += anchor_length;
            if include_group_anchor_targets && gop_index > 0 {
                rows.push(FusedNormalizedRow {
                    sample_id,
                    video_id: video_id as i64,
                    frame_idx: (gop_index as u32 * gop_size) as i32,
                    anchor_group_id,
                    target_ordinal: 0,
                    anchor_offset: anchor_offset as i64,
                    anchor_length: anchor_length as i64,
                    delta_offset: data_bytes as i64,
                    delta_length: 0,
                });
                sample_id += 1;
            }
            for (target_ordinal, delta) in gop.iter().enumerate().skip(1) {
                let delta_offset = data_bytes;
                output.write_all(delta)?;
                data_bytes += delta.len() as u64;
                rows.push(FusedNormalizedRow {
                    sample_id,
                    video_id: video_id as i64,
                    frame_idx: (gop_index as u32 * gop_size + target_ordinal as u32) as i32,
                    anchor_group_id,
                    target_ordinal: target_ordinal as i32,
                    anchor_offset: anchor_offset as i64,
                    anchor_length: anchor_length as i64,
                    delta_offset: delta_offset as i64,
                    delta_length: delta.len() as i64,
                });
                sample_id += 1;
            }
            anchor_group_id += 1;
        }
    }
    output.flush()?;
    let index = write_fused_normalized_index_to_bytes(&rows)?;
    std::fs::write(index_path, &index)?;
    Ok(FusedNormalizedBuildStats {
        videos: videos.len(),
        anchor_groups: anchor_group_id as usize,
        samples: rows.len(),
        data_bytes,
        index_bytes: index.len() as u64,
    })
}

/// Build one-copy two-level reference pages from source videos.
///
/// Each GOP has one root IDR. Every later page starts with a checkpoint P that
/// references the root, while ordinary targets may reference only the root and
/// current checkpoint. Payload bytes stay in display/page order. The index is
/// Ground Truth: it stores the exact root, checkpoint, target, and page extents
/// used by the reader rather than reconstructing them from a GOP constant.
#[cfg(feature = "ffmpeg")]
pub fn build_two_level_page_layout(
    videos: &[VideoInput],
    data_path: &Path,
    index_path: &Path,
    mut options: ChunkBuildOptions,
    gop_size: u32,
    page_size: u32,
) -> Result<TwoLevelPageBuildStats, Box<dyn std::error::Error>> {
    if videos.is_empty() {
        return Err("video list is empty".into());
    }
    if page_size < 2 || gop_size < page_size || !gop_size.is_multiple_of(page_size) {
        return Err("page_size must be at least two and divide gop_size".into());
    }
    if data_path == index_path {
        return Err("data_path and index_path must differ".into());
    }

    #[cfg(vclasp_patched_x264)]
    return build_two_level_stream_layout_impl(
        videos, data_path, index_path, options, gop_size, page_size,
    );

    #[cfg(not(vclasp_patched_x264))]
    {
        if let Some(parent) = data_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(parent) = index_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        options.output_path = data_path.to_path_buf();

        let mut output = File::create(data_path)?;
        let mut rows = Vec::new();
        let mut data_bytes = 0u64;
        let mut root_bytes = 0u64;
        let mut checkpoint_bytes = 0u64;
        let mut target_delta_bytes = 0u64;
        let mut sample_id = 0i64;
        let mut group_id = 0i64;
        let mut page_id = 0i64;
        let mut verified_samples = 0usize;

        for video in videos {
            let frames = decode_raw_video_frames(video, &options)?;
            for (local_group, group_frames) in frames.chunks(gop_size as usize).enumerate() {
                if group_frames.is_empty() {
                    continue;
                }

                struct EncodedPage {
                    checkpoint: usize,
                    stop: usize,
                    codec_config: Vec<u8>,
                    access_units: Vec<Vec<u8>>,
                }

                let checkpoints = if group_frames.len() == 1 {
                    Vec::new()
                } else {
                    std::iter::once(1usize)
                        .chain((page_size as usize..group_frames.len()).step_by(page_size as usize))
                        .collect::<Vec<_>>()
                };
                let mut pages = Vec::with_capacity(checkpoints.len());
                let mut canonical_config: Option<Vec<u8>> = None;
                let mut canonical_root: Option<Vec<u8>> = None;
                for (local_page, checkpoint) in checkpoints.iter().copied().enumerate() {
                    let stop = if local_page == 0 {
                        (page_size as usize).min(group_frames.len())
                    } else {
                        (checkpoint + page_size as usize).min(group_frames.len())
                    };
                    let mut session_frames = Vec::with_capacity(stop - checkpoint + 1);
                    session_frames.push(group_frames[0].as_slice());
                    session_frames.extend(group_frames[checkpoint..stop].iter().map(Vec::as_slice));
                    let raw = encode_page_session_frames(
                        &session_frames,
                        options.width as u32,
                        options.height as u32,
                        options.crf as u32,
                        gop_size,
                    )?;
                    let (codec_config, mut groups) =
                        extract_gops(&raw, StoragePolicy::FullGop { gop_size })?;
                    if groups.len() != 1 || groups[0].len() != session_frames.len() {
                        return Err(format!(
                        "page session checkpoint {checkpoint} produced groups={}/frames={}, expected one/{}",
                        groups.len(),
                        groups.first().map_or(0, Vec::len),
                        session_frames.len()
                    )
                    .into());
                    }
                    let access_units = groups.pop().expect("one page session group");
                    if canonical_config
                        .as_ref()
                        .is_some_and(|value| value != &codec_config)
                    {
                        return Err("page sessions produced different SPS/PPS".into());
                    }
                    if canonical_root
                        .as_ref()
                        .is_some_and(|value| value != &access_units[0])
                    {
                        return Err("page sessions produced different root IDR bytes".into());
                    }
                    canonical_config.get_or_insert_with(|| codec_config.clone());
                    canonical_root.get_or_insert_with(|| access_units[0].clone());
                    verify_page_session(&codec_config, &access_units)?;
                    verified_samples += stop - checkpoint;
                    pages.push(EncodedPage {
                        checkpoint,
                        stop,
                        codec_config,
                        access_units,
                    });
                }

                if pages.is_empty() {
                    let raw = encode_page_session_frames(
                        &[group_frames[0].as_slice()],
                        options.width as u32,
                        options.height as u32,
                        options.crf as u32,
                        gop_size,
                    )?;
                    let (codec_config, mut groups) =
                        extract_gops(&raw, StoragePolicy::FullGop { gop_size })?;
                    let access_units = groups.pop().ok_or("root-only session produced no group")?;
                    canonical_config = Some(codec_config);
                    canonical_root = access_units.first().cloned();
                }

                let codec_config = canonical_config.ok_or("two-level group has no codec config")?;
                let root = canonical_root.ok_or("two-level group has no root IDR")?;
                let root_offset = data_bytes;
                output.write_all(&codec_config)?;
                output.write_all(&root)?;
                let root_length = (codec_config.len() + root.len()) as u64;
                data_bytes += root_length;
                root_bytes += root_length;
                rows.push(TwoLevelPageRow {
                    sample_id,
                    video_id: video.video_id.clone(),
                    class_name: video.class_name.clone(),
                    frame_idx: (local_group as u32 * gop_size) as i32,
                    group_id,
                    page_id: -1,
                    target_ordinal: 0,
                    root_offset: root_offset as i64,
                    root_length: root_length as i64,
                    checkpoint_offset: root_offset as i64,
                    checkpoint_length: root_length as i64,
                    target_offset: root_offset as i64,
                    target_length: root_length as i64,
                    page_offset: root_offset as i64,
                    page_length: root_length as i64,
                    closure_records: 1,
                });
                sample_id += 1;
                verified_samples += 1;

                for page in pages {
                    debug_assert_eq!(page.codec_config, codec_config);
                    let page_offset = data_bytes;
                    let mut page_records = Vec::with_capacity(page.access_units.len() - 1);
                    for (local_ordinal, access_unit) in page.access_units.iter().enumerate().skip(1)
                    {
                        let offset = data_bytes;
                        output.write_all(access_unit)?;
                        let length = access_unit.len() as u64;
                        data_bytes += length;
                        if local_ordinal == 1 {
                            checkpoint_bytes += length;
                        } else {
                            target_delta_bytes += length;
                        }
                        page_records.push((offset, length));
                    }
                    let page_length = data_bytes - page_offset;
                    for target in page.checkpoint..page.stop {
                        let local_ordinal = target - page.checkpoint;
                        let checkpoint_record = page_records[0];
                        let target_record = page_records[local_ordinal];
                        let closure_records = if local_ordinal == 0 { 2 } else { 3 };
                        rows.push(TwoLevelPageRow {
                            sample_id,
                            video_id: video.video_id.clone(),
                            class_name: video.class_name.clone(),
                            frame_idx: (local_group as u32 * gop_size + target as u32) as i32,
                            group_id,
                            page_id,
                            target_ordinal: target as i32,
                            root_offset: root_offset as i64,
                            root_length: root_length as i64,
                            checkpoint_offset: checkpoint_record.0 as i64,
                            checkpoint_length: checkpoint_record.1 as i64,
                            target_offset: target_record.0 as i64,
                            target_length: target_record.1 as i64,
                            page_offset: page_offset as i64,
                            page_length: page_length as i64,
                            closure_records,
                        });
                        sample_id += 1;
                    }
                    page_id += 1;
                }
                group_id += 1;
            }
        }
        output.flush()?;
        let index = write_two_level_page_index_to_bytes(&rows)?;
        std::fs::write(index_path, &index)?;
        Ok(TwoLevelPageBuildStats {
            videos: videos.len(),
            groups: group_id as usize,
            pages: page_id as usize,
            samples: rows.len(),
            verified_samples,
            data_bytes,
            index_bytes: index.len() as u64,
            root_bytes,
            checkpoint_bytes,
            target_delta_bytes,
            max_closure_records: rows
                .iter()
                .map(|row| row.closure_records as usize)
                .max()
                .unwrap_or(0),
        })
    }
}

/// Build a page-addressable dependency graph in one legal H.264 session per
/// root group. Checkpoints are reference P frames; ordinary targets are
/// disposable P frames and therefore never enter the decoder DPB.
#[cfg(all(feature = "ffmpeg", vclasp_patched_x264))]
fn build_two_level_stream_layout_impl(
    videos: &[VideoInput],
    data_path: &Path,
    index_path: &Path,
    mut options: ChunkBuildOptions,
    gop_size: u32,
    page_size: u32,
) -> Result<TwoLevelPageBuildStats, Box<dyn std::error::Error>> {
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    options.output_path = data_path.to_path_buf();

    let mut output = File::create(data_path)?;
    let mut rows = Vec::new();
    let mut data_bytes = 0u64;
    let mut root_bytes = 0u64;
    let mut checkpoint_bytes = 0u64;
    let mut target_delta_bytes = 0u64;
    let mut sample_id = 0i64;
    let mut group_id = 0i64;
    let mut page_id = 0i64;
    let mut verified_samples = 0usize;

    for video in videos {
        let frames = decode_raw_video_frames(video, &options)?;
        for (local_group, group_frames) in frames.chunks(gop_size as usize).enumerate() {
            if group_frames.is_empty() {
                continue;
            }

            let raw = encode_two_level_stream_frames(
                group_frames,
                options.width as u32,
                options.height as u32,
                options.crf as u32,
                gop_size,
                page_size,
            )?;
            let (codec_config, mut groups) =
                extract_gops(&raw, StoragePolicy::FullGop { gop_size })?;
            if groups.len() != 1 || groups[0].len() != group_frames.len() {
                return Err(format!(
                    "two-level stream produced groups={}/frames={}, expected one/{}",
                    groups.len(),
                    groups.first().map_or(0, Vec::len),
                    group_frames.len()
                )
                .into());
            }
            let access_units = groups.pop().expect("one two-level stream group");
            verify_two_level_stream(&codec_config, &access_units, page_size as usize)?;
            verified_samples += access_units.len();

            let mut offsets = Vec::with_capacity(access_units.len());
            for (ordinal, access_unit) in access_units.iter().enumerate() {
                let offset = data_bytes;
                if ordinal == 0 {
                    output.write_all(&codec_config)?;
                    data_bytes += codec_config.len() as u64;
                }
                output.write_all(access_unit)?;
                data_bytes += access_unit.len() as u64;
                let length = access_unit.len() as u64
                    + if ordinal == 0 {
                        codec_config.len() as u64
                    } else {
                        0
                    };
                let record_offset = if ordinal == 0 {
                    offset
                } else {
                    data_bytes - access_unit.len() as u64
                };
                offsets.push((record_offset, length));
                if ordinal == 0 {
                    root_bytes += length;
                } else if ordinal.is_multiple_of(page_size as usize) {
                    checkpoint_bytes += length;
                } else {
                    target_delta_bytes += length;
                }
            }

            let root = offsets[0];
            rows.push(TwoLevelPageRow {
                sample_id,
                video_id: video.video_id.clone(),
                class_name: video.class_name.clone(),
                frame_idx: (local_group as u32 * gop_size) as i32,
                group_id,
                page_id: -1,
                target_ordinal: 0,
                root_offset: root.0 as i64,
                root_length: root.1 as i64,
                checkpoint_offset: root.0 as i64,
                checkpoint_length: root.1 as i64,
                target_offset: root.0 as i64,
                target_length: root.1 as i64,
                page_offset: root.0 as i64,
                page_length: root.1 as i64,
                closure_records: 1,
            });
            sample_id += 1;

            let page_count = access_units.len().div_ceil(page_size as usize);
            let group_page_base = page_id;
            for page in 0..page_count {
                let begin = if page == 0 {
                    1
                } else {
                    page * page_size as usize
                };
                let end = ((page + 1) * page_size as usize).min(access_units.len());
                if begin >= end {
                    continue;
                }
                let page_offset = offsets[begin].0;
                let page_end = offsets[end - 1].0 + offsets[end - 1].1;
                let checkpoint = if page == 0 {
                    0
                } else {
                    page * page_size as usize
                };
                for target in begin..end {
                    let target_record = offsets[target];
                    let checkpoint_record = offsets[checkpoint];
                    rows.push(TwoLevelPageRow {
                        sample_id,
                        video_id: video.video_id.clone(),
                        class_name: video.class_name.clone(),
                        frame_idx: (local_group as u32 * gop_size + target as u32) as i32,
                        group_id,
                        page_id: group_page_base + page as i64,
                        target_ordinal: target as i32,
                        root_offset: root.0 as i64,
                        root_length: root.1 as i64,
                        checkpoint_offset: checkpoint_record.0 as i64,
                        checkpoint_length: checkpoint_record.1 as i64,
                        target_offset: target_record.0 as i64,
                        target_length: target_record.1 as i64,
                        page_offset: page_offset as i64,
                        page_length: (page_end - page_offset) as i64,
                        closure_records: if checkpoint == 0 || target == checkpoint {
                            2
                        } else {
                            3
                        },
                    });
                    sample_id += 1;
                }
            }
            page_id += page_count as i64;
            group_id += 1;
        }
    }

    output.flush()?;
    let index = write_two_level_page_index_to_bytes(&rows)?;
    std::fs::write(index_path, &index)?;
    Ok(TwoLevelPageBuildStats {
        videos: videos.len(),
        groups: group_id as usize,
        pages: page_id as usize,
        samples: rows.len(),
        verified_samples,
        data_bytes,
        index_bytes: index.len() as u64,
        root_bytes,
        checkpoint_bytes,
        target_delta_bytes,
        max_closure_records: rows
            .iter()
            .map(|row| row.closure_records as usize)
            .max()
            .unwrap_or(0),
    })
}

fn encode_to_annex_b(
    video: &VideoInput,
    options: &ChunkBuildOptions,
    gop_size: u32,
    anchor_p: bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if anchor_p {
        return encode_via_x264(video, options, gop_size, X264ReferenceMode::AnchorP);
    }

    let x264_params = "repeat-headers=1:sliced-threads=0".to_string();
    let mut command = Command::new(&options.ffmpeg_path);
    command
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&video.source_path)
        .args([
            "-vf",
            &format!("scale={}:{}", options.width, options.height),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            &options.crf.to_string(),
            "-preset",
            &options.preset,
            "-g",
            &gop_size.to_string(),
            "-keyint_min",
            &gop_size.to_string(),
            "-sc_threshold",
            "0",
            "-x264-params",
            &x264_params,
            "-threads",
            "1",
            "-an",
        ]);
    if let Some(max_frames) = options.max_frames {
        command.args(["-frames:v", &max_frames.to_string()]);
    }
    let output = command.args(["-f", "h264", "pipe:1"]).output()?;

    if !output.status.success() {
        return Err(format!(
            "ffmpeg failed for {}: {}",
            video.source_path.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    if output.stdout.is_empty() {
        return Err(format!(
            "ffmpeg produced no output for {}",
            video.source_path.display()
        )
        .into());
    }
    Ok(output.stdout)
}

fn encode_via_x264(
    video: &VideoInput,
    options: &ChunkBuildOptions,
    gop_size: u32,
    reference_mode: X264ReferenceMode,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if matches!(reference_mode, X264ReferenceMode::AnchorP) {
        validate_anchor_p_group_size(gop_size)?;
    }
    let w = options.width as usize;
    let h = options.height as usize;
    let frame_size = w * h * 3 / 2;

    let mut command = Command::new(&options.ffmpeg_path);
    command
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&video.source_path)
        .args([
            "-vf",
            &format!("scale={}:{}", options.width, options.height),
            "-pix_fmt",
            "yuv420p",
            "-an",
        ]);
    if let Some(max_frames) = options.max_frames {
        command.args(["-frames:v", &max_frames.to_string()]);
    }
    let mut child = command
        .args(["-f", "rawvideo", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdout = child.stdout.take().expect("no stdout pipe");
    let mut encoder = match reference_mode {
        X264ReferenceMode::AnchorP => X264Encoder::new(
            options.width as u32,
            options.height as u32,
            options.crf as u32,
            gop_size,
            true,
        ),
        X264ReferenceMode::TwoLevel { page_size } => X264Encoder::new_two_level(
            options.width as u32,
            options.height as u32,
            options.crf as u32,
            gop_size,
            page_size,
        ),
    };

    let mut encoded = Vec::new();
    let mut frame_buf = vec![0u8; frame_size];
    let mut frame_idx = 0u32;

    loop {
        let mut offset = 0;
        while offset < frame_size {
            let n = stdout.read(&mut frame_buf[offset..])?;
            if n == 0 {
                break;
            }
            offset += n;
        }
        if offset == 0 {
            break;
        }
        if offset < frame_size {
            eprintln!(
                "warning: partial frame {} for {} (got {} bytes, expected {})",
                frame_idx,
                video.source_path.display(),
                offset,
                frame_size,
            );
            break;
        }

        let is_idr = frame_idx % gop_size == 0;
        let data = encoder
            .try_encode_frame(&frame_buf, is_idr)
            .map_err(|error| {
                format!(
                    "{} at source frame {frame_idx}: {error}",
                    video.source_path.display()
                )
            })?;
        if !data.is_empty() {
            encoded.extend_from_slice(&data);
        }
        frame_idx += 1;
    }

    loop {
        let data = encoder.try_flush().map_err(|error| {
            format!(
                "{} while flushing x264: {error}",
                video.source_path.display()
            )
        })?;
        if data.is_empty() {
            break;
        }
        encoded.extend_from_slice(&data);
    }

    let status = child.wait()?;
    if !status.success() {
        return Err(format!(
            "ffmpeg decode failed for {} (exit: {:?})",
            video.source_path.display(),
            status.code(),
        )
        .into());
    }

    if encoded.is_empty() {
        return Err(format!(
            "encode produced no output for {}",
            video.source_path.display()
        )
        .into());
    }

    Ok(encoded)
}

#[cfg(feature = "ffmpeg")]
fn decode_raw_video_frames(
    video: &VideoInput,
    options: &ChunkBuildOptions,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut command = Command::new(&options.ffmpeg_path);
    command
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&video.source_path)
        .args([
            "-vf",
            &format!("scale={}:{}", options.width, options.height),
            "-pix_fmt",
            "yuv420p",
            "-an",
        ]);
    if let Some(max_frames) = options.max_frames {
        command.args(["-frames:v", &max_frames.to_string()]);
    }
    let result = command.args(["-f", "rawvideo", "pipe:1"]).output()?;
    if !result.status.success() {
        return Err(format!(
            "ffmpeg decode failed for {}: {}",
            video.source_path.display(),
            String::from_utf8_lossy(&result.stderr)
        )
        .into());
    }
    let frame_size = options.width as usize * options.height as usize * 3 / 2;
    if result.stdout.is_empty() || result.stdout.len() % frame_size != 0 {
        return Err(format!(
            "raw decode for {} produced {} bytes, not a positive multiple of frame size {}",
            video.source_path.display(),
            result.stdout.len(),
            frame_size
        )
        .into());
    }
    Ok(result
        .stdout
        .chunks_exact(frame_size)
        .map(<[u8]>::to_vec)
        .collect())
}

#[cfg(feature = "ffmpeg")]
fn encode_page_session_frames(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    crf: u32,
    keyint: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if frames.is_empty() {
        return Err("page session has no frames".into());
    }
    let expected = width as usize * height as usize * 3 / 2;
    if frames.iter().any(|frame| frame.len() != expected) {
        return Err("page session contains an incorrectly sized YUV frame".into());
    }
    let mut encoder = X264Encoder::new_page_session(width, height, crf, keyint);
    let mut encoded = Vec::new();
    for (ordinal, frame) in frames.iter().enumerate() {
        encoded.extend_from_slice(&encoder.try_encode_frame(frame, ordinal == 0)?);
    }
    loop {
        let bytes = encoder.try_flush()?;
        if bytes.is_empty() {
            break;
        }
        encoded.extend_from_slice(&bytes);
    }
    if encoded.is_empty() {
        return Err("page session encoder produced no bytes".into());
    }
    Ok(encoded)
}

#[cfg(all(feature = "ffmpeg", vclasp_patched_x264))]
fn encode_two_level_stream_frames(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    crf: u32,
    gop_size: u32,
    page_size: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if frames.is_empty() {
        return Err("two-level stream has no frames".into());
    }
    let expected = width as usize * height as usize * 3 / 2;
    if frames.iter().any(|frame| frame.len() != expected) {
        return Err("two-level stream contains an incorrectly sized YUV frame".into());
    }
    let mut encoder = X264Encoder::new_two_level(width, height, crf, gop_size, page_size);
    let mut encoded = Vec::new();
    for (ordinal, frame) in frames.iter().enumerate() {
        encoded.extend_from_slice(&encoder.try_encode_frame(frame, ordinal == 0)?);
    }
    loop {
        let bytes = encoder.try_flush()?;
        if bytes.is_empty() {
            break;
        }
        encoded.extend_from_slice(&bytes);
    }
    if encoded.is_empty() {
        return Err("two-level stream encoder produced no bytes".into());
    }
    Ok(encoded)
}

fn validate_anchor_p_group_size(gop_size: u32) -> Result<(), Box<dyn std::error::Error>> {
    if gop_size == 0 || gop_size > MAX_ANCHOR_P_GROUP_FRAMES {
        return Err(format!(
            "Anchor-P group size {gop_size} is outside the supported range 1..={MAX_ANCHOR_P_GROUP_FRAMES}"
        )
        .into());
    }
    Ok(())
}

#[cfg(feature = "ffmpeg")]
fn verify_page_session(
    codec_config: &[u8],
    group: &[Vec<u8>],
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::decoder::{decode_gop_rgb24, DecoderConfig, DecoderPool};

    if group.is_empty() {
        return Err("page-session verification requires a non-empty group".into());
    }
    let full_record = group.concat();
    let mut full_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
    let full = decode_gop_rgb24(codec_config, &full_record, &mut full_pool)?;
    if full.len() != group.len() {
        return Err(format!(
            "two-level full stream decoded {} frames for {} access units",
            full.len(),
            group.len()
        )
        .into());
    }

    for target in 0..group.len() {
        let checkpoint = usize::from(target > 0);
        let mut closure = vec![0usize, checkpoint, target];
        closure.sort_unstable();
        closure.dedup();
        let selective = closure
            .iter()
            .flat_map(|ordinal| group[*ordinal].iter().copied())
            .collect::<Vec<_>>();
        let mut pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let decoded = decode_gop_rgb24(codec_config, &selective, &mut pool).map_err(|error| {
            format!(
                "two-level closure {:?} failed for target {target}: {error}",
                closure
            )
        })?;
        if decoded.len() != closure.len() {
            return Err(format!(
                "two-level target {target} decoded {} frames for closure {:?}",
                decoded.len(),
                closure
            )
            .into());
        }
        let selected = &decoded
            .last()
            .ok_or("two-level selective decode returned no frame")?
            .data;
        let expected = &full[target].data;
        if selected.len() != expected.len() {
            return Err(format!("two-level target {target} RGB size mismatch").into());
        }
        let mae = selected
            .iter()
            .zip(expected)
            .map(|(&left, &right)| (left as f64 - right as f64).abs())
            .sum::<f64>()
            / selected.len() as f64;
        if mae > 0.01 {
            return Err(format!(
                "two-level target {target} differs from the complete stream: MAE={mae}"
            )
            .into());
        }
    }
    Ok(())
}

#[cfg(all(feature = "ffmpeg", vclasp_patched_x264))]
fn verify_two_level_stream(
    codec_config: &[u8],
    group: &[Vec<u8>],
    page_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::decoder::{decode_gop_rgb24, DecoderConfig, DecoderPool};

    if group.is_empty() {
        return Err("two-level stream verification requires a non-empty group".into());
    }
    let full_record = group.concat();
    let mut full_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
    let full = decode_gop_rgb24(codec_config, &full_record, &mut full_pool)?;
    if full.len() != group.len() {
        return Err(format!(
            "two-level stream decoded {} frames for {} access units",
            full.len(),
            group.len()
        )
        .into());
    }

    for target in 0..group.len() {
        let checkpoint = target / page_size * page_size;
        let mut closure = vec![0usize, checkpoint, target];
        closure.sort_unstable();
        closure.dedup();
        let selective = closure
            .iter()
            .flat_map(|ordinal| group[*ordinal].iter().copied())
            .collect::<Vec<_>>();
        let mut pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let decoded = decode_gop_rgb24(codec_config, &selective, &mut pool).map_err(|error| {
            format!(
                "two-level stream closure {:?} failed for target {target}: {error}",
                closure
            )
        })?;
        if decoded.len() != closure.len() {
            return Err(format!(
                "two-level stream target {target} decoded {} frames for closure {:?}",
                decoded.len(),
                closure
            )
            .into());
        }
        let selected = &decoded
            .last()
            .ok_or("selective decode returned no frame")?
            .data;
        let expected = &full[target].data;
        let mae = selected
            .iter()
            .zip(expected)
            .map(|(&left, &right)| (left as f64 - right as f64).abs())
            .sum::<f64>()
            / selected.len() as f64;
        if mae > 0.01 {
            return Err(format!(
                "two-level stream target {target} differs from full decode: MAE={mae}"
            )
            .into());
        }
    }
    Ok(())
}

fn derive_encode_params(tier_policies: &HashMap<i32, StoragePolicy>) -> (u32, bool) {
    let mut max_gop = 8u32;
    let mut anchor_p = false;
    for (_, policy) in tier_policies {
        match policy {
            StoragePolicy::IdrOnly => {}
            StoragePolicy::FullGop { gop_size } if *gop_size > 0 => {
                max_gop = max_gop.max(*gop_size);
            }
            StoragePolicy::AnchorP { gop_size } if *gop_size > 0 => {
                max_gop = max_gop.max(*gop_size);
                anchor_p = true;
            }
            _ => {}
        }
    }
    (max_gop, anchor_p)
}

fn policy_dep_kind(policy: StoragePolicy) -> &'static str {
    match policy {
        StoragePolicy::IdrOnly => "idr",
        StoragePolicy::FullGop { .. } => "gop",
        StoragePolicy::AnchorP { .. } => "anchor_p",
    }
}

/// Extract GOPs (groups of NALs) from an Annex-B encoded stream according to policy.
///
/// For IdrOnly: each GOP is a single IDR NAL (non-IDR frames discarded).
/// For FullGop/AnchorP: each GOP groups the IDR with following P-frame NALs.
fn extract_gops(
    data: &[u8],
    policy: StoragePolicy,
) -> Result<(Vec<u8>, Vec<Vec<Vec<u8>>>), Box<dyn std::error::Error>> {
    let nals = split_annex_b_nals(data);

    let mut sps: Option<Vec<u8>> = None;
    let mut pps: Option<Vec<u8>> = None;
    let mut gops: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut current: Vec<Vec<u8>> = Vec::new();

    for (nal_type, nal) in nals {
        match nal_type {
            7 if sps.is_none() => sps = Some(nal),
            8 if pps.is_none() => pps = Some(nal),
            5 => {
                if !current.is_empty() {
                    gops.push(std::mem::take(&mut current));
                }
                current.push(nal);
            }
            1 => {
                match policy {
                    StoragePolicy::IdrOnly => { /* discard P frames */ }
                    StoragePolicy::FullGop { .. } | StoragePolicy::AnchorP { .. } => {
                        current.push(nal);
                    }
                }
            }
            _ => {}
        }
    }
    if !current.is_empty() {
        gops.push(current);
    }

    let mut sps_pps = sps.ok_or("encoded stream did not contain SPS")?;
    let pps = pps.ok_or("encoded stream did not contain PPS")?;
    sps_pps.extend_from_slice(&pps);

    if gops.is_empty() {
        return Err("encoded stream did not contain any IDR access units".into());
    }

    Ok((sps_pps, gops))
}

fn start_code_len(data: &[u8], pos: usize) -> usize {
    if pos + 4 <= data.len() && &data[pos..pos + 4] == b"\x00\x00\x00\x01" {
        4
    } else if pos + 3 <= data.len() && &data[pos..pos + 3] == b"\x00\x00\x01" {
        3
    } else {
        0
    }
}

fn split_annex_b_nals(data: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 < data.len() {
        let n = start_code_len(data, i);
        if n > 0 {
            starts.push((i, n));
            i += n;
        } else {
            i += 1;
        }
    }

    let mut out = Vec::new();
    for (idx, (sc_start, sc_len)) in starts.iter().copied().enumerate() {
        let nal_start = sc_start + sc_len;
        let nal_end = starts.get(idx + 1).map(|(p, _)| *p).unwrap_or(data.len());
        if nal_start >= nal_end {
            continue;
        }
        let nal_type = data[nal_start] & 0x1f;
        out.push((nal_type, data[sc_start..nal_end].to_vec()));
    }
    out
}

fn extract_sps_pps_and_idrs(
    data: &[u8],
) -> Result<(Vec<u8>, Vec<Vec<u8>>), Box<dyn std::error::Error>> {
    let mut sps: Option<Vec<u8>> = None;
    let mut pps: Option<Vec<u8>> = None;
    let mut idrs = Vec::new();
    for (nal_type, nal) in split_annex_b_nals(data) {
        match nal_type {
            7 if sps.is_none() => sps = Some(nal),
            8 if pps.is_none() => pps = Some(nal),
            5 => idrs.push(nal),
            _ => {}
        }
    }
    let mut sps_pps = sps.ok_or("encoded stream did not contain SPS")?;
    let pps = pps.ok_or("encoded stream did not contain PPS")?;
    if idrs.is_empty() {
        return Err("encoded stream did not contain IDR access units".into());
    }
    sps_pps.extend_from_slice(&pps);
    Ok((sps_pps, idrs))
}

fn adjust_offsets(rows: &mut [(Section, IndexRow)], tier0_len: u64, tier1_len: u64) {
    for (section, row) in rows.iter_mut() {
        let base = match section {
            Section::Tier0 => 0,
            Section::Tier1 => tier0_len,
            Section::Tier2 => tier0_len + tier1_len,
        };
        row.record_offset += base as i64;
    }
}

fn write_index_to_bytes(rows: &[IndexRow]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("tier", DataType::Int32, false),
        Field::new("video_id", DataType::Utf8, false),
        Field::new("class_name", DataType::Utf8, false),
        Field::new("record_offset", DataType::Int64, false),
        Field::new("record_length", DataType::Int64, false),
        Field::new("frame_idx", DataType::Int32, false),
        Field::new("codec_config_id", DataType::Int32, false),
        Field::new("dependency_kind", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.tier).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.video_id.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| r.class_name.as_str())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.record_offset).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.record_length).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.frame_idx).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.codec_config_id).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| r.dependency_kind.as_str())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;

    let mut writer = ArrowWriter::try_new(Vec::new(), schema, None)?;
    writer.write(&batch)?;
    Ok(writer.into_inner()?)
}

fn write_fused_normalized_index_to_bytes(
    rows: &[FusedNormalizedRow],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_id", DataType::Int64, false),
        Field::new("video_id", DataType::Int64, false),
        Field::new("frame_idx", DataType::Int32, false),
        Field::new("anchor_group_id", DataType::Int64, false),
        Field::new("target_ordinal", DataType::Int32, false),
        Field::new("anchor_offset", DataType::Int64, false),
        Field::new("anchor_length", DataType::Int64, false),
        Field::new("delta_offset", DataType::Int64, false),
        Field::new("delta_length", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.sample_id).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.video_id).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int32Array::from(
                rows.iter().map(|row| row.frame_idx).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.anchor_group_id)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int32Array::from(
                rows.iter()
                    .map(|row| row.target_ordinal)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.anchor_offset).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.anchor_length).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.delta_offset).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.delta_length).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;
    let mut writer = ArrowWriter::try_new(Vec::new(), schema, None)?;
    writer.write(&batch)?;
    Ok(writer.into_inner()?)
}

fn write_two_level_page_index_to_bytes(
    rows: &[TwoLevelPageRow],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_id", DataType::Int64, false),
        Field::new("video_id", DataType::Utf8, false),
        Field::new("class_name", DataType::Utf8, false),
        Field::new("frame_idx", DataType::Int32, false),
        Field::new("group_id", DataType::Int64, false),
        Field::new("page_id", DataType::Int64, false),
        Field::new("target_ordinal", DataType::Int32, false),
        Field::new("root_offset", DataType::Int64, false),
        Field::new("root_length", DataType::Int64, false),
        Field::new("checkpoint_offset", DataType::Int64, false),
        Field::new("checkpoint_length", DataType::Int64, false),
        Field::new("target_offset", DataType::Int64, false),
        Field::new("target_length", DataType::Int64, false),
        Field::new("page_offset", DataType::Int64, false),
        Field::new("page_length", DataType::Int64, false),
        Field::new("closure_records", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.sample_id).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.video_id.as_str())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.class_name.as_str())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int32Array::from(
                rows.iter().map(|row| row.frame_idx).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.group_id).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.page_id).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int32Array::from(
                rows.iter()
                    .map(|row| row.target_ordinal)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.root_offset).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.root_length).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.checkpoint_offset)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.checkpoint_length)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.target_offset).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.target_length).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.page_offset).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.page_length).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int32Array::from(
                rows.iter()
                    .map(|row| row.closure_records)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;
    let mut writer = ArrowWriter::try_new(Vec::new(), schema, None)?;
    writer.write(&batch)?;
    Ok(writer.into_inner()?)
}

fn append_file(src: &Path, dst: &mut File) -> Result<(), Box<dyn std::error::Error>> {
    let mut input = File::open(src)?;
    std::io::copy(&mut input, dst)?;
    Ok(())
}

fn cleanup_tmp(paths: &[PathBuf]) {
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ffmpeg")]
    fn rgb_mae(lhs: &[u8], rhs: &[u8]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter()
            .zip(rhs)
            .map(|(&left, &right)| (left as f64 - right as f64).abs())
            .sum::<f64>()
            / lhs.len() as f64
    }

    fn annex_b_stream(nals: &[(u8, usize)]) -> Vec<u8> {
        let mut out = Vec::new();
        for &(nal_type, payload_len) in nals {
            out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            out.push(nal_type);
            for _ in 1..payload_len {
                out.push(0);
            }
        }
        out
    }

    #[test]
    fn test_extract_gops_idr_only_discards_p_frames() {
        let data = annex_b_stream(&[(7, 4), (8, 4), (5, 8), (1, 4), (1, 4), (5, 8), (1, 4)]);
        let (sps_pps, gops) = extract_gops(&data, StoragePolicy::IdrOnly).unwrap();
        assert!(sps_pps.len() >= 8);
        assert_eq!(gops.len(), 2);
        assert_eq!(gops[0].len(), 1);
        assert_eq!(gops[1].len(), 1);
        assert_eq!(gops[0][0][4] & 0x1f, 5);
        assert_eq!(gops[1][0][4] & 0x1f, 5);
    }

    #[test]
    fn test_extract_gops_full_gop_keeps_all_frames() {
        let data = annex_b_stream(&[(7, 4), (8, 4), (5, 8), (1, 4), (1, 4), (5, 8), (1, 4)]);
        let (_sps_pps, gops) = extract_gops(&data, StoragePolicy::FullGop { gop_size: 8 }).unwrap();
        assert_eq!(gops.len(), 2);
        assert_eq!(gops[0].len(), 3);
        assert_eq!(gops[1].len(), 2);
        assert_eq!(gops[0][0][4] & 0x1f, 5);
        assert_eq!(gops[0][1][4] & 0x1f, 1);
        assert_eq!(gops[0][2][4] & 0x1f, 1);
    }

    #[test]
    fn test_extract_gops_anchor_p_keeps_all_frames() {
        let data = annex_b_stream(&[(7, 4), (8, 4), (5, 8), (1, 4), (5, 8)]);
        let (_sps_pps, gops) = extract_gops(&data, StoragePolicy::AnchorP { gop_size: 8 }).unwrap();
        assert_eq!(gops.len(), 2);
        assert_eq!(gops[0].len(), 2);
        assert_eq!(gops[1].len(), 1);
    }

    #[test]
    fn test_extract_gops_no_sps_is_error() {
        let data = annex_b_stream(&[(5, 8), (1, 4)]);
        let result = extract_gops(&data, StoragePolicy::IdrOnly);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_gops_no_idr_is_error() {
        let data = annex_b_stream(&[(7, 4), (8, 4), (1, 4)]);
        let result = extract_gops(&data, StoragePolicy::IdrOnly);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_gops_empty_is_error() {
        let data = annex_b_stream(&[]);
        let result = extract_gops(&data, StoragePolicy::FullGop { gop_size: 8 });
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_gops_single_gop() {
        let data = annex_b_stream(&[(7, 4), (8, 4), (5, 8)]);
        let (_sps_pps, gops) = extract_gops(&data, StoragePolicy::IdrOnly).unwrap();
        assert_eq!(gops.len(), 1);
        assert_eq!(gops[0].len(), 1);
    }

    #[test]
    fn test_split_annex_b_nals_empty() {
        let nals = split_annex_b_nals(b"");
        assert!(nals.is_empty());
    }

    #[test]
    fn test_split_annex_b_nals_basic() {
        let data = annex_b_stream(&[(7, 4), (8, 4), (5, 8)]);
        let nals = split_annex_b_nals(&data);
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0].0, 7);
        assert_eq!(nals[1].0, 8);
        assert_eq!(nals[2].0, 5);
    }

    #[test]
    fn test_split_annex_b_nals_detects_3_byte_start_code() {
        let mut data = vec![0x00, 0x00, 0x01, 0x67, 0x00];
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0x68, 0x00]);
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0x65, 0x00]);
        let nals = split_annex_b_nals(&data);
        assert_eq!(nals.len(), 3);
    }

    #[test]
    fn test_derive_encode_params_defaults() {
        let mut map = HashMap::new();
        map.insert(0, StoragePolicy::IdrOnly);
        map.insert(1, StoragePolicy::IdrOnly);
        let (gop, anchor) = derive_encode_params(&map);
        assert_eq!(gop, 8);
        assert!(!anchor);
    }

    #[test]
    fn test_default_build_has_no_frame_limit() {
        let options = ChunkBuildOptions::default_for_output(PathBuf::from("out.chunk"));
        assert_eq!(options.max_frames, None);
    }

    #[test]
    fn test_derive_encode_params_full_gop_sets_gop_size() {
        let mut map = HashMap::new();
        map.insert(2, StoragePolicy::FullGop { gop_size: 16 });
        let (gop, anchor) = derive_encode_params(&map);
        assert_eq!(gop, 16);
        assert!(!anchor);
    }

    #[test]
    fn test_derive_encode_params_anchor_p_sets_both() {
        let mut map = HashMap::new();
        map.insert(2, StoragePolicy::AnchorP { gop_size: 12 });
        let (gop, anchor) = derive_encode_params(&map);
        assert_eq!(gop, 12);
        assert!(anchor);
    }

    #[test]
    fn test_anchor_p_group_limit_is_rejected_before_encoding() {
        assert!(validate_anchor_p_group_size(MAX_ANCHOR_P_GROUP_FRAMES).is_ok());
        let error = validate_anchor_p_group_size(MAX_ANCHOR_P_GROUP_FRAMES + 1)
            .expect_err("stock x264 cannot keep more than sixteen direct references");
        assert!(error.to_string().contains("supported range"));
    }

    #[test]
    fn test_policy_dep_kind() {
        assert_eq!(policy_dep_kind(StoragePolicy::IdrOnly), "idr");
        assert_eq!(
            policy_dep_kind(StoragePolicy::FullGop { gop_size: 8 }),
            "gop"
        );
        assert_eq!(
            policy_dep_kind(StoragePolicy::AnchorP { gop_size: 8 }),
            "anchor_p"
        );
    }

    #[test]
    fn test_parse_storage_policy_enforces_anchor_p_bound() {
        assert_eq!(
            parse_storage_policy("anchor_p", MAX_ANCHOR_P_GROUP_FRAMES).unwrap(),
            StoragePolicy::AnchorP {
                gop_size: MAX_ANCHOR_P_GROUP_FRAMES
            }
        );
        assert!(parse_storage_policy("anchor_p", MAX_ANCHOR_P_GROUP_FRAMES + 1).is_err());
        assert!(parse_storage_policy("unknown", 8).is_err());
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    #[ignore = "requires the local Long-UCF200 real-video fixture"]
    fn test_anchor_p_real_video_selective_dependency() {
        use crate::decoder::{
            decode_closed_targets_continuous_rgb24, decode_gop_rgb24,
            decode_shared_anchor_deltas_rgb24, DecoderConfig, DecoderPool,
        };
        use std::hint::black_box;
        use std::time::Instant;

        let source =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/long_video_200/long_000.mp4");
        assert!(
            source.is_file(),
            "missing real-video fixture {}",
            source.display()
        );
        let video = VideoInput {
            video_id: "long_000".to_string(),
            class_name: "fixture".to_string(),
            source_path: source,
        };
        let mut options = ChunkBuildOptions::default_for_output(
            std::env::temp_dir().join("vclasp-anchor-p-real-video.chunk"),
        );
        options.width = 320;
        options.height = 240;
        options.crf = 23;
        options.max_frames = Some(8);

        let raw = encode_to_annex_b(&video, &options, 8, true)
            .expect("real-video Anchor-P encode must succeed");
        let (codec_config, gops) = extract_gops(&raw, StoragePolicy::AnchorP { gop_size: 8 })
            .expect("real-video Anchor-P stream must parse");
        assert_eq!(gops.len(), 1);
        assert_eq!(gops[0].len(), 8);
        let full_record = gops[0].concat();
        let mut full_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let full = decode_gop_rgb24(&codec_config, &full_record, &mut full_pool)
            .expect("full real-video Anchor-P GOP must decode");
        assert_eq!(full.len(), 8);

        for target in 1..gops[0].len() {
            let selective = [&gops[0][0][..], &gops[0][target][..]].concat();
            let mut pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
            let frames = decode_gop_rgb24(&codec_config, &selective, &mut pool)
                .unwrap_or_else(|error| panic!("real-video I+P{target} failed: {error}"));
            assert_eq!(frames.len(), 2);
            let mae = rgb_mae(&frames[1].data, &full[target].data);
            assert!(mae <= 0.01, "real-video I+P{target} MAE={mae}");
        }

        let fused_targets = [1usize, 3, 7];
        let mut fused = gops[0][0].clone();
        for target in fused_targets {
            fused.extend_from_slice(&gops[0][target]);
        }
        let mut fused_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let fused_frames = decode_gop_rgb24(&codec_config, &fused, &mut fused_pool)
            .expect("real-video fused Anchor-P targets must decode");
        assert_eq!(fused_frames.len(), fused_targets.len() + 1);
        for (output, target) in fused_frames.iter().skip(1).zip(fused_targets) {
            let mae = rgb_mae(&output.data, &full[target].data);
            assert!(mae <= 0.01, "real-video fused P{target} MAE={mae}");
        }

        let mut anchor = codec_config.clone();
        anchor.extend_from_slice(&gops[0][0]);
        let deltas: Vec<Vec<u8>> = fused_targets
            .iter()
            .map(|target| gops[0][*target].clone())
            .collect();
        let closed_records: Vec<Vec<u8>> = deltas
            .iter()
            .map(|delta| [&gops[0][0][..], &delta[..]].concat())
            .collect();
        let mut repeated_ms = Vec::new();
        let mut fused_ms = Vec::new();
        for round in 0..5 {
            let mut repeated_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
            let mut fused_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
            let mut run_repeated = || {
                let started = Instant::now();
                for _ in 0..20 {
                    let frames = decode_closed_targets_continuous_rgb24(
                        &codec_config,
                        &closed_records,
                        &mut repeated_pool,
                        2,
                    )
                    .expect("repeated Anchor+Delta decode must succeed");
                    black_box(frames);
                }
                started.elapsed().as_secs_f64() * 1e3
            };
            let mut run_fused = || {
                let started = Instant::now();
                for _ in 0..20 {
                    let frames =
                        decode_shared_anchor_deltas_rgb24(&anchor, &deltas, &mut fused_pool)
                            .expect("shared-Anchor fusion decode must succeed");
                    black_box(frames);
                }
                started.elapsed().as_secs_f64() * 1e3
            };
            if round % 2 == 0 {
                repeated_ms.push(run_repeated());
                fused_ms.push(run_fused());
            } else {
                fused_ms.push(run_fused());
                repeated_ms.push(run_repeated());
            }
        }
        repeated_ms.sort_by(f64::total_cmp);
        fused_ms.sort_by(f64::total_cmp);
        let repeated_median = repeated_ms[repeated_ms.len() / 2];
        let fused_median = fused_ms[fused_ms.len() / 2];
        eprintln!(
            "ANCHOR_P_FUSION repeated_20x_ms={repeated_median:.3} fused_20x_ms={fused_median:.3} speedup={:.3}",
            repeated_median / fused_median
        );
    }
}
