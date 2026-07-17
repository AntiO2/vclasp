use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::legacy_chunk_schema;
use crate::index::IndexReader;

pub const FORMAT_MAGIC: &str = "HVS";
pub const FORMAT_VERSION: u16 = 1;
pub const DEFAULT_CODEC: &str = "h264";
pub const DEFAULT_WIDTH: u16 = 320;
pub const DEFAULT_HEIGHT: u16 = 240;
pub const DEFAULT_FPS_NUM: u16 = 25;
pub const DEFAULT_FPS_DEN: u16 = 1;

#[derive(Debug, Clone)]
pub struct ChunkWriteConfig {
    pub codec: String,
    pub width: u16,
    pub height: u16,
    pub fps_num: u16,
    pub fps_den: u16,
    pub created_at: u64,
}

impl Default for ChunkWriteConfig {
    fn default() -> Self {
        Self {
            codec: DEFAULT_CODEC.to_string(),
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            fps_num: DEFAULT_FPS_NUM,
            fps_den: DEFAULT_FPS_DEN,
            created_at: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodecConfig {
    pub codec: String,
    pub width: u16,
    pub height: u16,
    pub fps_num: u8,
    pub fps_den: u8,
    pub sps_pps_offset: u32,
    pub sps_pps_length: u32,
}

#[derive(Debug, Clone)]
pub struct ChunkHeader {
    pub magic: String,
    pub version: u16,
    pub flags: u16,
    pub codec_config: CodecConfig,
    pub payload_length: u64,
    pub index_length: u64,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub struct ChunkLayout {
    pub sps_pps_start: usize,
    pub sps_pps_end: usize,
    pub payload_start: usize,
    pub payload_end: usize,
    pub index_start: usize,
}

pub struct ChunkReader {
    pub file: File,
    pub mmap: Mmap,
    pub header: ChunkHeader,
    pub layout: ChunkLayout,
    pub index: IndexReader,
    rng: fastrand::Rng,
}

impl ChunkReader {
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 4 {
            return Err("chunk is smaller than the 4-byte FlatBuffer length".into());
        }
        let fb_len = u32::from_le_bytes(mmap[0..4].try_into()?) as usize;
        if mmap.len() < 4 + fb_len {
            return Err(format!(
                "chunk is smaller than declared FlatBuffer header: len={}, fb_len={}",
                mmap.len(),
                fb_len
            )
            .into());
        }

        let fb = &mmap[4..4 + fb_len];
        if !flatbuffers::buffer_has_identifier(fb, legacy_chunk_schema::FILE_IDENTIFIER, false) {
            return Err("chunk FlatBuffer header has an invalid file identifier".into());
        }
        let meta = legacy_chunk_schema::root_as_chunk_header(fb);
        validate_header(&meta)?;

        let sps_pps_start = 4 + fb_len;
        let sps_pps_end = sps_pps_start + meta.sps_pps_length() as usize;
        let payload_start = sps_pps_end;
        let payload_end = payload_start + meta.payload_length() as usize;
        let index_start = payload_end;
        let index_end = index_start + meta.index_length() as usize;
        if mmap.len() != index_end {
            return Err(format!(
                "chunk length mismatch: len={}, expected={}",
                mmap.len(),
                index_end
            )
            .into());
        }

        let header = ChunkHeader {
            magic: meta.magic().unwrap_or("").to_string(),
            version: meta.format_version(),
            flags: 0,
            codec_config: CodecConfig {
                codec: meta.codec().unwrap_or("").to_string(),
                width: meta.width(),
                height: meta.height(),
                fps_num: meta.fps_num() as u8,
                fps_den: meta.fps_den() as u8,
                sps_pps_offset: 0,
                sps_pps_length: meta.sps_pps_length() as u32,
            },
            payload_length: meta.payload_length(),
            index_length: meta.index_length(),
            created_at: meta.created_at(),
        };

        let layout = ChunkLayout {
            sps_pps_start,
            sps_pps_end,
            payload_start,
            payload_end,
            index_start,
        };

        let index_data = &mmap[index_start..index_end];
        let index = IndexReader::from_parquet(index_data)?;

        Ok(ChunkReader {
            file,
            mmap,
            header,
            layout,
            index,
            rng: fastrand::Rng::new(),
        })
    }

    pub fn read_sps_pps(&mut self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(self.mmap[self.layout.sps_pps_start..self.layout.sps_pps_end].to_vec())
    }

    pub fn read_bytes(
        &mut self,
        offset: usize,
        length: usize,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let start = self.layout.payload_start + offset;
        let end = start + length;
        if end > self.layout.payload_end {
            return Err(format!(
                "payload range out of bounds: offset={}, length={}, payload_length={}",
                offset, length, self.header.payload_length
            )
            .into());
        }
        Ok(self.mmap[start..end].to_vec())
    }

    pub fn read_record(
        &mut self,
        video_id: &str,
        tier: i32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let (offset, length) = self.index.lookup_one(video_id, tier, &mut self.rng)?;
        self.read_bytes(offset as usize, length as usize)
    }

    pub fn read_all_records(
        &mut self,
        video_id: &str,
        tier: i32,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let offsets = self.index.lookup_all(video_id, tier)?;
        offsets
            .iter()
            .map(|&(off, len)| self.read_bytes(off as usize, len as usize))
            .collect()
    }

    /// Read `count` consecutive records for (video_id, tier) starting at `start_idx` within
    /// the matching record group. Uses `lookup_all` to get the full ordered list, then slices
    /// [start_idx..start_idx+count].
    pub fn read_records_range(
        &mut self,
        video_id: &str,
        tier: i32,
        start_idx: usize,
        count: usize,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let all = self.index.lookup_all(video_id, tier)?;
        if start_idx >= all.len() {
            return Err(format!(
                "range start {} exceeds record count {}",
                start_idx,
                all.len()
            )
            .into());
        }
        let end = (start_idx + count).min(all.len());
        all[start_idx..end]
            .iter()
            .map(|&(off, len)| self.read_bytes(off as usize, len as usize))
            .collect()
    }

    /// Return the number of records for (video_id, tier), or 0 if not found.
    pub fn record_count_for(&self, video_id: &str, tier: i32) -> usize {
        self.index
            .lookup_all(video_id, tier)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Read raw payload bytes at index-relative `offset` for `length` bytes.
    pub fn read_byte_range(
        &mut self,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let start = self.layout.payload_start + offset as usize;
        let end = start + length as usize;
        if end > self.mmap.len() {
            return Err(format!("byte range out of bounds").into());
        }
        Ok(self.mmap[start..end].to_vec())
    }

    /// Read a single deterministic record at index `idx` within the (video_id, tier) group.
    pub fn read_record_at(
        &mut self,
        video_id: &str,
        tier: i32,
        idx: usize,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let records = self.read_records_range(video_id, tier, idx, 1)?;
        records
            .into_iter()
            .next()
            .ok_or_else(|| format!("read_record_at: no record at idx={}", idx).into())
    }

    pub fn batch_lookup(
        &self,
        video_ids: &[String],
        tiers: &[i32],
    ) -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
        self.index.lookup_batch(video_ids, tiers)
    }
}

pub fn write_chunk_from_files(
    output_path: &Path,
    sps_pps_path: &Path,
    payload_path: &Path,
    index_path: &Path,
    created_at: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let sps_pps = std::fs::read(sps_pps_path)?;
    let index = std::fs::read(index_path)?;
    let mut payload = File::open(payload_path)?;
    let payload_len = payload.metadata()?.len();
    let mut config = ChunkWriteConfig::default();
    config.created_at = created_at;

    write_chunk_from_parts(
        output_path,
        &sps_pps,
        &mut payload,
        payload_len,
        &index,
        &config,
    )
}

pub fn write_chunk_from_parts<R: Read>(
    output_path: &Path,
    sps_pps: &[u8],
    payload: &mut R,
    payload_len: u64,
    index: &[u8],
    config: &ChunkWriteConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.codec != DEFAULT_CODEC {
        return Err(format!("unsupported codec for current writer: {}", config.codec).into());
    }
    if sps_pps.is_empty() {
        return Err("SPS/PPS bytes are empty".into());
    }
    if payload_len == 0 {
        return Err("payload is empty".into());
    }
    if index.is_empty() {
        return Err("index bytes are empty".into());
    }

    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let magic = builder.create_string(FORMAT_MAGIC);
    let codec = builder.create_string(&config.codec);
    let header = legacy_chunk_schema::create_chunk_header(
        &mut builder,
        &legacy_chunk_schema::ChunkHeaderArgs {
            magic,
            format_version: FORMAT_VERSION,
            codec,
            width: config.width,
            height: config.height,
            fps_num: config.fps_num,
            fps_den: config.fps_den,
            sps_pps_length: sps_pps.len() as u64,
            payload_length: payload_len,
            index_length: index.len() as u64,
            created_at: config.created_at,
        },
    );
    builder.finish(header, Some(legacy_chunk_schema::FILE_IDENTIFIER));
    let fb = builder.finished_data();

    let mut output = File::create(output_path)?;
    output.write_all(&(fb.len() as u32).to_le_bytes())?;
    output.write_all(fb)?;
    output.write_all(sps_pps)?;
    std::io::copy(payload, &mut output)?;
    output.write_all(index)?;
    output.flush()?;
    Ok(())
}

fn validate_header(header: &legacy_chunk_schema::ChunkHeader<'_>) -> Result<(), Box<dyn std::error::Error>> {
    if header.magic() != Some(FORMAT_MAGIC) {
        return Err(format!(
            "unsupported chunk magic: expected {}, got {}",
            FORMAT_MAGIC,
            header.magic().unwrap_or("<missing>")
        )
        .into());
    }
    if header.format_version() != FORMAT_VERSION {
        return Err(format!(
            "unsupported chunk format_version: expected {}, got {}",
            FORMAT_VERSION,
            header.format_version()
        )
        .into());
    }
    if header.codec() != Some(DEFAULT_CODEC) {
        return Err(format!(
            "unsupported codec: {}",
            header.codec().unwrap_or("<missing>")
        )
        .into());
    }
    if header.sps_pps_length() == 0 {
        return Err("header.sps_pps_length must be non-zero".into());
    }
    if header.payload_length() == 0 {
        return Err("header.payload_length must be non-zero".into());
    }
    if header.index_length() == 0 {
        return Err("header.index_length must be non-zero".into());
    }
    Ok(())
}
