use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use memmap2::Mmap;

use crate::chunk_schema;
use crate::index::IndexReader;

pub const FORMAT_MAGIC: &str = "VClasp";
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
    pub fps_num: u16,
    pub fps_den: u16,
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

pub(crate) fn checked_payload_range(
    payload_start: usize,
    payload_end: usize,
    offset: u64,
    length: u64,
) -> Result<std::ops::Range<usize>, Box<dyn std::error::Error>> {
    let start = payload_start
        .checked_add(usize::try_from(offset)?)
        .ok_or("payload offset overflow")?;
    let end = start
        .checked_add(usize::try_from(length)?)
        .ok_or("payload length overflow")?;
    if payload_start > payload_end || start > payload_end || end > payload_end {
        return Err(
            format!("payload range out of bounds: offset={offset}, length={length}").into(),
        );
    }
    Ok(start..end)
}

pub struct ChunkReader {
    pub file: File,
    pub mmap: Mmap,
    pub header: ChunkHeader,
    pub layout: ChunkLayout,
    index: OnceLock<Result<IndexReader, String>>,
    catalog: OnceLock<Result<Arc<crate::catalog::HierarchicalCatalog>, String>>,
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
        let header_end = 4usize.checked_add(fb_len).ok_or("header length overflow")?;
        if mmap.len() < header_end {
            return Err(format!(
                "chunk is smaller than declared FlatBuffer header: len={}, fb_len={}",
                mmap.len(),
                fb_len
            )
            .into());
        }

        let fb = &mmap[4..header_end];
        if fb.len() < 8
            || !flatbuffers::buffer_has_identifier(fb, chunk_schema::FILE_IDENTIFIER, false)
        {
            return Err("chunk FlatBuffer header has an invalid file identifier".into());
        }
        let meta = chunk_schema::root_as_chunk_header(fb)?;
        validate_header(&meta)?;

        let sps_pps_start = header_end;
        let section_end =
            |start: usize, length: u64| -> Result<usize, Box<dyn std::error::Error>> {
                start
                    .checked_add(usize::try_from(length)?)
                    .ok_or_else(|| "chunk section overflow".into())
            };
        let sps_pps_end = section_end(sps_pps_start, meta.sps_pps_length())?;
        let payload_start = sps_pps_end;
        let payload_end = section_end(payload_start, meta.payload_length())?;
        let index_start = payload_end;
        let index_end = section_end(index_start, meta.index_length())?;
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
                fps_num: meta.fps_num(),
                fps_den: meta.fps_den(),
                sps_pps_offset: 0,
                sps_pps_length: u32::try_from(meta.sps_pps_length())?,
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

        Ok(ChunkReader {
            file,
            mmap,
            header,
            layout,
            index: OnceLock::new(),
            catalog: OnceLock::new(),
            rng: fastrand::Rng::new(),
        })
    }

    /// Parse the production AU catalog once and share it across readers/lanes.
    pub(crate) fn catalog(&self) -> Result<Arc<crate::catalog::HierarchicalCatalog>, String> {
        self.catalog
            .get_or_init(|| {
                crate::catalog::HierarchicalCatalog::from_parquet(
                    &self.mmap[self.layout.index_start..],
                )
                .map_err(|error| error.to_string())
                .and_then(|catalog| {
                    for record in catalog.records() {
                        checked_payload_range(
                            self.layout.payload_start,
                            self.layout.payload_end,
                            record.offset,
                            record.length,
                        )
                        .map_err(|error| error.to_string())?;
                    }
                    Ok(Arc::new(catalog))
                })
            })
            .clone()
    }

    /// Explicit tier-control access; never used to construct production sessions.
    pub fn control_index(&self) -> Result<&IndexReader, Box<dyn std::error::Error>> {
        self.index
            .get_or_init(|| {
                IndexReader::from_parquet(&self.mmap[self.layout.index_start..])
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| error.clone().into())
    }

    pub fn read_sps_pps(&mut self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(self.mmap[self.layout.sps_pps_start..self.layout.sps_pps_end].to_vec())
    }

    pub fn read_bytes(
        &mut self,
        offset: usize,
        length: usize,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        self.read_byte_range(u64::try_from(offset)?, u64::try_from(length)?)
    }

    pub fn read_record(
        &mut self,
        video_id: &str,
        tier: i32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut rng = self.rng.clone();
        let (offset, length) = self.control_index()?.lookup_one(video_id, tier, &mut rng)?;
        self.rng = rng;
        self.read_bytes(offset as usize, length as usize)
    }

    pub fn read_all_records(
        &mut self,
        video_id: &str,
        tier: i32,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let offsets = self.control_index()?.lookup_all(video_id, tier)?;
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
        let all = self.control_index()?.lookup_all(video_id, tier)?;
        if start_idx >= all.len() {
            return Err(format!(
                "range start {} exceeds record count {}",
                start_idx,
                all.len()
            )
            .into());
        }
        let end = start_idx.saturating_add(count).min(all.len());
        all[start_idx..end]
            .iter()
            .map(|&(off, len)| self.read_bytes(off as usize, len as usize))
            .collect()
    }

    /// Return the number of records for (video_id, tier), or 0 if not found.
    pub fn record_count_for(&self, video_id: &str, tier: i32) -> usize {
        self.control_index()
            .and_then(|index| index.lookup_all(video_id, tier))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Read raw payload bytes at index-relative `offset` for `length` bytes.
    pub fn read_byte_range(
        &mut self,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let range = checked_payload_range(
            self.layout.payload_start,
            self.layout.payload_end,
            offset,
            length,
        )?;
        Ok(self.mmap[range].to_vec())
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
        self.control_index()?.lookup_batch(video_ids, tiers)
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
    let header = chunk_schema::create_chunk_header(
        &mut builder,
        &chunk_schema::ChunkHeaderArgs {
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
    builder.finish(header, Some(chunk_schema::FILE_IDENTIFIER));
    let fb = builder.finished_data();

    let parent = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut output = tempfile::NamedTempFile::new_in(parent)?;
    output.write_all(&u32::try_from(fb.len())?.to_le_bytes())?;
    output.write_all(fb)?;
    output.write_all(sps_pps)?;
    let copied = std::io::copy(&mut (&mut *payload).take(payload_len), &mut output)?;
    if copied != payload_len || payload.read(&mut [0u8; 1])? != 0 {
        return Err("payload stream does not match its declared length".into());
    }
    output.write_all(index)?;
    output.flush()?;
    output.persist(output_path)?;
    Ok(())
}

fn validate_header(
    header: &chunk_schema::ChunkHeader<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
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
    if header.width() == 0 || header.height() == 0 || header.fps_num() == 0 || header.fps_den() == 0
    {
        return Err("chunk dimensions and frame-rate components must be positive".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_bytes(magic_value: &str, version: u16, identifier: &str) -> Vec<u8> {
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let magic = builder.create_string(magic_value);
        let codec = builder.create_string(DEFAULT_CODEC);
        let header = chunk_schema::create_chunk_header(
            &mut builder,
            &chunk_schema::ChunkHeaderArgs {
                magic,
                format_version: version,
                codec,
                width: DEFAULT_WIDTH,
                height: DEFAULT_HEIGHT,
                fps_num: DEFAULT_FPS_NUM,
                fps_den: DEFAULT_FPS_DEN,
                sps_pps_length: 1,
                payload_length: 1,
                index_length: 1,
                created_at: 0,
            },
        );
        builder.finish(header, Some(identifier));
        builder.finished_data().to_vec()
    }

    #[test]
    fn current_header_identity_is_explicit() {
        let bytes = header_bytes(FORMAT_MAGIC, FORMAT_VERSION, chunk_schema::FILE_IDENTIFIER);
        assert!(flatbuffers::buffer_has_identifier(
            &bytes,
            chunk_schema::FILE_IDENTIFIER,
            false
        ));
        validate_header(&chunk_schema::root_as_chunk_header(&bytes).unwrap()).unwrap();
        assert_eq!(chunk_schema::FILE_IDENTIFIER, "VCSP");
        assert_eq!(FORMAT_MAGIC, "VClasp");
        assert_eq!(FORMAT_VERSION, 1);
    }

    #[test]
    fn header_rejects_other_identity_or_version() {
        let wrong_identifier = header_bytes(FORMAT_MAGIC, FORMAT_VERSION, "BAD!");
        assert!(!flatbuffers::buffer_has_identifier(
            &wrong_identifier,
            chunk_schema::FILE_IDENTIFIER,
            false
        ));

        let wrong_magic = header_bytes("INVALID", FORMAT_VERSION, chunk_schema::FILE_IDENTIFIER);
        assert!(
            validate_header(&chunk_schema::root_as_chunk_header(&wrong_magic).unwrap()).is_err()
        );

        let wrong_version = header_bytes(
            FORMAT_MAGIC,
            FORMAT_VERSION + 1,
            chunk_schema::FILE_IDENTIFIER,
        );
        assert!(
            validate_header(&chunk_schema::root_as_chunk_header(&wrong_version).unwrap()).is_err()
        );
    }

    #[test]
    fn malformed_headers_are_rejected_without_unchecked_access() {
        for length in 0..8 {
            assert!(chunk_schema::root_as_chunk_header(&vec![0; length]).is_err());
        }
        let mut bytes = header_bytes(FORMAT_MAGIC, FORMAT_VERSION, chunk_schema::FILE_IDENTIFIER);
        bytes[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(chunk_schema::root_as_chunk_header(&bytes).is_err());
    }

    #[test]
    fn payload_bounds_exclude_index_and_reject_overflow() {
        assert_eq!(checked_payload_range(10, 20, 5, 5).unwrap(), 15..20);
        assert!(checked_payload_range(10, 20, 10, 1).is_err());
        assert!(checked_payload_range(10, 20, u64::MAX, 1).is_err());
        assert!(checked_payload_range(10, 20, 1, u64::MAX).is_err());
    }

    #[test]
    fn payload_length_mismatch_preserves_existing_output() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("chunk");
        std::fs::write(&path, b"original").unwrap();
        for bytes in [&b"x"[..], &b"xxx"[..]] {
            let mut payload = bytes;
            assert!(write_chunk_from_parts(
                &path,
                b"sps",
                &mut payload,
                2,
                b"index",
                &ChunkWriteConfig::default()
            )
            .is_err());
            assert_eq!(std::fs::read(&path).unwrap(), b"original");
        }
    }
}
