//! FFmpeg/libavcodec-backed decode path.
//!
//! This module is behind the `ffmpeg` Cargo feature because it requires FFmpeg
//! development libraries (`libavcodec.pc`, `libavutil.pc`, `libswscale.pc`) at
//! build time. The chunk/index reader remains usable without this feature.

use ffmpeg_next as ffmpeg;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::ptr;
use std::sync::{Arc, Mutex};

/// Configuration for the FFmpeg decoder.
///
/// `num_threads` controls how many threads FFmpeg uses for H.264 decoding.
/// `0` (default) lets FFmpeg auto-detect the optimal thread count (typically
/// equal to the number of CPU cores).
#[derive(Clone)]
pub struct DecoderConfig {
    pub num_threads: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self { num_threads: 0 }
    }
}

fn hash_sps_pps(data: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Pool of FFmpeg decoders keyed by SPS/PPS hash.
///
/// Decoders are long-lived so that their internal thread pools stay warm
/// across GOP record boundaries. Call `get_or_create` before each decode
/// session; the returned decoder is flushed (via `avcodec_flush_buffers`)
/// and ready to accept new packets.
pub struct DecoderPool {
    decoders: HashMap<u64, ffmpeg::codec::decoder::Video>,
    lru: VecDeque<u64>,
    capacity: usize,
    config: DecoderConfig,
}

impl DecoderPool {
    pub fn new(config: DecoderConfig) -> Self {
        Self::with_capacity(config, usize::MAX)
    }

    pub fn with_capacity(config: DecoderConfig, capacity: usize) -> Self {
        assert!(capacity > 0, "decoder pool capacity must be positive");
        Self {
            decoders: HashMap::new(),
            lru: VecDeque::new(),
            capacity,
            config,
        }
    }

    pub fn len(&self) -> usize {
        self.decoders.len()
    }

    fn touch(&mut self, key: u64) {
        if let Some(position) = self.lru.iter().position(|value| *value == key) {
            self.lru.remove(position);
        }
        self.lru.push_back(key);
    }

    /// Update the thread count for newly created decoders.
    /// Existing cached decoders keep their original thread count.
    pub fn set_threads(&mut self, num: usize) {
        self.config.num_threads = num;
    }

    /// Retrieve or create a decoder for the given SPS/PPS bytes.
    ///
    /// * **Hit** — the existing decoder is flushed (reset) and returned.
    /// * **Miss** — a new decoder is created with the pool's threading config,
    ///   inserted into the map, flushed, and returned.
    pub fn get_or_create(
        &mut self,
        sps_pps: &[u8],
    ) -> Result<&mut ffmpeg::codec::decoder::Video, Box<dyn std::error::Error>> {
        let key = hash_sps_pps(sps_pps);
        if !self.decoders.contains_key(&key) {
            while self.decoders.len() >= self.capacity {
                if let Some(evicted) = self.lru.pop_front() {
                    self.decoders.remove(&evicted);
                }
            }
            self.decoders.insert(key, create_decoder(&self.config)?);
        }
        self.touch(key);
        let decoder = self.decoders.get_mut(&key).unwrap();
        decoder.flush();
        Ok(decoder)
    }
}

pub(crate) type SharedDecoderSlots = Arc<Vec<Mutex<DecoderPool>>>;

pub(crate) fn shared_decoder_slots(concurrency: usize) -> SharedDecoderSlots {
    Arc::new(
        (0..concurrency.max(1))
            .map(|_| {
                Mutex::new(DecoderPool::with_capacity(
                    DecoderConfig { num_threads: 1 },
                    1,
                ))
            })
            .collect(),
    )
}

/// Create a video decoder for the given codec with threading configured.
fn create_decoder(
    config: &DecoderConfig,
) -> Result<ffmpeg::codec::decoder::Video, Box<dyn std::error::Error>> {
    let codec =
        ffmpeg::codec::decoder::find(ffmpeg::codec::Id::H264).ok_or("H.264 decoder not found")?;
    let mut ctx = ffmpeg::codec::context::Context::new();
    ctx.set_threading(ffmpeg::codec::threading::Config {
        kind: ffmpeg::codec::threading::Type::Frame,
        count: config.num_threads,
    });
    Ok(ctx.decoder().open_as(codec)?.video()?)
}

#[derive(Clone)]
pub struct DecodedRgbFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

struct MemoryReader {
    data: *const u8,
    len: usize,
    position: usize,
}

unsafe extern "C" fn memory_read_packet(
    opaque: *mut std::ffi::c_void,
    buffer: *mut u8,
    buffer_size: i32,
) -> i32 {
    let reader = &mut *(opaque as *mut MemoryReader);
    if reader.position >= reader.len {
        return ffmpeg::Error::Eof.into();
    }
    let available = reader.len - reader.position;
    let count = available.min(buffer_size.max(0) as usize);
    ptr::copy_nonoverlapping(reader.data.add(reader.position), buffer, count);
    reader.position += count;
    count as i32
}

unsafe extern "C" fn memory_seek(opaque: *mut std::ffi::c_void, offset: i64, whence: i32) -> i64 {
    let reader = &mut *(opaque as *mut MemoryReader);
    let invalid_data = i32::from(ffmpeg::Error::InvalidData) as i64;
    if whence == ffmpeg::ffi::AVSEEK_SIZE {
        return reader.len as i64;
    }
    let mode = whence & !ffmpeg::ffi::AVSEEK_FORCE;
    let base = match mode {
        0 => 0i64,
        1 => reader.position as i64,
        2 => reader.len as i64,
        _ => return invalid_data,
    };
    let Some(position) = base.checked_add(offset) else {
        return invalid_data;
    };
    if position < 0 || position > reader.len as i64 {
        return invalid_data;
    }
    reader.position = position as usize;
    position
}

struct CustomIo(*mut ffmpeg::ffi::AVIOContext);

impl Drop for CustomIo {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffmpeg::ffi::avio_context_free(&mut self.0);
            }
        }
    }
}

/// Decode selected display-order frames from one in-memory MP4 segment.
///
/// This is a baseline adapter primitive: the caller chooses the segment and
/// target ordinals. It performs no dependency planning or record coalescing.
pub fn decode_mp4_selected_rgb24(
    data: &[u8],
    target_ordinals: &[usize],
    config: DecoderConfig,
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    use ffmpeg::media::Type;

    if data.is_empty() {
        return Err("MP4 payload is empty".into());
    }
    if target_ordinals.is_empty() {
        return Ok(Vec::new());
    }
    ffmpeg::init()?;

    let mut reader = MemoryReader {
        data: data.as_ptr(),
        len: data.len(),
        position: 0,
    };
    let io_buffer_size = 32 * 1024;
    let io_buffer = unsafe { ffmpeg::ffi::av_malloc(io_buffer_size) as *mut u8 };
    if io_buffer.is_null() {
        return Err("FFmpeg failed to allocate the custom AVIO buffer".into());
    }
    let io_context = unsafe {
        ffmpeg::ffi::avio_alloc_context(
            io_buffer,
            io_buffer_size as i32,
            0,
            (&mut reader as *mut MemoryReader).cast(),
            Some(memory_read_packet),
            None,
            Some(memory_seek),
        )
    };
    if io_context.is_null() {
        unsafe { ffmpeg::ffi::av_free(io_buffer.cast()) };
        return Err("FFmpeg failed to allocate a custom AVIO context".into());
    }
    let _custom_io = CustomIo(io_context);

    let mut format_context = unsafe { ffmpeg::ffi::avformat_alloc_context() };
    if format_context.is_null() {
        return Err("FFmpeg failed to allocate an MP4 format context".into());
    }
    unsafe {
        (*format_context).pb = io_context;
        (*format_context).flags |= ffmpeg::ffi::AVFMT_FLAG_CUSTOM_IO;
    }
    let open_result = unsafe {
        ffmpeg::ffi::avformat_open_input(
            &mut format_context,
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
        )
    };
    if open_result < 0 {
        if !format_context.is_null() {
            unsafe { ffmpeg::ffi::avformat_free_context(format_context) };
        }
        return Err(format!(
            "FFmpeg could not open the in-memory MP4: {}",
            ffmpeg::Error::from(open_result)
        )
        .into());
    }
    let stream_info_result =
        unsafe { ffmpeg::ffi::avformat_find_stream_info(format_context, ptr::null_mut()) };
    if stream_info_result < 0 {
        unsafe { ffmpeg::ffi::avformat_close_input(&mut format_context) };
        return Err(format!(
            "FFmpeg could not read in-memory MP4 stream information: {}",
            ffmpeg::Error::from(stream_info_result)
        )
        .into());
    }

    // Input owns and closes AVFormatContext. CustomIo remains separately owned
    // because AVFMT_FLAG_CUSTOM_IO prevents libavformat from freeing it.
    let mut input = unsafe { ffmpeg::format::context::Input::wrap(format_context) };
    let stream = input
        .streams()
        .best(Type::Video)
        .ok_or("in-memory MP4 contains no video stream")?;
    let stream_index = stream.index();
    let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
    context.set_threading(ffmpeg::codec::threading::Config {
        kind: ffmpeg::codec::threading::Type::Frame,
        count: config.num_threads,
    });
    let mut decoder = context.decoder().video()?;
    let requested = target_ordinals.iter().copied().collect::<HashSet<_>>();
    let mut selected = HashMap::with_capacity(requested.len());
    let mut output_ordinal = 0usize;
    let mut scaler = None;

    for (packet_stream, packet) in input.packets() {
        if packet_stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        loop {
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            if decoder.receive_frame(&mut decoded).is_err() {
                break;
            }
            store_selected(
                &mut scaler,
                &decoded,
                output_ordinal,
                &requested,
                &mut selected,
            )?;
            output_ordinal += 1;
        }
        if selected.len() == requested.len() {
            break;
        }
    }
    if selected.len() != requested.len() {
        decoder.send_eof()?;
        loop {
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            if decoder.receive_frame(&mut decoded).is_err() {
                break;
            }
            store_selected(
                &mut scaler,
                &decoded,
                output_ordinal,
                &requested,
                &mut selected,
            )?;
            output_ordinal += 1;
        }
    }
    target_ordinals
        .iter()
        .map(|ordinal| {
            selected.get(ordinal).cloned().ok_or_else(|| {
                format!("MP4 target ordinal {ordinal} outside decoded frame count {output_ordinal}")
                    .into()
            })
        })
        .collect()
}

/// Persistent decoder state for monotonic reads from one immutable compact
/// Prefix stream. Returned frames are removed immediately; only codec-delayed
/// frames remain buffered, so this is decoder state rather than an RGB cache.
pub struct PrefixCursor {
    decoder: ffmpeg::codec::decoder::Video,
    stream_key: u64,
    next_packet: usize,
    next_output: usize,
    pending: BTreeMap<usize, DecodedRgbFrame>,
    ended: bool,
}

impl PrefixCursor {
    pub fn new(
        sps_pps: &[u8],
        record: &[u8],
        config: DecoderConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        ffmpeg::init()?;
        Ok(Self {
            decoder: create_decoder(&config)?,
            stream_key: prefix_stream_key(sps_pps, record),
            next_packet: 0,
            next_output: 0,
            pending: BTreeMap::new(),
            ended: false,
        })
    }

    fn reset(
        &mut self,
        sps_pps: &[u8],
        record: &[u8],
        config: DecoderConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.decoder = create_decoder(&config)?;
        self.stream_key = prefix_stream_key(sps_pps, record);
        self.next_packet = 0;
        self.next_output = 0;
        self.pending.clear();
        self.ended = false;
        Ok(())
    }

    fn drain(
        &mut self,
        scaler: &mut Option<ffmpeg::software::scaling::context::Context>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let mut frame = ffmpeg::util::frame::video::Video::empty();
            if self.decoder.receive_frame(&mut frame).is_err() {
                break;
            }
            let mut converted = Vec::with_capacity(1);
            store_one(scaler, &frame, &mut converted)?;
            self.pending.insert(
                self.next_output,
                converted.pop().expect("store_one emits one frame"),
            );
            self.next_output += 1;
        }
        Ok(())
    }

    /// Returns `(frames, reset)` in the caller's ordinal order.
    pub fn decode_selected(
        &mut self,
        sps_pps: &[u8],
        record: &[u8],
        ordinals: &[usize],
    ) -> Result<(Vec<DecodedRgbFrame>, bool), Box<dyn std::error::Error>> {
        if ordinals.is_empty() {
            return Ok((Vec::new(), false));
        }
        let access_units = split_annex_b_access_units(record);
        let max_ordinal = *ordinals.iter().max().expect("non-empty ordinals");
        if max_ordinal >= access_units.len() {
            return Err(format!(
                "Prefix target ordinal {max_ordinal} exceeds {} access units",
                access_units.len()
            )
            .into());
        }
        let key = prefix_stream_key(sps_pps, record);
        let must_reset = key != self.stream_key
            || ordinals
                .iter()
                .any(|ordinal| *ordinal < self.next_output && !self.pending.contains_key(ordinal));
        if must_reset {
            self.reset(sps_pps, record, DecoderConfig { num_threads: 1 })?;
        }
        let mut scaler = None;

        while ordinals
            .iter()
            .any(|ordinal| !self.pending.contains_key(ordinal))
        {
            if self.next_packet < access_units.len() {
                let index = self.next_packet;
                let packet_data = if index == 0 {
                    let mut data = Vec::with_capacity(sps_pps.len() + access_units[index].len());
                    data.extend_from_slice(sps_pps);
                    data.extend_from_slice(&access_units[index]);
                    data
                } else {
                    access_units[index].clone()
                };
                let packet = ffmpeg::Packet::copy(&packet_data);
                loop {
                    match self.decoder.send_packet(&packet) {
                        Ok(()) => break,
                        Err(_) => {
                            let before = self.next_output;
                            self.drain(&mut scaler)?;
                            if self.next_output == before {
                                self.decoder.send_packet(&packet)?;
                                break;
                            }
                        }
                    }
                }
                self.next_packet += 1;
                self.drain(&mut scaler)?;
            } else if !self.ended {
                self.decoder.send_eof()?;
                self.ended = true;
                self.drain(&mut scaler)?;
            } else {
                return Err("Prefix cursor ended before all requested frames were decoded".into());
            }
        }

        let frames = ordinals
            .iter()
            .map(|ordinal| {
                self.pending
                    .remove(ordinal)
                    .ok_or_else(|| format!("Prefix cursor lost target ordinal {ordinal}").into())
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        Ok((frames, must_reset))
    }

    /// Predict decoder work without consuming packets or buffered frames.
    pub fn preview_ordinals(&self, ordinals: &[usize]) -> (usize, bool) {
        let Some(max_ordinal) = ordinals.iter().copied().max() else {
            return (0, false);
        };
        let must_reset = ordinals
            .iter()
            .any(|ordinal| *ordinal < self.next_output && !self.pending.contains_key(ordinal));
        if must_reset {
            (max_ordinal + 1, true)
        } else {
            let missing_after_cursor = max_ordinal
                .saturating_add(1)
                .saturating_sub(self.next_packet);
            (missing_after_cursor, false)
        }
    }
}

fn prefix_stream_key(sps_pps: &[u8], record: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    sps_pps.hash(&mut hasher);
    record.len().hash(&mut hasher);
    record
        .get(..record.len().min(256))
        .unwrap_or(record)
        .hash(&mut hasher);
    hasher.finish()
}

/// Decode one H.264 Annex-B access unit or closed decode group to RGB24.
///
/// The caller should pass decoder-ready bytes, typically:
///
/// ```text
/// SPS/PPS + record bytes
/// ```
///
/// The record must contain every byte required by a standard decoder.
pub fn decode_h264_annex_b_rgb24(
    data: &[u8],
    pool: &mut DecoderPool,
    sps_pps: &[u8],
) -> Result<DecodedRgbFrame, Box<dyn std::error::Error>> {
    ffmpeg::init()?;

    let decoder = pool.get_or_create(sps_pps)?;

    let packet = ffmpeg::Packet::copy(data);
    decoder.send_packet(&packet)?;
    decoder.send_eof()?;

    let mut decoded = ffmpeg::util::frame::video::Video::empty();
    decoder.receive_frame(&mut decoded)?;

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoded.format(),
        decoded.width(),
        decoded.height(),
        ffmpeg::format::Pixel::RGB24,
        decoded.width(),
        decoded.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )?;

    let mut rgb = ffmpeg::util::frame::video::Video::empty();
    scaler.run(&decoded, &mut rgb)?;

    Ok(frame_to_rgb24(&rgb))
}

pub fn decode_h264_annex_b_rgb24_batch(
    sps_pps: &[u8],
    records: &[Vec<u8>],
    pool: &mut DecoderPool,
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    ffmpeg::init()?;

    let mut frames: Vec<DecodedRgbFrame> = Vec::with_capacity(records.len());
    let mut scaler: Option<ffmpeg::software::scaling::context::Context> = None;

    for record in records {
        let decoder = pool.get_or_create(sps_pps)?;

        let mut data = Vec::from(sps_pps);
        data.extend_from_slice(record);

        let packet = ffmpeg::Packet::copy(&data);
        decoder.send_packet(&packet)?;
        decoder.send_eof()?;

        let mut decoded = ffmpeg::util::frame::video::Video::empty();
        decoder.receive_frame(&mut decoded)?;

        let scaler = scaler.get_or_insert_with(|| {
            ffmpeg::software::scaling::context::Context::get(
                decoded.format(),
                decoded.width(),
                decoded.height(),
                ffmpeg::format::Pixel::RGB24,
                decoded.width(),
                decoded.height(),
                ffmpeg::software::scaling::flag::Flags::BILINEAR,
            )
            .expect("failed to create YUV→RGB24 scaler inside batch decode")
        });

        let mut rgb = ffmpeg::util::frame::video::Video::empty();
        scaler.run(&decoded, &mut rgb)?;
        frames.push(frame_to_rgb24(&rgb));
    }

    Ok(frames)
}

/// Decode a multi-frame GOP record (IDR + P-frames) using a single decoder session.
///
/// `sps_pps` contains SPS and PPS NALs (with start codes). `record` contains
/// the remaining NALs in the GOP: first the IDR, then zero or more P-frames,
/// each with Annex-B start codes.
///
/// All frames share one decoder instance so that P-frames can reference the
/// IDR (and each other, for FullGop policy). Returns one RGB frame per access
/// unit in the record.
pub fn decode_gop_rgb24(
    sps_pps: &[u8],
    record: &[u8],
    pool: &mut DecoderPool,
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    ffmpeg::init()?;

    let decoder = pool.get_or_create(sps_pps)?;

    // Build packets: first = SPS/PPS + IDR, rest = P-frame NALs
    let nals = split_annex_b_access_units(record);
    let packets: Vec<Vec<u8>> = nals
        .iter()
        .enumerate()
        .map(|(i, au)| {
            if i == 0 {
                let mut d = Vec::from(sps_pps);
                d.extend_from_slice(au);
                d
            } else {
                au.to_vec()
            }
        })
        .collect();

    let mut scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    let mut frames: Vec<DecodedRgbFrame> = Vec::new();

    for pkt_data in &packets {
        let pkt = ffmpeg::Packet::copy(pkt_data);
        // Send packet — if decoder is full (EAGAIN), drain a frame and retry
        loop {
            match decoder.send_packet(&pkt) {
                Ok(()) => break,
                Err(_) => {
                    let mut f = ffmpeg::util::frame::video::Video::empty();
                    if decoder.receive_frame(&mut f).is_ok() {
                        store_one(&mut scaler, &f, &mut frames)?;
                    } else {
                        // Real error—propagate by retrying (will produce meaningful error)
                        decoder.send_packet(&pkt)?;
                    }
                }
            }
        }
    }

    // Flush remaining buffered frames
    decoder.send_eof()?;
    let mut f = ffmpeg::util::frame::video::Video::empty();
    while decoder.receive_frame(&mut f).is_ok() {
        store_one(&mut scaler, &f, &mut frames)?;
        f = ffmpeg::util::frame::video::Video::empty();
    }

    if frames.is_empty() {
        return Err("no frames decoded from GOP record".into());
    }
    Ok(frames)
}

/// Batch-decode multiple GOP records, reusing the same decoder from the pool
/// and sharing a single YUV→RGB scaler across all records.
///
/// `sps_pps` is read once and shared across all records. Each record is treated
/// as an independent GOP: first NAL gets SPS/PPS prepended, the decoder sends
/// all packets, drains frames via send_eof, then flushes before the next record.
/// Works for idr_only (single frame per record), anchor_p, and full_gop policies.
pub fn decode_gop_rgb24_batch(
    sps_pps: &[u8],
    records: &[Vec<u8>],
    pool: &mut DecoderPool,
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    ffmpeg::init()?;

    let mut scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    let mut frames: Vec<DecodedRgbFrame> = Vec::new();

    for record in records {
        let decoder = pool.get_or_create(sps_pps)?;

        let nals = split_annex_b_access_units(record);
        let packets: Vec<Vec<u8>> = nals
            .iter()
            .enumerate()
            .map(|(i, au)| {
                if i == 0 {
                    let mut d = Vec::from(sps_pps);
                    d.extend_from_slice(au);
                    d
                } else {
                    au.to_vec()
                }
            })
            .collect();

        for pkt_data in &packets {
            let pkt = ffmpeg::Packet::copy(pkt_data);
            loop {
                match decoder.send_packet(&pkt) {
                    Ok(()) => break,
                    Err(_) => {
                        let mut f = ffmpeg::util::frame::video::Video::empty();
                        if decoder.receive_frame(&mut f).is_ok() {
                            store_one(&mut scaler, &f, &mut frames)?;
                        } else {
                            decoder.send_packet(&pkt)?;
                        }
                    }
                }
            }
        }

        decoder.send_eof()?;
        let mut f = ffmpeg::util::frame::video::Video::empty();
        while decoder.receive_frame(&mut f).is_ok() {
            store_one(&mut scaler, &f, &mut frames)?;
            f = ffmpeg::util::frame::video::Video::empty();
        }

        decoder.flush();
    }

    if frames.is_empty() {
        return Err("no frames decoded from batch GOP records".into());
    }
    Ok(frames)
}

/// Decode every dependency in each closed record, but materialize only the
/// final target frame as RGB24. This avoids converting anchor/intermediate
/// frames that are required by the codec but not returned to the ML workload.
pub fn decode_gop_targets_rgb24_batch(
    sps_pps: &[u8],
    records: &[Vec<u8>],
    pool: &mut DecoderPool,
    expected_frames_per_record: usize,
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    if expected_frames_per_record == 0 {
        return Err("expected_frames_per_record must be positive".into());
    }
    ffmpeg::init()?;
    let mut scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    let mut targets = Vec::with_capacity(records.len());

    for record in records {
        let decoder = pool.get_or_create(sps_pps)?;
        let nals = split_annex_b_access_units(record);
        let mut decoded_count = 0usize;
        let mut last_frame: Option<ffmpeg::util::frame::video::Video> = None;

        for (index, access_unit) in nals.iter().enumerate() {
            let packet_data = if index == 0 {
                let mut data = Vec::with_capacity(sps_pps.len() + access_unit.len());
                data.extend_from_slice(sps_pps);
                data.extend_from_slice(access_unit);
                data
            } else {
                access_unit.clone()
            };
            let packet = ffmpeg::Packet::copy(&packet_data);
            loop {
                match decoder.send_packet(&packet) {
                    Ok(()) => break,
                    Err(_) => {
                        let mut frame = ffmpeg::util::frame::video::Video::empty();
                        if decoder.receive_frame(&mut frame).is_ok() {
                            decoded_count += 1;
                            last_frame = Some(frame);
                        } else {
                            decoder.send_packet(&packet)?;
                            break;
                        }
                    }
                }
            }
        }

        decoder.send_eof()?;
        loop {
            let mut frame = ffmpeg::util::frame::video::Video::empty();
            if decoder.receive_frame(&mut frame).is_err() {
                break;
            }
            decoded_count += 1;
            last_frame = Some(frame);
        }
        if decoded_count != expected_frames_per_record {
            return Err(format!(
                "decoded {} frames from closed record, expected {}",
                decoded_count, expected_frames_per_record
            )
            .into());
        }
        let target = last_frame.ok_or("closed record decoded no target frame")?;
        store_one(&mut scaler, &target, &mut targets)?;
        decoder.flush();
    }
    Ok(targets)
}

/// Decode one compact Prefix record once and return selected display-order
/// frames. Callers use this for multiple logical targets contained in the same
/// physical IPPP prefix instead of independently decoding every nested prefix.
pub fn decode_gop_selected_rgb24(
    sps_pps: &[u8],
    record: &[u8],
    pool: &mut DecoderPool,
    ordinals: &[usize],
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    decode_gop_selected_rgb24_impl(sps_pps, record, pool, ordinals, true)
}

/// Decode a complete GOP while converting only selected display-order frames
/// to RGB24. Unlike the compact Prefix path, the input may contain frames
/// after the largest requested ordinal.
pub fn decode_full_gop_selected_rgb24(
    sps_pps: &[u8],
    record: &[u8],
    pool: &mut DecoderPool,
    ordinals: &[usize],
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    decode_gop_selected_rgb24_impl(sps_pps, record, pool, ordinals, false)
}

fn decode_gop_selected_rgb24_impl(
    sps_pps: &[u8],
    record: &[u8],
    pool: &mut DecoderPool,
    ordinals: &[usize],
    require_truncated_prefix: bool,
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    if ordinals.is_empty() {
        return Ok(Vec::new());
    }
    let max_ordinal = *ordinals.iter().max().expect("non-empty ordinals");
    let requested = ordinals.iter().copied().collect::<HashSet<_>>();
    if requested.len() != ordinals.len() {
        return Err("duplicate Prefix target ordinal".into());
    }

    ffmpeg::init()?;
    let decoder = pool.get_or_create(sps_pps)?;
    let access_units = split_annex_b_access_units(record);
    let mut scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    let mut selected = HashMap::<usize, DecodedRgbFrame>::new();
    let mut decoded_count = 0usize;

    for (index, access_unit) in access_units.iter().enumerate() {
        let packet_data = if index == 0 {
            let mut data = Vec::with_capacity(sps_pps.len() + access_unit.len());
            data.extend_from_slice(sps_pps);
            data.extend_from_slice(access_unit);
            data
        } else {
            access_unit.clone()
        };
        let packet = ffmpeg::Packet::copy(&packet_data);
        loop {
            match decoder.send_packet(&packet) {
                Ok(()) => break,
                Err(_) => {
                    let mut frame = ffmpeg::util::frame::video::Video::empty();
                    if decoder.receive_frame(&mut frame).is_ok() {
                        store_selected(
                            &mut scaler,
                            &frame,
                            decoded_count,
                            &requested,
                            &mut selected,
                        )?;
                        decoded_count += 1;
                    } else {
                        decoder.send_packet(&packet)?;
                        break;
                    }
                }
            }
        }
    }

    decoder.send_eof()?;
    loop {
        let mut frame = ffmpeg::util::frame::video::Video::empty();
        if decoder.receive_frame(&mut frame).is_err() {
            break;
        }
        store_selected(
            &mut scaler,
            &frame,
            decoded_count,
            &requested,
            &mut selected,
        )?;
        decoded_count += 1;
    }
    decoder.flush();

    if require_truncated_prefix && decoded_count != max_ordinal + 1 {
        return Err(format!(
            "Prefix decoded {} frames, expected {} through target ordinal {}",
            decoded_count,
            max_ordinal + 1,
            max_ordinal
        )
        .into());
    }
    if max_ordinal >= decoded_count {
        return Err(format!(
            "target ordinal {} outside decoded frame count {}",
            max_ordinal, decoded_count
        )
        .into());
    }
    ordinals
        .iter()
        .map(|ordinal| {
            selected.remove(ordinal).ok_or_else(|| {
                format!("Prefix target ordinal {ordinal} was not materialized").into()
            })
        })
        .collect()
}

/// Decode independent closed records in one continuous decoder session.
/// Every record must start with an IDR and contain exactly
/// `expected_frames_per_record` access units. A new IDR resets codec reference
/// state, so no `send_eof`/`flush` is required between records.
pub fn decode_closed_targets_continuous_rgb24(
    sps_pps: &[u8],
    records: &[Vec<u8>],
    pool: &mut DecoderPool,
    expected_frames_per_record: usize,
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    if expected_frames_per_record == 0 {
        return Err("expected_frames_per_record must be positive".into());
    }
    decode_mixed_closed_targets_continuous_rgb24(
        sps_pps,
        records,
        pool,
        &vec![expected_frames_per_record; records.len()],
    )
}

/// Decode independent closed records through one persistent decoder while
/// validating each record's own frame count.
pub fn decode_mixed_closed_targets_continuous_rgb24(
    sps_pps: &[u8],
    records: &[Vec<u8>],
    pool: &mut DecoderPool,
    expected_frames_per_record: &[usize],
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    if records.len() != expected_frames_per_record.len() {
        return Err("closed records and expected frame counts differ in length".into());
    }
    if expected_frames_per_record.iter().any(|value| *value == 0) {
        return Err("expected frames per closed record must be positive".into());
    }
    ffmpeg::init()?;
    let decoder = pool.get_or_create(sps_pps)?;
    let mut scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    let mut targets = Vec::with_capacity(records.len());

    for (record, expected_frames) in records.iter().zip(expected_frames_per_record) {
        let nals = split_annex_b_access_units(record);
        if nals.len() != *expected_frames {
            return Err(format!(
                "closed record has {} access units, expected {}",
                nals.len(),
                expected_frames
            )
            .into());
        }
        let mut record_frames: Vec<ffmpeg::util::frame::video::Video> = Vec::new();
        for (index, access_unit) in nals.iter().enumerate() {
            let packet_data = if index == 0 {
                let mut data = Vec::with_capacity(sps_pps.len() + access_unit.len());
                data.extend_from_slice(sps_pps);
                data.extend_from_slice(access_unit);
                data
            } else {
                access_unit.clone()
            };
            let packet = ffmpeg::Packet::copy(&packet_data);
            loop {
                match decoder.send_packet(&packet) {
                    Ok(()) => break,
                    Err(_) => {
                        let mut frame = ffmpeg::util::frame::video::Video::empty();
                        if decoder.receive_frame(&mut frame).is_ok() {
                            record_frames.push(frame);
                        } else {
                            decoder.send_packet(&packet)?;
                            break;
                        }
                    }
                }
            }
            loop {
                let mut frame = ffmpeg::util::frame::video::Video::empty();
                if decoder.receive_frame(&mut frame).is_err() {
                    break;
                }
                record_frames.push(frame);
            }
        }
        if record_frames.len() != *expected_frames {
            return Err(format!(
                "continuous decode produced {} frames for one record, expected {}",
                record_frames.len(),
                expected_frames
            )
            .into());
        }
        let target = record_frames
            .last()
            .ok_or("continuous decode produced no target")?;
        store_one(&mut scaler, target, &mut targets)?;
    }

    decoder.send_eof()?;
    let mut trailing = ffmpeg::util::frame::video::Video::empty();
    if decoder.receive_frame(&mut trailing).is_ok() {
        return Err("continuous decode left an unexpected trailing frame".into());
    }
    decoder.flush();
    Ok(targets)
}

/// Decode one shared Anchor followed by independently anchor-referenced Delta
/// access units. Deltas must be supplied in their original display order.
/// The Anchor enters libavcodec once and is not converted to RGB; every Delta
/// output is materialized exactly once.
pub fn decode_shared_anchor_deltas_rgb24(
    anchor: &[u8],
    deltas: &[Vec<u8>],
    pool: &mut DecoderPool,
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    decode_shared_anchor_group_rgb24(anchor, deltas, false, pool)
}

/// Decode one bounded shared-Anchor group. When `include_anchor` is true, the
/// reconstructed IDR is returned before the requested Delta targets.
pub fn decode_shared_anchor_group_rgb24(
    anchor: &[u8],
    deltas: &[Vec<u8>],
    include_anchor: bool,
    pool: &mut DecoderPool,
) -> Result<Vec<DecodedRgbFrame>, Box<dyn std::error::Error>> {
    if deltas.is_empty() && !include_anchor {
        return Ok(Vec::new());
    }
    let (codec_config, idr) = extract_anchor_parts(anchor)?;
    let mut record = Vec::with_capacity(idr.len() + deltas.iter().map(Vec::len).sum::<usize>());
    record.extend_from_slice(&idr);
    for (index, delta) in deltas.iter().enumerate() {
        let units = split_annex_b_access_units(delta);
        if units.len() != 1 {
            return Err(format!(
                "Delta {index} contains {} access units, expected one",
                units.len()
            )
            .into());
        }
        let start = if units[0].starts_with(b"\x00\x00\x00\x01") {
            4
        } else if units[0].starts_with(b"\x00\x00\x01") {
            3
        } else {
            return Err(format!("Delta {index} lacks an Annex-B start code").into());
        };
        if units[0].get(start).map(|byte| byte & 0x1f) != Some(1) {
            return Err(format!("Delta {index} is not a non-IDR VCL access unit").into());
        }
        record.extend_from_slice(&units[0]);
    }
    let mut ordinals = Vec::with_capacity(deltas.len() + usize::from(include_anchor));
    if include_anchor {
        ordinals.push(0);
    }
    ordinals.extend(1..=deltas.len());
    decode_full_gop_selected_rgb24(&codec_config, &record, pool, &ordinals)
}

fn store_one(
    scaler: &mut Option<ffmpeg::software::scaling::context::Context>,
    decoded: &ffmpeg::util::frame::video::Video,
    frames: &mut Vec<DecodedRgbFrame>,
) -> Result<(), Box<dyn std::error::Error>> {
    let s = scaler.get_or_insert_with(|| {
        ffmpeg::software::scaling::context::Context::get(
            decoded.format(),
            decoded.width(),
            decoded.height(),
            ffmpeg::format::Pixel::RGB24,
            decoded.width(),
            decoded.height(),
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )
        .expect("failed to create YUV→RGB24 scaler")
    });
    let mut rgb = ffmpeg::util::frame::video::Video::empty();
    s.run(decoded, &mut rgb)?;
    frames.push(frame_to_rgb24(&rgb));
    Ok(())
}

fn store_selected(
    scaler: &mut Option<ffmpeg::software::scaling::context::Context>,
    decoded: &ffmpeg::util::frame::video::Video,
    ordinal: usize,
    requested: &HashSet<usize>,
    selected: &mut HashMap<usize, DecodedRgbFrame>,
) -> Result<(), Box<dyn std::error::Error>> {
    if requested.contains(&ordinal) {
        let mut frame = Vec::with_capacity(1);
        store_one(scaler, decoded, &mut frame)?;
        selected.insert(
            ordinal,
            frame.pop().expect("store_one must append one RGB frame"),
        );
    }
    Ok(())
}

/// Split an Annex-B byte slice into individual access units (NALs with start codes).
fn split_annex_b_access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let sc_len = start_code_len(data, i);
        if sc_len == 0 {
            i += 1;
            continue;
        }
        let start = i;
        i += sc_len;
        while i < data.len() {
            let n = start_code_len(data, i);
            if n > 0 {
                break;
            }
            i += 1;
        }
        units.push(data[start..i].to_vec());
    }
    units
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

/// Extract SPS/PPS and IDR access units from one independently decodable
/// Annex-B anchor record. The returned byte slices retain their start codes.
pub fn extract_anchor_parts(
    anchor: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), Box<dyn std::error::Error>> {
    let mut codec_config = Vec::new();
    let mut idr = Vec::new();
    for unit in split_annex_b_access_units(anchor) {
        let start = if unit.starts_with(b"\x00\x00\x00\x01") {
            4
        } else if unit.starts_with(b"\x00\x00\x01") {
            3
        } else {
            continue;
        };
        if start >= unit.len() {
            continue;
        }
        match unit[start] & 0x1f {
            7 | 8 => codec_config.extend_from_slice(&unit),
            5 => idr.extend_from_slice(&unit),
            _ => {}
        }
    }
    if codec_config.is_empty() {
        return Err("anchor record did not contain SPS/PPS".into());
    }
    if idr.is_empty() {
        return Err("anchor record did not contain an IDR access unit".into());
    }
    Ok((codec_config, idr))
}

/// Split one self-contained closed record into decoder configuration and VCL
/// access units. Non-VCL metadata is intentionally excluded from the decode
/// record; every retained VCL NAL corresponds to one frame in the restricted
/// no-B-frame Pair representation.
pub fn extract_closed_record_parts(
    data: &[u8],
) -> Result<(Vec<u8>, Vec<u8>, usize), Box<dyn std::error::Error>> {
    let mut codec_config = Vec::new();
    let mut record = Vec::new();
    let mut frame_count = 0;
    let mut has_idr = false;
    for unit in split_annex_b_access_units(data) {
        let start = if unit.starts_with(b"\x00\x00\x00\x01") {
            4
        } else if unit.starts_with(b"\x00\x00\x01") {
            3
        } else {
            continue;
        };
        if start >= unit.len() {
            continue;
        }
        match unit[start] & 0x1f {
            7 | 8 => codec_config.extend_from_slice(&unit),
            nal_type @ (1 | 5) => {
                has_idr |= nal_type == 5;
                frame_count += 1;
                record.extend_from_slice(&unit);
            }
            _ => {}
        }
    }
    if codec_config.is_empty() {
        return Err("closed record did not contain SPS/PPS".into());
    }
    if !has_idr {
        return Err("closed record did not contain an IDR access unit".into());
    }
    if frame_count == 0 {
        return Err("closed record did not contain VCL access units".into());
    }
    Ok((codec_config, record, frame_count))
}

fn frame_to_rgb24(rgb: &ffmpeg::util::frame::video::Video) -> DecodedRgbFrame {
    let width = rgb.width();
    let height = rgb.height();
    let row_bytes = width as usize * 3;
    let stride = rgb.stride(0);
    let plane = rgb.data(0);

    let mut out = Vec::with_capacity(row_bytes * height as usize);
    for y in 0..height as usize {
        let start = y * stride;
        let end = start + row_bytes;
        out.extend_from_slice(&plane[start..end]);
    }

    DecodedRgbFrame {
        data: out,
        width,
        height,
    }
}

#[cfg(all(test, feature = "ffmpeg"))]
mod tests {
    use super::*;
    use crate::chunk::ChunkReader;
    use crate::encoder::X264Encoder;
    use std::path::Path;
    use std::time::Instant;

    fn open_chunk() -> ChunkReader {
        let path = std::env::var("VCLASP_TEST_CHUNK")
            .expect("VCLASP_TEST_CHUNK is required for fixture tests");
        ChunkReader::open(Path::new(&path)).expect("failed to open VClasp test chunk")
    }

    fn test_video() -> &'static str {
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
    }

    /// Stage-level decode microbenchmark.
    ///
    /// Times each phase of the GOP batch decode pipeline to identify where
    /// the ~4.8× performance gap vs. decord originates:
    ///   1. NAL splitting + SPS/PPS prepend
    ///   2. decoder flush (pool get_or_create)
    ///   3. ffmpeg Packet::copy
    ///   4. send_packet + receive_frame (actual H.264 decode)
    ///   5. swscale YUV→RGB24 conversion
    ///   6. frame_to_rgb24 (stride-corrected copy)
    #[test]
    #[ignore = "requires VCLASP_TEST_CHUNK"]
    fn bench_decode_stages() {
        ffmpeg::init().unwrap();
        let mut chunk = open_chunk();
        let sps_pps = chunk.read_sps_pps().unwrap();
        let n_records = 21;

        let mut records: Vec<Vec<u8>> = Vec::with_capacity(n_records);
        for _ in 0..n_records {
            records.push(chunk.read_record(test_video(), 2).unwrap());
        }

        let n_reps = 50;

        // ---- stage 1: NAL split + SPS/PPS prepend + alloc ----
        let t0 = Instant::now();
        for _ in 0..n_reps {
            for rec in &records {
                let nals = split_annex_b_access_units(rec);
                for (i, au) in nals.iter().enumerate() {
                    let _d = if i == 0 {
                        let mut d = sps_pps.clone();
                        d.extend_from_slice(au);
                        d
                    } else {
                        au.to_vec()
                    };
                }
            }
        }
        let stage1 = t0.elapsed().as_secs_f64() / (n_reps as f64 * n_records as f64);
        eprintln!(
            "stage1 (NAL split+SPS/PPS+alloc): {:.6} s/rec = {:.3} ms/rec",
            stage1,
            stage1 * 1000.0
        );

        // ---- stage 2-6: full decode pipeline measured externally ----
        let mut pool = DecoderPool::new(DecoderConfig { num_threads: 0 });
        // warmup
        let _ = decode_gop_rgb24_batch(&sps_pps, &records, &mut pool).unwrap();

        let t0 = Instant::now();
        for _ in 0..n_reps {
            let _ = decode_gop_rgb24_batch(&sps_pps, &records, &mut pool).unwrap();
        }
        let total = t0.elapsed().as_secs_f64() / (n_reps as f64 * n_records as f64);
        eprintln!(
            "stage2-6 (full batch decode): {:.6} s/rec = {:.3} ms/rec",
            total,
            total * 1000.0
        );

        // ---- stage 2+3+4: decode to YUV (no swscale) ----
        // Direct ffmpeg API to measure just send/receive without swscale.
        let t0 = Instant::now();
        for _ in 0..n_reps {
            for rec in &records {
                let dec = pool.get_or_create(&sps_pps).unwrap();
                let nals = split_annex_b_access_units(rec);
                for (i, au) in nals.iter().enumerate() {
                    let pkt_data = if i == 0 {
                        let mut d = sps_pps.clone();
                        d.extend_from_slice(au);
                        d
                    } else {
                        au.to_vec()
                    };
                    let pkt = ffmpeg::Packet::copy(&pkt_data);
                    loop {
                        match dec.send_packet(&pkt) {
                            Ok(()) => break,
                            Err(_) => {
                                let mut f = ffmpeg::util::frame::video::Video::empty();
                                let _ = dec.receive_frame(&mut f);
                            }
                        }
                    }
                }
                let _ = dec.send_eof();
                let mut f = ffmpeg::util::frame::video::Video::empty();
                while dec.receive_frame(&mut f).is_ok() {
                    // YUV frame received, do NOT convert to RGB
                    f = ffmpeg::util::frame::video::Video::empty();
                }
                dec.flush();
            }
        }
        let stage234 = t0.elapsed().as_secs_f64() / (n_reps as f64 * n_records as f64);
        eprintln!(
            "stage2+3+4 (flush+NAL+send+recv H.264): {:.6} s/rec = {:.3} ms/rec",
            stage234,
            stage234 * 1000.0
        );

        // ---- stage 5+6: swscale YUV→RGB + stride-corrected frame copy ----
        let dec = pool.get_or_create(&sps_pps).unwrap();
        let nals = split_annex_b_access_units(&records[0]);
        let pkt_data = {
            let mut d = sps_pps.clone();
            d.extend_from_slice(&nals[0]);
            d
        };
        let pkt = ffmpeg::Packet::copy(&pkt_data);
        dec.send_packet(&pkt).unwrap();
        dec.send_eof().unwrap();
        let mut yuv = ffmpeg::util::frame::video::Video::empty();
        dec.receive_frame(&mut yuv).unwrap();
        dec.flush();

        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            yuv.format(),
            yuv.width(),
            yuv.height(),
            ffmpeg::format::Pixel::RGB24,
            yuv.width(),
            yuv.height(),
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )
        .unwrap();

        // stage 5: swscale only
        let t0 = Instant::now();
        for _ in 0..(n_reps * n_records) {
            let mut rgb = ffmpeg::util::frame::video::Video::empty();
            scaler.run(&yuv, &mut rgb).unwrap();
            let _ = rgb.data(0).len();
        }
        let stage5 = t0.elapsed().as_secs_f64() / (n_reps as f64 * n_records as f64);
        eprintln!(
            "stage5 (swscale YUV→RGB24): {:.6} s/rec = {:.3} ms/rec",
            stage5,
            stage5 * 1000.0
        );

        // stage 5+6: swscale + stride-corrected copy
        let t0 = Instant::now();
        for _ in 0..(n_reps * n_records) {
            let mut rgb = ffmpeg::util::frame::video::Video::empty();
            scaler.run(&yuv, &mut rgb).unwrap();
            let _ = frame_to_rgb24(&rgb);
        }
        let stage56 = t0.elapsed().as_secs_f64() / (n_reps as f64 * n_records as f64);
        eprintln!(
            "stage5+6 (swscale+stride-copy): {:.6} s/rec = {:.3} ms/rec",
            stage56,
            stage56 * 1000.0
        );

        let stage6 = stage56 - stage5;
        eprintln!(
            "stage6 (stride-corrected frame copy): {:.6} s/rec = {:.3} ms/rec",
            stage6,
            stage6 * 1000.0
        );

        let derived_overhead = total - stage234 - stage56;
        eprintln!(
            "---\nderived overhead (pool flush + SPS/PPS prepend + Vec alloc): {:.6} s/rec = {:.3} ms/rec",
            derived_overhead, derived_overhead * 1000.0
        );
        eprintln!(
            "Breakdown:  H.264={:.1}%  swscale={:.1}%  stride-copy={:.1}%  overhead={:.1}%",
            stage234 / total * 100.0,
            stage5 / total * 100.0,
            stage6 / total * 100.0,
            derived_overhead / total * 100.0,
        );
        eprintln!(
            "BENCH: {:.3} ms/rec = H264({:.3}ms) + swscale({:.3}ms) + copy({:.3}ms) + ovhd({:.3}ms)",
            total * 1000.0,
            stage234 * 1000.0,
            stage5 * 1000.0,
            stage6 * 1000.0,
            derived_overhead * 1000.0,
        );
    }

    #[test]
    #[ignore = "requires VCLASP_TEST_CHUNK"]
    fn test_single_frame_works() {
        let mut chunk = open_chunk();
        let sps_pps = chunk.read_sps_pps().unwrap();
        let record = chunk.read_record(test_video(), 0).unwrap();
        let mut data = sps_pps.clone();
        data.extend_from_slice(&record);
        let mut pool = DecoderPool::new(DecoderConfig::default());
        let frame = decode_h264_annex_b_rgb24(&data, &mut pool, &sps_pps).unwrap();
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_eq!(frame.data.len(), 320 * 240 * 3);
    }

    #[test]
    #[ignore = "requires VCLASP_TEST_CHUNK"]
    fn test_batch_decode_two() {
        let mut chunk = open_chunk();
        let sps_pps = chunk.read_sps_pps().unwrap();
        let records: Vec<Vec<u8>> = vec![
            chunk.read_record(test_video(), 0).unwrap(),
            chunk.read_record(test_video(), 2).unwrap(),
        ];
        let mut pool = DecoderPool::new(DecoderConfig::default());
        let frames = decode_h264_annex_b_rgb24_batch(&sps_pps, &records, &mut pool).unwrap();
        assert_eq!(frames.len(), 2, "expected 2 frames, got {}", frames.len());
        for f in &frames {
            assert_eq!(f.width, 320);
            assert_eq!(f.height, 240);
            assert_eq!(f.data.len(), 320 * 240 * 3);
        }
    }

    #[test]
    #[ignore = "requires VCLASP_TEST_CHUNK"]
    fn test_batch_decode_five() {
        let mut chunk = open_chunk();
        let sps_pps = chunk.read_sps_pps().unwrap();
        let mut records = Vec::new();
        for _ in 0..5 {
            records.push(chunk.read_record(test_video(), 2).unwrap());
        }
        let mut pool = DecoderPool::new(DecoderConfig::default());
        let frames = decode_h264_annex_b_rgb24_batch(&sps_pps, &records, &mut pool).unwrap();
        assert_eq!(frames.len(), 5, "expected 5 frames, got {}", frames.len());
    }

    #[test]
    #[ignore = "requires VCLASP_TEST_CHUNK"]
    fn test_decode_gop_single_frame() {
        // A single-frame GOP (IDR only) should decode to 1 frame.
        let mut chunk = open_chunk();
        let sps_pps = chunk.read_sps_pps().unwrap();
        let record = chunk.read_record(test_video(), 0).unwrap();
        let mut pool = DecoderPool::new(DecoderConfig::default());
        let frames = decode_gop_rgb24(&sps_pps, &record, &mut pool).unwrap();
        assert_eq!(frames.len(), 1, "single-frame GOP should yield 1 frame");
        assert_eq!(frames[0].width, 320);
        assert_eq!(frames[0].height, 240);
        assert_eq!(frames[0].data.len(), 320 * 240 * 3);
    }

    #[test]
    #[ignore = "requires VCLASP_TEST_CHUNK"]
    fn test_decode_gop_roundtrip_matches_single_decode() {
        // decode_gop_rgb24 on an IDR-only record should match decode_h264_annex_b_rgb24.
        let mut chunk = open_chunk();
        let sps_pps = chunk.read_sps_pps().unwrap();
        let record = chunk.read_record(test_video(), 0).unwrap();

        // Single-frame decode (old path)
        let mut single_data = sps_pps.clone();
        single_data.extend_from_slice(&record);
        let mut pool = DecoderPool::new(DecoderConfig::default());
        let single = decode_h264_annex_b_rgb24(&single_data, &mut pool, &sps_pps).unwrap();

        // GOP decode (new path)
        let gop_frames = decode_gop_rgb24(&sps_pps, &record, &mut pool).unwrap();
        assert_eq!(gop_frames.len(), 1);

        // Pixel data should match
        assert_eq!(gop_frames[0].data, single.data);
    }

    #[test]
    #[ignore = "requires VCLASP_TEST_CHUNK"]
    fn test_decode_gop_batch_three_records() {
        let mut chunk = open_chunk();
        let sps_pps = chunk.read_sps_pps().unwrap();
        let records: Vec<Vec<u8>> = (0..3)
            .map(|_| chunk.read_record(test_video(), 2).unwrap())
            .collect();
        let mut pool = DecoderPool::new(DecoderConfig::default());
        let frames = decode_gop_rgb24_batch(&sps_pps, &records, &mut pool).unwrap();
        assert_eq!(frames.len(), 3, "expected 3 frames, got {}", frames.len());
        for f in &frames {
            assert_eq!(f.width, 320);
            assert_eq!(f.height, 240);
            assert_eq!(f.data.len(), 320 * 240 * 3);
        }
    }

    #[test]
    #[ignore = "requires VCLASP_TEST_CHUNK"]
    fn test_target_only_batch_matches_full_decode_for_idr_records() {
        let mut chunk = open_chunk();
        let sps_pps = chunk.read_sps_pps().unwrap();
        let records: Vec<Vec<u8>> = (0..3)
            .map(|_| chunk.read_record(test_video(), 2).unwrap())
            .collect();
        let mut full_pool = DecoderPool::new(DecoderConfig::default());
        let full = decode_gop_rgb24_batch(&sps_pps, &records, &mut full_pool).unwrap();
        let mut target_pool = DecoderPool::new(DecoderConfig::default());
        let targets =
            decode_gop_targets_rgb24_batch(&sps_pps, &records, &mut target_pool, 1).unwrap();
        assert_eq!(targets.len(), records.len());
        for (target, expected) in targets.iter().zip(full.iter()) {
            assert_eq!(target.data, expected.data);
        }
    }

    fn nal_type(nal: &[u8]) -> u8 {
        nal_header(nal) & 0x1F
    }

    fn nal_ref_idc(nal: &[u8]) -> u8 {
        (nal_header(nal) >> 5) & 0x03
    }

    fn nal_header(nal: &[u8]) -> u8 {
        if nal.len() >= 5 && nal[..4] == [0, 0, 0, 1] {
            nal[4]
        } else if nal.len() >= 4 && nal[..3] == [0, 0, 1] {
            nal[3]
        } else {
            panic!("NAL does not start with Annex-B start code");
        }
    }

    fn mean_absolute_error(lhs: &[u8], rhs: &[u8]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter()
            .zip(rhs)
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / lhs.len() as f64
    }

    fn synthetic_yuv420(width: usize, height: usize, frame: usize) -> Vec<u8> {
        let mut yuv = vec![0u8; width * height * 3 / 2];
        for y in 0..height {
            for x in 0..width {
                yuv[y * width + x] = 16 + ((x * 3 + y * 5 + frame * 17) % 220) as u8;
            }
        }
        let chroma = width * height / 4;
        for index in 0..chroma {
            yuv[width * height + index] = 96 + ((index + frame * 7) % 48) as u8;
            yuv[width * height + chroma + index] = 104 + ((index * 3 + frame * 11) % 40) as u8;
        }
        yuv
    }

    #[test]
    fn test_anchor_p_encoder_selective_dependency() {
        let width = 64u32;
        let height = 64u32;
        let frame_count = 8usize;
        let mut encoder = X264Encoder::new(width, height, 23, frame_count as u32, true);
        let mut encoded = Vec::new();
        for frame in 0..frame_count {
            encoded.extend_from_slice(&encoder.encode_frame(
                &synthetic_yuv420(width as usize, height as usize, frame),
                frame == 0,
            ));
        }
        loop {
            let bytes = encoder.flush();
            if bytes.is_empty() {
                break;
            }
            encoded.extend_from_slice(&bytes);
        }

        let (codec_config, record, encoded_frames) =
            extract_closed_record_parts(&encoded).expect("Anchor-P stream must be self-contained");
        assert_eq!(encoded_frames, frame_count);
        let access_units = split_annex_b_access_units(&record);
        assert_eq!(access_units.len(), frame_count);

        let mut full_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let full = decode_gop_rgb24(&codec_config, &record, &mut full_pool)
            .expect("full Anchor-P stream must decode");
        assert_eq!(full.len(), frame_count);

        for target in 1..frame_count {
            let selective = [&access_units[0][..], &access_units[target][..]].concat();
            let mut selective_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
            let frames = decode_gop_rgb24(&codec_config, &selective, &mut selective_pool)
                .unwrap_or_else(|error| panic!("I+P{target} failed: {error}"));
            assert_eq!(frames.len(), 2, "I+P{target} must decode two frames");
            let mae = mean_absolute_error(&frames[1].data, &full[target].data);
            assert!(
                mae <= 0.01,
                "I+P{target} differs from full Anchor-P stream: MAE={mae}"
            );
        }

        let mixed_records = vec![
            access_units[0].clone(),
            [&access_units[0][..], &access_units[7][..]].concat(),
        ];
        let mut mixed_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let mixed = decode_mixed_closed_targets_continuous_rgb24(
            &codec_config,
            &mixed_records,
            &mut mixed_pool,
            &[1, 2],
        )
        .expect("mixed Anchor-only and Anchor+Delta Pair records must decode");
        assert_eq!(mixed.len(), 2);
        assert_eq!(mixed[0].data, full[0].data);
        assert!(mean_absolute_error(&mixed[1].data, &full[7].data) <= 0.01);

        let fused_targets = [1usize, 3, 7];
        let mut fused = access_units[0].clone();
        for target in fused_targets {
            fused.extend_from_slice(&access_units[target]);
        }
        let mut fused_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let fused_frames = decode_gop_rgb24(&codec_config, &fused, &mut fused_pool)
            .expect("non-contiguous shared-Anchor stream must decode");
        assert_eq!(fused_frames.len(), fused_targets.len() + 1);
        for (output, target) in fused_frames.iter().skip(1).zip(fused_targets) {
            let mae = mean_absolute_error(&output.data, &full[target].data);
            assert!(
                mae <= 0.01,
                "fused I+P targets differ at P{target}: MAE={mae}"
            );
        }

        let mut anchor = codec_config.clone();
        anchor.extend_from_slice(&access_units[0]);
        let deltas: Vec<Vec<u8>> = fused_targets
            .iter()
            .map(|target| access_units[*target].clone())
            .collect();
        let mut api_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let api_frames = decode_shared_anchor_deltas_rgb24(&anchor, &deltas, &mut api_pool)
            .expect("shared-Anchor decoder API must decode");
        assert_eq!(api_frames.len(), fused_targets.len());
        for (output, target) in api_frames.iter().zip(fused_targets) {
            let mae = mean_absolute_error(&output.data, &full[target].data);
            assert!(mae <= 0.01, "shared-Anchor API P{target} MAE={mae}");
        }

        let reverse_targets = [7usize, 3, 1];
        let reverse_deltas = reverse_targets
            .iter()
            .map(|target| access_units[*target].clone())
            .collect::<Vec<_>>();
        let mut reverse_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let reverse_error =
            match decode_shared_anchor_deltas_rgb24(&anchor, &reverse_deltas, &mut reverse_pool) {
                Ok(_) => {
                    panic!("reverse Delta ordinals must not be treated as a legal fused stream")
                }
                Err(error) => error,
            };
        assert!(
            reverse_error
                .to_string()
                .contains("outside decoded frame count"),
            "unexpected reverse-ordinal error: {reverse_error}"
        );

        let mut anchor_api_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let anchor_and_targets =
            decode_shared_anchor_group_rgb24(&anchor, &deltas, true, &mut anchor_api_pool)
                .expect("shared-Anchor group API must decode its Anchor target");
        assert_eq!(anchor_and_targets.len(), fused_targets.len() + 1);
        assert_eq!(anchor_and_targets[0].data, full[0].data);
        for (output, target) in anchor_and_targets.iter().skip(1).zip(fused_targets) {
            let mae = mean_absolute_error(&output.data, &full[target].data);
            assert!(mae <= 0.01, "Anchor-inclusive API P{target} MAE={mae}");
        }
    }

    #[test]
    fn test_two_level_anchor_pages_decode_from_root_checkpoint_target() {
        let width = 64u32;
        let height = 64u32;
        let frame_count = if cfg!(vclasp_patched_x264) {
            64usize
        } else {
            32usize
        };
        let page_size = 8usize;
        let mut encoder =
            X264Encoder::new_two_level(width, height, 23, frame_count as u32, page_size as u32);
        let mut encoded = Vec::new();
        for frame in 0..frame_count {
            encoded.extend_from_slice(&encoder.encode_frame(
                &synthetic_yuv420(width as usize, height as usize, frame),
                frame == 0,
            ));
        }
        loop {
            let bytes = encoder.flush();
            if bytes.is_empty() {
                break;
            }
            encoded.extend_from_slice(&bytes);
        }

        let (codec_config, record, encoded_frames) =
            extract_closed_record_parts(&encoded).expect("two-level stream must be self-contained");
        assert_eq!(encoded_frames, frame_count);
        let access_units = split_annex_b_access_units(&record);
        assert_eq!(access_units.len(), frame_count);
        if cfg!(vclasp_patched_x264) {
            assert_eq!(nal_type(&access_units[0]), 5, "root must be the only IDR");
            assert!(
                nal_ref_idc(&access_units[0]) > 0,
                "root IDR must be retained as a reference"
            );
            for (frame, access_unit) in access_units.iter().enumerate().skip(1) {
                assert_eq!(
                    nal_type(access_unit),
                    1,
                    "two-level encoder inserted an unexpected IDR at frame {frame}"
                );
                if frame.is_multiple_of(page_size) {
                    assert!(
                        nal_ref_idc(access_unit) > 0,
                        "checkpoint P{frame} must be retained as a reference"
                    );
                } else {
                    assert_eq!(
                        nal_ref_idc(access_unit),
                        0,
                        "ordinary target P{frame} must be non-reference"
                    );
                }
            }
        }

        let mut full_pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
        let full = decode_gop_rgb24(&codec_config, &record, &mut full_pool)
            .expect("full two-level stream must decode");
        assert_eq!(full.len(), frame_count);

        for target in 1..frame_count {
            let checkpoint = target / page_size * page_size;
            let mut selective = access_units[0].clone();
            if checkpoint > 0 {
                selective.extend_from_slice(&access_units[checkpoint]);
            }
            if target != checkpoint {
                selective.extend_from_slice(&access_units[target]);
            }
            let expected_frames =
                1 + usize::from(checkpoint > 0) + usize::from(target != checkpoint);
            let mut pool = DecoderPool::new(DecoderConfig { num_threads: 1 });
            let decoded =
                decode_gop_rgb24(&codec_config, &selective, &mut pool).unwrap_or_else(|error| {
                    panic!("root+checkpoint+target failed for target {target}: {error}")
                });
            assert_eq!(decoded.len(), expected_frames);
            let mae = mean_absolute_error(&decoded.last().unwrap().data, &full[target].data);
            assert!(
                mae <= 0.01,
                "two-level selective target {target} differs from full stream: MAE={mae}"
            );
        }
    }

    #[test]
    fn test_anchor_p_bounded_groups_cover_sixty_four_frames() {
        let width = 64u32;
        let height = 64u32;
        let frame_count = 64usize;
        let group_size = crate::encoder::MAX_ANCHOR_P_GROUP_FRAMES as usize;
        let mut encoder = X264Encoder::new(width, height, 23, group_size as u32, true);
        let mut encoded = Vec::new();
        for frame in 0..frame_count {
            encoded.extend_from_slice(
                &encoder
                    .try_encode_frame(
                        &synthetic_yuv420(width as usize, height as usize, frame),
                        frame % group_size == 0,
                    )
                    .expect("bounded Anchor-P group must encode"),
            );
        }
        loop {
            let bytes = encoder
                .try_flush()
                .expect("bounded Anchor-P flush must succeed");
            if bytes.is_empty() {
                break;
            }
            encoded.extend_from_slice(&bytes);
        }

        let (_, record, encoded_frames) =
            extract_closed_record_parts(&encoded).expect("segmented stream must parse");
        assert_eq!(encoded_frames, frame_count);
        let access_units = split_annex_b_access_units(&record);
        assert_eq!(access_units.len(), frame_count);
        for frame in 0..frame_count {
            let expected = if frame % group_size == 0 { 5 } else { 1 };
            assert_eq!(nal_type(&access_units[frame]), expected, "frame {frame}");
        }
    }

    #[test]
    #[ignore = "fixture predates the corrected monotonic-PTS Anchor-P encoder"]
    fn test_anchor_p_target_decodes_without_intermediate_p_frames() {
        let fixture = std::env::var("VCLASP_ANCHOR_P_TEST_CHUNK")
            .expect("VCLASP_ANCHOR_P_TEST_CHUNK is required");
        let path = Path::new(&fixture);
        let mut chunk = ChunkReader::open(path).expect("failed to open anchor_p chunk");
        let sps_pps = chunk.read_sps_pps().unwrap();
        let video = "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi";

        let records = chunk.read_all_records(video, 2).unwrap();
        assert!(records.len() >= 3, "need at least 3 anchor_p records");

        for (rec_idx, record) in records.iter().enumerate().take(3) {
            let nals = split_annex_b_access_units(record);
            assert_eq!(nal_type(&nals[0]), 5, "first NAL must be IDR (type 5)");
            let n_pframes = nals.len() - 1;
            assert!(n_pframes >= 1, "record must have at least 1 P-frame");

            let mut full_pool = DecoderPool::new(DecoderConfig::default());
            let full_frames = decode_gop_rgb24(&sps_pps, record, &mut full_pool)
                .expect("full anchor-P GOP must decode");
            assert_eq!(
                full_frames.len(),
                nals.len(),
                "one output frame is expected for every access unit"
            );

            for k in 1..=n_pframes {
                let selective_record: Vec<u8> = [&nals[0], &nals[k]]
                    .into_iter()
                    .flat_map(|nal| nal.iter())
                    .copied()
                    .collect();
                let mut selective_pool = DecoderPool::new(DecoderConfig::default());
                let frames = decode_gop_rgb24(&sps_pps, &selective_record, &mut selective_pool)
                    .unwrap_or_else(|e| {
                        panic!(
                            "I+P{} failed without intermediate P frames in record {}: {}",
                            k, rec_idx, e
                        )
                    });
                assert_eq!(
                    frames.len(),
                    2,
                    "I+P{} should yield exactly two frames, got {}",
                    k,
                    frames.len()
                );
                let mae = mean_absolute_error(&frames[1].data, &full_frames[k].data);
                assert!(
                    mae <= 0.01,
                    "I+P{} does not reproduce full-GOP frame in record {}: MAE={}",
                    k,
                    rec_idx,
                    mae
                );
            }
        }
    }

    #[test]
    fn test_extract_closed_record_parts_separates_config_and_vcl() {
        let mut payload = Vec::new();
        for nal in [
            vec![0, 0, 0, 1, 0x67, 1],
            vec![0, 0, 0, 1, 0x68, 2],
            vec![0, 0, 1, 0x06, 3],
            vec![0, 0, 1, 0x65, 4],
            vec![0, 0, 1, 0x41, 5],
        ] {
            payload.extend(nal);
        }
        let (config, record, frames) = extract_closed_record_parts(&payload).unwrap();
        assert_eq!(frames, 2);
        assert!(config.windows(2).any(|window| window == [0x67, 1]));
        assert!(config.windows(2).any(|window| window == [0x68, 2]));
        assert!(record.windows(2).any(|window| window == [0x65, 4]));
        assert!(record.windows(2).any(|window| window == [0x41, 5]));
        assert!(!record.contains(&0x06));
    }

    #[test]
    #[ignore = "requires VCLASP_TEST_CHUNK"]
    fn test_decode_gop_batch_pool_reuse() {
        let mut chunk = open_chunk();
        let sps_pps = chunk.read_sps_pps().unwrap();
        let records: Vec<Vec<u8>> = (0..5)
            .map(|_| chunk.read_record(test_video(), 2).unwrap())
            .collect();
        let mut pool = DecoderPool::new(DecoderConfig::default());
        let frames = decode_gop_rgb24_batch(&sps_pps, &records, &mut pool).unwrap();
        assert_eq!(frames.len(), 5, "expected 5 frames, got {}", frames.len());
        for f in &frames {
            assert_eq!(f.width, 320);
            assert_eq!(f.height, 240);
        }
        assert_eq!(pool.decoders.len(), 1, "pool should have 1 cached decoder");
    }
}
