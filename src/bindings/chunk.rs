use super::*;

#[pymethods]
impl VClaspChunk {
    /// Open a VClasp chunk from a local file path.
    #[new]
    pub(super) fn open(path: &str) -> PyResult<Self> {
        let reader = chunk::ChunkReader::open(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        #[cfg(feature = "ffmpeg")]
        let hierarchical_catalog = Some(
            reader
                .catalog()
                .map_err(|error| PyErr::new::<pyo3::exceptions::PyValueError, _>(error))?,
        );
        Ok(VClaspChunk {
            inner: reader,
            #[cfg(feature = "experiment-controls")]
            path: path.to_string(),
            #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
            decoder_pool: decoder::DecoderPool::new(decoder::DecoderConfig::default()),
            #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
            cached_sps: None,
            #[cfg(feature = "ffmpeg")]
            hierarchical_catalog,
        })
    }

    /// Total number of records in the chunk index.
    pub(super) fn record_count(&self) -> PyResult<usize> {
        self.inner
            .catalog()
            .map(|catalog| catalog.len())
            .map_err(|error| PyErr::new::<pyo3::exceptions::PyValueError, _>(error))
    }

    /// Get codec configuration: (codec, width, height, fps_num, fps_den, sps_pps_len).
    pub(super) fn codec_info(&self) -> (String, u16, u16, u16, u16, u32) {
        let h = &self.inner.header;
        let cc = &h.codec_config;
        (
            cc.codec.clone(),
            cc.width,
            cc.height,
            cc.fps_num,
            cc.fps_den,
            cc.sps_pps_length,
        )
    }

    /// Return the chunk format version from the FlatBuffer header.
    pub(super) fn format_version(&self) -> u16 {
        self.inner.header.version
    }

    /// Read SPS/PPS bytes from the chunk.
    pub(super) fn read_sps_pps(&mut self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let data = self
            .inner
            .read_sps_pps()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &data).unbind())
    }

    /// Return the embedded Parquet AU index without reparsing the video stream.
    pub(super) fn read_au_index(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new_bound(py, &self.inner.mmap[self.inner.layout.index_start..]).unbind()
    }

    /// Read a single record blob by (video_id, tier) — returns one random matching blob.
    /// For deterministic access, use read_record_at(video_id, tier, idx).
    #[cfg(feature = "experiment-controls")]
    pub(super) fn read_record(
        &mut self,
        py: Python<'_>,
        video_id: &str,
        tier: i32,
    ) -> PyResult<Py<PyBytes>> {
        let data = self
            .inner
            .read_record(video_id, tier)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &data).unbind())
    }

    /// Read a single deterministic record at index `idx` within the (video_id, tier) group.
    #[cfg(feature = "experiment-controls")]
    pub(super) fn read_record_at(
        &mut self,
        py: Python<'_>,
        video_id: &str,
        tier: i32,
        idx: usize,
    ) -> PyResult<Py<PyBytes>> {
        let data = self
            .inner
            .read_record_at(video_id, tier, idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &data).unbind())
    }

    /// Return the number of records for (video_id, tier), or 0 if not found.
    #[cfg(feature = "experiment-controls")]
    pub(super) fn record_count_for(&self, video_id: &str, tier: i32) -> usize {
        self.inner.record_count_for(video_id, tier)
    }

    /// Enumerate logical video IDs registered in the closure catalog.
    #[cfg(feature = "ffmpeg")]
    pub(super) fn video_ids(&self) -> PyResult<Vec<String>> {
        let catalog = self.hierarchical_catalog.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "chunk does not contain a valid closure catalog",
            )
        })?;
        let mut video_ids = catalog
            .records()
            .map(|record| record.video_id.clone())
            .collect::<Vec<_>>();
        video_ids.sort();
        video_ids.dedup();
        Ok(video_ids)
    }

    /// Enumerate registered display-order frame indices for one logical video.
    #[cfg(feature = "ffmpeg")]
    pub(super) fn frame_indices(&self, video_id: &str) -> PyResult<Vec<i32>> {
        let catalog = self.hierarchical_catalog.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "chunk does not contain a valid closure catalog",
            )
        })?;
        let mut frame_indices = catalog
            .records()
            .filter(|record| record.video_id == video_id)
            .map(|record| record.frame_idx)
            .collect::<Vec<_>>();
        if frame_indices.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                video_id.to_string(),
            ));
        }
        frame_indices.sort_unstable();
        frame_indices.dedup();
        Ok(frame_indices)
    }

    /// Read a specific byte range from the payload.
    #[cfg(feature = "experiment-controls")]
    pub(super) fn read_bytes(
        &mut self,
        py: Python<'_>,
        offset: u64,
        length: u64,
    ) -> PyResult<Py<PyBytes>> {
        let data = self
            .inner
            .read_bytes(offset as usize, length as usize)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &data).unbind())
    }

    /// Set the number of threads for FFmpeg decoder.
    /// `0` (default) lets FFmpeg auto-detect the optimal thread count.
    ///
    /// This only affects decoders created *after* the call. Existing cached
    /// decoders keep their original thread count.
    #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
    pub(super) fn set_decoder_threads(&mut self, num: usize) {
        self.decoder_pool.set_threads(num);
    }

    /// Decode one target from the bitstream-grounded closure stored in the
    /// hierarchical index. The target is addressed in source display order.
    #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
    pub(super) fn decode_hierarchical_target_rgb24(
        &mut self,
        py: Python<'_>,
        video_id: &str,
        frame_idx: i32,
    ) -> PyResult<(Py<PyBytes>, u32, u32, usize, u64)> {
        let catalog = self.hierarchical_catalog.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "chunk has no hierarchical dependency index",
            )
        })?;
        let target = catalog
            .target(video_id, frame_idx)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "unknown hierarchical target ({video_id}, {frame_idx})"
                ))
            })?
            .clone();
        let dependencies = target
            .closure_record_ids
            .iter()
            .map(|record_id| {
                catalog
                    .record(*record_id)
                    .cloned()
                    .ok_or_else(|| format!("missing dependency record {record_id}"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let mut record = Vec::new();
        let mut fetched_bytes = 0u64;
        for dependency in &dependencies {
            let sample = self
                .inner
                .read_bytes(dependency.offset as usize, dependency.length as usize)
                .map_err(|error| PyErr::new::<pyo3::exceptions::PyIOError, _>(error.to_string()))?;
            fetched_bytes += sample.len() as u64;
            record.extend_from_slice(
                &hierarchical_ingest::mp4_sample_to_annex_b(&sample, dependency.nal_length_size)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?,
            );
        }
        let sps_pps = self
            .get_sps_pps()
            .map_err(|error| PyErr::new::<pyo3::exceptions::PyIOError, _>(error.to_string()))?;
        let mut self_contained = Vec::with_capacity(sps_pps.len() + record.len());
        self_contained.extend_from_slice(&sps_pps);
        self_contained.extend_from_slice(&record);
        let (codec_config, vcl_record, frame_count) =
            decoder::extract_closed_record_parts(&self_contained).map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(error.to_string())
            })?;
        if target.target_output_ordinal >= frame_count {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "target output ordinal {} outside closure frame count {}",
                target.target_output_ordinal, frame_count
            )));
        }
        let mut frames = py
            .allow_threads(|| {
                decoder::decode_full_gop_selected_rgb24(
                    &codec_config,
                    &vcl_record,
                    &mut self.decoder_pool,
                    &[target.target_output_ordinal],
                )
                .map_err(|error| error.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frame = frames.pop().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "hierarchical closure decoded no target frame",
            )
        })?;
        Ok((
            PyBytes::new_bound(py, &frame.data).unbind(),
            frame.width,
            frame.height,
            dependencies.len(),
            fetched_bytes,
        ))
    }

    /// Decode a record through Rust FFmpeg/libavcodec and return (rgb24_bytes, width, height).
    ///
    /// This method is available only when compiled with `--features ffmpeg`.
    #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
    pub(super) fn decode_record_rgb24(
        &mut self,
        py: Python<'_>,
        video_id: &str,
        tier: i32,
    ) -> PyResult<(Py<PyBytes>, u32, u32)> {
        let sps_pps = self
            .get_sps_pps()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let record = self
            .inner
            .read_record(video_id, tier)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut data = Vec::with_capacity(sps_pps.len() + record.len());
        data.extend_from_slice(&sps_pps);
        data.extend_from_slice(&record);

        let frame = decoder::decode_h264_annex_b_rgb24(&data, &mut self.decoder_pool, &sps_pps)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok((
            PyBytes::new_bound(py, &frame.data).unbind(),
            frame.width,
            frame.height,
        ))
    }

    /// Batch-decode N records from a single (video_id, tier) in one decoder session.
    ///
    /// Reads `count` records, concatenates them with per-record SPS/PPS prefixes,
    /// and decodes all frames through a single FFmpeg decoder + scaler session.
    /// Returns one `(rgb_bytes, width, height)` tuple per decoded frame.
    #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
    pub(super) fn decode_records_batch_rgb24(
        &mut self,
        py: Python<'_>,
        video_id: &str,
        tier: i32,
        count: usize,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        let sps_pps = self
            .get_sps_pps()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let mut records: Vec<Vec<u8>> = Vec::with_capacity(count);
        for _ in 0..count {
            let record = self
                .inner
                .read_record(video_id, tier)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            records.push(record);
        }

        let frames =
            decoder::decode_h264_annex_b_rgb24_batch(&sps_pps, &records, &mut self.decoder_pool)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(frames
            .into_iter()
            .map(|f| (PyBytes::new_bound(py, &f.data).unbind(), f.width, f.height))
            .collect())
    }

    /// Batch lookup: resolve many (video_id, tier) pairs and return their blob byte ranges.
    /// Uses SIMD-accelerated columnar scan for large batches.
    #[cfg(feature = "experiment-controls")]
    pub(super) fn batch_lookup(
        &self,
        video_ids: Vec<String>,
        tiers: Vec<i32>,
    ) -> PyResult<Vec<(u64, u64)>> {
        if video_ids.len() != tiers.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "video_ids and tiers must have same length",
            ));
        }
        self.inner
            .batch_lookup(&video_ids, &tiers)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a GOP record (multi-frame) and return one RGB24 frame per access unit.
    ///
    /// Tier 2 records with full_gop or anchor_p policy store multiple frames per
    /// record. This method decodes all frames using a single decoder session and
    /// returns `(rgb24_bytes, width, height)` for each frame.
    #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
    pub(super) fn decode_gop_record_rgb24(
        &mut self,
        py: Python<'_>,
        video_id: &str,
        tier: i32,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        let sps_pps = self
            .get_sps_pps()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let record = self
            .inner
            .read_record(video_id, tier)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let frames = decoder::decode_gop_rgb24(&sps_pps, &record, &mut self.decoder_pool)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(frames
            .into_iter()
            .map(|f| (PyBytes::new_bound(py, &f.data).unbind(), f.width, f.height))
            .collect())
    }

    /// Decode multiple consecutive GOP records in one Rust call, returning one RGB24
    /// frame per access unit across all records.
    ///
    /// Reads `count` consecutive records for the same (video_id, tier) starting at
    /// `start_idx` (0-based within the matching record group), then batch-decodes
    /// all frames through a single FFmpeg decoder + scaler session.
    /// Returns `[(rgb24_bytes, width, height), ...]` per decoded frame.
    #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
    pub(super) fn decode_gop_records_batch_rgb24(
        &mut self,
        py: Python<'_>,
        video_id: &str,
        tier: i32,
        start_idx: usize,
        count: usize,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        let sps_pps = self
            .get_sps_pps()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let records = self
            .inner
            .read_records_range(video_id, tier, start_idx, count)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let frames = decoder::decode_gop_rgb24_batch(&sps_pps, &records, &mut self.decoder_pool)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(frames
            .into_iter()
            .map(|f| (PyBytes::new_bound(py, &f.data).unbind(), f.width, f.height))
            .collect())
    }

    /// Decode ALL records for a (video_id, tier) in a single index lookup + batch decode.
    ///
    /// Single-scan variant: reads all matching records with one index scan,
    /// then batch-decodes all of them. Returns `[(rgb24_bytes, width, height), ...]`.
    #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
    pub(super) fn decode_gop_all_records_rgb24(
        &mut self,
        py: Python<'_>,
        video_id: &str,
        tier: i32,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        let sps_pps = self
            .get_sps_pps()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let records = self
            .inner
            .read_all_records(video_id, tier)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let frames = decoder::decode_gop_rgb24_batch(&sps_pps, &records, &mut self.decoder_pool)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(frames
            .into_iter()
            .map(|f| (PyBytes::new_bound(py, &f.data).unbind(), f.width, f.height))
            .collect())
    }

    /// Decode a single GOP record at deterministic index `idx`.
    /// Reads the record at idx, prepends SPS/PPS, decodes all frames.
    #[cfg(all(feature = "ffmpeg", feature = "experiment-controls"))]
    pub(super) fn decode_gop_record_at(
        &mut self,
        py: Python<'_>,
        video_id: &str,
        tier: i32,
        idx: usize,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        let sps_pps = self
            .get_sps_pps()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let record = self
            .inner
            .read_record_at(video_id, tier, idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let frames = decoder::decode_gop_rgb24(&sps_pps, &record, &mut self.decoder_pool)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(frames
            .into_iter()
            .map(|f| (PyBytes::new_bound(py, &f.data).unbind(), f.width, f.height))
            .collect())
    }

    /// Create a logical scheduler auto-configured from this chunk's index.
    /// The scheduler infers gop_size, tier1_stride, and frames-per-record
    /// automatically — no manual parameters needed.
    #[cfg(feature = "experiment-controls")]
    pub(super) fn create_logical_scheduler(&self) -> PyResult<PyLogicalBatchDecoder> {
        let reader = chunk::ChunkReader::open(Path::new(&self.path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let sched = scheduler::LogicalBatchDecoder::new(reader, self.path.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyLogicalBatchDecoder { inner: sched })
    }
}
