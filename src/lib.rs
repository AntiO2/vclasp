//! VClasp: dependency-aware video access for object stores
//!
//! Reads VClasp v1 chunks: FlatBuffer header + raw payload + Parquet columnar index.
//! Supports local mmap/pread and authenticated S3/MinIO Range GET in Rust.
//!
//! Chunk layout:
//!   [4B fb_size][flatbuffer header][SPS/PPS bytes][payload blobs][Parquet index]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::{Path, PathBuf};

#[cfg(feature = "ffmpeg")]
#[path = "planning/adaptive_planner.rs"]
mod adaptive_planner;
#[path = "storage/backend.rs"]
mod backend;
#[path = "ingest/builder.rs"]
mod builder;
#[path = "format/chunk.rs"]
pub mod chunk;
#[path = "format/chunk_schema.rs"]
mod chunk_schema;
#[path = "planning/cost_selector.rs"]
mod cost_selector;
#[cfg(feature = "ffmpeg")]
#[path = "codec/decoder.rs"]
mod decoder;
#[cfg(feature = "ffmpeg")]
#[path = "execution/dependency_sampler.rs"]
mod dependency_sampler;
#[path = "codec/encoder.rs"]
mod encoder;
#[cfg(feature = "ffmpeg")]
#[path = "execution/fragment_scheduler.rs"]
mod fragment_scheduler;
#[cfg(feature = "ffmpeg")]
#[path = "ingest/hierarchical_ingest.rs"]
mod hierarchical_ingest;
#[path = "planning/hierarchical_layout.rs"]
mod hierarchical_layout;
#[cfg(feature = "ffmpeg")]
#[path = "execution/hierarchical_scheduler.rs"]
mod hierarchical_scheduler;
#[path = "format/index.rs"]
pub mod index;
#[path = "planning/materialization.rs"]
mod materialization;
#[cfg(feature = "ffmpeg")]
#[path = "execution/normalized_scheduler.rs"]
mod normalized_scheduler;
#[cfg(feature = "ffmpeg")]
#[path = "execution/pair_scheduler.rs"]
mod pair_scheduler;
#[path = "planning/planner.rs"]
mod planner;
#[cfg(feature = "ffmpeg")]
#[path = "execution/portfolio_scheduler.rs"]
mod portfolio_scheduler;
#[path = "planning/representation.rs"]
pub mod representation;
#[cfg(feature = "ffmpeg")]
#[path = "storage/saturation.rs"]
mod saturation;
#[cfg(feature = "ffmpeg")]
#[path = "execution/scheduler.rs"]
mod scheduler;
#[path = "execution/simd.rs"]
mod simd;

pub use planner::{plan_byte_ranges, unique_covered_bytes, PlannedRecord, RangePlan, RecordRange};

/// Python-facing VClasp chunk reader.
#[pyclass]
pub struct VClaspChunk {
    inner: chunk::ChunkReader,
    #[cfg(feature = "ffmpeg")]
    path: String,
    #[cfg(feature = "ffmpeg")]
    decoder_pool: decoder::DecoderPool,
    #[cfg(feature = "ffmpeg")]
    cached_sps: Option<Vec<u8>>,
    #[cfg(feature = "ffmpeg")]
    hierarchical_catalog: Option<hierarchical_ingest::HierarchicalCatalog>,
}

/// Decoder adapter for closed H.264 Annex-B records independent of a chunk.
#[cfg(feature = "ffmpeg")]
#[pyclass(name = "H264Decoder")]
pub struct PyH264Decoder {
    decoder_pool: decoder::DecoderPool,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "AnchorDeltaBatchExecutor")]
pub struct PyNormalizedBatchExecutor {
    inner: normalized_scheduler::NormalizedBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "AIStoreAnchorDeltaBatchExecutor")]
pub struct PyAIStoreNormalizedBatchExecutor {
    inner: normalized_scheduler::NormalizedBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "LocalAnchorDeltaBatchExecutor")]
pub struct PyLocalNormalizedBatchExecutor {
    inner: normalized_scheduler::NormalizedBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "AdaptiveBatchExecutor")]
pub struct PyAdaptiveBatchExecutor {
    inner: adaptive_planner::AdaptiveBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "S3BatchExecutor")]
pub struct PyS3HierarchicalBatchExecutor {
    inner: hierarchical_scheduler::HierarchicalBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "LocalBatchExecutor")]
pub struct PyLocalHierarchicalBatchExecutor {
    inner: hierarchical_scheduler::HierarchicalBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "AIStoreBatchExecutor")]
pub struct PyAIStoreHierarchicalBatchExecutor {
    inner: hierarchical_scheduler::HierarchicalBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "S3ExecutorPool")]
pub struct PyS3HierarchicalExecutorPool {
    workers: Vec<std::sync::Mutex<hierarchical_scheduler::HierarchicalBatchExecutor>>,
}

#[cfg(feature = "ffmpeg")]
fn normalized_descriptors_from_python(
    descriptors: Vec<(u64, u64, u64, u64, u64, u64)>,
    fusion_metadata: Option<Vec<(u64, u64, usize)>>,
    fuse_shared_anchors: bool,
) -> PyResult<Vec<normalized_scheduler::NormalizedDescriptor>> {
    let metadata = fusion_metadata
        .unwrap_or_default()
        .into_iter()
        .map(|(sample_id, anchor_group_id, target_ordinal)| {
            (sample_id, (anchor_group_id, target_ordinal))
        })
        .collect::<std::collections::HashMap<_, _>>();
    if fuse_shared_anchors && metadata.len() != descriptors.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "fusion metadata covers {} samples, expected {}",
            metadata.len(),
            descriptors.len()
        )));
    }
    descriptors
        .into_iter()
        .map(
            |(sample_id, video_id, anchor_offset, anchor_length, delta_offset, delta_length)| {
                let (anchor_group_id, target_ordinal) = match metadata.get(&sample_id).copied() {
                    Some(value) => value,
                    None if fuse_shared_anchors => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "fusion metadata is missing sample {sample_id}"
                        )));
                    }
                    None => (video_id, 1),
                };
                Ok(normalized_scheduler::NormalizedDescriptor {
                    sample_id,
                    video_id,
                    anchor_group_id,
                    target_ordinal,
                    anchor_offset,
                    anchor_length,
                    delta_offset,
                    delta_length,
                })
            },
        )
        .collect()
}

#[cfg(feature = "ffmpeg")]
fn normalized_decode_schedule_from_python(
    fuse_shared_anchors: bool,
    adaptive_anchor_work_units: Option<f64>,
    adaptive_delta_work_units: f64,
) -> PyResult<normalized_scheduler::DecodeSchedule> {
    if fuse_shared_anchors && adaptive_anchor_work_units.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "forced fusion and adaptive fusion are mutually exclusive",
        ));
    }
    Ok(match adaptive_anchor_work_units {
        Some(anchor_work_units) => normalized_scheduler::DecodeSchedule::Adaptive {
            anchor_work_units,
            delta_work_units: adaptive_delta_work_units,
        },
        None if fuse_shared_anchors => normalized_scheduler::DecodeSchedule::Fused,
        None => normalized_scheduler::DecodeSchedule::Repeated,
    })
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "PairBatchExecutor")]
pub struct PyPairBatchExecutor {
    inner: pair_scheduler::ClosedRecordBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "AIStorePairBatchExecutor")]
pub struct PyAIStorePairBatchExecutor {
    inner: pair_scheduler::ClosedRecordBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "PrefixBatchExecutor")]
pub struct PyPrefixBatchExecutor {
    inner: pair_scheduler::ClosedRecordBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "AIStorePrefixBatchExecutor")]
pub struct PyAIStorePrefixBatchExecutor {
    inner: pair_scheduler::ClosedRecordBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "LocalClosedRecordBatchExecutor")]
pub struct PyLocalClosedRecordBatchExecutor {
    inner: pair_scheduler::ClosedRecordBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "AIStoreFragmentBatchExecutor")]
pub struct PyAIStoreFragmentBatchExecutor {
    inner: fragment_scheduler::FragmentBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "S3FragmentBatchExecutor")]
pub struct PyS3FragmentBatchExecutor {
    inner: fragment_scheduler::FragmentBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
#[pyclass(name = "BudgetedPairBatchExecutor")]
pub struct PyBudgetedPairBatchExecutor {
    inner: portfolio_scheduler::BudgetedPairBatchExecutor,
}

#[cfg(feature = "ffmpeg")]
fn batch_stats_dict(stats: &representation::BatchStats) -> std::collections::HashMap<String, u64> {
    let values = [
        ("logical_samples", stats.logical_samples as u64),
        ("unique_targets", stats.unique_targets as u64),
        ("unique_videos", stats.unique_videos as u64),
        ("dependency_records", stats.dependency_records as u64),
        ("target_ordinal_sum", stats.target_ordinal_sum as u64),
        ("target_ordinal_max", stats.target_ordinal_max as u64),
        ("planned_ranges", stats.planned_ranges as u64),
        ("planned_useful_bytes", stats.planned_useful_bytes),
        ("planned_fetched_bytes", stats.planned_fetched_bytes),
        (
            "cache_race_fallback_records",
            stats.cache_race_fallback_records as u64,
        ),
        ("unique_records", stats.unique_records as u64),
        ("physical_ranges", stats.physical_ranges as u64),
        ("client_requests", stats.client_requests as u64),
        ("server_entries", stats.server_entries as u64),
        ("useful_bytes", stats.useful_bytes),
        ("fetched_bytes", stats.fetched_bytes),
        ("overfetch_bytes", stats.overfetch_bytes),
        ("decoded_targets", stats.decoded_targets as u64),
        ("decoded_frames", stats.decoded_frames as u64),
        ("decode_groups", stats.decode_groups as u64),
        (
            "anchor_decode_invocations",
            stats.anchor_decode_invocations as u64,
        ),
        ("fused_decode_groups", stats.fused_decode_groups as u64),
        (
            "repeated_decode_groups",
            stats.repeated_decode_groups as u64,
        ),
        ("decoded_cache_hits", stats.decoded_cache_hits as u64),
        ("decoded_cache_misses", stats.decoded_cache_misses as u64),
        ("encoded_cache_hits", stats.encoded_cache_hits as u64),
        ("encoded_cache_misses", stats.encoded_cache_misses as u64),
        ("anchor_cache_hits", stats.anchor_cache_hits as u64),
        ("anchor_cache_misses", stats.anchor_cache_misses as u64),
        ("delta_cache_hits", stats.delta_cache_hits as u64),
        ("delta_cache_misses", stats.delta_cache_misses as u64),
        (
            "encoded_cache_resident_bytes",
            stats.encoded_cache_resident_bytes,
        ),
        (
            "anchor_cache_resident_bytes",
            stats.anchor_cache_resident_bytes,
        ),
        (
            "delta_cache_resident_bytes",
            stats.delta_cache_resident_bytes,
        ),
        (
            "decoded_cache_resident_bytes",
            stats.decoded_cache_resident_bytes,
        ),
        ("decoder_state_hits", stats.decoder_state_hits as u64),
        ("decoder_state_misses", stats.decoder_state_misses as u64),
        ("decoder_state_resets", stats.decoder_state_resets as u64),
        (
            "decoder_state_resident",
            stats.decoder_state_resident as u64,
        ),
        ("materialized_targets", stats.materialized_targets as u64),
        ("fallback_targets", stats.fallback_targets as u64),
        (
            "materialization_budget_bytes",
            stats.materialization_budget_bytes,
        ),
        (
            "materialization_used_bytes",
            stats.materialization_used_bytes,
        ),
        (
            "portfolio_parallel_branches",
            stats.portfolio_parallel_branches as u64,
        ),
        ("global_io_concurrency", stats.global_io_concurrency as u64),
        ("global_decoder_slots", stats.global_decoder_slots as u64),
        (
            "portfolio_branch_overlap_ns",
            stats.portfolio_branch_overlap_ns,
        ),
        ("pair_branch_total_ns", stats.pair_branch_total_ns),
        ("fallback_branch_total_ns", stats.fallback_branch_total_ns),
        ("pair_branch_fetch_wall_ns", stats.pair_branch_fetch_wall_ns),
        (
            "fallback_branch_fetch_wall_ns",
            stats.fallback_branch_fetch_wall_ns,
        ),
        ("pair_branch_decode_ns", stats.pair_branch_decode_ns),
        ("fallback_branch_decode_ns", stats.fallback_branch_decode_ns),
        ("resolve_ns", stats.resolve_ns),
        ("cache_lookup_ns", stats.cache_lookup_ns),
        ("plan_ns", stats.plan_ns),
        ("fetch_wall_ns", stats.fetch_wall_ns),
        ("fetch_service_ns_sum", stats.fetch_service_ns_sum),
        ("range_queue_ns_sum", stats.range_queue_ns_sum),
        ("extract_ns", stats.extract_ns),
        ("assemble_ns", stats.assemble_ns),
        ("decode_ns", stats.decode_ns),
        ("rgb_convert_ns", stats.rgb_convert_ns),
        ("assemble_decode_ns", stats.assemble_ns + stats.decode_ns),
        ("fetch_decode_overlap_ns", stats.fetch_decode_overlap_ns),
        ("reorder_ns", stats.reorder_ns),
        ("total_ns", stats.total_ns),
        ("time_to_first_ready_ns", stats.time_to_first_ready_ns),
        ("completion_selected", u64::from(stats.completion_selected)),
    ];
    values
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

#[cfg(feature = "ffmpeg")]
fn batch_features_dict(
    features: &representation::BatchFeatures,
) -> std::collections::HashMap<String, u64> {
    [
        ("logical_requests", features.logical_requests as u64),
        ("unique_targets", features.unique_targets as u64),
        ("unique_videos", features.unique_videos as u64),
        ("dependency_records", features.dependency_records as u64),
        ("unique_records", features.unique_records as u64),
        ("useful_bytes", features.useful_bytes),
        ("physical_ranges", features.physical_ranges as u64),
        ("fetched_bytes", features.fetched_bytes),
        ("overfetch_bytes", features.overfetch_bytes),
        ("mean_range_bytes", features.mean_range_bytes),
        ("max_range_bytes", features.max_range_bytes),
        ("address_span_bytes", features.address_span_bytes),
        (
            "contiguous_range_pairs",
            features.contiguous_range_pairs as u64,
        ),
        (
            "unique_anchor_records",
            features.unique_anchor_records as u64,
        ),
        ("anchor_reuse_hits", features.anchor_reuse_hits as u64),
        ("target_ordinal_sum", features.target_ordinal_sum as u64),
        ("target_ordinal_max", features.target_ordinal_max as u64),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect()
}

#[cfg(feature = "ffmpeg")]
fn mechanistic_model_from_python(
    values: &std::collections::HashMap<String, f64>,
) -> PyResult<adaptive_planner::MechanisticModel> {
    let required = |key: &str| {
        values.get(key).copied().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "mechanistic model is missing {key}"
            ))
        })
    };
    let raw_concurrency = required("io_concurrency")?;
    if !raw_concurrency.is_finite()
        || raw_concurrency < 1.0
        || raw_concurrency.fract() != 0.0
        || raw_concurrency > usize::MAX as f64
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "io_concurrency must be a positive integer",
        ));
    }
    let model = adaptive_planner::MechanisticModel {
        request_latency_ns: required("request_latency_ns")?,
        bandwidth_bytes_per_ns: required("bandwidth_bytes_per_ns")?,
        io_concurrency: raw_concurrency as usize,
        anchor_decode_ns: required("anchor_decode_ns")?,
        delta_decode_ns: required("delta_decode_ns")?,
        prefix_frame_decode_ns: required("prefix_frame_decode_ns")?,
        prefix_reset_ns: required("prefix_reset_ns")?,
        normalized_fixed_ns: required("normalized_fixed_ns")?,
        prefix_fixed_ns: required("prefix_fixed_ns")?,
        normalized_overlap: required("normalized_overlap")?,
        prefix_overlap: required("prefix_overlap")?,
    };
    model
        .validate()
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(model)
}

#[cfg(feature = "ffmpeg")]
fn adaptive_estimate_dict(
    estimate: &adaptive_planner::PlanEstimate,
) -> std::collections::HashMap<String, f64> {
    [
        ("total_ns", estimate.total_ns),
        ("io_ns", estimate.io_ns),
        ("decode_ns", estimate.decode_ns),
        ("physical_ranges", estimate.physical_ranges as f64),
        ("useful_bytes", estimate.useful_bytes as f64),
        ("fetched_bytes", estimate.fetched_bytes as f64),
        ("overfetch_bytes", estimate.overfetch_bytes as f64),
        ("anchor_decodes", estimate.anchor_decodes as f64),
        ("delta_decodes", estimate.delta_decodes as f64),
        ("prefix_frames", estimate.prefix_frames as f64),
        ("prefix_resets", estimate.prefix_resets as f64),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect()
}

#[pyclass(name = "ByteCache")]
pub struct PyByteCache {
    inner: planner::ByteCache,
}

/// Python adapter for authenticated, bounded-concurrency S3 Range GETs.
#[pyclass(name = "S3RangeReader")]
pub struct PyS3RangeReader {
    inner: backend::S3Backend,
}

/// Thin Python batch adapter over the same multi-object Rust transport exposed
/// to C/C++. Planning and per-range I/O remain in native code.
#[pyclass(name = "S3ObjectStoreReader")]
pub struct PyS3ObjectStoreReader {
    inner: backend::S3ObjectStoreClient,
}

/// Thin Python adapter over the Rust mmap backend for label-free storage probes.
#[pyclass(name = "LocalRangeReader")]
pub struct PyLocalRangeReader {
    inner: backend::LocalBackend,
}

/// Python adapter for one-request AIStore MOSS/GetBatch range retrieval.
#[pyclass(name = "AIStoreGetBatchReader")]
pub struct PyAIStoreGetBatchReader {
    inner: backend::AIStoreGetBatchBackend,
}

#[pymethods]
impl PyAIStoreGetBatchReader {
    #[new]
    #[pyo3(signature = (endpoint, bucket, object_key, provider="ais".to_string()))]
    fn new(
        endpoint: String,
        bucket: String,
        object_key: String,
        provider: String,
    ) -> PyResult<Self> {
        let inner = backend::AIStoreGetBatchBackend::new(endpoint, bucket, object_key, provider, 0)
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
        Ok(Self { inner })
    }

    fn fetch_ranges(&self, py: Python<'_>, ranges: Vec<(u64, u64)>) -> PyResult<Vec<Py<PyBytes>>> {
        let buffers = py
            .allow_threads(|| {
                backend::StorageBackend::read_byte_ranges(&self.inner, &ranges)
                    .map_err(|error| error.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(buffers
            .iter()
            .map(|buffer| PyBytes::new_bound(py, buffer).unbind())
            .collect())
    }

    fn fetch_object_ranges(
        &self,
        py: Python<'_>,
        ranges: Vec<(String, u64, u64)>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let ranges = ranges
            .into_iter()
            .map(|(object_key, offset, length)| backend::AIStoreObjectRange {
                object_key,
                offset,
                length,
            })
            .collect::<Vec<_>>();
        let buffers = py
            .allow_threads(|| {
                self.inner
                    .client()
                    .fetch_object_ranges(&ranges)
                    .map_err(|error| error.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(buffers
            .iter()
            .map(|buffer| PyBytes::new_bound(py, buffer).unbind())
            .collect())
    }
}

#[pymethods]
impl PyLocalRangeReader {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let file = std::fs::File::open(&path).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyOSError, _>(format!(
                "failed to open local probe payload {path}: {error}"
            ))
        })?;
        // SAFETY: the reader owns the read-only mapping and formal probes use
        // immutable payload artifacts for the reader lifetime.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyOSError, _>(format!(
                "failed to mmap local probe payload {path}: {error}"
            ))
        })?;
        let inner = backend::LocalBackend::new(mmap, 0);
        Ok(Self { inner })
    }

    fn fetch_ranges(&self, py: Python<'_>, ranges: Vec<(u64, u64)>) -> PyResult<Vec<Py<PyBytes>>> {
        let buffers = py
            .allow_threads(|| {
                backend::StorageBackend::read_byte_ranges(&self.inner, &ranges)
                    .map_err(|error| error.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(buffers
            .iter()
            .map(|buffer| PyBytes::new_bound(py, buffer).unbind())
            .collect())
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyAIStoreFragmentBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, endpoint, bucket,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        encoded_cache_bytes=0, decoded_cache_bytes=0,
                        decode_concurrency=8, width=320, height=240,
                        provider="ais".to_string()))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, u64, String, u64, u64, usize, Vec<u8>, bool)>,
        endpoint: String,
        bucket: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
        provider: String,
    ) -> PyResult<Self> {
        let descriptors = descriptors
            .into_iter()
            .map(
                |(
                    sample_id,
                    video_id,
                    fragment_id,
                    object_key,
                    offset,
                    length,
                    target_ordinal,
                    codec_config,
                    mp4_length_prefixed,
                )| fragment_scheduler::FragmentDescriptor {
                    sample_id,
                    video_id,
                    fragment_id,
                    object_key,
                    offset,
                    length,
                    target_ordinal,
                    codec_config,
                    mp4_length_prefixed,
                    mp4_container: false,
                },
            )
            .collect();
        let backend =
            backend::AIStoreGetBatchClient::new(endpoint, bucket, provider).map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
        let inner = fragment_scheduler::FragmentBatchExecutor::new(
            descriptors,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (targets, gops, gop_size, endpoint, bucket,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        encoded_cache_bytes=0, decoded_cache_bytes=0,
                        decode_concurrency=8, width=320, height=240,
                        provider="ais".to_string()))]
    #[allow(clippy::too_many_arguments)]
    fn from_gop_index(
        targets: Vec<(u64, u64, u64)>,
        gops: Vec<(u64, u64, u64, String, u64, u64, Vec<u8>, bool)>,
        gop_size: u64,
        endpoint: String,
        bucket: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
        provider: String,
    ) -> PyResult<Self> {
        let targets = targets
            .into_iter()
            .map(
                |(sample_id, video_id, target_frame)| fragment_scheduler::LogicalFragmentTarget {
                    sample_id,
                    video_id,
                    target_frame,
                },
            )
            .collect();
        let gops = gops
            .into_iter()
            .map(
                |(
                    video_id,
                    gop_ordinal,
                    fragment_id,
                    object_key,
                    offset,
                    length,
                    codec_config,
                    mp4_length_prefixed,
                )| fragment_scheduler::GopDescriptor {
                    video_id,
                    gop_ordinal,
                    fragment_id,
                    object_key,
                    offset,
                    length,
                    codec_config,
                    mp4_length_prefixed,
                },
            )
            .collect();
        let backend =
            backend::AIStoreGetBatchClient::new(endpoint, bucket, provider).map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
        let inner = fragment_scheduler::FragmentBatchExecutor::new_gop_index(
            targets,
            gops,
            gop_size,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (descriptors, endpoint, bucket,
                        encoded_cache_bytes=0, decoded_cache_bytes=0,
                        decode_concurrency=8, width=320, height=240,
                        provider="ais".to_string()))]
    #[allow(clippy::too_many_arguments)]
    fn from_mp4_segments(
        descriptors: Vec<(u64, u64, String, u64, usize)>,
        endpoint: String,
        bucket: String,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
        provider: String,
    ) -> PyResult<Self> {
        let descriptors = descriptors
            .into_iter()
            .map(
                |(sample_id, video_id, object_key, object_length, target_ordinal)| {
                    fragment_scheduler::FragmentDescriptor {
                        sample_id,
                        video_id,
                        fragment_id: video_id,
                        object_key,
                        offset: 0,
                        length: object_length,
                        target_ordinal,
                        codec_config: Vec::new(),
                        mp4_length_prefixed: false,
                        mp4_container: true,
                    }
                },
            )
            .collect();
        let backend =
            backend::AIStoreGetBatchClient::new(endpoint, bucket, provider).map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
        let inner = fragment_scheduler::FragmentBatchExecutor::new(
            descriptors,
            Box::new(backend),
            None,
            None,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyS3FragmentBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, endpoint, bucket, access_key_id,
                        secret_access_key, merge_threshold_bytes=None,
                        max_range_bytes=None, encoded_cache_bytes=0,
                        decoded_cache_bytes=0, decode_concurrency=8,
                        width=320, height=240, region="us-east-1".to_string(),
                        max_concurrency=8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, u64, String, u64, u64, usize, Vec<u8>, bool)>,
        endpoint: String,
        bucket: String,
        access_key_id: String,
        secret_access_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
        region: String,
        max_concurrency: usize,
    ) -> PyResult<Self> {
        let descriptors = descriptors
            .into_iter()
            .map(
                |(
                    sample_id,
                    video_id,
                    fragment_id,
                    object_key,
                    offset,
                    length,
                    target_ordinal,
                    codec_config,
                    mp4_length_prefixed,
                )| fragment_scheduler::FragmentDescriptor {
                    sample_id,
                    video_id,
                    fragment_id,
                    object_key,
                    offset,
                    length,
                    target_ordinal,
                    codec_config,
                    mp4_length_prefixed,
                    mp4_container: false,
                },
            )
            .collect();
        let backend = backend::S3ObjectStoreClient::new(
            endpoint,
            bucket,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let inner = fragment_scheduler::FragmentBatchExecutor::new(
            descriptors,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (targets, gops, gop_size, endpoint, bucket, access_key_id,
                        secret_access_key, merge_threshold_bytes=None,
                        max_range_bytes=None, encoded_cache_bytes=0,
                        decoded_cache_bytes=0, decode_concurrency=8,
                        width=320, height=240, region="us-east-1".to_string(),
                        max_concurrency=8))]
    #[allow(clippy::too_many_arguments)]
    fn from_gop_index(
        targets: Vec<(u64, u64, u64)>,
        gops: Vec<(u64, u64, u64, String, u64, u64, Vec<u8>, bool)>,
        gop_size: u64,
        endpoint: String,
        bucket: String,
        access_key_id: String,
        secret_access_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
        region: String,
        max_concurrency: usize,
    ) -> PyResult<Self> {
        let targets = targets
            .into_iter()
            .map(
                |(sample_id, video_id, target_frame)| fragment_scheduler::LogicalFragmentTarget {
                    sample_id,
                    video_id,
                    target_frame,
                },
            )
            .collect();
        let gops = gops
            .into_iter()
            .map(
                |(
                    video_id,
                    gop_ordinal,
                    fragment_id,
                    object_key,
                    offset,
                    length,
                    codec_config,
                    mp4_length_prefixed,
                )| fragment_scheduler::GopDescriptor {
                    video_id,
                    gop_ordinal,
                    fragment_id,
                    object_key,
                    offset,
                    length,
                    codec_config,
                    mp4_length_prefixed,
                },
            )
            .collect();
        let backend = backend::S3ObjectStoreClient::new(
            endpoint,
            bucket,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let inner = fragment_scheduler::FragmentBatchExecutor::new_gop_index(
            targets,
            gops,
            gop_size,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (descriptors, endpoint, bucket, access_key_id,
                        secret_access_key, encoded_cache_bytes=0,
                        decoded_cache_bytes=0, decode_concurrency=8,
                        width=320, height=240, region="us-east-1".to_string(),
                        max_concurrency=8))]
    #[allow(clippy::too_many_arguments)]
    fn from_mp4_segments(
        descriptors: Vec<(u64, u64, String, u64, usize)>,
        endpoint: String,
        bucket: String,
        access_key_id: String,
        secret_access_key: String,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
        region: String,
        max_concurrency: usize,
    ) -> PyResult<Self> {
        let descriptors = descriptors
            .into_iter()
            .map(
                |(sample_id, video_id, object_key, object_length, target_ordinal)| {
                    fragment_scheduler::FragmentDescriptor {
                        sample_id,
                        video_id,
                        fragment_id: video_id,
                        object_key,
                        offset: 0,
                        length: object_length,
                        target_ordinal,
                        codec_config: Vec::new(),
                        mp4_length_prefixed: false,
                        mp4_container: true,
                    }
                },
            )
            .collect();
        let backend = backend::S3ObjectStoreClient::new(
            endpoint,
            bucket,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let inner = fragment_scheduler::FragmentBatchExecutor::new(
            descriptors,
            Box::new(backend),
            None,
            None,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[pymethods]
impl PyS3RangeReader {
    #[new]
    #[pyo3(signature = (endpoint, bucket, object_key, access_key_id, secret_access_key,
                        region="us-east-1".to_string(), max_concurrency=8))]
    fn new(
        endpoint: String,
        bucket: String,
        object_key: String,
        access_key_id: String,
        secret_access_key: String,
        region: String,
        max_concurrency: usize,
    ) -> PyResult<Self> {
        let inner = backend::S3Backend::new(
            endpoint,
            bucket,
            object_key,
            0,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    fn fetch_ranges(&self, py: Python<'_>, ranges: Vec<(u64, u64)>) -> PyResult<Vec<Py<PyBytes>>> {
        let buffers = py
            .allow_threads(|| {
                backend::StorageBackend::read_byte_ranges(&self.inner, &ranges)
                    .map_err(|e| e.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(buffers
            .iter()
            .map(|buffer| PyBytes::new_bound(py, buffer).unbind())
            .collect())
    }

    /// Identical to `fetch_ranges`, with one `(index, started_ns,
    /// completed_ns)` tuple per range for transport breakdown experiments.
    fn fetch_ranges_profiled(
        &self,
        py: Python<'_>,
        ranges: Vec<(u64, u64)>,
    ) -> PyResult<(Vec<Py<PyBytes>>, Vec<(usize, u64, u64, u64)>)> {
        let (buffers, timings) = py
            .allow_threads(|| {
                let mut buffers: Vec<Option<Vec<u8>>> =
                    std::iter::repeat_with(|| None).take(ranges.len()).collect();
                let mut timings = Vec::with_capacity(ranges.len());
                backend::StorageBackend::for_each_byte_range(
                    &self.inner,
                    &ranges,
                    &mut |completed| {
                        timings.push((
                            completed.index,
                            completed.started_ns,
                            completed.first_byte_ns,
                            completed.completed_ns,
                        ));
                        buffers[completed.index] = Some(completed.bytes);
                        Ok(())
                    },
                )
                .map_err(|error| error.to_string())?;
                let buffers = buffers
                    .into_iter()
                    .map(|buffer| {
                        buffer.ok_or_else(|| "S3 backend omitted a requested range".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok::<_, String>((buffers, timings))
            })
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
        let py_buffers = buffers
            .iter()
            .map(|buffer| PyBytes::new_bound(py, buffer).unbind())
            .collect();
        Ok((py_buffers, timings))
    }
}

#[pymethods]
impl PyS3ObjectStoreReader {
    #[new]
    #[pyo3(signature = (endpoint, bucket, access_key_id, secret_access_key,
                        region="us-east-1".to_string(), max_concurrency=8))]
    fn new(
        endpoint: String,
        bucket: String,
        access_key_id: String,
        secret_access_key: String,
        region: String,
        max_concurrency: usize,
    ) -> PyResult<Self> {
        let inner = backend::S3ObjectStoreClient::new(
            endpoint,
            bucket,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        Ok(Self { inner })
    }

    fn fetch_object_ranges(
        &self,
        py: Python<'_>,
        ranges: Vec<(String, u64, u64)>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let ranges = ranges
            .into_iter()
            .map(|(object_key, offset, length)| backend::ObjectRange {
                object_key,
                offset,
                length,
            })
            .collect::<Vec<_>>();
        let buffers = py
            .allow_threads(|| {
                self.inner
                    .fetch_object_ranges(&ranges)
                    .map_err(|error| error.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(buffers
            .iter()
            .map(|buffer| PyBytes::new_bound(py, buffer).unbind())
            .collect())
    }

    /// Generic baseline backed by object_store::ObjectStore::get_ranges.
    fn fetch_object_ranges_vectored(
        &self,
        py: Python<'_>,
        ranges: Vec<(String, u64, u64)>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let ranges = ranges
            .into_iter()
            .map(|(object_key, offset, length)| backend::ObjectRange {
                object_key,
                offset,
                length,
            })
            .collect::<Vec<_>>();
        let buffers = py
            .allow_threads(|| {
                self.inner
                    .fetch_object_ranges_vectored(&ranges)
                    .map_err(|error| error.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(buffers
            .iter()
            .map(|buffer| PyBytes::new_bound(py, buffer).unbind())
            .collect())
    }

    fn head_object(&self, object_key: String) -> PyResult<(u64, Option<String>, Option<String>)> {
        let head = self.inner.head_object(&object_key).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
        })?;
        Ok((head.size, head.e_tag, head.version))
    }

    fn fetch_object_ranges_profiled(
        &self,
        py: Python<'_>,
        ranges: Vec<(String, u64, u64)>,
    ) -> PyResult<(Vec<Py<PyBytes>>, Vec<(usize, u64, u64, u64)>)> {
        let ranges = ranges
            .into_iter()
            .map(|(object_key, offset, length)| backend::ObjectRange {
                object_key,
                offset,
                length,
            })
            .collect::<Vec<_>>();
        let (buffers, timings) = py
            .allow_threads(|| {
                let mut buffers: Vec<Option<Vec<u8>>> =
                    std::iter::repeat_with(|| None).take(ranges.len()).collect();
                let mut timings = Vec::with_capacity(ranges.len());
                self.inner
                    .for_each_object_range(&ranges, &mut |completed| {
                        timings.push((
                            completed.index,
                            completed.started_ns,
                            completed.first_byte_ns,
                            completed.completed_ns,
                        ));
                        buffers[completed.index] = Some(completed.bytes);
                        Ok(())
                    })
                    .map_err(|error| error.to_string())?;
                let buffers = buffers
                    .into_iter()
                    .map(|buffer| {
                        buffer.ok_or_else(|| "S3 client omitted a requested range".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok::<_, String>((buffers, timings))
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let py_buffers = buffers
            .iter()
            .map(|buffer| PyBytes::new_bound(py, buffer).unbind())
            .collect();
        Ok((py_buffers, timings))
    }
}

#[pymethods]
impl PyByteCache {
    #[new]
    fn new(capacity_bytes: usize) -> Self {
        Self {
            inner: planner::ByteCache::new(capacity_bytes),
        }
    }

    fn get(&mut self, py: Python<'_>, record_id: u64) -> Option<Py<PyBytes>> {
        self.inner
            .get(record_id)
            .map(|value| PyBytes::new_bound(py, &value).unbind())
    }

    fn put(&mut self, record_id: u64, value: &[u8]) -> bool {
        self.inner.put(record_id, value.to_vec())
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn stats(&self) -> std::collections::HashMap<String, u64> {
        self.inner.stats()
    }
}

#[pyfunction(name = "plan_byte_ranges")]
#[pyo3(signature = (records, merge_threshold_bytes=None, max_range_bytes=None))]
fn plan_byte_ranges_py(
    records: Vec<(u64, u64, u64)>,
    merge_threshold_bytes: Option<u64>,
    max_range_bytes: Option<u64>,
) -> PyResult<Vec<(u64, u64, Vec<(u64, u64, u64)>)>> {
    let records: Vec<planner::RecordRange> = records
        .into_iter()
        .map(|(record_id, offset, length)| planner::RecordRange {
            record_id,
            offset,
            length,
        })
        .collect();
    let plans = planner::plan_byte_ranges(&records, merge_threshold_bytes, max_range_bytes)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(plans
        .into_iter()
        .map(|plan| {
            (
                plan.offset,
                plan.length,
                plan.records
                    .into_iter()
                    .map(|record| (record.record_id, record.relative_offset, record.length))
                    .collect(),
            )
        })
        .collect())
}

#[pyfunction]
fn select_pair_materialization(
    candidates: Vec<(u64, u64)>,
    calibration_trace: Vec<u64>,
    budget_bytes: u64,
) -> PyResult<(Vec<u64>, u64, u64, usize, usize)> {
    let candidates = candidates
        .into_iter()
        .map(|(sample_id, bytes)| materialization::PairCandidate { sample_id, bytes })
        .collect::<Vec<_>>();
    let selected =
        materialization::select_profiled_pairs(&candidates, &calibration_trace, budget_bytes)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((
        selected.sample_ids,
        selected.used_bytes,
        selected.budget_bytes,
        selected.calibration_requests,
        selected.calibration_unique,
    ))
}

/// Materialize closed Pair records from Ground Truth Normalized extents.
///
/// Rust owns all payload reads, validation, ordering, and atomic output. Python
/// supplies structured index rows and may persist the returned Pair index.
#[pyfunction]
#[pyo3(signature = (source_path, output_path, descriptors, selected_sample_ids=None))]
fn materialize_pair_records(
    source_path: &str,
    output_path: &str,
    descriptors: Vec<(u64, u64, u64, u64, u64, u64, usize)>,
    selected_sample_ids: Option<Vec<u64>>,
) -> PyResult<(Vec<(u64, u64, u64, u64, usize)>, u64)> {
    let descriptors = descriptors
        .into_iter()
        .map(
            |(
                sample_id,
                video_id,
                anchor_offset,
                anchor_length,
                delta_offset,
                delta_length,
                target_ordinal,
            )| materialization::PairSourceDescriptor {
                sample_id,
                video_id,
                anchor_offset,
                anchor_length,
                delta_offset,
                delta_length,
                target_ordinal,
            },
        )
        .collect::<Vec<_>>();
    let materialized = materialization::materialize_pairs(
        Path::new(source_path),
        Path::new(output_path),
        &descriptors,
        selected_sample_ids.as_deref(),
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok((
        materialized
            .descriptors
            .into_iter()
            .map(|descriptor| {
                (
                    descriptor.sample_id,
                    descriptor.video_id,
                    descriptor.offset,
                    descriptor.length,
                    descriptor.target_ordinal,
                )
            })
            .collect(),
        materialized.bytes,
    ))
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyH264Decoder {
    #[new]
    fn new(num_threads: usize) -> Self {
        Self {
            decoder_pool: decoder::DecoderPool::new(decoder::DecoderConfig { num_threads }),
        }
    }

    /// Decode a closed record and return its final RGB24 frame.
    ///
    /// `codec_config` contains SPS/PPS Annex-B NALs. `record` starts with an
    /// IDR and may contain dependent P slices, such as one normalized I+Dk
    /// closure or an ordinary GOP prefix.
    fn decode_record_target_rgb24(
        &mut self,
        py: Python<'_>,
        codec_config: &[u8],
        record: &[u8],
    ) -> PyResult<(Py<PyBytes>, u32, u32)> {
        let frames = decoder::decode_gop_rgb24(codec_config, record, &mut self.decoder_pool)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let frame = frames.last().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("record decoded no frames")
        })?;
        Ok((
            PyBytes::new_bound(py, &frame.data).unbind(),
            frame.width,
            frame.height,
        ))
    }

    /// Decode one self-contained Annex-B fragment and return one selected
    /// display-order frame. The fragment must contain SPS/PPS and begin with
    /// an IDR; `target_ordinal` is zero-based within the fragment.
    fn decode_self_contained_record_selected_rgb24(
        &mut self,
        py: Python<'_>,
        payload: &[u8],
        target_ordinal: usize,
    ) -> PyResult<(Py<PyBytes>, u32, u32)> {
        let (codec_config, record, frame_count) = decoder::extract_closed_record_parts(payload)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        if target_ordinal >= frame_count {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "target ordinal {} outside fragment frame count {}",
                target_ordinal, frame_count
            )));
        }
        let frames = py
            .allow_threads(|| {
                decoder::decode_full_gop_selected_rgb24(
                    &codec_config,
                    &record,
                    &mut self.decoder_pool,
                    &[target_ordinal],
                )
                .map_err(|e| e.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frame = frames.into_iter().next().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "selected-fragment decode produced no frame",
            )
        })?;
        Ok((
            PyBytes::new_bound(py, &frame.data).unbind(),
            frame.width,
            frame.height,
        ))
    }

    /// Decode one complete self-contained GOP once and return multiple
    /// selected display-order frames in the requested order.
    fn decode_self_contained_record_selected_many_rgb24(
        &mut self,
        py: Python<'_>,
        payload: &[u8],
        target_ordinals: Vec<usize>,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        let (codec_config, record, frame_count) = decoder::extract_closed_record_parts(payload)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        if let Some(target_ordinal) = target_ordinals
            .iter()
            .copied()
            .find(|ordinal| *ordinal >= frame_count)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "target ordinal {} outside fragment frame count {}",
                target_ordinal, frame_count
            )));
        }
        let frames = py
            .allow_threads(|| {
                decoder::decode_full_gop_selected_rgb24(
                    &codec_config,
                    &record,
                    &mut self.decoder_pool,
                    &target_ordinals,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(frames
            .into_iter()
            .map(|frame| {
                (
                    PyBytes::new_bound(py, &frame.data).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect())
    }

    /// Decode selected display-order frames from one complete MP4 segment.
    /// The caller supplies the segment and ordinals; this adapter performs no
    /// dependency planning, indexing, caching, or range coalescing.
    fn decode_mp4_selected_rgb24(
        &self,
        py: Python<'_>,
        payload: &[u8],
        target_ordinals: Vec<usize>,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        let frames = py
            .allow_threads(|| {
                decoder::decode_mp4_selected_rgb24(
                    payload,
                    &target_ordinals,
                    decoder::DecoderConfig { num_threads: 1 },
                )
                .map_err(|error| error.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(frames
            .into_iter()
            .map(|frame| {
                (
                    PyBytes::new_bound(py, &frame.data).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect())
    }

    /// Batch-decode fixed-size closed records and return the final frame of
    /// each record. Records share one codec configuration and decoder pool.
    fn decode_fixed_records_target_rgb24(
        &mut self,
        py: Python<'_>,
        codec_config: &[u8],
        records: Vec<Vec<u8>>,
        frames_per_record: usize,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        if frames_per_record == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "frames_per_record must be positive",
            ));
        }
        let frames = py
            .allow_threads(|| {
                decoder::decode_closed_targets_continuous_rgb24(
                    codec_config,
                    &records,
                    &mut self.decoder_pool,
                    frames_per_record,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        if frames.len() != records.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "decoded {} targets from {} records",
                frames.len(),
                records.len(),
            )));
        }
        Ok(frames
            .iter()
            .map(|frame| {
                (
                    PyBytes::new_bound(py, &frame.data).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect())
    }

    /// Decode normalized Anchor+Delta records. Anchor parsing and record
    /// assembly stay in Rust; only final RGB targets cross the Python boundary.
    fn decode_anchor_delta_records_target_rgb24(
        &mut self,
        py: Python<'_>,
        anchor: &[u8],
        deltas: Vec<Vec<u8>>,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        let (codec_config, idr) = decoder::extract_anchor_parts(anchor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let records: Vec<Vec<u8>> = deltas
            .iter()
            .map(|delta| {
                let mut record = Vec::with_capacity(idr.len() + delta.len());
                record.extend_from_slice(&idr);
                record.extend_from_slice(delta);
                record
            })
            .collect();
        let frames = py
            .allow_threads(|| {
                decoder::decode_closed_targets_continuous_rgb24(
                    &codec_config,
                    &records,
                    &mut self.decoder_pool,
                    2,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        if frames.len() != records.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "decoded {} targets from {} Anchor+Delta records",
                frames.len(),
                records.len(),
            )));
        }
        Ok(frames
            .iter()
            .map(|frame| {
                (
                    PyBytes::new_bound(py, &frame.data).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect())
    }

    /// Decode an ordered union of independently anchor-referenced Deltas.
    /// The shared Anchor enters libavcodec once for the entire group.
    fn decode_shared_anchor_deltas_target_rgb24(
        &mut self,
        py: Python<'_>,
        anchor: &[u8],
        deltas: Vec<Vec<u8>>,
    ) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
        let frames = py
            .allow_threads(|| {
                decoder::decode_shared_anchor_deltas_rgb24(anchor, &deltas, &mut self.decoder_pool)
                    .map_err(|error| error.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        if frames.len() != deltas.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "decoded {} targets from {} shared-Anchor Deltas",
                frames.len(),
                deltas.len(),
            )));
        }
        Ok(frames
            .iter()
            .map(|frame| {
                (
                    PyBytes::new_bound(py, &frame.data).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect())
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyNormalizedBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, endpoint, bucket, object_key, access_key_id,
                        secret_access_key, merge_threshold_bytes=None,
                        max_range_bytes=None, encoded_cache_bytes=0,
                        delta_cache_bytes=0, decoded_cache_bytes=0,
                        decode_concurrency=8,
                        decode_microbatch_targets=4, width=320, height=240,
                        region="us-east-1".to_string(), max_concurrency=8,
                        fusion_metadata=None, fuse_shared_anchors=false,
                        adaptive_anchor_work_units=None,
                        adaptive_delta_work_units=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, u64, u64, u64, u64)>,
        endpoint: String,
        bucket: String,
        object_key: String,
        access_key_id: String,
        secret_access_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        delta_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        decode_microbatch_targets: usize,
        width: u32,
        height: u32,
        region: String,
        max_concurrency: usize,
        fusion_metadata: Option<Vec<(u64, u64, usize)>>,
        fuse_shared_anchors: bool,
        adaptive_anchor_work_units: Option<f64>,
        adaptive_delta_work_units: f64,
    ) -> PyResult<Self> {
        let decode_schedule = normalized_decode_schedule_from_python(
            fuse_shared_anchors,
            adaptive_anchor_work_units,
            adaptive_delta_work_units,
        )?;
        let descriptors = normalized_descriptors_from_python(
            descriptors,
            fusion_metadata,
            decode_schedule.requires_group_metadata(),
        )?;
        let backend = backend::S3Backend::new(
            endpoint,
            bucket,
            object_key,
            0,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let inner = normalized_scheduler::NormalizedBatchExecutor::new_with_decode_schedule(
            descriptors,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            delta_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            decode_microbatch_targets,
            decode_schedule,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn plan_features(&self, batch: Vec<u64>) -> PyResult<std::collections::HashMap<String, u64>> {
        self.inner
            .plan_features(&batch)
            .map(|features| batch_features_dict(&features))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    #[pyo3(signature = (
        batches, candidates, first_batch_slo_ms, cost_model,
        wave_request_overhead_ns
    ))]
    fn plan_lookahead(
        &self,
        batches: Vec<Vec<u64>>,
        candidates: Vec<usize>,
        first_batch_slo_ms: f64,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
    ) -> PyResult<(usize, bool, Vec<std::collections::HashMap<String, f64>>)> {
        normalized_lookahead_from_python(
            &self.inner,
            batches,
            candidates,
            first_batch_slo_ms,
            cost_model,
            wave_request_overhead_ns,
        )
    }

    #[pyo3(signature = (
        batches, candidates, first_batch_slo_ms, cost_model,
        wave_request_overhead_ns, completion_driven=true
    ))]
    fn execute_planned_lookahead(
        &mut self,
        py: Python<'_>,
        batches: Vec<Vec<u64>>,
        candidates: Vec<usize>,
        first_batch_slo_ms: f64,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        completion_driven: bool,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        usize,
        bool,
        u64,
        Vec<std::collections::HashMap<String, f64>>,
    )> {
        execute_normalized_lookahead_from_python(
            &mut self.inner,
            py,
            batches,
            candidates,
            first_batch_slo_ms,
            cost_model,
            wave_request_overhead_ns,
            completion_driven,
        )
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
        completion_driven: bool,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch, completion_driven))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyAIStoreNormalizedBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, endpoint, bucket, object_key,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        encoded_cache_bytes=0, delta_cache_bytes=0,
                        decoded_cache_bytes=0, decode_concurrency=8,
                        decode_microbatch_targets=4,
                        width=320, height=240, provider="ais".to_string(),
                        fusion_metadata=None, fuse_shared_anchors=false,
                        adaptive_anchor_work_units=None,
                        adaptive_delta_work_units=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, u64, u64, u64, u64)>,
        endpoint: String,
        bucket: String,
        object_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        delta_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        decode_microbatch_targets: usize,
        width: u32,
        height: u32,
        provider: String,
        fusion_metadata: Option<Vec<(u64, u64, usize)>>,
        fuse_shared_anchors: bool,
        adaptive_anchor_work_units: Option<f64>,
        adaptive_delta_work_units: f64,
    ) -> PyResult<Self> {
        let decode_schedule = normalized_decode_schedule_from_python(
            fuse_shared_anchors,
            adaptive_anchor_work_units,
            adaptive_delta_work_units,
        )?;
        let descriptors = normalized_descriptors_from_python(
            descriptors,
            fusion_metadata,
            decode_schedule.requires_group_metadata(),
        )?;
        let backend =
            backend::AIStoreGetBatchBackend::new(endpoint, bucket, object_key, provider, 0)
                .map_err(|error| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
                })?;
        let inner = normalized_scheduler::NormalizedBatchExecutor::new_with_decode_schedule(
            descriptors,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            delta_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            decode_microbatch_targets,
            decode_schedule,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn plan_features(&self, batch: Vec<u64>) -> PyResult<std::collections::HashMap<String, u64>> {
        self.inner
            .plan_features(&batch)
            .map(|features| batch_features_dict(&features))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    #[pyo3(signature = (
        batches, candidates, first_batch_slo_ms, cost_model,
        wave_request_overhead_ns
    ))]
    fn plan_lookahead(
        &self,
        batches: Vec<Vec<u64>>,
        candidates: Vec<usize>,
        first_batch_slo_ms: f64,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
    ) -> PyResult<(usize, bool, Vec<std::collections::HashMap<String, f64>>)> {
        normalized_lookahead_from_python(
            &self.inner,
            batches,
            candidates,
            first_batch_slo_ms,
            cost_model,
            wave_request_overhead_ns,
        )
    }

    #[pyo3(signature = (
        batches, candidates, first_batch_slo_ms, cost_model,
        wave_request_overhead_ns, completion_driven=true
    ))]
    fn execute_planned_lookahead(
        &mut self,
        py: Python<'_>,
        batches: Vec<Vec<u64>>,
        candidates: Vec<usize>,
        first_batch_slo_ms: f64,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        completion_driven: bool,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        usize,
        bool,
        u64,
        Vec<std::collections::HashMap<String, f64>>,
    )> {
        execute_normalized_lookahead_from_python(
            &mut self.inner,
            py,
            batches,
            candidates,
            first_batch_slo_ms,
            cost_model,
            wave_request_overhead_ns,
            completion_driven,
        )
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
        completion_driven: bool,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch, completion_driven))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyLocalNormalizedBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, path,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        encoded_cache_bytes=0, delta_cache_bytes=0,
                        decoded_cache_bytes=0, decode_concurrency=8,
                        decode_microbatch_targets=4,
                        width=320, height=240, fusion_metadata=None,
                        fuse_shared_anchors=false,
                        adaptive_anchor_work_units=None,
                        adaptive_delta_work_units=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, u64, u64, u64, u64)>,
        path: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        delta_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        decode_microbatch_targets: usize,
        width: u32,
        height: u32,
        fusion_metadata: Option<Vec<(u64, u64, usize)>>,
        fuse_shared_anchors: bool,
        adaptive_anchor_work_units: Option<f64>,
        adaptive_delta_work_units: f64,
    ) -> PyResult<Self> {
        let decode_schedule = normalized_decode_schedule_from_python(
            fuse_shared_anchors,
            adaptive_anchor_work_units,
            adaptive_delta_work_units,
        )?;
        let descriptors = normalized_descriptors_from_python(
            descriptors,
            fusion_metadata,
            decode_schedule.requires_group_metadata(),
        )?;
        let file = std::fs::File::open(&path).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyOSError, _>(format!(
                "failed to open normalized payload {path}: {error}"
            ))
        })?;
        // SAFETY: the executor owns the read-only mapping and the benchmark
        // contract requires immutable payload artifacts for its lifetime.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyOSError, _>(format!(
                "failed to mmap normalized payload {path}: {error}"
            ))
        })?;
        let inner = normalized_scheduler::NormalizedBatchExecutor::new_with_decode_schedule(
            descriptors,
            Box::new(backend::LocalBackend::new(mmap, 0)),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            delta_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            decode_microbatch_targets,
            decode_schedule,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn plan_features(&self, batch: Vec<u64>) -> PyResult<std::collections::HashMap<String, u64>> {
        self.inner
            .plan_features(&batch)
            .map(|features| batch_features_dict(&features))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    #[pyo3(signature = (
        batches, candidates, first_batch_slo_ms, cost_model,
        wave_request_overhead_ns
    ))]
    fn plan_lookahead(
        &self,
        batches: Vec<Vec<u64>>,
        candidates: Vec<usize>,
        first_batch_slo_ms: f64,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
    ) -> PyResult<(usize, bool, Vec<std::collections::HashMap<String, f64>>)> {
        normalized_lookahead_from_python(
            &self.inner,
            batches,
            candidates,
            first_batch_slo_ms,
            cost_model,
            wave_request_overhead_ns,
        )
    }

    #[pyo3(signature = (
        batches, candidates, first_batch_slo_ms, cost_model,
        wave_request_overhead_ns, completion_driven=true
    ))]
    fn execute_planned_lookahead(
        &mut self,
        py: Python<'_>,
        batches: Vec<Vec<u64>>,
        candidates: Vec<usize>,
        first_batch_slo_ms: f64,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        completion_driven: bool,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        usize,
        bool,
        u64,
        Vec<std::collections::HashMap<String, f64>>,
    )> {
        execute_normalized_lookahead_from_python(
            &mut self.inner,
            py,
            batches,
            candidates,
            first_batch_slo_ms,
            cost_model,
            wave_request_overhead_ns,
            completion_driven,
        )
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
        completion_driven: bool,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch, completion_driven))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyPairBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, endpoint, bucket, object_key, access_key_id,
                        secret_access_key, merge_threshold_bytes=None,
                        max_range_bytes=None, encoded_cache_bytes=0,
                        decoded_cache_bytes=0, decode_concurrency=8,
                        width=320, height=240, region="us-east-1".to_string(),
                        max_concurrency=8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, u64, u64)>,
        endpoint: String,
        bucket: String,
        object_key: String,
        access_key_id: String,
        secret_access_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
        region: String,
        max_concurrency: usize,
    ) -> PyResult<Self> {
        let descriptors = descriptors
            .into_iter()
            .map(
                |(sample_id, video_id, offset, length)| pair_scheduler::ClosedRecordDescriptor {
                    sample_id,
                    video_id,
                    offset,
                    length,
                    target_ordinal: 1,
                },
            )
            .collect();
        let backend = backend::S3Backend::new(
            endpoint,
            bucket,
            object_key,
            0,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let inner = pair_scheduler::ClosedRecordBatchExecutor::new(
            descriptors,
            representation::Representation::Pair,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            false,
            0,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn plan_features(&self, batch: Vec<u64>) -> PyResult<std::collections::HashMap<String, u64>> {
        self.inner
            .plan_features(&batch)
            .map(|features| batch_features_dict(&features))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyAIStorePairBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, endpoint, bucket, object_key,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        encoded_cache_bytes=0, decoded_cache_bytes=0,
                        decode_concurrency=8, width=320, height=240,
                        provider="ais".to_string()))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, u64, u64)>,
        endpoint: String,
        bucket: String,
        object_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        width: u32,
        height: u32,
        provider: String,
    ) -> PyResult<Self> {
        let descriptors = descriptors
            .into_iter()
            .map(
                |(sample_id, video_id, offset, length)| pair_scheduler::ClosedRecordDescriptor {
                    sample_id,
                    video_id,
                    offset,
                    length,
                    target_ordinal: 1,
                },
            )
            .collect();
        let backend =
            backend::AIStoreGetBatchBackend::new(endpoint, bucket, object_key, provider, 0)
                .map_err(|error| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
                })?;
        let inner = pair_scheduler::ClosedRecordBatchExecutor::new(
            descriptors,
            representation::Representation::Pair,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            false,
            0,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn plan_features(&self, batch: Vec<u64>) -> PyResult<std::collections::HashMap<String, u64>> {
        self.inner
            .plan_features(&batch)
            .map(|features| batch_features_dict(&features))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyBudgetedPairBatchExecutor {
    #[new]
    #[pyo3(signature = (normalized_descriptors, pair_descriptors, endpoint, bucket,
                        normalized_object_key, pair_object_key, access_key_id,
                        secret_access_key, materialization_budget_bytes,
                        materialization_used_bytes, merge_threshold_bytes=None,
                        max_range_bytes=None, normalized_encoded_cache_bytes=0,
                        pair_encoded_cache_bytes=0, normalized_decoded_cache_bytes=0,
                        pair_decoded_cache_bytes=0, decode_concurrency=8,
                        decode_microbatch_targets=4, width=320, height=240,
                        region="us-east-1".to_string(), max_concurrency=8,
                        fusion_metadata=None, pair_target_ordinals=None,
                        fuse_shared_anchors=false,
                        adaptive_anchor_work_units=None,
                        adaptive_delta_work_units=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        normalized_descriptors: Vec<(u64, u64, u64, u64, u64, u64)>,
        pair_descriptors: Vec<(u64, u64, u64, u64)>,
        endpoint: String,
        bucket: String,
        normalized_object_key: String,
        pair_object_key: String,
        access_key_id: String,
        secret_access_key: String,
        materialization_budget_bytes: u64,
        materialization_used_bytes: u64,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        normalized_encoded_cache_bytes: usize,
        pair_encoded_cache_bytes: usize,
        normalized_decoded_cache_bytes: usize,
        pair_decoded_cache_bytes: usize,
        decode_concurrency: usize,
        decode_microbatch_targets: usize,
        width: u32,
        height: u32,
        region: String,
        max_concurrency: usize,
        fusion_metadata: Option<Vec<(u64, u64, usize)>>,
        pair_target_ordinals: Option<Vec<(u64, usize)>>,
        fuse_shared_anchors: bool,
        adaptive_anchor_work_units: Option<f64>,
        adaptive_delta_work_units: f64,
    ) -> PyResult<Self> {
        let decode_schedule = normalized_decode_schedule_from_python(
            fuse_shared_anchors,
            adaptive_anchor_work_units,
            adaptive_delta_work_units,
        )?;
        let normalized_descriptors = normalized_descriptors_from_python(
            normalized_descriptors,
            fusion_metadata,
            decode_schedule.requires_group_metadata(),
        )?;
        let require_pair_target_ordinals = pair_target_ordinals.is_some();
        let pair_target_ordinal_rows = pair_target_ordinals.unwrap_or_default();
        let pair_target_ordinals = pair_target_ordinal_rows
            .iter()
            .copied()
            .collect::<std::collections::HashMap<_, _>>();
        if pair_target_ordinals.len() != pair_target_ordinal_rows.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pair target-ordinal metadata contains duplicate sample IDs",
            ));
        }
        if pair_target_ordinals.values().any(|ordinal| *ordinal > 1) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "closed Pair target ordinals must be 0 or 1",
            ));
        }
        let pair_ids = pair_descriptors
            .iter()
            .map(|(sample_id, _, _, _)| *sample_id)
            .collect::<std::collections::HashSet<_>>();
        if pair_target_ordinals
            .keys()
            .any(|sample_id| !pair_ids.contains(sample_id))
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pair target-ordinal metadata references an unknown sample",
            ));
        }
        let pair_descriptors = pair_descriptors
            .into_iter()
            .map(|(sample_id, video_id, offset, length)| {
                let target_ordinal = match pair_target_ordinals.get(&sample_id).copied() {
                    Some(value) => value,
                    None if require_pair_target_ordinals => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Pair target-ordinal metadata is missing sample {sample_id}"
                        )));
                    }
                    None => 1,
                };
                Ok(pair_scheduler::ClosedRecordDescriptor {
                    sample_id,
                    video_id,
                    offset,
                    length,
                    target_ordinal,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let shared_client = backend::S3ObjectStoreClient::new(
            endpoint,
            bucket,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let normalized_backend =
            backend::S3Backend::from_shared_client(shared_client.clone(), normalized_object_key, 0)
                .map_err(|error| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
                })?;
        let pair_backend =
            backend::S3Backend::from_shared_client(shared_client, pair_object_key, 0).map_err(
                |error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()),
            )?;
        let inner = portfolio_scheduler::BudgetedPairBatchExecutor::new(
            normalized_descriptors,
            pair_descriptors,
            Box::new(normalized_backend),
            Box::new(pair_backend),
            merge_threshold_bytes,
            max_range_bytes,
            normalized_encoded_cache_bytes,
            pair_encoded_cache_bytes,
            normalized_decoded_cache_bytes,
            pair_decoded_cache_bytes,
            decode_concurrency,
            decode_microbatch_targets,
            decode_schedule,
            max_concurrency,
            materialization_budget_bytes,
            materialization_used_bytes,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
        completion_driven: bool,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch, completion_driven))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyPrefixBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, endpoint, bucket, object_key, access_key_id,
                        secret_access_key, merge_threshold_bytes=None,
                        max_range_bytes=None, encoded_cache_bytes=0,
                        decoded_cache_bytes=0, decode_concurrency=8,
                        prefix_streaming=false,
                        prefix_cursor_capacity=64,
                        width=320, height=240, region="us-east-1".to_string(),
                        max_concurrency=8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, usize, u64, u64)>,
        endpoint: String,
        bucket: String,
        object_key: String,
        access_key_id: String,
        secret_access_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        prefix_streaming: bool,
        prefix_cursor_capacity: usize,
        width: u32,
        height: u32,
        region: String,
        max_concurrency: usize,
    ) -> PyResult<Self> {
        let descriptors = descriptors
            .into_iter()
            .map(|(sample_id, video_id, target_ordinal, offset, length)| {
                pair_scheduler::ClosedRecordDescriptor {
                    sample_id,
                    video_id,
                    offset,
                    length,
                    target_ordinal,
                }
            })
            .collect();
        let backend = backend::S3Backend::new(
            endpoint,
            bucket,
            object_key,
            0,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let inner = pair_scheduler::ClosedRecordBatchExecutor::new(
            descriptors,
            representation::Representation::Prefix,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            prefix_streaming,
            prefix_cursor_capacity,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn plan_features(&self, batch: Vec<u64>) -> PyResult<std::collections::HashMap<String, u64>> {
        self.inner
            .plan_features(&batch)
            .map(|features| batch_features_dict(&features))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyAIStorePrefixBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, endpoint, bucket, object_key,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        encoded_cache_bytes=0, decoded_cache_bytes=0,
                        decode_concurrency=8, prefix_streaming=false,
                        prefix_cursor_capacity=64, width=320, height=240,
                        provider="ais".to_string()))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, usize, u64, u64)>,
        endpoint: String,
        bucket: String,
        object_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        prefix_streaming: bool,
        prefix_cursor_capacity: usize,
        width: u32,
        height: u32,
        provider: String,
    ) -> PyResult<Self> {
        let descriptors = descriptors
            .into_iter()
            .map(|(sample_id, video_id, target_ordinal, offset, length)| {
                pair_scheduler::ClosedRecordDescriptor {
                    sample_id,
                    video_id,
                    offset,
                    length,
                    target_ordinal,
                }
            })
            .collect();
        let backend =
            backend::AIStoreGetBatchBackend::new(endpoint, bucket, object_key, provider, 0)
                .map_err(|error| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
                })?;
        let inner = pair_scheduler::ClosedRecordBatchExecutor::new(
            descriptors,
            representation::Representation::Prefix,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            prefix_streaming,
            prefix_cursor_capacity,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn plan_features(&self, batch: Vec<u64>) -> PyResult<std::collections::HashMap<String, u64>> {
        self.inner
            .plan_features(&batch)
            .map(|features| batch_features_dict(&features))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyLocalClosedRecordBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, representation, path,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        encoded_cache_bytes=0, decoded_cache_bytes=0,
                        decode_concurrency=8, prefix_streaming=false,
                        prefix_cursor_capacity=64,
                        width=320, height=240))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, usize, u64, u64)>,
        representation: &str,
        path: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        encoded_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        prefix_streaming: bool,
        prefix_cursor_capacity: usize,
        width: u32,
        height: u32,
    ) -> PyResult<Self> {
        let representation = match representation.to_ascii_lowercase().as_str() {
            "prefix" => representation::Representation::Prefix,
            "pair" => representation::Representation::Pair,
            value => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unsupported closed-record representation: {value}"
                )))
            }
        };
        let descriptors = descriptors
            .into_iter()
            .map(|(sample_id, video_id, target_ordinal, offset, length)| {
                pair_scheduler::ClosedRecordDescriptor {
                    sample_id,
                    video_id,
                    offset,
                    length,
                    target_ordinal,
                }
            })
            .collect();
        let file = std::fs::File::open(&path).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyOSError, _>(format!(
                "failed to open closed-record payload {path}: {error}"
            ))
        })?;
        // SAFETY: the executor owns the read-only mapping and the benchmark
        // contract requires immutable payload artifacts for its lifetime.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyOSError, _>(format!(
                "failed to mmap closed-record payload {path}: {error}"
            ))
        })?;
        let inner = pair_scheduler::ClosedRecordBatchExecutor::new(
            descriptors,
            representation,
            Box::new(backend::LocalBackend::new(mmap, 0)),
            merge_threshold_bytes,
            max_range_bytes,
            encoded_cache_bytes,
            decoded_cache_bytes,
            decode_concurrency,
            prefix_streaming,
            prefix_cursor_capacity,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn plan_features(&self, batch: Vec<u64>) -> PyResult<std::collections::HashMap<String, u64>> {
        self.inner
            .plan_features(&batch)
            .map(|features| batch_features_dict(&features))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute(&batch))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((frames, batch_stats_dict(&stats)))
    }
}

/// Python-facing logical batch decoder (auto-configured from chunk).
#[cfg(feature = "ffmpeg")]
#[pyclass(name = "LogicalBatchDecoder")]
pub struct PyLogicalBatchDecoder {
    inner: scheduler::LogicalBatchDecoder,
}

/// Python-facing logical scheduler (epoch-level coalescing).
#[cfg(feature = "ffmpeg")]
#[pyclass(name = "LogicalScheduler")]
pub struct PyLogicalScheduler {
    inner: scheduler::LogicalScheduler,
}

#[cfg(feature = "ffmpeg")]
impl VClaspChunk {
    fn get_sps_pps(&mut self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if self.cached_sps.is_none() {
            let data = self.inner.read_sps_pps()?;
            self.cached_sps = Some(data);
        }
        Ok(self.cached_sps.as_ref().unwrap().clone())
    }
}

#[pymethods]
impl VClaspChunk {
    /// Open a VClasp chunk from a local file path.
    #[new]
    fn open(path: &str) -> PyResult<Self> {
        let reader = chunk::ChunkReader::open(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        #[cfg(feature = "ffmpeg")]
        let hierarchical_catalog = hierarchical_ingest::HierarchicalCatalog::from_parquet(
            &reader.mmap[reader.layout.index_start..],
        )
        .ok();
        Ok(VClaspChunk {
            inner: reader,
            #[cfg(feature = "ffmpeg")]
            path: path.to_string(),
            #[cfg(feature = "ffmpeg")]
            decoder_pool: decoder::DecoderPool::new(decoder::DecoderConfig::default()),
            #[cfg(feature = "ffmpeg")]
            cached_sps: None,
            #[cfg(feature = "ffmpeg")]
            hierarchical_catalog,
        })
    }

    /// Total number of records in the chunk index.
    fn record_count(&self) -> usize {
        self.inner.index.record_count()
    }

    /// Get codec configuration: (codec, width, height, fps_num, fps_den, sps_pps_len).
    fn codec_info(&self) -> (String, u16, u16, u8, u8, u32) {
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
    fn format_version(&self) -> u16 {
        self.inner.header.version
    }

    /// Read SPS/PPS bytes from the chunk.
    fn read_sps_pps(&mut self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let data = self
            .inner
            .read_sps_pps()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &data).unbind())
    }

    /// Read a single record blob by (video_id, tier) — returns one random matching blob.
    /// For deterministic access, use read_record_at(video_id, tier, idx).
    fn read_record(&mut self, py: Python<'_>, video_id: &str, tier: i32) -> PyResult<Py<PyBytes>> {
        let data = self
            .inner
            .read_record(video_id, tier)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &data).unbind())
    }

    /// Read a single deterministic record at index `idx` within the (video_id, tier) group.
    fn read_record_at(
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
    fn record_count_for(&self, video_id: &str, tier: i32) -> usize {
        self.inner.record_count_for(video_id, tier)
    }

    /// Read a specific byte range from the payload.
    fn read_bytes(&mut self, py: Python<'_>, offset: u64, length: u64) -> PyResult<Py<PyBytes>> {
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
    #[cfg(feature = "ffmpeg")]
    fn set_decoder_threads(&mut self, num: usize) {
        self.decoder_pool.set_threads(num);
    }

    /// Decode one target from the bitstream-grounded closure stored in the
    /// hierarchical index. The target is addressed in source display order.
    #[cfg(feature = "ffmpeg")]
    fn decode_hierarchical_target_rgb24(
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
    #[cfg(feature = "ffmpeg")]
    fn decode_record_rgb24(
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
    #[cfg(feature = "ffmpeg")]
    fn decode_records_batch_rgb24(
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
    fn batch_lookup(&self, video_ids: Vec<String>, tiers: Vec<i32>) -> PyResult<Vec<(u64, u64)>> {
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
    #[cfg(feature = "ffmpeg")]
    fn decode_gop_record_rgb24(
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
    #[cfg(feature = "ffmpeg")]
    fn decode_gop_records_batch_rgb24(
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
    #[cfg(feature = "ffmpeg")]
    fn decode_gop_all_records_rgb24(
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
    #[cfg(feature = "ffmpeg")]
    fn decode_gop_record_at(
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
    #[cfg(feature = "ffmpeg")]
    fn create_logical_scheduler(&self) -> PyResult<PyLogicalBatchDecoder> {
        let reader = chunk::ChunkReader::open(Path::new(&self.path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let sched = scheduler::LogicalBatchDecoder::new(reader, self.path.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyLogicalBatchDecoder { inner: sched })
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyLogicalBatchDecoder {
    /// Schedule a batch of logical requests and return decoded frames.
    /// Input: [(sample_id, video_id, tier, frame_idx), ...]
    /// Output: [(sample_id, rgb24_bytes, width, height), ...] in input order.
    fn schedule(
        &mut self,
        py: Python<'_>,
        requests: Vec<(i64, String, i32, i32)>,
    ) -> PyResult<Vec<(i64, Py<PyBytes>, u32, u32)>> {
        let logical: Vec<scheduler::LogicalRequest> = requests
            .iter()
            .map(|(sid, vid, tier, frame)| scheduler::LogicalRequest {
                sample_id: *sid,
                video_id: vid.clone(),
                tier: *tier,
                frame_idx: *frame,
            })
            .collect();

        let frames = self
            .inner
            .schedule(&logical)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(frames
            .into_iter()
            .map(|f| {
                let pybytes = PyBytes::new_bound(py, &f.rgb_bytes).unbind();
                (f.sample_id, pybytes, f.width, f.height)
            })
            .collect())
    }

    fn gop_size(&self) -> usize {
        self.inner.gop_size()
    }
    fn tier1_stride(&self) -> usize {
        self.inner.tier1_stride()
    }

    /// Create a LogicalScheduler from this decoder.
    /// `merge_threshold_kb`: byte gap threshold, None = no coalescing.
    /// `backend`: "local" (default) or "minio".
    /// S3/MinIO params are passed to object_store's AmazonS3Builder.
    #[pyo3(signature = (merge_threshold_kb=None, backend="local".to_string(),
                         minio_endpoint=None, minio_bucket=None, minio_key=None,
                         access_key_id=None, secret_access_key=None,
                         region="us-east-1".to_string(), max_concurrency=8,
                         completion_driven=false, decode_microbatch_records=1,
                         range_priority="offset".to_string()))]
    fn create_scheduler(
        &mut self,
        merge_threshold_kb: Option<usize>,
        backend: String,
        minio_endpoint: Option<String>,
        minio_bucket: Option<String>,
        minio_key: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        region: String,
        max_concurrency: usize,
        completion_driven: bool,
        decode_microbatch_records: usize,
        range_priority: String,
    ) -> PyResult<PyLogicalScheduler> {
        let threshold = merge_threshold_kb.map(|kb| kb * 1024);
        let reader = chunk::ChunkReader::open(Path::new(self.inner.path()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let payload_start = reader.layout.payload_start;
        let dec = scheduler::LogicalBatchDecoder::new(reader, self.inner.path().to_string())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mut sched = scheduler::LogicalScheduler::new(dec, threshold)
            .with_completion_driven(completion_driven)
            .with_decode_microbatch_records(decode_microbatch_records)
            .with_range_priority(&range_priority)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        if backend == "minio" || backend == "s3" {
            let ep = minio_endpoint.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("minio_endpoint required")
            })?;
            let bucket = minio_bucket.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("minio_bucket required")
            })?;
            let key = minio_key.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("minio_key required")
            })?;
            let access_key = access_key_id.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("access_key_id required")
            })?;
            let secret_key = secret_access_key.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("secret_access_key required")
            })?;
            let mb = backend::S3Backend::new(
                ep,
                bucket,
                key,
                payload_start,
                access_key,
                secret_key,
                region,
                max_concurrency,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            sched = sched.with_backend(Box::new(mb));
        } else if backend != "local" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown backend {backend:?}; expected local, minio, or s3"
            )));
        }

        Ok(PyLogicalScheduler { inner: sched })
    }

    fn path(&self) -> &str {
        self.inner.path()
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyLogicalScheduler {
    /// Execute an epoch trace with smart coalescing.
    /// Input: [(sample_id, video_id, tier, frame_idx), ...]
    /// Output: ([(sample_id, rgb24_bytes, width, height), ...], stats_dict)
    fn execute(
        &mut self,
        py: Python<'_>,
        trace: Vec<(i64, String, i32, i32)>,
    ) -> PyResult<(
        Vec<(i64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, usize>,
    )> {
        let logical: Vec<scheduler::LogicalRequest> = trace
            .iter()
            .map(|(sid, vid, tier, frame)| scheduler::LogicalRequest {
                sample_id: *sid,
                video_id: vid.clone(),
                tier: *tier,
                frame_idx: *frame,
            })
            .collect();

        let (frames, stats) = self
            .inner
            .execute(&logical)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let py_frames: Vec<(i64, Py<PyBytes>, u32, u32)> = frames
            .into_iter()
            .map(|f| {
                let pb = PyBytes::new_bound(py, &f.rgb_bytes).unbind();
                (f.sample_id, pb, f.width, f.height)
            })
            .collect();

        let mut stats_dict = std::collections::HashMap::new();
        stats_dict.insert("unique_records".into(), stats.unique_records);
        stats_dict.insert("physical_ranges".into(), stats.physical_ranges);
        stats_dict.insert("fetched_bytes".into(), stats.fetched_bytes as usize);
        stats_dict.insert("useful_bytes".into(), stats.useful_bytes as usize);
        stats_dict.insert("overfetch_bytes".into(), stats.overfetch_bytes as usize);
        stats_dict.insert("records_decoded".into(), stats.records_decoded);
        stats_dict.insert("decode_batches".into(), stats.decode_batches);
        stats_dict.insert("total_frames".into(), stats.total_frames);
        stats_dict.insert("max_fetch_records".into(), stats.max_fetch_records);
        stats_dict.insert("resolve_ns".into(), stats.resolve_ns as usize);
        stats_dict.insert("plan_ns".into(), stats.plan_ns as usize);
        stats_dict.insert("fetch_wall_ns".into(), stats.fetch_wall_ns as usize);
        stats_dict.insert(
            "fetch_service_ns_sum".into(),
            stats.fetch_service_ns_sum as usize,
        );
        stats_dict.insert(
            "range_queue_ns_sum".into(),
            stats.range_queue_ns_sum as usize,
        );
        stats_dict.insert("extract_ns".into(), stats.extract_ns as usize);
        stats_dict.insert("decode_ns".into(), stats.decode_ns as usize);
        stats_dict.insert(
            "fetch_decode_overlap_ns".into(),
            stats.fetch_decode_overlap_ns as usize,
        );
        stats_dict.insert("reorder_ns".into(), stats.reorder_ns as usize);
        stats_dict.insert("total_ns".into(), stats.total_ns as usize);
        stats_dict.insert(
            "time_to_first_ready_target_ns".into(),
            stats.time_to_first_ready_target_ns as usize,
        );

        Ok((py_frames, stats_dict))
    }

    /// Global offset-based execution: resolves all requests to byte offsets,
    /// sorts by physical chunk offset, coalesces adjacent ranges globally
    /// (across video/tier boundaries).
    fn execute_global(
        &mut self,
        py: Python<'_>,
        trace: Vec<(i64, String, i32, i32)>,
    ) -> PyResult<(
        Vec<(i64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, usize>,
    )> {
        let logical: Vec<scheduler::LogicalRequest> = trace
            .iter()
            .map(|(sid, vid, tier, frame)| scheduler::LogicalRequest {
                sample_id: *sid,
                video_id: vid.clone(),
                tier: *tier,
                frame_idx: *frame,
            })
            .collect();

        let (frames, stats) = self
            .inner
            .execute_global(&logical)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let py_frames: Vec<(i64, Py<PyBytes>, u32, u32)> = frames
            .into_iter()
            .map(|f| {
                let pb = PyBytes::new_bound(py, &f.rgb_bytes).unbind();
                (f.sample_id, pb, f.width, f.height)
            })
            .collect();

        let mut stats_dict = std::collections::HashMap::new();
        stats_dict.insert("unique_records".into(), stats.unique_records);
        stats_dict.insert("physical_ranges".into(), stats.physical_ranges);
        stats_dict.insert("fetched_bytes".into(), stats.fetched_bytes as usize);
        stats_dict.insert("useful_bytes".into(), stats.useful_bytes as usize);
        stats_dict.insert("overfetch_bytes".into(), stats.overfetch_bytes as usize);
        stats_dict.insert("records_decoded".into(), stats.records_decoded);
        stats_dict.insert("decode_batches".into(), stats.decode_batches);
        stats_dict.insert("total_frames".into(), stats.total_frames);
        stats_dict.insert("max_fetch_records".into(), stats.max_fetch_records);
        stats_dict.insert("resolve_ns".into(), stats.resolve_ns as usize);
        stats_dict.insert("plan_ns".into(), stats.plan_ns as usize);
        stats_dict.insert("fetch_wall_ns".into(), stats.fetch_wall_ns as usize);
        stats_dict.insert(
            "fetch_service_ns_sum".into(),
            stats.fetch_service_ns_sum as usize,
        );
        stats_dict.insert(
            "range_queue_ns_sum".into(),
            stats.range_queue_ns_sum as usize,
        );
        stats_dict.insert("extract_ns".into(), stats.extract_ns as usize);
        stats_dict.insert("decode_ns".into(), stats.decode_ns as usize);
        stats_dict.insert(
            "fetch_decode_overlap_ns".into(),
            stats.fetch_decode_overlap_ns as usize,
        );
        stats_dict.insert("reorder_ns".into(), stats.reorder_ns as usize);
        stats_dict.insert("total_ns".into(), stats.total_ns as usize);
        stats_dict.insert(
            "time_to_first_ready_target_ns".into(),
            stats.time_to_first_ready_target_ns as usize,
        );

        Ok((py_frames, stats_dict))
    }

    fn gop_size(&self) -> usize {
        self.inner.gop_size()
    }
    fn tier1_stride(&self) -> usize {
        self.inner.tier1_stride()
    }
}

/// Write a latest-format VClasp chunk from already built component files.
///
/// Layout:
///   [4B flatbuffer_size][FlatBuffer header][SPS/PPS bytes][payload blobs][Parquet index]
#[pyfunction]
fn write_chunk_from_files(
    output_path: &str,
    sps_pps_path: &str,
    payload_path: &str,
    index_path: &str,
    created_at: u64,
) -> PyResult<()> {
    chunk::write_chunk_from_files(
        Path::new(output_path),
        Path::new(sps_pps_path),
        Path::new(payload_path),
        Path::new(index_path),
        created_at,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Build a latest-format VClasp chunk from source videos.
///
/// Python passes only metadata and paths. Rust performs encode, tier selection,
/// Annex-B parsing, Parquet index construction, and final chunk assembly.
#[pyfunction(name = "build_tiered_chunk")]
#[pyo3(signature = (
     videos,
     output_path,
     ffmpeg_path=None,
     tier_policies=None,
     gop_size=8,
     tier1_stride_idrs=4,
     width=320,
     height=240,
     crf=23,
     preset="veryfast".to_string(),
     max_frames=None,
     progress_every=100
 ))]
fn build_chunk_from_videos(
    videos: Vec<(String, String, String)>,
    output_path: &str,
    ffmpeg_path: Option<String>,
    tier_policies: Option<std::collections::HashMap<i32, String>>,
    gop_size: u32,
    tier1_stride_idrs: usize,
    width: u16,
    height: u16,
    crf: u8,
    preset: String,
    max_frames: Option<u32>,
    progress_every: usize,
) -> PyResult<(usize, usize, usize, usize, usize, u64, u64, u64)> {
    let inputs: Vec<builder::VideoInput> = videos
        .into_iter()
        .map(|(video_id, class_name, source_path)| builder::VideoInput {
            video_id,
            class_name,
            source_path: PathBuf::from(source_path),
        })
        .collect();
    let mut options = builder::ChunkBuildOptions::default_for_output(PathBuf::from(output_path));
    if let Some(path) = ffmpeg_path {
        options.ffmpeg_path = PathBuf::from(path);
    }

    // Policy validation belongs to the Rust builder core. PyO3 only adapts
    // structured Python arguments to the core options.
    let policies = tier_policies.unwrap_or_default();
    for (tier, policy_name) in policies {
        let pol = builder::parse_storage_policy(&policy_name, gop_size).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid policy for tier {tier}: {error}"
            ))
        })?;
        options.tier_policies.insert(tier, pol);
    }
    options.tier1_stride_idrs = tier1_stride_idrs;
    options.width = width;
    options.height = height;
    options.crf = crf;
    options.preset = preset;
    options.max_frames = max_frames;
    options.progress_every = progress_every;

    let stats = builder::build_chunk_from_videos(&inputs, &options)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok((
        stats.videos,
        stats.records,
        stats.tier0_records,
        stats.tier1_records,
        stats.tier2_records,
        stats.payload_bytes,
        stats.index_bytes,
        stats.chunk_bytes,
    ))
}

/// Build one hierarchical-B chunk from source videos.
///
/// Python supplies source metadata and immutable build parameters. Rust owns
/// controlled encoding, packet extraction, dependency indexing, and chunk
/// assembly.
#[cfg(feature = "ffmpeg")]
#[pyfunction(name = "build_chunk")]
#[pyo3(signature = (
    videos,
    output_path,
    baseline_mp4_dir=None,
    ffmpeg_path="ffmpeg".to_string(),
    ffprobe_path="ffprobe".to_string(),
    gop_size=64,
    max_frames=512,
    width=320,
    height=240,
    fps=25,
    crf=23,
    preset="veryfast".to_string()
))]
#[allow(clippy::too_many_arguments)]
fn build_hierarchical_chunk(
    py: Python<'_>,
    videos: Vec<(String, String, String)>,
    output_path: &str,
    baseline_mp4_dir: Option<String>,
    ffmpeg_path: String,
    ffprobe_path: String,
    gop_size: u32,
    max_frames: u32,
    width: u16,
    height: u16,
    fps: u16,
    crf: u8,
    preset: String,
) -> PyResult<(usize, usize, usize, u64, u64, u64, usize)> {
    let inputs = videos
        .into_iter()
        .map(|(video_id, class_name, source_path)| builder::VideoInput {
            video_id,
            class_name,
            source_path: PathBuf::from(source_path),
        })
        .collect::<Vec<_>>();
    let options = hierarchical_ingest::HierarchicalBuildOptions {
        output_path: PathBuf::from(output_path),
        baseline_mp4_dir: baseline_mp4_dir.map(PathBuf::from),
        ffmpeg_path: PathBuf::from(ffmpeg_path),
        ffprobe_path: PathBuf::from(ffprobe_path),
        gop_size,
        max_frames,
        width,
        height,
        fps,
        crf,
        preset,
    };
    let stats = py
        .allow_threads(|| {
            hierarchical_ingest::build_hierarchical_chunk(&inputs, &options)
                .map_err(|error| error.to_string())
        })
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
    Ok((
        stats.videos,
        stats.records,
        stats.targets,
        stats.payload_bytes,
        stats.index_bytes,
        stats.chunk_bytes,
        stats.max_closure_records,
    ))
}

/// Build the bounded experimental Anchor/Delta layout used by the closure-
/// fusion gate. Rust owns decode, encode, NAL parsing, data writing and index
/// construction; Python supplies paths and immutable build parameters only.
#[pyfunction(name = "build_anchor_delta_layout")]
#[pyo3(signature = (
    videos,
    data_path,
    index_path,
    ffmpeg_path=None,
    gop_size=64,
    width=320,
    height=240,
    crf=23,
    preset="veryfast".to_string(),
    max_frames=Some(64),
    include_group_anchor_targets=false
))]
fn build_fused_normalized_layout(
    videos: Vec<(String, String, String)>,
    data_path: &str,
    index_path: &str,
    ffmpeg_path: Option<String>,
    gop_size: u32,
    width: u16,
    height: u16,
    crf: u8,
    preset: String,
    max_frames: Option<u32>,
    include_group_anchor_targets: bool,
) -> PyResult<(usize, usize, usize, u64, u64)> {
    let inputs = videos
        .into_iter()
        .map(|(video_id, class_name, source_path)| builder::VideoInput {
            video_id,
            class_name,
            source_path: PathBuf::from(source_path),
        })
        .collect::<Vec<_>>();
    let mut options = builder::ChunkBuildOptions::default_for_output(PathBuf::from(data_path));
    if let Some(path) = ffmpeg_path {
        options.ffmpeg_path = PathBuf::from(path);
    }
    options.width = width;
    options.height = height;
    options.crf = crf;
    options.preset = preset;
    options.max_frames = max_frames;
    let stats = builder::build_fused_normalized_layout(
        &inputs,
        Path::new(data_path),
        Path::new(index_path),
        options,
        gop_size,
        include_group_anchor_targets,
    )
    .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
    Ok((
        stats.videos,
        stats.anchor_groups,
        stats.samples,
        stats.data_bytes,
        stats.index_bytes,
    ))
}

/// Build and verify one-copy two-level reference pages. Rust owns source
/// decode, x264 reference control, physical placement, index construction, and
/// per-target libavcodec verification. Python supplies only paths and immutable
/// build parameters.
#[cfg(feature = "ffmpeg")]
#[pyfunction(name = "build_reference_page_layout")]
#[pyo3(signature = (
    videos,
    data_path,
    index_path,
    ffmpeg_path=None,
    gop_size=64,
    page_size=8,
    width=320,
    height=240,
    crf=23,
    preset="veryfast".to_string(),
    max_frames=Some(64)
))]
#[allow(clippy::too_many_arguments)]
fn build_two_level_page_layout(
    py: Python<'_>,
    videos: Vec<(String, String, String)>,
    data_path: &str,
    index_path: &str,
    ffmpeg_path: Option<String>,
    gop_size: u32,
    page_size: u32,
    width: u16,
    height: u16,
    crf: u8,
    preset: String,
    max_frames: Option<u32>,
) -> PyResult<(
    usize,
    usize,
    usize,
    usize,
    usize,
    u64,
    u64,
    u64,
    u64,
    u64,
    usize,
)> {
    let inputs = videos
        .into_iter()
        .map(|(video_id, class_name, source_path)| builder::VideoInput {
            video_id,
            class_name,
            source_path: PathBuf::from(source_path),
        })
        .collect::<Vec<_>>();
    let mut options = builder::ChunkBuildOptions::default_for_output(PathBuf::from(data_path));
    if let Some(path) = ffmpeg_path {
        options.ffmpeg_path = PathBuf::from(path);
    }
    options.width = width;
    options.height = height;
    options.crf = crf;
    options.preset = preset;
    options.max_frames = max_frames;
    let data_path = PathBuf::from(data_path);
    let index_path = PathBuf::from(index_path);
    let stats = py
        .allow_threads(|| {
            builder::build_two_level_page_layout(
                &inputs,
                &data_path,
                &index_path,
                options,
                gop_size,
                page_size,
            )
            .map_err(|error| error.to_string())
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok((
        stats.videos,
        stats.groups,
        stats.pages,
        stats.samples,
        stats.verified_samples,
        stats.data_bytes,
        stats.index_bytes,
        stats.root_bytes,
        stats.checkpoint_bytes,
        stats.target_delta_bytes,
        stats.max_closure_records,
    ))
}

/// Select a constrained portfolio candidate from measured calibration metrics.
///
/// Candidate IDs are kept separate from numeric fields so the Python adapter
/// remains structured without moving planning logic out of Rust.
#[pyfunction]
fn select_portfolio_candidate(
    candidates: Vec<(String, std::collections::HashMap<String, f64>)>,
    prices: std::collections::HashMap<String, f64>,
    constraints: std::collections::HashMap<String, f64>,
) -> PyResult<(
    String,
    Vec<(
        String,
        bool,
        f64,
        Vec<String>,
        std::collections::HashMap<String, f64>,
    )>,
)> {
    fn required(values: &std::collections::HashMap<String, f64>, key: &str) -> PyResult<f64> {
        values.get(key).copied().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "missing required selector field {key}"
            ))
        })
    }
    let candidates = candidates
        .into_iter()
        .map(|(id, values)| {
            Ok(cost_selector::Candidate {
                id,
                portfolio_storage_bytes: required(&values, "portfolio_storage_bytes")?,
                metadata_bytes: values.get("metadata_bytes").copied().unwrap_or(0.0),
                ingestion_ns: required(&values, "ingestion_ns")?,
                requests_per_sample: required(&values, "requests_per_sample")?,
                fetched_bytes_per_sample: required(&values, "fetched_bytes_per_sample")?,
                decode_ns_per_sample: required(&values, "decode_ns_per_sample")?,
                p95_ns: required(&values, "p95_ns")?,
                p99_ns: required(&values, "p99_ns")?,
                throughput_samples_s: required(&values, "throughput_samples_s")?,
                cache_bytes: required(&values, "cache_bytes")?,
                model_utility: required(&values, "model_utility")?,
                switch_cost_usd: values.get("switch_cost_usd").copied().unwrap_or(0.0),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let prices = cost_selector::PriceVector {
        storage_usd_per_byte_month: required(&prices, "storage_usd_per_byte_month")?,
        retention_months: required(&prices, "retention_months")?,
        request_usd_per_request: required(&prices, "request_usd_per_request")?,
        transfer_usd_per_byte: required(&prices, "transfer_usd_per_byte")?,
        decode_compute_usd_per_ns: required(&prices, "decode_compute_usd_per_ns")?,
        ingestion_compute_usd_per_ns: required(&prices, "ingestion_compute_usd_per_ns")?,
        workload_samples: required(&prices, "workload_samples")?,
        amortization_runs: required(&prices, "amortization_runs")?,
        slo_penalty_usd_per_ns: required(&prices, "slo_penalty_usd_per_ns")?,
    };
    let optional = |key: &str| constraints.get(key).copied();
    let constraints = cost_selector::Constraints {
        max_p95_ns: optional("max_p95_ns"),
        max_p99_ns: optional("max_p99_ns"),
        min_throughput_samples_s: optional("min_throughput_samples_s"),
        max_portfolio_storage_bytes: optional("max_portfolio_storage_bytes"),
        max_cache_bytes: optional("max_cache_bytes"),
        min_model_utility: optional("min_model_utility"),
        target_p95_ns: optional("target_p95_ns"),
        target_p99_ns: optional("target_p99_ns"),
    };
    let selection = cost_selector::select(candidates, prices, constraints)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok((
        selection.selected_id,
        selection
            .candidates
            .into_iter()
            .map(|candidate| {
                (
                    candidate.id,
                    candidate.eligible,
                    candidate.total_cost_usd,
                    candidate.rejection_reasons,
                    candidate.breakdown,
                )
            })
            .collect(),
    ))
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyAdaptiveBatchExecutor {
    #[new]
    #[pyo3(signature = (
        normalized_descriptors, fusion_metadata, prefix_descriptors,
        endpoint, bucket, normalized_object_key, prefix_object_key,
        access_key_id, secret_access_key, model,
        max_range_bytes=Some(1_048_576), anchor_cache_bytes=8_388_608,
        prefix_cache_bytes=0, decoded_cache_bytes=0,
        decode_concurrency=8, decode_microbatch_targets=4,
        prefix_cursor_capacity=64, width=320, height=240,
        region="us-east-1".to_string(), max_concurrency=8,
        completion_driven=true, fuse_shared_anchors=true,
        allow_cross_encoding_prefix=false, use_native_vectored_io=false,
        derive_normalized_span=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        normalized_descriptors: Vec<(u64, u64, u64, u64, u64, u64)>,
        fusion_metadata: Vec<(u64, u64, usize)>,
        prefix_descriptors: Vec<(u64, u64, usize, u64, u64)>,
        endpoint: String,
        bucket: String,
        normalized_object_key: String,
        prefix_object_key: String,
        access_key_id: String,
        secret_access_key: String,
        model: std::collections::HashMap<String, f64>,
        max_range_bytes: Option<u64>,
        anchor_cache_bytes: usize,
        prefix_cache_bytes: usize,
        decoded_cache_bytes: usize,
        decode_concurrency: usize,
        decode_microbatch_targets: usize,
        prefix_cursor_capacity: usize,
        width: u32,
        height: u32,
        region: String,
        max_concurrency: usize,
        completion_driven: bool,
        fuse_shared_anchors: bool,
        allow_cross_encoding_prefix: bool,
        use_native_vectored_io: bool,
        derive_normalized_span: bool,
    ) -> PyResult<Self> {
        let model = mechanistic_model_from_python(&model)?;
        if model.io_concurrency != max_concurrency {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "model io_concurrency must equal max_concurrency",
            ));
        }
        let decode_schedule = if fuse_shared_anchors {
            normalized_scheduler::DecodeSchedule::Fused
        } else {
            normalized_scheduler::DecodeSchedule::Repeated
        };
        if derive_normalized_span && normalized_object_key != prefix_object_key {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Normalized contiguous-span execution must use the Normalized object key",
            ));
        }
        let normalized_descriptors = normalized_descriptors_from_python(
            normalized_descriptors,
            Some(fusion_metadata),
            decode_schedule.requires_group_metadata(),
        )?;
        let prefix_descriptors = if derive_normalized_span {
            if !prefix_descriptors.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Normalized span descriptors are derived in Rust and must not be supplied",
                ));
            }
            adaptive_planner::derive_normalized_span_descriptors(&normalized_descriptors)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?
        } else {
            prefix_descriptors
                .into_iter()
                .map(|(sample_id, video_id, target_ordinal, offset, length)| {
                    pair_scheduler::ClosedRecordDescriptor {
                        sample_id,
                        video_id,
                        offset,
                        length,
                        target_ordinal,
                    }
                })
                .collect::<Vec<_>>()
        };
        let client = backend::S3ObjectStoreClient::new(
            endpoint,
            bucket,
            access_key_id,
            secret_access_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let normalized_backend = if use_native_vectored_io {
            backend::S3Backend::from_shared_client_vectored(
                client.clone(),
                normalized_object_key,
                0,
            )
        } else {
            backend::S3Backend::from_shared_client(client.clone(), normalized_object_key, 0)
        }
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let prefix_backend = backend::S3Backend::from_shared_client(client, prefix_object_key, 0)
            .map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
        })?;
        let decoder_slots = decoder::shared_decoder_slots(decode_concurrency);
        let normalized =
            normalized_scheduler::NormalizedBatchExecutor::new_with_decode_schedule_and_slots(
                normalized_descriptors,
                Box::new(normalized_backend),
                None,
                max_range_bytes,
                anchor_cache_bytes,
                0,
                decoded_cache_bytes,
                decode_microbatch_targets,
                decode_schedule,
                decoder_slots.clone(),
                width,
                height,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let prefix = pair_scheduler::ClosedRecordBatchExecutor::new_with_decoder_slots(
            prefix_descriptors,
            representation::Representation::Prefix,
            Box::new(prefix_backend),
            Some(0),
            max_range_bytes,
            prefix_cache_bytes,
            decoded_cache_bytes,
            decoder_slots,
            true,
            prefix_cursor_capacity,
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let inner = adaptive_planner::AdaptiveBatchExecutor::new(
            normalized,
            prefix,
            model,
            completion_driven,
            allow_cross_encoding_prefix,
            derive_normalized_span,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn explain_plan(
        &self,
        batch: Vec<u64>,
    ) -> PyResult<(
        String,
        u64,
        std::collections::HashMap<String, f64>,
        Vec<(String, std::collections::HashMap<String, f64>)>,
    )> {
        let decision = self
            .inner
            .plan(&batch)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok((
            decision.selected.mode.id(),
            decision.selector_ns,
            adaptive_estimate_dict(&decision.selected),
            decision
                .alternatives
                .iter()
                .map(|estimate| (estimate.mode.id(), adaptive_estimate_dict(estimate)))
                .collect(),
        ))
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        std::collections::HashMap<String, f64>,
        Vec<(String, std::collections::HashMap<String, f64>)>,
    )> {
        let (frames, decision, stats) = py
            .allow_threads(|| self.inner.execute(&batch))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                (
                    frame.sample_id,
                    PyBytes::new_bound(py, &frame.rgb).unbind(),
                    frame.width,
                    frame.height,
                )
            })
            .collect();
        Ok((
            frames,
            batch_stats_dict(&stats),
            decision.selected.mode.id(),
            adaptive_estimate_dict(&decision.selected),
            decision
                .alternatives
                .iter()
                .map(|estimate| (estimate.mode.id(), adaptive_estimate_dict(estimate)))
                .collect(),
        ))
    }

    fn execute_forced(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
        mode: String,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let mode = match mode.as_str() {
            "prefix_stream" => adaptive_planner::PlanMode::Prefix,
            "normalized_contiguous_span" => adaptive_planner::PlanMode::NormalizedSpan,
            "normalized_group_span" => adaptive_planner::PlanMode::NormalizedGroupSpan,
            "normalized_exact" => adaptive_planner::PlanMode::Normalized {
                merge_threshold_bytes: None,
            },
            value if value.starts_with("normalized_gap_") => {
                let threshold = value["normalized_gap_".len()..]
                    .parse::<u64>()
                    .map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "invalid normalized mode {value}"
                        ))
                    })?;
                adaptive_planner::PlanMode::Normalized {
                    merge_threshold_bytes: Some(threshold),
                }
            }
            value => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown adaptive mode {value}"
                )))
            }
        };
        let (frames, stats) = py
            .allow_threads(|| self.inner.execute_forced(&batch, mode))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok((
            frames
                .into_iter()
                .map(|frame| {
                    (
                        frame.sample_id,
                        PyBytes::new_bound(py, &frame.rgb).unbind(),
                        frame.width,
                        frame.height,
                    )
                })
                .collect(),
            batch_stats_dict(&stats),
        ))
    }
}

#[cfg(feature = "ffmpeg")]
fn hierarchical_metadata_from_chunk(
    path: &str,
) -> PyResult<(hierarchical_ingest::HierarchicalCatalog, Vec<u8>, usize)> {
    let mut reader = chunk::ChunkReader::open(Path::new(path))
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyIOError, _>(error.to_string()))?;
    let catalog = hierarchical_ingest::HierarchicalCatalog::from_parquet(
        &reader.mmap[reader.layout.index_start..],
    )
    .map_err(|error| PyErr::new::<pyo3::exceptions::PyValueError, _>(error.to_string()))?;
    let codec_config = reader
        .read_sps_pps()
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyIOError, _>(error.to_string()))?;
    Ok((catalog, codec_config, reader.layout.payload_start))
}

#[cfg(feature = "ffmpeg")]
fn hierarchical_model_from_python(
    values: &std::collections::HashMap<String, f64>,
    wave_request_overhead_ns: Vec<f64>,
) -> PyResult<hierarchical_layout::HierarchicalCostModel> {
    let required = |name: &str| {
        values.get(name).copied().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "hierarchical cost model is missing {name}"
            ))
        })
    };
    let io_concurrency = required("io_concurrency")?;
    if io_concurrency.fract() != 0.0 || io_concurrency < 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "io_concurrency must be a positive integer",
        ));
    }
    Ok(hierarchical_layout::HierarchicalCostModel {
        request_latency_ns: required("request_latency_ns")?,
        bandwidth_bytes_per_ns: required("bandwidth_bytes_per_ns")?,
        io_concurrency: io_concurrency as usize,
        wave_request_overhead_ns,
        selection_tolerance_ns: required("selection_tolerance_ns")?,
        decode_fixed_ns: required("decode_fixed_ns")?,
        decode_access_unit_ns: required("decode_access_unit_ns")?,
        fetch_decode_overlap: required("fetch_decode_overlap")?,
    })
}

#[cfg(feature = "ffmpeg")]
fn normalized_lookahead_alternatives(
    decision: &hierarchical_layout::DependencyLookaheadDecision,
) -> Vec<std::collections::HashMap<String, f64>> {
    decision
        .alternatives
        .iter()
        .map(|estimate| {
            let mode = match estimate.plan.mode {
                hierarchical_layout::HierarchicalReadMode::SparseClosure => 0.0,
                hierarchical_layout::HierarchicalReadMode::ContiguousRegion => 1.0,
            };
            [
                ("lookahead_batches", estimate.lookahead_batches as f64),
                ("logical_samples", estimate.logical_samples as f64),
                ("first_batch_ranges", estimate.first_batch_ranges as f64),
                (
                    "first_batch_fetched_bytes",
                    estimate.first_batch_fetched_bytes as f64,
                ),
                (
                    "first_batch_access_units",
                    estimate.first_batch_access_units as f64,
                ),
                ("first_batch_io_ns", estimate.first_batch_io_ns),
                ("first_batch_decode_ns", estimate.first_batch_decode_ns),
                ("first_batch_total_ns", estimate.first_batch_total_ns),
                ("predicted_total_ns", estimate.predicted_total_ns),
                (
                    "predicted_samples_per_second",
                    estimate.predicted_samples_per_second,
                ),
                ("plan_mode", mode),
                ("plan_ranges", estimate.plan.ranges.len() as f64),
                ("plan_useful_bytes", estimate.plan.useful_bytes as f64),
                ("plan_fetched_bytes", estimate.plan.fetched_bytes as f64),
                (
                    "plan_access_units",
                    estimate.plan.access_units_submitted as f64,
                ),
            ]
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect()
        })
        .collect()
}

#[cfg(feature = "ffmpeg")]
fn normalized_lookahead_from_python(
    inner: &normalized_scheduler::NormalizedBatchExecutor,
    batches: Vec<Vec<u64>>,
    candidates: Vec<usize>,
    first_batch_slo_ms: f64,
    cost_model: std::collections::HashMap<String, f64>,
    wave_request_overhead_ns: Vec<f64>,
) -> PyResult<(usize, bool, Vec<std::collections::HashMap<String, f64>>)> {
    if !first_batch_slo_ms.is_finite() || first_batch_slo_ms <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "first_batch_slo_ms must be finite and positive",
        ));
    }
    let model = hierarchical_model_from_python(&cost_model, wave_request_overhead_ns)?;
    let decision = inner
        .choose_lookahead(
            &batches,
            &candidates,
            first_batch_slo_ms * 1_000_000.0,
            &model,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let alternatives = normalized_lookahead_alternatives(&decision);
    Ok((
        decision.selected.lookahead_batches,
        decision.selected_under_slo,
        alternatives,
    ))
}

#[cfg(feature = "ffmpeg")]
fn execute_normalized_lookahead_from_python(
    inner: &mut normalized_scheduler::NormalizedBatchExecutor,
    py: Python<'_>,
    batches: Vec<Vec<u64>>,
    candidates: Vec<usize>,
    first_batch_slo_ms: f64,
    cost_model: std::collections::HashMap<String, f64>,
    wave_request_overhead_ns: Vec<f64>,
    completion_driven: bool,
) -> PyResult<(
    Vec<(u64, Py<PyBytes>, u32, u32)>,
    std::collections::HashMap<String, u64>,
    usize,
    bool,
    u64,
    Vec<std::collections::HashMap<String, f64>>,
)> {
    if !first_batch_slo_ms.is_finite() || first_batch_slo_ms <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "first_batch_slo_ms must be finite and positive",
        ));
    }
    let model = hierarchical_model_from_python(&cost_model, wave_request_overhead_ns)?;
    let result = py
        .allow_threads(|| {
            inner.execute_planned_lookahead(
                &batches,
                &candidates,
                first_batch_slo_ms * 1_000_000.0,
                &model,
                completion_driven,
            )
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    let selected = result.decision.selected.lookahead_batches;
    let selected_under_slo = result.decision.selected_under_slo;
    let alternatives = normalized_lookahead_alternatives(&result.decision);
    let frames = result
        .frames
        .into_iter()
        .map(|frame| {
            (
                frame.sample_id,
                PyBytes::new_bound(py, &frame.rgb).unbind(),
                frame.width,
                frame.height,
            )
        })
        .collect();
    Ok((
        frames,
        batch_stats_dict(&result.stats),
        selected,
        selected_under_slo,
        result.first_batch_ready_ns,
        alternatives,
    ))
}

#[cfg(feature = "ffmpeg")]
fn hierarchical_stats_dict(
    stats: &hierarchical_scheduler::HierarchicalBatchStats,
) -> std::collections::HashMap<String, u64> {
    [
        ("logical_targets", stats.logical_targets as u64),
        ("unique_targets", stats.unique_targets as u64),
        ("physical_ranges", stats.physical_ranges as u64),
        ("client_requests", stats.client_requests as u64),
        ("useful_bytes", stats.useful_bytes),
        ("fetched_bytes", stats.fetched_bytes),
        (
            "submitted_access_units",
            stats.submitted_access_units as u64,
        ),
        ("decode_groups", stats.decode_groups as u64),
        ("streaming_cache_hits", stats.streaming_cache_hits as u64),
        (
            "streaming_cache_misses",
            stats.streaming_cache_misses as u64,
        ),
        (
            "streaming_cache_resident_bytes",
            stats.streaming_cache_resident_bytes,
        ),
        (
            "streaming_cache_budget_bytes",
            stats.streaming_cache_budget_bytes,
        ),
        ("decoder_state_resets", stats.decoder_state_resets as u64),
        ("plan_ns", stats.plan_ns),
        ("fetch_ns", stats.fetch_ns),
        ("assemble_ns", stats.assemble_ns),
        ("decode_ns", stats.decode_ns),
        ("total_ns", stats.total_ns),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value))
    .collect()
}

#[cfg(feature = "ffmpeg")]
fn execute_hierarchical_python(
    inner: &mut hierarchical_scheduler::HierarchicalBatchExecutor,
    py: Python<'_>,
    targets: Vec<(u64, String, i32)>,
    action: Option<hierarchical_scheduler::HierarchicalAction>,
) -> PyResult<(
    Vec<(u64, Py<PyBytes>, u32, u32)>,
    std::collections::HashMap<String, u64>,
    String,
    f64,
)> {
    let targets = targets
        .into_iter()
        .map(
            |(sample_id, video_id, frame_idx)| hierarchical_scheduler::LogicalTarget {
                sample_id,
                video_id,
                frame_idx,
            },
        )
        .collect::<Vec<_>>();
    let (outputs, stats) = py
        .allow_threads(|| match action {
            Some(action) => inner.execute_action(&targets, action),
            None => inner.execute(&targets),
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    let frames = outputs
        .into_iter()
        .map(|output| {
            (
                output.sample_id,
                PyBytes::new_bound(py, &output.frame.data).unbind(),
                output.frame.width,
                output.frame.height,
            )
        })
        .collect();
    Ok((
        frames,
        hierarchical_stats_dict(&stats),
        stats.mode.to_string(),
        stats.predicted_total_ns,
    ))
}

#[cfg(feature = "ffmpeg")]
fn execute_hierarchical_incremental_python(
    inner: &mut hierarchical_scheduler::HierarchicalBatchExecutor,
    py: Python<'_>,
    batches: Vec<Vec<(u64, String, i32)>>,
) -> PyResult<(
    Vec<Vec<(u64, Py<PyBytes>, u32, u32)>>,
    Vec<u64>,
    Vec<u64>,
    std::collections::HashMap<String, u64>,
    String,
    f64,
)> {
    let batches = batches
        .into_iter()
        .map(|batch| {
            batch
                .into_iter()
                .map(
                    |(sample_id, video_id, frame_idx)| hierarchical_scheduler::LogicalTarget {
                        sample_id,
                        video_id,
                        frame_idx,
                    },
                )
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let result = py
        .allow_threads(|| inner.execute_incremental_window(&batches))
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    let frames = result
        .batches
        .into_iter()
        .map(|batch| {
            batch
                .into_iter()
                .map(|output| {
                    (
                        output.sample_id,
                        PyBytes::new_bound(py, &output.frame.data).unbind(),
                        output.frame.width,
                        output.frame.height,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok((
        frames,
        result.batch_ready_ns,
        result.ordered_delivery_ns,
        hierarchical_stats_dict(&result.stats),
        result.stats.mode.to_string(),
        result.stats.predicted_total_ns,
    ))
}

#[cfg(feature = "ffmpeg")]
fn plan_dependency_sampler_python(
    inner: &hierarchical_scheduler::HierarchicalBatchExecutor,
    base_order: Vec<u64>,
    batch_size: usize,
    lookahead_samples: usize,
) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
    let plan = inner
        .plan_dependency_sampler(&base_order, batch_size, lookahead_samples)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let stats = [
        ("predicted_total_ns", plan.predicted_total_ns),
        ("logical_samples", base_order.len() as f64),
        ("batches", plan.batches.len() as f64),
        ("batch_size", batch_size as f64),
        ("lookahead_samples", lookahead_samples as f64),
        ("max_displacement", plan.max_displacement as f64),
        (
            "mean_absolute_displacement",
            plan.mean_absolute_displacement,
        ),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value))
    .collect();
    Ok((plan.batches, stats))
}

#[cfg(feature = "ffmpeg")]
fn plan_label_preserving_dependency_sampler_python(
    inner: &hierarchical_scheduler::HierarchicalBatchExecutor,
    base_order: Vec<(u64, i64, Vec<u64>)>,
    batch_size: usize,
    lookahead_batches: usize,
) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
    let samples = base_order
        .into_iter()
        .map(
            |(logical_id, class_id, target_ids)| dependency_sampler::LabeledDependencySample {
                logical_id,
                class_id,
                target_ids,
            },
        )
        .collect::<Vec<_>>();
    let plan = inner
        .plan_label_preserving_dependency_sampler(&samples, batch_size, lookahead_batches)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let stats = [
        ("predicted_total_ns", plan.predicted_total_ns),
        ("logical_samples", samples.len() as f64),
        ("batches", plan.batches.len() as f64),
        ("batch_size", batch_size as f64),
        ("lookahead_batches", lookahead_batches as f64),
        ("max_displacement", plan.max_displacement as f64),
        (
            "mean_absolute_displacement",
            plan.mean_absolute_displacement,
        ),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value))
    .collect();
    Ok((plan.batches, stats))
}

#[cfg(feature = "ffmpeg")]
fn plan_label_preserving_logical_sampler_python(
    inner: &hierarchical_scheduler::HierarchicalBatchExecutor,
    base_order: Vec<(u64, i64, Vec<(String, i32)>)>,
    batch_size: usize,
    lookahead_batches: usize,
) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
    let logical_samples = base_order.len();
    let plan = inner
        .plan_label_preserving_logical_sampler(&base_order, batch_size, lookahead_batches)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let stats = [
        ("predicted_total_ns", plan.predicted_total_ns),
        ("logical_samples", logical_samples as f64),
        ("batches", plan.batches.len() as f64),
        ("batch_size", batch_size as f64),
        ("lookahead_batches", lookahead_batches as f64),
        ("max_displacement", plan.max_displacement as f64),
        (
            "mean_absolute_displacement",
            plan.mean_absolute_displacement,
        ),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value))
    .collect();
    Ok((plan.batches, stats))
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyLocalHierarchicalBatchExecutor {
    #[new]
    #[pyo3(signature = (
        chunk_path, cost_model, wave_request_overhead_ns,
        max_merge_gap_bytes=None, max_range_bytes=None, decoder_threads=1,
        incremental_decode_slots=1, incremental_batch_deadline_fences=false,
        streaming_cache_bytes=1048576, streaming_min_contiguous_targets=16
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        chunk_path: &str,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        decoder_threads: usize,
        incremental_decode_slots: usize,
        incremental_batch_deadline_fences: bool,
        streaming_cache_bytes: usize,
        streaming_min_contiguous_targets: usize,
    ) -> PyResult<Self> {
        let (catalog, codec_config, payload_start) = hierarchical_metadata_from_chunk(chunk_path)?;
        let file = std::fs::File::open(chunk_path).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyOSError, _>(format!(
                "failed to open hierarchical chunk {chunk_path}: {error}"
            ))
        })?;
        // SAFETY: the executor owns this read-only mapping, and benchmark
        // chunks are immutable for the lifetime of an executor.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyOSError, _>(format!(
                "failed to mmap hierarchical chunk {chunk_path}: {error}"
            ))
        })?;
        let inner = hierarchical_scheduler::HierarchicalBatchExecutor::new(
            catalog,
            Box::new(backend::LocalBackend::new(mmap, payload_start)),
            codec_config,
            hierarchical_model_from_python(&cost_model, wave_request_overhead_ns)?,
            max_merge_gap_bytes,
            max_range_bytes,
            decoder_threads,
            incremental_decode_slots,
            incremental_batch_deadline_fences,
            streaming_cache_bytes,
            streaming_min_contiguous_targets,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        targets: Vec<(u64, String, i32)>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        execute_hierarchical_python(&mut self.inner, py, targets, None)
    }

    fn execute_incremental_window(
        &mut self,
        py: Python<'_>,
        batches: Vec<Vec<(u64, String, i32)>>,
    ) -> PyResult<(
        Vec<Vec<(u64, Py<PyBytes>, u32, u32)>>,
        Vec<u64>,
        Vec<u64>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        execute_hierarchical_incremental_python(&mut self.inner, py, batches)
    }

    fn plan_dependency_sampler(
        &self,
        base_order: Vec<u64>,
        batch_size: usize,
        lookahead_samples: usize,
    ) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
        plan_dependency_sampler_python(&self.inner, base_order, batch_size, lookahead_samples)
    }

    fn plan_label_preserving_dependency_sampler(
        &self,
        base_order: Vec<(u64, i64, Vec<u64>)>,
        batch_size: usize,
        lookahead_batches: usize,
    ) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
        plan_label_preserving_dependency_sampler_python(
            &self.inner,
            base_order,
            batch_size,
            lookahead_batches,
        )
    }

    fn plan_label_preserving_logical_sampler(
        &self,
        base_order: Vec<(u64, i64, Vec<(String, i32)>)>,
        batch_size: usize,
        lookahead_batches: usize,
    ) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
        plan_label_preserving_logical_sampler_python(
            &self.inner,
            base_order,
            batch_size,
            lookahead_batches,
        )
    }

    fn choose_lookahead(
        &self,
        batches: Vec<Vec<(u64, String, i32)>>,
        candidates: Vec<usize>,
        first_batch_slo_ms: f64,
    ) -> PyResult<(usize, bool, Vec<(usize, f64, f64, u64, usize)>)> {
        let batches = batches
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(
                        |(sample_id, video_id, frame_idx)| hierarchical_scheduler::LogicalTarget {
                            sample_id,
                            video_id,
                            frame_idx,
                        },
                    )
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let decision = self
            .inner
            .choose_lookahead(&batches, &candidates, first_batch_slo_ms * 1e6)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let alternatives = decision
            .alternatives
            .iter()
            .map(|estimate| {
                (
                    estimate.lookahead_batches,
                    estimate.predicted_samples_per_second,
                    estimate.first_batch_total_ns / 1e6,
                    estimate.first_batch_fetched_bytes,
                    estimate.first_batch_ranges,
                )
            })
            .collect();
        Ok((
            decision.selected.lookahead_batches,
            decision.selected_under_slo,
            alternatives,
        ))
    }

    #[pyo3(signature = (targets, action, fixed_gap_bytes=16384))]
    fn execute_forced(
        &mut self,
        py: Python<'_>,
        targets: Vec<(u64, String, i32)>,
        action: &str,
        fixed_gap_bytes: u64,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        let action = match action {
            "region_all" => hierarchical_scheduler::HierarchicalAction::RegionAll,
            "region_selective" => hierarchical_scheduler::HierarchicalAction::RegionSelective,
            "exact_closure" => hierarchical_scheduler::HierarchicalAction::ExactClosure,
            "fixed_gap_closure" => {
                hierarchical_scheduler::HierarchicalAction::FixedGapClosure(fixed_gap_bytes)
            }
            value => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown hierarchical action {value}"
                )))
            }
        };
        execute_hierarchical_python(&mut self.inner, py, targets, Some(action))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyS3HierarchicalBatchExecutor {
    #[new]
    #[pyo3(signature = (
        local_chunk_path, object_key, endpoint, bucket, access_key, secret_key,
        cost_model, wave_request_overhead_ns, region="us-east-1".to_string(),
        max_concurrency=8, max_merge_gap_bytes=None, max_range_bytes=None,
        decoder_threads=1, incremental_decode_slots=1,
        incremental_batch_deadline_fences=false, streaming_cache_bytes=1048576,
        streaming_min_contiguous_targets=16
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        local_chunk_path: &str,
        object_key: String,
        endpoint: String,
        bucket: String,
        access_key: String,
        secret_key: String,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        region: String,
        max_concurrency: usize,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        decoder_threads: usize,
        incremental_decode_slots: usize,
        incremental_batch_deadline_fences: bool,
        streaming_cache_bytes: usize,
        streaming_min_contiguous_targets: usize,
    ) -> PyResult<Self> {
        let (catalog, codec_config, payload_start) =
            hierarchical_metadata_from_chunk(local_chunk_path)?;
        let backend = backend::S3Backend::new(
            endpoint,
            bucket,
            object_key,
            payload_start,
            access_key,
            secret_key,
            region,
            max_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let inner = hierarchical_scheduler::HierarchicalBatchExecutor::new(
            catalog,
            Box::new(backend),
            codec_config,
            hierarchical_model_from_python(&cost_model, wave_request_overhead_ns)?,
            max_merge_gap_bytes,
            max_range_bytes,
            decoder_threads,
            incremental_decode_slots,
            incremental_batch_deadline_fences,
            streaming_cache_bytes,
            streaming_min_contiguous_targets,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        targets: Vec<(u64, String, i32)>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        execute_hierarchical_python(&mut self.inner, py, targets, None)
    }

    fn execute_incremental_window(
        &mut self,
        py: Python<'_>,
        batches: Vec<Vec<(u64, String, i32)>>,
    ) -> PyResult<(
        Vec<Vec<(u64, Py<PyBytes>, u32, u32)>>,
        Vec<u64>,
        Vec<u64>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        execute_hierarchical_incremental_python(&mut self.inner, py, batches)
    }

    fn plan_dependency_sampler(
        &self,
        base_order: Vec<u64>,
        batch_size: usize,
        lookahead_samples: usize,
    ) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
        plan_dependency_sampler_python(&self.inner, base_order, batch_size, lookahead_samples)
    }

    fn plan_label_preserving_dependency_sampler(
        &self,
        base_order: Vec<(u64, i64, Vec<u64>)>,
        batch_size: usize,
        lookahead_batches: usize,
    ) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
        plan_label_preserving_dependency_sampler_python(
            &self.inner,
            base_order,
            batch_size,
            lookahead_batches,
        )
    }

    fn plan_label_preserving_logical_sampler(
        &self,
        base_order: Vec<(u64, i64, Vec<(String, i32)>)>,
        batch_size: usize,
        lookahead_batches: usize,
    ) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
        plan_label_preserving_logical_sampler_python(
            &self.inner,
            base_order,
            batch_size,
            lookahead_batches,
        )
    }

    fn choose_lookahead(
        &self,
        batches: Vec<Vec<(u64, String, i32)>>,
        candidates: Vec<usize>,
        first_batch_slo_ms: f64,
    ) -> PyResult<(usize, bool, Vec<(usize, f64, f64, u64, usize)>)> {
        let batches = batches
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(
                        |(sample_id, video_id, frame_idx)| hierarchical_scheduler::LogicalTarget {
                            sample_id,
                            video_id,
                            frame_idx,
                        },
                    )
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let decision = self
            .inner
            .choose_lookahead(&batches, &candidates, first_batch_slo_ms * 1e6)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let alternatives = decision
            .alternatives
            .iter()
            .map(|estimate| {
                (
                    estimate.lookahead_batches,
                    estimate.predicted_samples_per_second,
                    estimate.first_batch_total_ns / 1e6,
                    estimate.first_batch_fetched_bytes,
                    estimate.first_batch_ranges,
                )
            })
            .collect();
        Ok((
            decision.selected.lookahead_batches,
            decision.selected_under_slo,
            alternatives,
        ))
    }

    #[pyo3(signature = (targets, action, fixed_gap_bytes=16384))]
    fn execute_forced(
        &mut self,
        py: Python<'_>,
        targets: Vec<(u64, String, i32)>,
        action: &str,
        fixed_gap_bytes: u64,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        let action = match action {
            "region_all" => hierarchical_scheduler::HierarchicalAction::RegionAll,
            "region_selective" => hierarchical_scheduler::HierarchicalAction::RegionSelective,
            "exact_closure" => hierarchical_scheduler::HierarchicalAction::ExactClosure,
            "fixed_gap_closure" => {
                hierarchical_scheduler::HierarchicalAction::FixedGapClosure(fixed_gap_bytes)
            }
            value => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown hierarchical action {value}"
                )))
            }
        };
        execute_hierarchical_python(&mut self.inner, py, targets, Some(action))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyAIStoreHierarchicalBatchExecutor {
    #[new]
    #[pyo3(signature = (
        local_chunk_path, object_key, endpoint, bucket, cost_model,
        wave_request_overhead_ns, provider="ais".to_string(),
        max_merge_gap_bytes=None, max_range_bytes=None, decoder_threads=1,
        incremental_decode_slots=1, incremental_batch_deadline_fences=false,
        streaming_cache_bytes=1048576,
        streaming_min_contiguous_targets=16
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        local_chunk_path: &str,
        object_key: String,
        endpoint: String,
        bucket: String,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        provider: String,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        decoder_threads: usize,
        incremental_decode_slots: usize,
        incremental_batch_deadline_fences: bool,
        streaming_cache_bytes: usize,
        streaming_min_contiguous_targets: usize,
    ) -> PyResult<Self> {
        let (catalog, codec_config, payload_start) =
            hierarchical_metadata_from_chunk(local_chunk_path)?;
        let backend = backend::AIStoreGetBatchBackend::new(
            endpoint,
            bucket,
            object_key,
            provider,
            payload_start,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let inner = hierarchical_scheduler::HierarchicalBatchExecutor::new(
            catalog,
            Box::new(backend),
            codec_config,
            hierarchical_model_from_python(&cost_model, wave_request_overhead_ns)?,
            max_merge_gap_bytes,
            max_range_bytes,
            decoder_threads,
            incremental_decode_slots,
            incremental_batch_deadline_fences,
            streaming_cache_bytes,
            streaming_min_contiguous_targets,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        targets: Vec<(u64, String, i32)>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        execute_hierarchical_python(&mut self.inner, py, targets, None)
    }

    fn execute_incremental_window(
        &mut self,
        py: Python<'_>,
        batches: Vec<Vec<(u64, String, i32)>>,
    ) -> PyResult<(
        Vec<Vec<(u64, Py<PyBytes>, u32, u32)>>,
        Vec<u64>,
        Vec<u64>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        execute_hierarchical_incremental_python(&mut self.inner, py, batches)
    }

    fn plan_dependency_sampler(
        &self,
        base_order: Vec<u64>,
        batch_size: usize,
        lookahead_samples: usize,
    ) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
        plan_dependency_sampler_python(&self.inner, base_order, batch_size, lookahead_samples)
    }

    fn plan_label_preserving_dependency_sampler(
        &self,
        base_order: Vec<(u64, i64, Vec<u64>)>,
        batch_size: usize,
        lookahead_batches: usize,
    ) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
        plan_label_preserving_dependency_sampler_python(
            &self.inner,
            base_order,
            batch_size,
            lookahead_batches,
        )
    }

    fn plan_label_preserving_logical_sampler(
        &self,
        base_order: Vec<(u64, i64, Vec<(String, i32)>)>,
        batch_size: usize,
        lookahead_batches: usize,
    ) -> PyResult<(Vec<Vec<u64>>, std::collections::HashMap<String, f64>)> {
        plan_label_preserving_logical_sampler_python(
            &self.inner,
            base_order,
            batch_size,
            lookahead_batches,
        )
    }

    #[pyo3(signature = (targets, action, fixed_gap_bytes=16384))]
    fn execute_forced(
        &mut self,
        py: Python<'_>,
        targets: Vec<(u64, String, i32)>,
        action: &str,
        fixed_gap_bytes: u64,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        let action = match action {
            "region_all" => hierarchical_scheduler::HierarchicalAction::RegionAll,
            "region_selective" => hierarchical_scheduler::HierarchicalAction::RegionSelective,
            "exact_closure" => hierarchical_scheduler::HierarchicalAction::ExactClosure,
            "fixed_gap_closure" => {
                hierarchical_scheduler::HierarchicalAction::FixedGapClosure(fixed_gap_bytes)
            }
            value => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown hierarchical action {value}"
                )))
            }
        };
        execute_hierarchical_python(&mut self.inner, py, targets, Some(action))
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyS3HierarchicalExecutorPool {
    #[new]
    #[pyo3(signature = (
        local_chunk_path, object_key, endpoint, bucket, access_key, secret_key,
        cost_model, wave_request_overhead_ns, workers,
        region="us-east-1".to_string(), global_io_concurrency=8,
        max_merge_gap_bytes=None, max_range_bytes=None, decoder_threads=1,
        streaming_cache_bytes=1048576, streaming_min_contiguous_targets=16
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        local_chunk_path: &str,
        object_key: String,
        endpoint: String,
        bucket: String,
        access_key: String,
        secret_key: String,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        workers: usize,
        region: String,
        global_io_concurrency: usize,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        decoder_threads: usize,
        streaming_cache_bytes: usize,
        streaming_min_contiguous_targets: usize,
    ) -> PyResult<Self> {
        if workers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "hierarchical executor pool requires at least one worker",
            ));
        }
        let (catalog, codec_config, payload_start) =
            hierarchical_metadata_from_chunk(local_chunk_path)?;
        let client = backend::S3ObjectStoreClient::new(
            endpoint,
            bucket,
            access_key,
            secret_key,
            region,
            global_io_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let model = hierarchical_model_from_python(&cost_model, wave_request_overhead_ns)?;
        let per_worker_streaming_cache_bytes = streaming_cache_bytes / workers;
        let mut executors = Vec::with_capacity(workers);
        for _ in 0..workers {
            let backend = backend::S3Backend::from_shared_client(
                client.clone(),
                object_key.clone(),
                payload_start,
            )
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
            executors.push(std::sync::Mutex::new(
                hierarchical_scheduler::HierarchicalBatchExecutor::new(
                    catalog.clone(),
                    Box::new(backend),
                    codec_config.clone(),
                    model.clone(),
                    max_merge_gap_bytes,
                    max_range_bytes,
                    decoder_threads,
                    1,
                    false,
                    per_worker_streaming_cache_bytes,
                    streaming_min_contiguous_targets,
                )
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?,
            ));
        }
        Ok(Self { workers: executors })
    }

    fn execute(
        &self,
        py: Python<'_>,
        worker: usize,
        targets: Vec<(u64, String, i32)>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        let slot = self.workers.get(worker).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "worker {worker} outside executor pool of size {}",
                self.workers.len()
            ))
        })?;
        let mut executor = slot.lock().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "hierarchical executor worker lock is poisoned",
            )
        })?;
        execute_hierarchical_python(&mut executor, py, targets, None)
    }

    #[pyo3(signature = (worker, targets, action, fixed_gap_bytes=16384))]
    fn execute_forced(
        &self,
        py: Python<'_>,
        worker: usize,
        targets: Vec<(u64, String, i32)>,
        action: &str,
        fixed_gap_bytes: u64,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        let action = match action {
            "region_all" => hierarchical_scheduler::HierarchicalAction::RegionAll,
            "region_selective" => hierarchical_scheduler::HierarchicalAction::RegionSelective,
            "exact_closure" => hierarchical_scheduler::HierarchicalAction::ExactClosure,
            "fixed_gap_closure" => {
                hierarchical_scheduler::HierarchicalAction::FixedGapClosure(fixed_gap_bytes)
            }
            value => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown hierarchical action {value}"
                )))
            }
        };
        let slot = self.workers.get(worker).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "worker {worker} outside executor pool of size {}",
                self.workers.len()
            ))
        })?;
        let mut executor = slot.lock().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "hierarchical executor worker lock is poisoned",
            )
        })?;
        execute_hierarchical_python(&mut executor, py, targets, Some(action))
    }
}

/// Python module entry point.
#[pymodule]
fn vclasp(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "TWO_LEVEL_LAYOUT_MODE",
        if cfg!(vclasp_patched_x264) {
            "single_h264_session"
        } else {
            "independent_page_sessions"
        },
    )?;
    m.add_class::<VClaspChunk>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyH264Decoder>()?;
    #[cfg(feature = "ffmpeg")]
    saturation::register(m)?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyNormalizedBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyLocalNormalizedBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyAdaptiveBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyLocalHierarchicalBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyS3HierarchicalBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyAIStoreHierarchicalBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyS3HierarchicalExecutorPool>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyAIStoreNormalizedBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyPairBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyAIStorePairBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyPrefixBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyAIStorePrefixBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyAIStoreFragmentBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyS3FragmentBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyLocalClosedRecordBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyBudgetedPairBatchExecutor>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyLogicalBatchDecoder>()?;
    #[cfg(feature = "ffmpeg")]
    m.add_class::<PyLogicalScheduler>()?;
    m.add_class::<PyByteCache>()?;
    m.add_class::<PyS3RangeReader>()?;
    m.add_class::<PyS3ObjectStoreReader>()?;
    m.add_class::<PyLocalRangeReader>()?;
    m.add_class::<PyAIStoreGetBatchReader>()?;
    m.add_function(wrap_pyfunction!(write_chunk_from_files, m)?)?;
    m.add_function(wrap_pyfunction!(build_chunk_from_videos, m)?)?;
    #[cfg(feature = "ffmpeg")]
    m.add_function(wrap_pyfunction!(build_hierarchical_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(build_fused_normalized_layout, m)?)?;
    #[cfg(feature = "ffmpeg")]
    m.add_function(wrap_pyfunction!(build_two_level_page_layout, m)?)?;
    m.add_function(wrap_pyfunction!(plan_byte_ranges_py, m)?)?;
    m.add_function(wrap_pyfunction!(select_pair_materialization, m)?)?;
    m.add_function(wrap_pyfunction!(materialize_pair_records, m)?)?;
    m.add_function(wrap_pyfunction!(select_portfolio_candidate, m)?)?;
    Ok(())
}
