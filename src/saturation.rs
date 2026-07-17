//! Shared-resource PyO3 executors for workload-native saturation experiments.
//!
//! Each concurrent client owns request-local scheduler state but shares one S3
//! transport semaphore/runtime, one decoder-slot pool, and one bounded Anchor
//! cache. The Anchor budget is global rather than multiplied per client.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::backend;
use crate::decoder;
use crate::fragment_scheduler;
use crate::normalized_scheduler;
use crate::pair_scheduler;
use crate::representation;

#[pyclass]
pub struct PySharedS3ObjectStream {
    inner: backend::S3ObjectStream,
}

#[pymethods]
impl PySharedS3ObjectStream {
    fn next_chunk(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyBytes>>> {
        let chunk = py
            .allow_threads(|| self.inner.next_chunk().map_err(|error| error.to_string()))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(chunk.map(|bytes| PyBytes::new_bound(py, &bytes).unbind()))
    }

    fn bytes_received(&self) -> u64 {
        self.inner.bytes_received()
    }
}

#[pyclass]
pub struct PySharedS3ExecutionResources {
    client: backend::S3ObjectStoreClient,
    decoder_slots: decoder::SharedDecoderSlots,
    anchor_cache: crate::planner::SharedByteCache,
    delta_cache: crate::planner::SharedByteCache,
    io_concurrency: usize,
    decode_concurrency: usize,
}

#[pyclass]
pub struct PySharedAIStoreExecutionResources {
    client: backend::SharedAIStoreGetBatchClient,
    decoder_slots: decoder::SharedDecoderSlots,
    anchor_cache: crate::planner::SharedByteCache,
    delta_cache: crate::planner::SharedByteCache,
    io_concurrency: usize,
    decode_concurrency: usize,
}

impl PySharedAIStoreExecutionResources {
    fn client(&self) -> backend::SharedAIStoreGetBatchClient {
        self.client.clone()
    }

    fn decoder_slots(&self) -> decoder::SharedDecoderSlots {
        self.decoder_slots.clone()
    }

    fn anchor_cache(&self) -> crate::planner::SharedByteCache {
        self.anchor_cache.clone()
    }

    fn delta_cache(&self) -> crate::planner::SharedByteCache {
        self.delta_cache.clone()
    }
}

#[pymethods]
impl PySharedAIStoreExecutionResources {
    #[new]
    #[pyo3(signature = (endpoint, bucket, provider="ais".to_string(),
                        io_concurrency=8, decode_concurrency=8,
                        anchor_cache_bytes=0, delta_cache_bytes=0))]
    fn new(
        endpoint: String,
        bucket: String,
        provider: String,
        io_concurrency: usize,
        decode_concurrency: usize,
        anchor_cache_bytes: usize,
        delta_cache_bytes: usize,
    ) -> PyResult<Self> {
        if decode_concurrency == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "decode_concurrency must be positive",
            ));
        }
        let client =
            backend::SharedAIStoreGetBatchClient::new(endpoint, bucket, provider, io_concurrency)
                .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
        Ok(Self {
            client,
            decoder_slots: decoder::shared_decoder_slots(decode_concurrency),
            anchor_cache: crate::planner::shared_byte_cache(anchor_cache_bytes),
            delta_cache: crate::planner::shared_byte_cache(delta_cache_bytes),
            io_concurrency,
            decode_concurrency,
        })
    }

    #[getter]
    fn io_concurrency(&self) -> usize {
        self.io_concurrency
    }

    #[getter]
    fn decode_concurrency(&self) -> usize {
        self.decode_concurrency
    }
}

impl PySharedS3ExecutionResources {
    fn client(&self) -> backend::S3ObjectStoreClient {
        self.client.clone()
    }

    fn decoder_slots(&self) -> decoder::SharedDecoderSlots {
        self.decoder_slots.clone()
    }

    fn anchor_cache(&self) -> crate::planner::SharedByteCache {
        self.anchor_cache.clone()
    }

    fn delta_cache(&self) -> crate::planner::SharedByteCache {
        self.delta_cache.clone()
    }
}

#[pymethods]
impl PySharedS3ExecutionResources {
    #[new]
    #[pyo3(signature = (endpoint, bucket, access_key_id, secret_access_key,
                        region="us-east-1".to_string(), io_concurrency=8,
                        decode_concurrency=8, anchor_cache_bytes=0,
                        delta_cache_bytes=0))]
    fn new(
        endpoint: String,
        bucket: String,
        access_key_id: String,
        secret_access_key: String,
        region: String,
        io_concurrency: usize,
        decode_concurrency: usize,
        anchor_cache_bytes: usize,
        delta_cache_bytes: usize,
    ) -> PyResult<Self> {
        if decode_concurrency == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "decode_concurrency must be positive",
            ));
        }
        let client = backend::S3ObjectStoreClient::new(
            endpoint,
            bucket,
            access_key_id,
            secret_access_key,
            region,
            io_concurrency,
        )
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        Ok(Self {
            client,
            decoder_slots: decoder::shared_decoder_slots(decode_concurrency),
            anchor_cache: crate::planner::shared_byte_cache(anchor_cache_bytes),
            delta_cache: crate::planner::shared_byte_cache(delta_cache_bytes),
            io_concurrency,
            decode_concurrency,
        })
    }

    #[getter]
    fn io_concurrency(&self) -> usize {
        self.io_concurrency
    }

    #[getter]
    fn decode_concurrency(&self) -> usize {
        self.decode_concurrency
    }

    /// Fetch ranges through the same client/runtime/semaphore shared by every
    /// saturation executor. This is intentionally generic so native baseline
    /// adapters can share transport without inheriting VClasp planning.
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
                self.client
                    .fetch_object_ranges(&ranges)
                    .map_err(|error| error.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(buffers
            .iter()
            .map(|buffer| PyBytes::new_bound(py, buffer).unbind())
            .collect())
    }

    fn head_object(&self, object_key: String) -> PyResult<(u64, Option<String>, Option<String>)> {
        let head = self.client.head_object(&object_key).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
        })?;
        Ok((head.size, head.e_tag, head.version))
    }

    #[pyo3(signature = (object_key, buffered_chunks=2))]
    fn open_object_stream(
        &self,
        object_key: String,
        buffered_chunks: usize,
    ) -> PyResult<PySharedS3ObjectStream> {
        let inner = self
            .client
            .open_object_stream(&object_key, buffered_chunks)
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
        Ok(PySharedS3ObjectStream { inner })
    }
}

#[pyclass]
pub struct PySharedNormalizedBatchExecutor {
    inner: normalized_scheduler::NormalizedBatchExecutor,
    dependency_group_spans: bool,
}

#[pyclass]
pub struct PySharedAIStoreNormalizedBatchExecutor {
    inner: normalized_scheduler::NormalizedBatchExecutor,
    dependency_group_spans: bool,
}

#[pymethods]
impl PySharedAIStoreNormalizedBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, resources, object_key,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        delta_cache_bytes=0,
                        decode_microbatch_targets=4, width=320, height=240,
                        fusion_metadata=None, fuse_shared_anchors=false,
                        adaptive_anchor_work_units=None,
                        adaptive_delta_work_units=1.0,
                        dependency_group_spans=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, u64, u64, u64, u64)>,
        resources: PyRef<'_, PySharedAIStoreExecutionResources>,
        object_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        delta_cache_bytes: usize,
        decode_microbatch_targets: usize,
        width: u32,
        height: u32,
        fusion_metadata: Option<Vec<(u64, u64, usize)>>,
        fuse_shared_anchors: bool,
        adaptive_anchor_work_units: Option<f64>,
        adaptive_delta_work_units: f64,
        dependency_group_spans: bool,
    ) -> PyResult<Self> {
        let decode_schedule = crate::normalized_decode_schedule_from_python(
            fuse_shared_anchors,
            adaptive_anchor_work_units,
            adaptive_delta_work_units,
        )?;
        let descriptors = crate::normalized_descriptors_from_python(
            descriptors,
            fusion_metadata,
            decode_schedule.requires_group_metadata(),
        )?;
        let backend =
            backend::AIStoreGetBatchBackend::from_shared_client(resources.client(), object_key, 0)
                .map_err(|error| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
                })?;
        let mut inner = normalized_scheduler::NormalizedBatchExecutor::new_with_decode_schedule_and_slots_and_anchor_cache(
            descriptors,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            resources.anchor_cache(),
            delta_cache_bytes,
            0,
            decode_microbatch_targets,
            decode_schedule,
            resources.decoder_slots(),
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        inner.set_shared_delta_cache(resources.delta_cache());
        Ok(Self {
            inner,
            dependency_group_spans,
        })
    }

    fn execute(
        &mut self,
        py: Python<'_>,
        batch: Vec<u64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
    )> {
        let dependency_group_spans = self.dependency_group_spans;
        let (frames, stats) = py
            .allow_threads(|| {
                if dependency_group_spans {
                    self.inner
                        .execute_with_dependency_group_spans(&batch, false)
                } else {
                    self.inner.execute(&batch, false)
                }
            })
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
        Ok((frames, crate::batch_stats_dict(&stats)))
    }

    #[pyo3(signature = (
        batches, candidates, first_batch_slo_ms, cost_model,
        wave_request_overhead_ns
    ))]
    fn execute_planned_lookahead(
        &mut self,
        py: Python<'_>,
        batches: Vec<Vec<u64>>,
        candidates: Vec<usize>,
        first_batch_slo_ms: f64,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        usize,
        bool,
        u64,
        Vec<std::collections::HashMap<String, f64>>,
    )> {
        crate::execute_normalized_lookahead_from_python(
            &mut self.inner,
            py,
            batches,
            candidates,
            first_batch_slo_ms,
            cost_model,
            wave_request_overhead_ns,
            false,
        )
    }
}

#[pymethods]
impl PySharedNormalizedBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, resources, object_key,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        delta_cache_bytes=0,
                        decode_microbatch_targets=4, width=320, height=240,
                        fusion_metadata=None, fuse_shared_anchors=false,
                        adaptive_anchor_work_units=None,
                        adaptive_delta_work_units=1.0,
                        dependency_group_spans=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, u64, u64, u64, u64)>,
        resources: PyRef<'_, PySharedS3ExecutionResources>,
        object_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        delta_cache_bytes: usize,
        decode_microbatch_targets: usize,
        width: u32,
        height: u32,
        fusion_metadata: Option<Vec<(u64, u64, usize)>>,
        fuse_shared_anchors: bool,
        adaptive_anchor_work_units: Option<f64>,
        adaptive_delta_work_units: f64,
        dependency_group_spans: bool,
    ) -> PyResult<Self> {
        let decode_schedule = crate::normalized_decode_schedule_from_python(
            fuse_shared_anchors,
            adaptive_anchor_work_units,
            adaptive_delta_work_units,
        )?;
        let descriptors = crate::normalized_descriptors_from_python(
            descriptors,
            fusion_metadata,
            decode_schedule.requires_group_metadata(),
        )?;
        let backend = backend::S3Backend::from_shared_client(resources.client(), object_key, 0)
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
        let mut inner = normalized_scheduler::NormalizedBatchExecutor::new_with_decode_schedule_and_slots_and_anchor_cache(
            descriptors,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            resources.anchor_cache(),
            delta_cache_bytes,
            0,
            decode_microbatch_targets,
            decode_schedule,
            resources.decoder_slots(),
            width,
            height,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        inner.set_shared_delta_cache(resources.delta_cache());
        Ok(Self {
            inner,
            dependency_group_spans,
        })
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
        let dependency_group_spans = self.dependency_group_spans;
        let (frames, stats) = py
            .allow_threads(|| {
                if dependency_group_spans {
                    self.inner
                        .execute_with_dependency_group_spans(&batch, completion_driven)
                } else {
                    self.inner.execute(&batch, completion_driven)
                }
            })
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
        Ok((frames, crate::batch_stats_dict(&stats)))
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
        crate::execute_normalized_lookahead_from_python(
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
}

#[pyclass]
pub struct PySharedPrefixBatchExecutor {
    inner: pair_scheduler::ClosedRecordBatchExecutor,
}

#[pymethods]
impl PySharedPrefixBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, resources, object_key,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        prefix_streaming=false, prefix_cursor_capacity=64,
                        width=320, height=240))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        descriptors: Vec<(u64, u64, usize, u64, u64)>,
        resources: PyRef<'_, PySharedS3ExecutionResources>,
        object_key: String,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        prefix_streaming: bool,
        prefix_cursor_capacity: usize,
        width: u32,
        height: u32,
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
        let backend = backend::S3Backend::from_shared_client(resources.client(), object_key, 0)
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            })?;
        let inner = pair_scheduler::ClosedRecordBatchExecutor::new_with_decoder_slots(
            descriptors,
            representation::Representation::Prefix,
            Box::new(backend),
            merge_threshold_bytes,
            max_range_bytes,
            0,
            0,
            resources.decoder_slots(),
            prefix_streaming,
            prefix_cursor_capacity,
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
        Ok((frames, crate::batch_stats_dict(&stats)))
    }
}

#[pyclass]
pub struct PySharedS3FragmentBatchExecutor {
    inner: fragment_scheduler::FragmentBatchExecutor,
}

#[pymethods]
impl PySharedS3FragmentBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, resources, merge_threshold_bytes=None,
                        max_range_bytes=None, width=320, height=240))]
    fn new(
        descriptors: Vec<(u64, u64, u64, String, u64, u64, usize, Vec<u8>, bool)>,
        resources: PyRef<'_, PySharedS3ExecutionResources>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        width: u32,
        height: u32,
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
        let inner = fragment_scheduler::FragmentBatchExecutor::new(
            descriptors,
            Box::new(resources.client()),
            merge_threshold_bytes,
            max_range_bytes,
            0,
            0,
            1,
            width,
            height,
        )
        .and_then(|inner| inner.with_shared_decoder_slots(resources.decoder_slots()))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (targets, gops, gop_size, resources,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        width=320, height=240))]
    #[allow(clippy::too_many_arguments)]
    fn from_gop_index(
        targets: Vec<(u64, u64, u64)>,
        gops: Vec<(u64, u64, u64, String, u64, u64, Vec<u8>, bool)>,
        gop_size: u64,
        resources: PyRef<'_, PySharedS3ExecutionResources>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        width: u32,
        height: u32,
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
        let inner = fragment_scheduler::FragmentBatchExecutor::new_gop_index(
            targets,
            gops,
            gop_size,
            Box::new(resources.client()),
            merge_threshold_bytes,
            max_range_bytes,
            0,
            0,
            1,
            width,
            height,
        )
        .and_then(|inner| inner.with_shared_decoder_slots(resources.decoder_slots()))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (descriptors, resources, width=320, height=240))]
    fn from_mp4_segments(
        descriptors: Vec<(u64, u64, String, u64, usize)>,
        resources: PyRef<'_, PySharedS3ExecutionResources>,
        width: u32,
        height: u32,
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
        let inner = fragment_scheduler::FragmentBatchExecutor::new(
            descriptors,
            Box::new(resources.client()),
            None,
            None,
            0,
            0,
            1,
            width,
            height,
        )
        .and_then(|inner| inner.with_shared_decoder_slots(resources.decoder_slots()))
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
        Ok((frames, crate::batch_stats_dict(&stats)))
    }
}

#[pyclass]
pub struct PySharedAIStoreFragmentBatchExecutor {
    inner: fragment_scheduler::FragmentBatchExecutor,
}

#[pymethods]
impl PySharedAIStoreFragmentBatchExecutor {
    #[new]
    #[pyo3(signature = (descriptors, resources, merge_threshold_bytes=None,
                        max_range_bytes=None, width=320, height=240))]
    fn new(
        descriptors: Vec<(u64, u64, u64, String, u64, u64, usize, Vec<u8>, bool)>,
        resources: PyRef<'_, PySharedAIStoreExecutionResources>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        width: u32,
        height: u32,
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
        let inner = fragment_scheduler::FragmentBatchExecutor::new(
            descriptors,
            Box::new(resources.client()),
            merge_threshold_bytes,
            max_range_bytes,
            0,
            0,
            1,
            width,
            height,
        )
        .and_then(|inner| inner.with_shared_decoder_slots(resources.decoder_slots()))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (targets, gops, gop_size, resources,
                        merge_threshold_bytes=None, max_range_bytes=None,
                        width=320, height=240))]
    #[allow(clippy::too_many_arguments)]
    fn from_gop_index(
        targets: Vec<(u64, u64, u64)>,
        gops: Vec<(u64, u64, u64, String, u64, u64, Vec<u8>, bool)>,
        gop_size: u64,
        resources: PyRef<'_, PySharedAIStoreExecutionResources>,
        merge_threshold_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        width: u32,
        height: u32,
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
        let inner = fragment_scheduler::FragmentBatchExecutor::new_gop_index(
            targets,
            gops,
            gop_size,
            Box::new(resources.client()),
            merge_threshold_bytes,
            max_range_bytes,
            0,
            0,
            1,
            width,
            height,
        )
        .and_then(|inner| inner.with_shared_decoder_slots(resources.decoder_slots()))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (descriptors, resources, width=320, height=240))]
    fn from_mp4_segments(
        descriptors: Vec<(u64, u64, String, u64, usize)>,
        resources: PyRef<'_, PySharedAIStoreExecutionResources>,
        width: u32,
        height: u32,
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
        let inner = fragment_scheduler::FragmentBatchExecutor::new(
            descriptors,
            Box::new(resources.client()),
            None,
            None,
            0,
            0,
            1,
            width,
            height,
        )
        .and_then(|inner| inner.with_shared_decoder_slots(resources.decoder_slots()))
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
        Ok((frames, crate::batch_stats_dict(&stats)))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySharedS3ObjectStream>()?;
    m.add_class::<PySharedS3ExecutionResources>()?;
    m.add_class::<PySharedAIStoreExecutionResources>()?;
    m.add_class::<PySharedNormalizedBatchExecutor>()?;
    m.add_class::<PySharedAIStoreNormalizedBatchExecutor>()?;
    m.add_class::<PySharedPrefixBatchExecutor>()?;
    m.add_class::<PySharedS3FragmentBatchExecutor>()?;
    m.add_class::<PySharedAIStoreFragmentBatchExecutor>()?;
    Ok(())
}
