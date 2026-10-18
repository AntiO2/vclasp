use super::*;

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyPendingBatch {
    pub(super) fn result(
        &self,
        py: Python<'_>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        let pending = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "pending VClasp batch result was already consumed",
                )
            })?;
        let result = py.allow_threads(|| pending.wait());
        PyVClaspSession::batch_result_to_python(py, result)
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyPendingWindow {
    pub(super) fn result(
        &self,
        py: Python<'_>,
    ) -> PyResult<(
        Vec<Vec<(u64, Py<PyBytes>, u32, u32)>>,
        Vec<u64>,
        Vec<u64>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        let pending = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "pending VClasp window result was already consumed",
                )
            })?;
        let result = py.allow_threads(|| pending.wait());
        PyVClaspSession::window_result_to_python(py, result)
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyVClaspPipeline {
    /// Submit one already-sampled training batch. This releases the GIL while
    /// waiting for bounded pipeline capacity.
    pub(super) fn submit(&self, py: Python<'_>, targets: Vec<(u64, String, i32)>) -> PyResult<u64> {
        let logical = targets
            .into_iter()
            .map(
                |(sample_id, video_id, frame_idx)| hierarchical_scheduler::LogicalTarget {
                    sample_id,
                    video_id,
                    frame_idx,
                },
            )
            .collect::<Vec<_>>();
        py.allow_threads(|| self.inner.submit(logical))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Consume the next batch in submission order. Physical execution remains
    /// concurrent and may complete out of order behind this ordered boundary.
    #[allow(clippy::type_complexity)]
    pub(super) fn take(
        &self,
        py: Python<'_>,
    ) -> PyResult<
        Option<(
            u64,
            Vec<(u64, Py<PyBytes>, u32, u32)>,
            std::collections::HashMap<String, u64>,
            String,
            f64,
            u64,
        )>,
    > {
        let Some(batch) = py.allow_threads(|| self.inner.take()) else {
            return Ok(None);
        };
        let (frames, stats, mode, predicted_ns) =
            PyVClaspSession::batch_result_to_python(py, batch.result)?;
        Ok(Some((
            batch.sequence,
            frames,
            stats,
            mode,
            predicted_ns,
            batch.residence_ns,
        )))
    }

    pub(super) fn close(&self) {
        self.inner.close();
    }

    pub(super) fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    pub(super) fn metrics_snapshot(&self) -> std::collections::HashMap<String, u64> {
        let metrics = self.inner.metrics_snapshot();
        [
            ("submitted_batches", metrics.submitted_batches),
            ("submitted_targets", metrics.submitted_targets),
            ("delivered_batches", metrics.delivered_batches),
            ("delivered_targets", metrics.delivered_targets),
            ("outstanding_batches", metrics.outstanding_batches as u64),
            ("outstanding_targets", metrics.outstanding_targets as u64),
            (
                "peak_outstanding_batches",
                metrics.peak_outstanding_batches as u64,
            ),
            (
                "peak_outstanding_targets",
                metrics.peak_outstanding_targets as u64,
            ),
            ("producer_wait_ns", metrics.producer_wait_ns),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect()
    }
}

#[cfg(feature = "ffmpeg")]
#[pymethods]
impl PyVClaspSession {
    pub(super) fn resident_state_snapshot(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .resident_state_snapshot()
            .into_iter()
            .enumerate()
            .map(|(lane, state)| {
                let row = pyo3::types::PyDict::new_bound(py);
                for (name, value) in state {
                    row.set_item(name, value)?;
                }
                row.set_item("lane_id", lane)?;
                row.set_item("scope", "last_completed_execution_per_lane")?;
                row.set_item(
                    "clock",
                    if cfg!(target_os = "linux") {
                        "CLOCK_MONOTONIC"
                    } else {
                        "unavailable"
                    },
                )?;
                Ok(row.into_py(py))
            })
            .collect()
    }
    pub(super) fn set_stage_profiling(&self, enabled: bool) {
        self.inner.set_stage_profiling(enabled);
    }

    pub(super) fn take_decode_job_trace(&self, py: Python<'_>) -> PyResult<(Vec<PyObject>, u64)> {
        let (events, dropped) = self.inner.drain_decode_job_trace();
        let mut rows = Vec::with_capacity(events.len());
        for (lane, event) in events {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("lane_id", lane)?;
            row.set_item("job_id", event.job_id)?;
            row.set_item("record_ids", event.record_ids)?;
            row.set_item("target_record_ids", event.target_record_ids)?;
            row.set_item("consumer_sample_ids", event.consumer_sample_ids)?;
            row.set_item("execution_path", event.execution_path)?;
            row.set_item(
                "decoder_context_address",
                (event.profile.context_address != 0).then_some(event.profile.context_address),
            )?;
            row.set_item("submitted_ns", event.submitted_ns)?;
            row.set_item("worker_started_ns", event.worker_started_ns)?;
            row.set_item("resource_acquired_ns", event.resource_acquired_ns)?;
            row.set_item("completed_ns", event.completed_ns)?;
            row.set_item(
                "clock",
                if cfg!(target_os = "linux") {
                    "CLOCK_MONOTONIC"
                } else {
                    "unavailable"
                },
            )?;
            row.set_item(
                "codec_metrics",
                event
                    .profile
                    .metric_pairs()
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>(),
            )?;
            rows.push(row.into_py(py));
        }
        Ok((rows, dropped))
    }
    #[staticmethod]
    #[pyo3(signature = (
        chunk_path, cost_model, wave_request_overhead_ns,
        max_callers=1, max_pending_calls=None, max_inflight_bytes=268435456, max_inflight_windows=None,
        admission_quiet_us=0,
        max_merge_gap_bytes=None, max_range_bytes=None, decoder_threads=1,
        global_decode_concurrency=8, resident_encoded_bytes=1048576,
        resident_read_ahead_bytes=65536, resident_cursor_capacity=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn local(
        py: Python<'_>,
        chunk_path: &str,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        max_callers: usize,
        max_pending_calls: Option<usize>,
        max_inflight_bytes: usize,
        max_inflight_windows: Option<usize>,
        admission_quiet_us: u64,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        decoder_threads: usize,
        global_decode_concurrency: usize,
        resident_encoded_bytes: usize,
        resident_read_ahead_bytes: usize,
        resident_cursor_capacity: Option<usize>,
    ) -> PyResult<Self> {
        let config = session_config_from_python(
            &cost_model,
            wave_request_overhead_ns,
            max_callers,
            max_pending_calls,
            max_inflight_bytes,
            max_inflight_windows,
            admission_quiet_us,
            max_merge_gap_bytes,
            max_range_bytes,
            decoder_threads,
            global_decode_concurrency,
            None,
            resident_encoded_bytes,
            resident_read_ahead_bytes,
            resident_cursor_capacity,
        )?;
        Ok(Self {
            inner: py
                .allow_threads(|| session::VClaspSession::open_local(chunk_path, config))
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (
        local_chunk_path, object_key, endpoint, bucket, cost_model,
        wave_request_overhead_ns, max_callers=8,
        max_pending_calls=None, max_inflight_bytes=268435456, max_inflight_windows=None,
        provider="ais".to_string(), admission_quiet_us=250,
        max_merge_gap_bytes=None, max_range_bytes=None, decoder_threads=1,
        global_decode_concurrency=8, resident_encoded_bytes=1048576,
        resident_read_ahead_bytes=65536, resident_cursor_capacity=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn aistore(
        py: Python<'_>,
        local_chunk_path: &str,
        object_key: String,
        endpoint: String,
        bucket: String,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        max_callers: usize,
        max_pending_calls: Option<usize>,
        max_inflight_bytes: usize,
        max_inflight_windows: Option<usize>,
        provider: String,
        admission_quiet_us: u64,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        decoder_threads: usize,
        global_decode_concurrency: usize,
        resident_encoded_bytes: usize,
        resident_read_ahead_bytes: usize,
        resident_cursor_capacity: Option<usize>,
    ) -> PyResult<Self> {
        let config = session_config_from_python(
            &cost_model,
            wave_request_overhead_ns,
            max_callers,
            max_pending_calls,
            max_inflight_bytes,
            max_inflight_windows,
            admission_quiet_us,
            max_merge_gap_bytes,
            max_range_bytes,
            decoder_threads,
            global_decode_concurrency,
            None,
            resident_encoded_bytes,
            resident_read_ahead_bytes,
            resident_cursor_capacity,
        )?;
        Ok(Self {
            inner: py
                .allow_threads(|| {
                    session::VClaspSession::open_aistore(
                        local_chunk_path,
                        session::AIStoreConfig {
                            endpoint,
                            bucket,
                            object_key,
                            provider,
                        },
                        config,
                    )
                })
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?,
        })
    }

    #[new]
    #[pyo3(signature = (
        local_chunk_path, object_key, endpoint, bucket, access_key, secret_key,
        cost_model, wave_request_overhead_ns, max_callers,
        max_pending_calls=None, max_inflight_bytes=268435456, max_inflight_windows=None,
        region="us-east-1".to_string(), global_io_concurrency=8,
        global_decode_concurrency=8,
        admission_quiet_us=250,
        max_merge_gap_bytes=None, max_range_bytes=None, decoder_threads=1,
        cursor_decoder_threads=None,
        resident_encoded_bytes=1048576, resident_read_ahead_bytes=65536,
        resident_cursor_capacity=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        py: Python<'_>,
        local_chunk_path: &str,
        object_key: String,
        endpoint: String,
        bucket: String,
        access_key: String,
        secret_key: String,
        cost_model: std::collections::HashMap<String, f64>,
        wave_request_overhead_ns: Vec<f64>,
        max_callers: usize,
        max_pending_calls: Option<usize>,
        max_inflight_bytes: usize,
        max_inflight_windows: Option<usize>,
        region: String,
        global_io_concurrency: usize,
        global_decode_concurrency: usize,
        admission_quiet_us: u64,
        max_merge_gap_bytes: Option<u64>,
        max_range_bytes: Option<u64>,
        decoder_threads: usize,
        cursor_decoder_threads: Option<usize>,
        resident_encoded_bytes: usize,
        resident_read_ahead_bytes: usize,
        resident_cursor_capacity: Option<usize>,
    ) -> PyResult<Self> {
        let config = session_config_from_python(
            &cost_model,
            wave_request_overhead_ns,
            max_callers,
            max_pending_calls,
            max_inflight_bytes,
            max_inflight_windows,
            admission_quiet_us,
            max_merge_gap_bytes,
            max_range_bytes,
            decoder_threads,
            global_decode_concurrency,
            cursor_decoder_threads,
            resident_encoded_bytes,
            resident_read_ahead_bytes,
            resident_cursor_capacity,
        )?;
        if config.cost_model.io_concurrency != global_io_concurrency {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "cost-model I/O concurrency {} differs from the execution budget {}",
                config.cost_model.io_concurrency, global_io_concurrency
            )));
        }
        Ok(Self {
            inner: py
                .allow_threads(|| {
                    session::VClaspSession::open_s3(
                        local_chunk_path,
                        session::S3Config {
                            endpoint,
                            bucket,
                            object_key,
                            access_key_id: access_key,
                            secret_access_key: secret_key,
                            region,
                            max_concurrency: global_io_concurrency,
                        },
                        config,
                    )
                })
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?,
        })
    }

    #[cfg(feature = "experiment-controls")]
    pub(super) fn execute_forced(
        &self,
        py: Python<'_>,
        targets: Vec<(u64, String, i32)>,
        action: String,
        fixed_gap_bytes: u64,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        let action = hierarchical_action_from_python(&action, fixed_gap_bytes)?;
        let logical = targets
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
            .allow_threads(|| self.inner.execute_forced(logical, action))
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

    pub(super) fn execute(
        &self,
        py: Python<'_>,
        targets: Vec<(u64, String, i32)>,
    ) -> PyResult<(
        Vec<(u64, Py<PyBytes>, u32, u32)>,
        std::collections::HashMap<String, u64>,
        String,
        f64,
    )> {
        self.execute_batch_internal(py, targets)
    }

    /// Admit work without waiting for completion. The returned handle may be
    /// resolved later, after additional requests have entered the same bounded
    /// planning window.
    pub(super) fn submit(
        &self,
        py: Python<'_>,
        targets: Vec<(u64, String, i32)>,
    ) -> PyResult<PyPendingBatch> {
        let logical = targets
            .into_iter()
            .map(
                |(sample_id, video_id, frame_idx)| hierarchical_scheduler::LogicalTarget {
                    sample_id,
                    video_id,
                    frame_idx,
                },
            )
            .collect::<Vec<_>>();
        let pending = py
            .allow_threads(|| self.inner.submit(logical))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(PyPendingBatch {
            inner: Mutex::new(Some(pending)),
        })
    }

    /// Create the default bounded producer/consumer interface for training.
    #[pyo3(signature = (max_outstanding_batches=8, max_outstanding_targets=None))]
    pub(super) fn pipeline(
        &self,
        max_outstanding_batches: usize,
        max_outstanding_targets: Option<usize>,
    ) -> PyResult<PyVClaspPipeline> {
        let inner = session::VClaspPipeline::new(
            self.inner.clone(),
            session::PipelineConfig {
                max_outstanding_batches,
                max_outstanding_targets,
            },
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(PyVClaspPipeline { inner })
    }

    /// Admit an already-sampled bounded window without waiting. Logical batch
    /// boundaries and order are restored after joint physical execution.
    pub(super) fn submit_window(
        &self,
        py: Python<'_>,
        batches: Vec<Vec<(u64, String, i32)>>,
    ) -> PyResult<PyPendingWindow> {
        if batches.is_empty() || batches.iter().any(Vec::is_empty) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "VClasp request window requires non-empty batches",
            ));
        }
        let logical = batches
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
        let pending = py
            .allow_threads(|| self.inner.submit_window(logical))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(PyPendingWindow {
            inner: Mutex::new(Some(pending)),
        })
    }

    pub(super) fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.synchronize())
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    pub(super) fn metrics_snapshot(&self) -> std::collections::HashMap<String, u64> {
        let metrics = self.inner.metrics_snapshot();
        [
            ("completed_windows", metrics.completed_windows),
            ("resident_path_windows", metrics.resident_path_windows),
            ("pending_calls", metrics.pending_calls),
            ("pending_peak_calls", metrics.pending_peak_calls),
            (
                "pending_admission_wait_ns",
                metrics.pending_admission_wait_ns,
            ),
            ("pending_call_capacity", metrics.pending_call_capacity),
            (
                "transient_buffer_capacity_bytes",
                metrics.transient_buffer_capacity_bytes,
            ),
            (
                "transient_buffer_reserved_bytes",
                metrics.transient_buffer_reserved_bytes,
            ),
            (
                "transient_buffer_peak_bytes",
                metrics.transient_buffer_peak_bytes,
            ),
            ("transient_buffer_wait_ns", metrics.transient_buffer_wait_ns),
            ("stateless_path_windows", metrics.stateless_path_windows),
            ("other_path_windows", metrics.other_path_windows),
            ("logical_targets", metrics.logical_targets),
            ("physical_ranges", metrics.physical_ranges),
            ("client_requests", metrics.client_requests),
            ("useful_bytes", metrics.useful_bytes),
            ("fetched_bytes", metrics.fetched_bytes),
            ("submitted_access_units", metrics.submitted_access_units),
            ("decoded_access_units", metrics.decoded_access_units),
            ("decode_groups", metrics.decode_groups),
            ("resident_encoded_hits", metrics.resident_encoded_hits),
            ("resident_encoded_misses", metrics.resident_encoded_misses),
            (
                "resident_encoded_evictions",
                metrics.resident_encoded_evictions,
            ),
            ("resident_cursor_hits", metrics.resident_cursor_hits),
            ("resident_cursor_misses", metrics.resident_cursor_misses),
            ("decoder_state_resets", metrics.decoder_state_resets),
            ("plan_ns", metrics.plan_ns),
            ("fetch_ns", metrics.fetch_ns),
            ("assemble_ns", metrics.assemble_ns),
            (
                "incremental_range_buffer_peak_bytes",
                metrics.incremental_range_buffer_peak_bytes,
            ),
            ("decode_ns", metrics.decode_ns),
            ("range_ttfb_ns_sum", metrics.range_ttfb_ns_sum),
            ("range_service_ns_sum", metrics.range_service_ns_sum),
            ("range_timing_samples", metrics.range_timing_samples),
            ("range_max_in_flight", metrics.range_max_in_flight),
        ]
        .into_iter()
        .chain(incremental_metric_pairs(&metrics.incremental))
        .map(|(name, value)| (name.to_string(), value))
        .chain(metrics.incremental.codec.metric_pairs())
        .collect()
    }

    /// Collect bounded physical-range and decoder-submission traces.
    #[cfg(feature = "experiment-controls")]
    pub(super) fn start_execution_trace(
        &self,
        max_range_events: usize,
        max_decode_events: usize,
    ) -> PyResult<()> {
        self.inner
            .start_execution_trace(max_range_events, max_decode_events)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Sample distinct shared decode budgets without coordinator admission.
    #[cfg(feature = "experiment-controls")]
    pub(super) fn decoder_activity(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.inner
            .decode_activity()
            .into_iter()
            .map(|state| {
                let row = PyDict::new_bound(py);
                row.set_item("active_jobs", state.active_jobs)?;
                row.set_item("queued_jobs", state.queued_jobs)?;
                row.set_item("available_threads", state.available_threads)?;
                row.set_item("lifetime_peak_active_jobs", state.peak_active_jobs)?;
                row.set_item("completed_jobs", state.completed_jobs)?;
                row.set_item("acquisition_wait_ns", state.acquisition_wait_ns)?;
                Ok(row.unbind())
            })
            .collect()
    }

    #[cfg(feature = "experiment-controls")]
    pub(super) fn io_activity(&self, py: Python<'_>) -> PyResult<Vec<Option<Py<PyDict>>>> {
        self.inner
            .io_activity()
            .into_iter()
            .map(|pressure| {
                pressure
                    .map(|value| object_store_pressure_to_python(py, value))
                    .transpose()
            })
            .collect()
    }

    /// Drain complete windows; stop capture after all work finishes by default.
    #[cfg(feature = "experiment-controls")]
    #[pyo3(signature = (stop=true))]
    pub(super) fn take_execution_trace(
        &self,
        py: Python<'_>,
        stop: bool,
    ) -> PyResult<(Vec<Py<PyDict>>, usize, usize, Vec<Py<PyDict>>, usize, usize)> {
        let trace = if stop {
            self.inner.take_execution_trace()
        } else {
            self.inner.drain_execution_trace()
        };
        let mut ranges = Vec::with_capacity(trace.range_events.len());
        for event in trace.range_events {
            let row = PyDict::new_bound(py);
            row.set_item("window_id", event.window_id)?;
            row.set_item("range_index", event.range.range_index)?;
            row.set_item("planned_payload_offset", event.range.planned_payload_offset)?;
            row.set_item("planned_length", event.range.planned_length)?;
            row.set_item("physical_request_id", event.range.physical_request_id)?;
            row.set_item("physical_object_offset", event.range.physical_object_offset)?;
            row.set_item("physical_object_length", event.range.physical_object_length)?;
            row.set_item("started_ns", event.range.started_ns)?;
            row.set_item("first_byte_ns", event.range.first_byte_ns)?;
            row.set_item("completed_ns", event.range.completed_ns)?;
            row.set_item("monotonic_timing", event.range.monotonic_timing)?;
            row.set_item("physical_requests", event.range.physical_requests)?;
            row.set_item("fetched_bytes", event.range.fetched_bytes)?;
            row.set_item("consumer_sample_ids", event.range.consumer_sample_ids)?;
            ranges.push(row.unbind());
        }
        let mut decodes = Vec::with_capacity(trace.decode_events.len());
        for event in trace.decode_events {
            let row = PyDict::new_bound(py);
            row.set_item("window_id", event.window_id)?;
            row.set_item("record_id", event.decode.record_id)?;
            row.set_item("consumer_sample_ids", event.decode.consumer_sample_ids)?;
            decodes.push(row.unbind());
        }
        Ok((
            ranges,
            trace.expected_requests,
            trace.dropped_range_events,
            decodes,
            trace.expected_submitted_aus,
            trace.dropped_decode_events,
        ))
    }

    /// Execute one already-sampled request window through the same global
    /// planner and resident-state owner used by concurrent submissions.
    pub(super) fn execute_window(
        &self,
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
        self.execute_window_internal(py, batches)
    }
}
