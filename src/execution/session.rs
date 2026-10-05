pub use crate::hierarchical_layout::HierarchicalCostModel as CostModel;
#[cfg(feature = "experiment-controls")]
use crate::hierarchical_scheduler::DecodeTraceEvent;
#[cfg(feature = "experiment-controls")]
use crate::hierarchical_scheduler::HierarchicalAction;
#[cfg(feature = "experiment-controls")]
use crate::hierarchical_scheduler::RangeTraceEvent;
pub use crate::hierarchical_scheduler::{
    HierarchicalBatchStats as ExecutionStats, HierarchicalIncrementalWindow as WindowOutput,
    HierarchicalOutput as FrameOutput, LogicalTarget as Target, ResidentStateBudget,
};
pub use crate::runtime_feedback::RuntimeFeedbackConfig;
use crate::{
    backend::{
        AIStoreGetBatchBackend, GlobalRangeBrokerConfig, LocalBackend, S3Backend,
        S3ObjectStoreClient,
    },
    chunk::ChunkReader,
    decoder::DecodeBudget,
    hierarchical_ingest::HierarchicalCatalog,
    hierarchical_scheduler::HierarchicalBatchExecutor,
};
use memmap2::Mmap;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "experiment-controls")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub type BatchResult = Result<(Vec<FrameOutput>, ExecutionStats), String>;
pub type WindowResult = Result<WindowOutput, String>;

/// One admitted batch whose physical work may overlap other admitted batches.
///
/// Submission and completion are separate so a single producer can expose
/// bounded future work without creating one thread per outstanding batch.
pub struct PendingBatch {
    receiver: Receiver<BatchResult>,
}

impl PendingBatch {
    pub fn wait(self) -> BatchResult {
        self.receiver
            .recv()
            .map_err(|_| "VClasp session dropped execution response".to_string())?
    }
}

pub struct PendingWindow {
    receiver: Receiver<WindowResult>,
}

impl PendingWindow {
    pub fn wait(self) -> WindowResult {
        self.receiver
            .recv()
            .map_err(|_| "VClasp session dropped window response".to_string())?
    }
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Batches admitted but not yet returned by `take`.
    pub max_outstanding_batches: usize,
    /// Optional target-count bound across all admitted, unconsumed batches.
    pub max_outstanding_targets: Option<usize>,
}

impl PipelineConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_outstanding_batches == 0 {
            return Err("pipeline batch capacity must be positive".to_string());
        }
        if self.max_outstanding_targets == Some(0) {
            return Err("pipeline target capacity must be positive when set".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    pub submitted_batches: u64,
    pub submitted_targets: u64,
    pub delivered_batches: u64,
    pub delivered_targets: u64,
    pub outstanding_batches: usize,
    pub outstanding_targets: usize,
    pub peak_outstanding_batches: usize,
    pub peak_outstanding_targets: usize,
    pub producer_wait_ns: u64,
}

pub struct PipelineBatch {
    pub sequence: u64,
    pub target_count: usize,
    pub residence_ns: u64,
    pub result: BatchResult,
}

struct PipelineEntry {
    sequence: u64,
    target_count: usize,
    submitted_at: Instant,
    pending: PendingBatch,
}

struct PipelineState {
    accepting: bool,
    next_sequence: u64,
    queue: VecDeque<PipelineEntry>,
    metrics: PipelineMetrics,
}

trait PipelineSubmitter: Send + Sync {
    fn submit_batch(&self, targets: Vec<Target>) -> Result<PendingBatch, String>;
}

struct PipelineShared {
    submitter: Arc<dyn PipelineSubmitter>,
    config: PipelineConfig,
    state: Mutex<PipelineState>,
    capacity_changed: Condvar,
    item_available: Condvar,
    /// Ordered delivery is a single logical stream even if multiple consumer
    /// threads accidentally call `take`.
    take_gate: Mutex<()>,
}

/// Bounded producer/consumer interface over one process-wide session.
///
/// `submit` exposes already-sampled future batches to the shared planner and
/// applies backpressure when the configured unconsumed-work budget is full.
/// Physical execution may finish out of order; `take` always delivers batches
/// in submission order.
#[derive(Clone)]
pub struct VClaspPipeline {
    shared: Arc<PipelineShared>,
}

impl VClaspPipeline {
    pub fn new(session: VClaspSession, config: PipelineConfig) -> Result<Self, String> {
        Self::from_submitter(Arc::new(session), config)
    }

    fn from_submitter(
        submitter: Arc<dyn PipelineSubmitter>,
        config: PipelineConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            shared: Arc::new(PipelineShared {
                submitter,
                config,
                state: Mutex::new(PipelineState {
                    accepting: true,
                    next_sequence: 0,
                    queue: VecDeque::new(),
                    metrics: PipelineMetrics::default(),
                }),
                capacity_changed: Condvar::new(),
                item_available: Condvar::new(),
                take_gate: Mutex::new(()),
            }),
        })
    }

    /// Submit one logical batch. This blocks only while the pipeline's bounded
    /// unconsumed-work budget is full.
    pub fn submit(&self, targets: Vec<Target>) -> Result<u64, String> {
        if targets.is_empty() {
            return Err("VClasp pipeline requires at least one target".to_string());
        }
        let target_count = targets.len();
        if self
            .shared
            .config
            .max_outstanding_targets
            .is_some_and(|capacity| target_count > capacity)
        {
            return Err(format!(
                "batch contains {target_count} targets, exceeding pipeline target capacity {}",
                self.shared.config.max_outstanding_targets.unwrap()
            ));
        }

        let wait_started = Instant::now();
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while state.accepting && !self.has_capacity(&state, target_count) {
            state = self
                .shared
                .capacity_changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        if !state.accepting {
            return Err("VClasp pipeline is closed".to_string());
        }
        let producer_wait_ns = elapsed_ns(wait_started);

        // Keep the queue lock while handing the command to the session so
        // concurrent producers cannot reorder sequence assignment and enqueue.
        let pending = self.shared.submitter.submit_batch(targets)?;
        let sequence = state.next_sequence;
        state.next_sequence = state.next_sequence.saturating_add(1);
        state.queue.push_back(PipelineEntry {
            sequence,
            target_count,
            submitted_at: Instant::now(),
            pending,
        });
        state.metrics.submitted_batches += 1;
        state.metrics.submitted_targets += target_count as u64;
        state.metrics.outstanding_batches += 1;
        state.metrics.outstanding_targets += target_count;
        state.metrics.peak_outstanding_batches = state
            .metrics
            .peak_outstanding_batches
            .max(state.metrics.outstanding_batches);
        state.metrics.peak_outstanding_targets = state
            .metrics
            .peak_outstanding_targets
            .max(state.metrics.outstanding_targets);
        state.metrics.producer_wait_ns = state
            .metrics
            .producer_wait_ns
            .saturating_add(producer_wait_ns);
        drop(state);
        self.shared.item_available.notify_one();
        Ok(sequence)
    }

    /// Return the next submitted batch. After `close`, this returns `None`
    /// once all previously submitted batches have been drained.
    pub fn take(&self) -> Option<PipelineBatch> {
        let _take_guard = self
            .shared
            .take_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            loop {
                if let Some(entry) = state.queue.pop_front() {
                    break entry;
                }
                if !state.accepting {
                    return None;
                }
                state = self
                    .shared
                    .item_available
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        };

        let result = entry.pending.wait();
        let residence_ns = elapsed_ns(entry.submitted_at);
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.metrics.delivered_batches += 1;
        state.metrics.delivered_targets += entry.target_count as u64;
        state.metrics.outstanding_batches = state.metrics.outstanding_batches.saturating_sub(1);
        state.metrics.outstanding_targets = state
            .metrics
            .outstanding_targets
            .saturating_sub(entry.target_count);
        drop(state);
        self.shared.capacity_changed.notify_all();
        Some(PipelineBatch {
            sequence: entry.sequence,
            target_count: entry.target_count,
            residence_ns,
            result,
        })
    }

    pub fn close(&self) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.accepting = false;
        drop(state);
        self.shared.capacity_changed.notify_all();
        self.shared.item_available.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        !self
            .shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .accepting
    }

    pub fn metrics_snapshot(&self) -> PipelineMetrics {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .metrics
            .clone()
    }

    fn has_capacity(&self, state: &PipelineState, target_count: usize) -> bool {
        state.metrics.outstanding_batches < self.shared.config.max_outstanding_batches
            && self
                .shared
                .config
                .max_outstanding_targets
                .is_none_or(|capacity| {
                    state
                        .metrics
                        .outstanding_targets
                        .saturating_add(target_count)
                        <= capacity
                })
    }
}

#[derive(Debug, Clone, Default)]
pub struct SessionMetrics {
    pub completed_windows: u64,
    pub resident_path_windows: u64,
    pub stateless_path_windows: u64,
    pub other_path_windows: u64,
    pub logical_targets: u64,
    pub physical_ranges: u64,
    pub client_requests: u64,
    pub useful_bytes: u64,
    pub fetched_bytes: u64,
    pub submitted_access_units: u64,
    pub decoded_access_units: u64,
    pub decode_groups: u64,
    pub resident_encoded_hits: u64,
    pub resident_encoded_misses: u64,
    pub resident_encoded_evictions: u64,
    pub resident_cursor_hits: u64,
    pub resident_cursor_misses: u64,
    pub decoder_state_resets: u64,
    pub plan_ns: u64,
    pub fetch_ns: u64,
    pub assemble_ns: u64,
    pub decode_ns: u64,
    pub range_ttfb_ns_sum: u64,
    pub range_service_ns_sum: u64,
    pub range_timing_samples: u64,
    pub range_max_in_flight: u64,
}

#[cfg(feature = "experiment-controls")]
#[derive(Debug, Clone)]
pub struct SessionRangeTraceEvent {
    pub window_id: u64,
    pub range: RangeTraceEvent,
}

#[cfg(feature = "experiment-controls")]
#[derive(Debug, Clone)]
pub struct SessionDecodeTraceEvent {
    pub window_id: u64,
    pub decode: DecodeTraceEvent,
}

#[cfg(feature = "experiment-controls")]
pub struct SessionExecutionTrace {
    pub range_events: Vec<SessionRangeTraceEvent>,
    pub expected_requests: usize,
    pub dropped_range_events: usize,
    pub decode_events: Vec<SessionDecodeTraceEvent>,
    pub expected_submitted_aus: usize,
    pub dropped_decode_events: usize,
}

#[cfg(feature = "experiment-controls")]
#[derive(Default)]
struct ExecutionTraceBuffer {
    range_events: Vec<SessionRangeTraceEvent>,
    decode_events: Vec<SessionDecodeTraceEvent>,
    max_range_events: usize,
    max_decode_events: usize,
    expected_requests: usize,
    expected_submitted_aus: usize,
    dropped_range_events: usize,
    dropped_decode_events: usize,
}

#[cfg(feature = "experiment-controls")]
struct ExecutionTraceState {
    enabled: Arc<AtomicBool>,
    next_window_id: AtomicU64,
    buffer: Mutex<ExecutionTraceBuffer>,
}

#[cfg(feature = "experiment-controls")]
impl ExecutionTraceState {
    fn new() -> Self {
        Self {
            enabled: Arc::new(AtomicBool::new(false)),
            next_window_id: AtomicU64::new(0),
            buffer: Mutex::new(ExecutionTraceBuffer::default()),
        }
    }

    fn observe(&self, window: &WindowOutput) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let window_id = self.next_window_id.fetch_add(1, Ordering::Relaxed);
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        buffer.expected_requests += window.stats.physical_ranges;
        buffer.expected_submitted_aus += window.stats.submitted_access_units;
        for event in &window.range_events {
            if buffer.range_events.len() < buffer.max_range_events {
                buffer.range_events.push(SessionRangeTraceEvent {
                    window_id,
                    range: event.clone(),
                });
            } else {
                buffer.dropped_range_events += 1;
            }
        }
        for event in &window.decode_events {
            if buffer.decode_events.len() < buffer.max_decode_events {
                buffer.decode_events.push(SessionDecodeTraceEvent {
                    window_id,
                    decode: event.clone(),
                });
            } else {
                buffer.dropped_decode_events += 1;
            }
        }
    }

    fn start(&self, max_range_events: usize, max_decode_events: usize) -> Result<(), String> {
        if max_range_events == 0 || max_decode_events == 0 {
            return Err("execution trace capacities must be positive".to_string());
        }
        self.enabled.store(false, Ordering::Relaxed);
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *buffer = ExecutionTraceBuffer {
            max_range_events,
            max_decode_events,
            ..Default::default()
        };
        self.enabled.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn take(&self) -> SessionExecutionTrace {
        self.enabled.store(false, Ordering::Relaxed);
        self.drain()
    }

    fn drain(&self) -> SessionExecutionTrace {
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        SessionExecutionTrace {
            range_events: std::mem::take(&mut buffer.range_events),
            expected_requests: std::mem::take(&mut buffer.expected_requests),
            dropped_range_events: std::mem::take(&mut buffer.dropped_range_events),
            decode_events: std::mem::take(&mut buffer.decode_events),
            expected_submitted_aus: std::mem::take(&mut buffer.expected_submitted_aus),
            dropped_decode_events: std::mem::take(&mut buffer.dropped_decode_events),
        }
    }
}

impl SessionMetrics {
    fn observe(&mut self, stats: &ExecutionStats) {
        self.completed_windows += 1;
        match stats.mode {
            "global_window_adaptive" => self.resident_path_windows += 1,
            "window_closure_session"
            | "window_closure_session_fixed16_fallback"
            | "window_region_session"
            | "window_session" => self.stateless_path_windows += 1,
            _ => self.other_path_windows += 1,
        }
        self.logical_targets += stats.logical_targets as u64;
        self.physical_ranges += stats.physical_ranges as u64;
        self.client_requests += stats.client_requests as u64;
        self.useful_bytes += stats.useful_bytes;
        self.fetched_bytes += stats.fetched_bytes;
        self.submitted_access_units += stats.submitted_access_units as u64;
        self.decoded_access_units += stats.decoded_access_units as u64;
        self.decode_groups += stats.decode_groups as u64;
        self.resident_encoded_hits += stats.encoded_cache_hits as u64;
        self.resident_encoded_misses += stats.encoded_cache_misses as u64;
        self.resident_encoded_evictions += stats.encoded_cache_evictions;
        self.resident_cursor_hits += stats.resident_cursor_hits as u64;
        self.resident_cursor_misses += stats.resident_cursor_misses as u64;
        self.decoder_state_resets += stats.decoder_state_resets as u64;
        self.plan_ns += stats.plan_ns;
        self.fetch_ns += stats.fetch_ns;
        self.assemble_ns += stats.assemble_ns;
        self.decode_ns += stats.decode_ns;
        self.range_ttfb_ns_sum += stats.range_ttfb_ns_sum;
        self.range_service_ns_sum += stats.range_service_ns_sum;
        self.range_timing_samples += stats.range_timing_samples as u64;
        self.range_max_in_flight = self
            .range_max_in_flight
            .max(stats.range_max_in_flight as u64);
    }
}

/// Backend-neutral runtime configuration for one process-wide session.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub cost_model: CostModel,
    pub runtime_feedback: RuntimeFeedbackConfig,
    pub max_callers: usize,
    /// Maximum logical API calls waiting for admission. This is process-wide
    /// backpressure, not a worker count.
    pub max_pending_calls: usize,
    /// Maximum immutable execution windows that may overlap. This bounds
    /// pipeline state; it is independent of workload names or access modes.
    pub max_inflight_windows: usize,
    pub admission_quiet: Duration,
    pub max_merge_gap_bytes: Option<u64>,
    pub max_range_bytes: Option<u64>,
    /// Threads used by one libavcodec operation.
    pub decoder_threads: usize,
    /// Process-wide decoder-thread budget shared by all decode jobs and live cursors.
    pub global_decode_threads: usize,
    /// Decoder threads reserved while advancing one live cursor.
    pub cursor_decoder_threads: usize,
    pub resident_state: ResidentStateBudget,
}

impl SessionConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.cost_model.validate()?;
        self.runtime_feedback.validate()?;
        if self.max_callers == 0 {
            return Err("VClasp session requires at least one admitted caller".to_string());
        }
        if self.max_pending_calls < self.max_callers {
            return Err("pending-call capacity must be at least the admission width".to_string());
        }
        if self.max_inflight_windows == 0 {
            return Err("in-flight window capacity must be positive".to_string());
        }
        if self.admission_quiet > Duration::from_secs(1) {
            return Err("session admission quiet window must not exceed one second".to_string());
        }
        if self.decoder_threads == 0 {
            return Err("decoder threads must be positive".to_string());
        }
        if self.global_decode_threads == 0 {
            return Err("global decoder-thread budget must be positive".to_string());
        }
        if self.cursor_decoder_threads == 0
            || self.cursor_decoder_threads > self.global_decode_threads
        {
            return Err(format!(
                "cursor decoder threads must be within 1..={}",
                self.global_decode_threads
            ));
        }
        if self.decoder_threads > self.global_decode_threads {
            return Err(
                "decoder threads per operation exceed the global decoder-thread budget".to_string(),
            );
        }
        if self.max_range_bytes == Some(0) {
            return Err("maximum range bytes must be positive when set".to_string());
        }
        Ok(())
    }
}

/// Connection parameters for an S3-compatible object store.
#[derive(Debug, Clone)]
pub struct S3Config {
    pub endpoint: String,
    pub bucket: String,
    pub object_key: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub region: String,
    pub max_concurrency: usize,
}

/// Connection parameters for the AIStore multi-range transport.
#[derive(Debug, Clone)]
pub struct AIStoreConfig {
    pub endpoint: String,
    pub bucket: String,
    pub object_key: String,
    pub provider: String,
}

struct ChunkMetadata {
    catalog: HierarchicalCatalog,
    codec_config: Vec<u8>,
    payload_start: usize,
}

enum Command {
    Execute {
        targets: Vec<Target>,
        response: SyncSender<BatchResult>,
    },
    #[cfg(feature = "experiment-controls")]
    ExecuteForced {
        targets: Vec<Target>,
        action: HierarchicalAction,
        response: SyncSender<BatchResult>,
    },
    ExecuteWindow {
        batches: Vec<Vec<Target>>,
        response: SyncSender<WindowResult>,
    },
    Barrier {
        response: SyncSender<Result<(), String>>,
    },
    Shutdown,
}

/// Process-wide VClasp control plane.
///
/// Every concurrent caller enters this coordinator before physical planning.
/// The owned executor is the single authority for the catalog, cost feedback,
/// encoded AUs, live decoder/DPB state, and global execution budgets.
struct SessionInner {
    sender: SyncSender<Command>,
    thread: Mutex<Option<JoinHandle<()>>>,
    metrics: Arc<Mutex<SessionMetrics>>,
    #[cfg(feature = "experiment-controls")]
    execution_trace: Arc<ExecutionTraceState>,
    #[cfg(feature = "experiment-controls")]
    decode_budgets: Vec<Arc<DecodeBudget>>,
    #[cfg(feature = "experiment-controls")]
    backends: Vec<Arc<dyn crate::backend::StorageBackend>>,
    #[cfg(feature = "experiment-controls")]
    cache_observations: Vec<crate::planner::CacheObservation>,
    #[cfg(feature = "experiment-controls")]
    cursor_observations: Vec<crate::hierarchical_scheduler::CursorObservation>,
}

#[derive(Clone)]
pub struct VClaspSession {
    inner: Arc<SessionInner>,
}

impl VClaspSession {
    pub(crate) fn new(
        executors: Vec<HierarchicalBatchExecutor>,
        max_callers: usize,
        max_pending_calls: usize,
        admission_quiet: Duration,
    ) -> Result<Self, String> {
        if max_callers == 0 {
            return Err("VClasp session requires at least one admitted caller".to_string());
        }
        if admission_quiet > Duration::from_secs(1) {
            return Err("session admission quiet window must not exceed one second".to_string());
        }
        if max_pending_calls < max_callers {
            return Err("pending-call capacity must be at least the admission width".to_string());
        }
        if executors.is_empty() {
            return Err("VClasp session requires at least one execution lane".to_string());
        }
        #[cfg(feature = "experiment-controls")]
        let execution_trace = Arc::new(ExecutionTraceState::new());
        #[cfg(feature = "experiment-controls")]
        let decode_budgets = {
            let mut handles = Vec::new();
            for executor in &executors {
                let handle = executor.decode_budget_handle();
                if !handles.iter().any(|other| Arc::ptr_eq(other, &handle)) {
                    handles.push(handle);
                }
            }
            handles
        };
        #[cfg(feature = "experiment-controls")]
        let backends = {
            let mut handles = Vec::new();
            for executor in &executors {
                let handle = executor.backend_handle();
                if !handles.iter().any(|other| Arc::ptr_eq(other, &handle)) {
                    handles.push(handle);
                }
            }
            handles
        };
        #[cfg(feature = "experiment-controls")]
        let mut cache_observations = Vec::new();
        #[cfg(feature = "experiment-controls")]
        let mut cursor_observations = Vec::new();
        #[cfg(feature = "experiment-controls")]
        let executors = executors
            .into_iter()
            .map(|mut executor| {
                cache_observations.push(executor.cache_observation());
                cursor_observations.push(executor.cursor_observation());
                executor.with_execution_trace_enabled(Arc::clone(&execution_trace.enabled))
            })
            .collect();
        let (sender, receiver) = mpsc::sync_channel(max_pending_calls);
        let metrics = Arc::new(Mutex::new(SessionMetrics::default()));
        let coordinator_metrics = Arc::clone(&metrics);
        #[cfg(feature = "experiment-controls")]
        let coordinator_execution_trace = Arc::clone(&execution_trace);
        let thread = thread::spawn(move || {
            run_admission(
                executors,
                receiver,
                max_callers,
                admission_quiet,
                coordinator_metrics,
                #[cfg(feature = "experiment-controls")]
                coordinator_execution_trace,
            )
        });
        Ok(Self {
            inner: Arc::new(SessionInner {
                sender,
                thread: Mutex::new(Some(thread)),
                metrics,
                #[cfg(feature = "experiment-controls")]
                execution_trace,
                #[cfg(feature = "experiment-controls")]
                decode_budgets,
                #[cfg(feature = "experiment-controls")]
                backends,
                #[cfg(feature = "experiment-controls")]
                cache_observations,
                #[cfg(feature = "experiment-controls")]
                cursor_observations,
            }),
        })
    }

    /// Opens a local mmap-backed VClasp chunk.
    pub fn open_local(path: impl AsRef<Path>, config: SessionConfig) -> Result<Self, String> {
        config.validate()?;
        let mut reader = ChunkReader::open(path.as_ref()).map_err(|error| error.to_string())?;
        let metadata = read_chunk_metadata(&mut reader)?;
        // SAFETY: the backend owns this read-only mapping and callers must keep
        // registered chunks immutable for the session lifetime.
        let mmap = unsafe { Mmap::map(&reader.file) }.map_err(|error| error.to_string())?;
        let backend = LocalBackend::new(mmap, metadata.payload_start);
        Self::from_parts(metadata, Box::new(backend), config)
    }

    /// Opens one VClasp object through an S3-compatible transport. The local
    /// chunk supplies the immutable header and index; payload spans are read
    /// from `storage.object_key`.
    pub fn open_s3(
        local_chunk_path: impl AsRef<Path>,
        storage: S3Config,
        config: SessionConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        if storage.max_concurrency == 0 {
            return Err("S3 I/O concurrency must be positive".to_string());
        }
        if config.cost_model.io_concurrency != storage.max_concurrency {
            return Err(format!(
                "cost-model I/O concurrency {} differs from the S3 execution budget {}",
                config.cost_model.io_concurrency, storage.max_concurrency
            ));
        }
        let mut reader =
            ChunkReader::open(local_chunk_path.as_ref()).map_err(|error| error.to_string())?;
        let local_chunk_bytes = reader
            .file
            .metadata()
            .map_err(|error| error.to_string())?
            .len();
        let metadata = read_chunk_metadata(&mut reader)?;
        let client = S3ObjectStoreClient::new(
            storage.endpoint,
            storage.bucket,
            storage.access_key_id,
            storage.secret_access_key,
            storage.region,
            storage.max_concurrency,
        )
        .map_err(|error| error.to_string())?;
        let remote = client
            .head_object(&storage.object_key)
            .map_err(|error| error.to_string())?;
        validate_registered_object_size(&storage.object_key, local_chunk_bytes, remote.size)?;
        let calibrated_wave_ns = config
            .cost_model
            .wave_request_overhead_ns
            .iter()
            .copied()
            .fold(config.cost_model.request_latency_ns, f64::max)
            .clamp(0.0, u64::MAX as f64) as u64;
        let backend =
            S3Backend::from_shared_client(client, storage.object_key, metadata.payload_start)
                .and_then(|backend| {
                    backend.with_global_range_broker(GlobalRangeBrokerConfig {
                        expected_concurrent_callers: config.max_inflight_windows,
                        collection_delay: config.admission_quiet,
                        saturation_collection_budget: Duration::from_nanos(calibrated_wave_ns),
                        merge_gap_bytes: config.max_merge_gap_bytes.unwrap_or(0),
                        max_range_bytes: config.max_range_bytes.unwrap_or(u64::MAX),
                    })
                })
                .map_err(|error| error.to_string())?;
        Self::from_parts(metadata, Box::new(backend), config)
    }

    /// Opens one VClasp object through AIStore's multi-range transport.
    pub fn open_aistore(
        local_chunk_path: impl AsRef<Path>,
        storage: AIStoreConfig,
        config: SessionConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let mut reader =
            ChunkReader::open(local_chunk_path.as_ref()).map_err(|error| error.to_string())?;
        let metadata = read_chunk_metadata(&mut reader)?;
        let backend = AIStoreGetBatchBackend::new(
            storage.endpoint,
            storage.bucket,
            storage.object_key,
            storage.provider,
            metadata.payload_start,
        )
        .map_err(|error| error.to_string())?;
        Self::from_parts(metadata, Box::new(backend), config)
    }

    fn from_parts(
        metadata: ChunkMetadata,
        backend: Box<dyn crate::backend::StorageBackend>,
        config: SessionConfig,
    ) -> Result<Self, String> {
        let layout = Arc::new(metadata.catalog.to_layout_index()?);
        let catalog = Arc::new(metadata.catalog);
        let decode_budget = DecodeBudget::new(config.global_decode_threads);
        let total_decoder_slots = config.global_decode_threads / config.decoder_threads;
        let lane_count = config
            .max_inflight_windows
            .min(config.max_callers)
            .min(total_decoder_slots)
            .max(1);
        let backend: Arc<dyn crate::backend::StorageBackend> = Arc::from(backend);
        let decoder_slots = crate::decoder::shared_decoder_slots_with_threads(
            total_decoder_slots,
            config.decoder_threads,
        );
        let next_decoder_slot = Arc::new(AtomicUsize::new(0));
        let mut executors = Vec::with_capacity(lane_count);
        let mut shared_feedback = None;
        for lane in 0..lane_count {
            let resident_budget = ResidentStateBudget {
                encoded_bytes: config.resident_state.encoded_bytes / lane_count
                    + usize::from(lane < config.resident_state.encoded_bytes % lane_count),
                read_ahead_bytes: config.resident_state.read_ahead_bytes / lane_count
                    + usize::from(lane < config.resident_state.read_ahead_bytes % lane_count),
                live_cursors: config.resident_state.live_cursors / lane_count
                    + usize::from(lane < config.resident_state.live_cursors % lane_count),
            };
            let mut executor = HierarchicalBatchExecutor::new_with_shared_backend(
                Arc::clone(&catalog),
                Arc::clone(&layout),
                Arc::clone(&backend),
                metadata.codec_config.clone(),
                config.cost_model.clone(),
                config.max_merge_gap_bytes,
                config.max_range_bytes,
                config.decoder_threads,
                1,
                false,
                resident_budget,
            )?
            .with_shared_decode_resources(
                Arc::clone(&decoder_slots),
                Arc::clone(&decode_budget),
                Arc::clone(&next_decoder_slot),
                config.cursor_decoder_threads,
            )?;
            if let Some(feedback) = &shared_feedback {
                executor = executor.with_shared_runtime_feedback(Arc::clone(feedback));
            } else {
                executor =
                    executor.with_runtime_feedback_config(config.runtime_feedback.clone())?;
                shared_feedback = Some(executor.runtime_feedback_handle());
            }
            executors.push(executor);
        }
        Self::new(
            executors,
            config.max_callers,
            config.max_pending_calls,
            config.admission_quiet,
        )
    }

    pub fn execute(&self, targets: Vec<Target>) -> BatchResult {
        self.submit(targets)?.wait()
    }

    pub fn submit(&self, targets: Vec<Target>) -> Result<PendingBatch, String> {
        if targets.is_empty() {
            return Err("VClasp batch requires at least one target".to_string());
        }
        let (response, receiver) = mpsc::sync_channel(1);
        self.inner
            .sender
            .send(Command::Execute { targets, response })
            .map_err(|_| "VClasp session coordinator stopped".to_string())?;
        Ok(PendingBatch { receiver })
    }

    pub fn execute_window(&self, batches: Vec<Vec<Target>>) -> WindowResult {
        self.submit_window(batches)?.wait()
    }

    pub fn submit_window(&self, batches: Vec<Vec<Target>>) -> Result<PendingWindow, String> {
        if batches.is_empty() || batches.iter().any(Vec::is_empty) {
            return Err("VClasp request window requires non-empty batches".to_string());
        }
        let (response, receiver) = mpsc::sync_channel(1);
        self.inner
            .sender
            .send(Command::ExecuteWindow { batches, response })
            .map_err(|_| "VClasp session coordinator stopped".to_string())?;
        Ok(PendingWindow { receiver })
    }

    /// Wait for previously admitted work to finish, including window accounting.
    /// Stop concurrent producers before using this as a measurement boundary.
    pub fn synchronize(&self) -> Result<(), String> {
        let (response, receiver) = mpsc::sync_channel(1);
        self.inner
            .sender
            .send(Command::Barrier { response })
            .map_err(|_| "VClasp session coordinator stopped".to_string())?;
        receiver
            .recv()
            .map_err(|_| "VClasp session dropped barrier response".to_string())?
    }

    pub fn metrics_snapshot(&self) -> SessionMetrics {
        self.inner
            .metrics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    #[cfg(feature = "experiment-controls")]
    pub fn start_execution_trace(
        &self,
        max_range_events: usize,
        max_decode_events: usize,
    ) -> Result<(), String> {
        self.inner
            .execution_trace
            .start(max_range_events, max_decode_events)
    }

    /// Call only after all submitted work has completed; both traces are bounded.
    #[cfg(feature = "experiment-controls")]
    pub fn take_execution_trace(&self) -> SessionExecutionTrace {
        self.inner.execution_trace.take()
    }

    /// Drain complete observed windows without disabling capture.
    #[cfg(feature = "experiment-controls")]
    pub fn drain_execution_trace(&self) -> SessionExecutionTrace {
        self.inner.execution_trace.drain()
    }

    #[cfg(feature = "experiment-controls")]
    pub(crate) fn cursor_activity(&self) -> Vec<crate::hierarchical_scheduler::CursorActivity> {
        self.inner
            .cursor_observations
            .iter()
            .map(|observation| {
                observation
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
            })
            .collect()
    }

    #[cfg(feature = "experiment-controls")]
    pub(crate) fn cache_activity(&self) -> Vec<crate::planner::CacheObservationState> {
        self.inner
            .cache_observations
            .iter()
            .map(|observation| {
                observation
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone()
            })
            .collect()
    }

    #[cfg(feature = "experiment-controls")]
    pub(crate) fn decode_activity(&self) -> Vec<crate::decoder::DecodeBudgetState> {
        self.inner
            .decode_budgets
            .iter()
            .map(|budget| budget.snapshot())
            .collect()
    }

    #[cfg(feature = "experiment-controls")]
    pub(crate) fn io_activity(&self) -> Vec<Option<crate::backend::ObjectStorePressure>> {
        self.inner
            .backends
            .iter()
            .map(|backend| backend.object_store_pressure())
            .collect()
    }

    #[cfg(feature = "experiment-controls")]
    pub(crate) fn execute_forced(
        &self,
        targets: Vec<Target>,
        action: HierarchicalAction,
    ) -> BatchResult {
        let (response, receiver) = mpsc::sync_channel(1);
        self.inner
            .sender
            .send(Command::ExecuteForced {
                targets,
                action,
                response,
            })
            .map_err(|_| "VClasp session coordinator stopped".to_string())?;
        receiver
            .recv()
            .map_err(|_| "VClasp session dropped forced response".to_string())?
    }
}

fn read_chunk_metadata(reader: &mut ChunkReader) -> Result<ChunkMetadata, String> {
    let catalog = HierarchicalCatalog::from_parquet(&reader.mmap[reader.layout.index_start..])
        .map_err(|error| error.to_string())?;
    let codec_config = reader.read_sps_pps().map_err(|error| error.to_string())?;
    Ok(ChunkMetadata {
        catalog,
        codec_config,
        payload_start: reader.layout.payload_start,
    })
}

fn validate_registered_object_size(
    object_key: &str,
    local_chunk_bytes: u64,
    remote_object_bytes: u64,
) -> Result<(), String> {
    if remote_object_bytes != local_chunk_bytes {
        return Err(format!(
            "S3 object {object_key} has {remote_object_bytes} bytes but the registered local chunk has {local_chunk_bytes} bytes"
        ));
    }
    Ok(())
}

impl PipelineSubmitter for VClaspSession {
    fn submit_batch(&self, targets: Vec<Target>) -> Result<PendingBatch, String> {
        self.submit(targets)
    }
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        let _ = self.sender.send(Command::Shutdown);
        if let Ok(mut thread) = self.thread.lock() {
            if let Some(thread) = thread.take() {
                let _ = thread.join();
            }
        }
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

enum SessionSubmission {
    Batch {
        targets: Vec<Target>,
        response: SyncSender<BatchResult>,
    },
    Window {
        batches: Vec<Vec<Target>>,
        response: SyncSender<WindowResult>,
    },
}

impl SessionSubmission {
    fn append_batches(&self, output: &mut Vec<Vec<Target>>) {
        match self {
            Self::Batch { targets, .. } => output.push(targets.clone()),
            Self::Window { batches, .. } => output.extend(batches.iter().cloned()),
        }
    }

    fn logical_target_count(&self) -> usize {
        match self {
            Self::Batch { targets, .. } => targets.len(),
            Self::Window { batches, .. } => batches.iter().map(Vec::len).sum(),
        }
    }

    fn send_error(self, error: String) {
        match self {
            Self::Batch { response, .. } => {
                let _ = response.send(Err(error));
            }
            Self::Window { response, .. } => {
                let _ = response.send(Err(error));
            }
        }
    }

    fn send_ready_batch(
        &self,
        outputs: Vec<FrameOutput>,
        ready_ns: u64,
        admission_ns: u64,
        joint_submissions: usize,
    ) -> Result<(), Vec<FrameOutput>> {
        let Self::Batch { response, targets } = self else {
            return Err(outputs);
        };
        let stats = ExecutionStats {
            logical_targets: targets.len(),
            mode: "global_session_pipeline_secondary",
            total_ns: ready_ns.saturating_add(admission_ns),
            session_admission_ns: admission_ns,
            session_joint_submissions: joint_submissions,
            ..Default::default()
        };
        response
            .send(Ok((outputs, stats)))
            .map_err(|error| match error.0 {
                Ok((outputs, _)) => outputs,
                Err(_) => Vec::new(),
            })
    }
}

/// The state owner drains one rolling queue. Calls arriving during physical
/// execution remain ungrouped until the current plan completes, so early
/// completions cannot fragment the next global planning window.
fn run_admission(
    executors: Vec<HierarchicalBatchExecutor>,
    receiver: Receiver<Command>,
    max_callers: usize,
    admission_quiet: Duration,
    metrics: Arc<Mutex<SessionMetrics>>,
    #[cfg(feature = "experiment-controls")] execution_trace: Arc<ExecutionTraceState>,
) {
    let mut lanes = executors
        .into_iter()
        .map(|executor| {
            ExecutionLane::spawn(
                executor,
                Arc::clone(&metrics),
                #[cfg(feature = "experiment-controls")]
                Arc::clone(&execution_trace),
            )
        })
        .collect::<Vec<_>>();
    let mut next_lane = 0usize;
    let mut deferred = VecDeque::new();
    loop {
        let command = match deferred.pop_front() {
            Some(command) => command,
            None => match receiver.recv() {
                Ok(command) => command,
                Err(_) => break,
            },
        };
        let first_submission = match command {
            Command::Execute { targets, response } => {
                SessionSubmission::Batch { targets, response }
            }
            Command::ExecuteWindow { batches, response } => {
                SessionSubmission::Window { batches, response }
            }
            Command::Barrier { response } => {
                let _ = response.send(synchronize_lanes(&lanes));
                continue;
            }
            #[cfg(feature = "experiment-controls")]
            Command::ExecuteForced {
                targets,
                action,
                response,
            } => {
                if lanes[0]
                    .sender
                    .send(LaneCommand::Forced {
                        targets,
                        action,
                        response,
                    })
                    .is_err()
                {
                    break;
                }
                continue;
            }
            Command::Shutdown => break,
        };

        let admission_started = Instant::now();
        let submissions = collect_admission_cohort(
            first_submission,
            &receiver,
            max_callers,
            admission_quiet,
            &mut deferred,
        );
        let admission_ns = admission_started.elapsed().as_nanos() as u64;
        let route_key = common_stream_key(&submissions);
        let lane_index = match route_key {
            Some(key) => stable_stream_lane(&key, lanes.len()),
            None => {
                let selected = next_lane % lanes.len();
                next_lane = next_lane.wrapping_add(1);
                selected
            }
        };
        if let Err(error) = lanes[lane_index].sender.send(LaneCommand::Execute {
            submissions,
            admission_ns,
        }) {
            if let LaneCommand::Execute { submissions, .. } = error.0 {
                for submission in submissions {
                    submission.send_error("VClasp execution lane stopped".to_string());
                }
            }
            break;
        }
    }
    for lane in &lanes {
        let _ = lane.sender.send(LaneCommand::Shutdown);
    }
    for lane in &mut lanes {
        if let Some(thread) = lane.thread.take() {
            let _ = thread.join();
        }
    }
}

fn collect_admission_cohort(
    first_submission: SessionSubmission,
    receiver: &Receiver<Command>,
    max_callers: usize,
    admission_quiet: Duration,
    deferred: &mut VecDeque<Command>,
) -> Vec<SessionSubmission> {
    let mut submissions = vec![first_submission];
    while submissions.len() < max_callers {
        match receiver.recv_timeout(admission_quiet) {
            Ok(Command::Execute { targets, response }) => {
                submissions.push(SessionSubmission::Batch { targets, response });
            }
            Ok(Command::ExecuteWindow { batches, response }) => {
                submissions.push(SessionSubmission::Window { batches, response });
            }
            Ok(other) => {
                // A barrier/control command must fence off later submissions.
                deferred.push_back(other);
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    submissions
}

fn stable_stream_lane(stream_key: &str, lane_count: usize) -> usize {
    // FNV-1a keeps routing deterministic without retaining one map entry per
    // video. A single-stream request therefore reaches the same DPB owner.
    let hash = stream_key
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
    (hash as usize) % lane_count
}

fn common_stream_key(submissions: &[SessionSubmission]) -> Option<String> {
    let mut key = None::<&str>;
    for submission in submissions {
        let batches = match submission {
            SessionSubmission::Batch { targets, .. } => std::slice::from_ref(targets),
            SessionSubmission::Window { batches, .. } => batches.as_slice(),
        };
        for batch in batches {
            for target in batch {
                match key {
                    None => key = Some(&target.video_id),
                    Some(current) if current == target.video_id => {}
                    Some(_) => return None,
                }
            }
        }
    }
    key.map(str::to_owned)
}

enum LaneCommand {
    Execute {
        submissions: Vec<SessionSubmission>,
        admission_ns: u64,
    },
    #[cfg(feature = "experiment-controls")]
    Forced {
        targets: Vec<Target>,
        action: HierarchicalAction,
        response: SyncSender<BatchResult>,
    },
    Barrier {
        response: SyncSender<()>,
    },
    Shutdown,
}

struct ExecutionLane {
    sender: mpsc::Sender<LaneCommand>,
    thread: Option<JoinHandle<()>>,
}

fn synchronize_lanes(lanes: &[ExecutionLane]) -> Result<(), String> {
    let mut acknowledgements = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let (response, receiver) = mpsc::sync_channel(1);
        lane.sender
            .send(LaneCommand::Barrier { response })
            .map_err(|_| "VClasp execution lane stopped before barrier".to_string())?;
        acknowledgements.push(receiver);
    }
    for receiver in acknowledgements {
        receiver
            .recv()
            .map_err(|_| "VClasp execution lane dropped barrier response".to_string())?;
    }
    Ok(())
}

impl ExecutionLane {
    fn spawn(
        mut executor: HierarchicalBatchExecutor,
        metrics: Arc<Mutex<SessionMetrics>>,
        #[cfg(feature = "experiment-controls")] execution_trace: Arc<ExecutionTraceState>,
    ) -> Self {
        let (sender, receiver) = mpsc::channel();
        let thread = thread::spawn(move || {
            while let Ok(command) = receiver.recv() {
                match command {
                    LaneCommand::Execute {
                        submissions,
                        admission_ns,
                    } => execute_joint(
                        &mut executor,
                        submissions,
                        admission_ns,
                        Arc::clone(&metrics),
                        #[cfg(feature = "experiment-controls")]
                        Arc::clone(&execution_trace),
                    ),
                    #[cfg(feature = "experiment-controls")]
                    LaneCommand::Forced {
                        targets,
                        action,
                        response,
                    } => {
                        let _ = response.send(executor.execute_action(&targets, action));
                    }
                    LaneCommand::Barrier { response } => {
                        let _ = response.send(());
                    }
                    LaneCommand::Shutdown => break,
                }
            }
        });
        Self {
            sender,
            thread: Some(thread),
        }
    }
}

fn execute_joint(
    executor: &mut HierarchicalBatchExecutor,
    submissions: Vec<SessionSubmission>,
    admission_ns: u64,
    metrics: Arc<Mutex<SessionMetrics>>,
    #[cfg(feature = "experiment-controls")] execution_trace: Arc<ExecutionTraceState>,
) {
    let mut batches = Vec::new();
    let mut batch_routes = Vec::new();
    for (submission_index, submission) in submissions.iter().enumerate() {
        let first_batch = batches.len();
        submission.append_batches(&mut batches);
        batch_routes.extend((first_batch..batches.len()).map(|_| submission_index));
    }
    let mut sent_early = vec![false; submissions.len()];
    let result =
        executor.execute_window_streaming(&batches, &mut |batch_index, outputs, ready_ns| {
            let submission_index = batch_routes[batch_index];
            // Concurrent one-batch callers are released at data readiness.
            // Window-wide physical accounting belongs to the session rather
            // than forcing an arbitrary caller to wait for the slowest range.
            if sent_early[submission_index] {
                return Some(outputs);
            }
            match submissions[submission_index].send_ready_batch(
                outputs,
                ready_ns,
                admission_ns,
                submissions.len(),
            ) {
                Ok(()) => {
                    sent_early[submission_index] = true;
                    None
                }
                Err(outputs) => Some(outputs),
            }
        });
    if let Ok(window) = &result {
        metrics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .observe(&window.stats);
        #[cfg(feature = "experiment-controls")]
        execution_trace.observe(window);
    }
    distribute_joint_result_with_sent(result, submissions, admission_ns, &sent_early);
}

#[cfg(test)]
fn distribute_joint_result(
    result: WindowResult,
    submissions: Vec<SessionSubmission>,
    admission_ns: u64,
) {
    let sent = vec![false; submissions.len()];
    distribute_joint_result_with_sent(result, submissions, admission_ns, &sent);
}

fn distribute_joint_result_with_sent(
    result: WindowResult,
    submissions: Vec<SessionSubmission>,
    admission_ns: u64,
    sent_early: &[bool],
) {
    match result {
        Ok(window) => {
            let mut outputs = window.batches.into_iter();
            let mut ready = window.batch_ready_ns.into_iter();
            let mut global_stats = window.stats;
            global_stats.session_admission_ns = admission_ns;
            global_stats.session_joint_submissions = submissions.len();
            global_stats.total_ns = global_stats.total_ns.saturating_add(admission_ns);
            let attributed = attribute_joint_stats(
                &global_stats,
                &submissions
                    .iter()
                    .map(SessionSubmission::logical_target_count)
                    .collect::<Vec<_>>(),
            );
            for (submission_index, (submission, stats)) in
                submissions.into_iter().zip(attributed).enumerate()
            {
                match submission {
                    SessionSubmission::Batch { response, .. } => {
                        let output = outputs.next().ok_or_else(|| {
                            "global session changed submission cardinality".to_string()
                        });
                        let _ = ready.next();
                        if sent_early.get(submission_index).copied().unwrap_or(false) {
                            continue;
                        }
                        let _ = response.send(output.map(|output| (output, stats)));
                    }
                    SessionSubmission::Window {
                        batches, response, ..
                    } => {
                        let mut result_batches = Vec::with_capacity(batches.len());
                        let mut batch_ready_ns = Vec::with_capacity(batches.len());
                        let mut error = None;
                        for _ in 0..batches.len() {
                            match (outputs.next(), ready.next()) {
                                (Some(batch), Some(ready_ns)) => {
                                    result_batches.push(batch);
                                    batch_ready_ns.push(ready_ns);
                                }
                                _ => {
                                    error = Some(
                                        "global session changed window cardinality".to_string(),
                                    );
                                    break;
                                }
                            }
                        }
                        if let Some(error) = error {
                            let _ = response.send(Err(error));
                            continue;
                        }
                        let mut ordered_delivery_ns = Vec::with_capacity(batch_ready_ns.len());
                        let mut previous = 0;
                        for ready_ns in &batch_ready_ns {
                            previous = previous.max(*ready_ns);
                            ordered_delivery_ns.push(previous);
                        }
                        let _ = response.send(Ok(WindowOutput {
                            batches: result_batches,
                            batch_ready_ns,
                            ordered_delivery_ns,
                            stats,
                            #[cfg(feature = "experiment-controls")]
                            range_events: Vec::new(),
                            #[cfg(feature = "experiment-controls")]
                            decode_events: Vec::new(),
                        }));
                    }
                }
            }
            debug_assert!(outputs.next().is_none());
            debug_assert!(ready.next().is_none());
        }
        Err(error) => {
            for submission in submissions {
                submission.send_error(error.clone());
            }
        }
    }
}

fn attribute_joint_stats(global: &ExecutionStats, logical_counts: &[usize]) -> Vec<ExecutionStats> {
    logical_counts
        .iter()
        .copied()
        .enumerate()
        .map(|(index, logical_targets)| {
            if index == 0 {
                let mut stats = global.clone();
                stats.logical_targets = logical_targets;
                stats
            } else {
                ExecutionStats {
                    logical_targets,
                    mode: "global_session_joint_secondary",
                    total_ns: global.total_ns,
                    predicted_total_ns: global.predicted_total_ns,
                    session_admission_ns: global.session_admission_ns,
                    session_joint_submissions: global.session_joint_submissions,
                    ..Default::default()
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn session_counts_observed_window_paths_without_inference_from_hits() {
        let mut metrics = super::SessionMetrics::default();
        for mode in [
            "global_window_adaptive",
            "window_closure_session",
            "unknown",
        ] {
            metrics.observe(&super::ExecutionStats {
                mode,
                ..Default::default()
            });
        }
        assert_eq!(metrics.resident_path_windows, 1);
        assert_eq!(metrics.stateless_path_windows, 1);
        assert_eq!(metrics.other_path_windows, 1);
        assert_eq!(metrics.completed_windows, 3);
        assert_eq!(metrics.resident_cursor_hits, 0);
    }
    #[test]
    fn session_cache_counters_follow_physical_windows() {
        let mut metrics = super::SessionMetrics::default();
        for (hits, misses, targets) in [(0, 7, 32), (5, 2, 64)] {
            metrics.observe(&super::ExecutionStats {
                encoded_cache_hits: hits,
                encoded_cache_misses: misses,
                logical_targets: targets,
                ..Default::default()
            });
        }
        assert_eq!(metrics.completed_windows, 2);
        assert_eq!(metrics.logical_targets, 96);
        assert_eq!(metrics.resident_encoded_hits, 5);
        assert_eq!(metrics.resident_encoded_misses, 9);
    }
    #[cfg(feature = "experiment-controls")]
    use super::ExecutionTraceState;
    use super::{
        attribute_joint_stats, common_stream_key, distribute_joint_result,
        distribute_joint_result_with_sent, stable_stream_lane, validate_registered_object_size,
        BatchResult, ExecutionStats, FrameOutput, PendingBatch, PipelineConfig, PipelineSubmitter,
        SessionSubmission, Target, VClaspPipeline, WindowOutput,
    };
    use super::{collect_admission_cohort, Command};
    use super::{synchronize_lanes, ExecutionLane, LaneCommand};
    use crate::decoder::DecodedRgbFrame;
    #[cfg(feature = "experiment-controls")]
    use crate::hierarchical_scheduler::{DecodeTraceEvent, RangeTraceEvent};
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn barrier_waits_for_accounting_on_every_lane() {
        let (arrived, arrivals) = mpsc::channel();
        let (completed, completions) = mpsc::channel();
        let mut releases = Vec::new();
        let mut lanes = Vec::new();
        let mut workers = Vec::new();
        for index in 0..2 {
            let (sender, receiver) = mpsc::channel();
            let (release, gate) = mpsc::channel();
            let arrived = arrived.clone();
            let completed = completed.clone();
            workers.push(std::thread::spawn(move || {
                let LaneCommand::Barrier { response } = receiver.recv().unwrap() else {
                    panic!("expected a barrier after early frame delivery");
                };
                arrived.send(index).unwrap();
                // The frames are already returned, but window accounting is pending.
                gate.recv_timeout(Duration::from_secs(5)).unwrap();
                response.send(()).unwrap();
                completed.send(index).unwrap();
            }));
            releases.push(release);
            lanes.push(ExecutionLane {
                sender,
                thread: None,
            });
        }
        let (result, results) = mpsc::channel();
        let barrier = std::thread::spawn(move || {
            result.send(synchronize_lanes(&lanes)).unwrap();
        });
        for _ in 0..2 {
            arrivals.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        assert!(matches!(results.try_recv(), Err(mpsc::TryRecvError::Empty)));
        releases[0].send(()).unwrap();
        assert_eq!(completions.recv_timeout(Duration::from_secs(5)).unwrap(), 0);
        assert!(matches!(results.try_recv(), Err(mpsc::TryRecvError::Empty)));
        releases[1].send(()).unwrap();
        results
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
        barrier.join().unwrap();
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn barrier_reports_a_stopped_lane() {
        let (sender, receiver) = mpsc::channel();
        drop(receiver);
        let lane = ExecutionLane {
            sender,
            thread: None,
        };
        assert!(synchronize_lanes(&[lane]).is_err());
    }

    #[test]
    fn admission_cohort_does_not_cross_a_barrier() {
        let (sender, receiver) = mpsc::channel();
        let (response, _) = mpsc::sync_channel(1);
        sender.send(Command::Barrier { response }).unwrap();
        let (response, _) = mpsc::sync_channel(1);
        sender
            .send(Command::Execute {
                targets: Vec::new(),
                response,
            })
            .unwrap();
        let (response, _) = mpsc::sync_channel(1);
        let mut deferred = std::collections::VecDeque::new();
        let submissions = collect_admission_cohort(
            SessionSubmission::Batch {
                targets: Vec::new(),
                response,
            },
            &receiver,
            8,
            Duration::ZERO,
            &mut deferred,
        );
        assert_eq!(submissions.len(), 1);
        assert!(matches!(
            deferred.pop_front(),
            Some(Command::Barrier { .. })
        ));
        assert!(matches!(receiver.try_recv(), Ok(Command::Execute { .. })));
    }

    #[cfg(feature = "experiment-controls")]
    #[test]
    fn execution_trace_preserves_consumers_and_reports_capacity_loss() {
        let trace = ExecutionTraceState::new();
        assert!(trace.start(0, 1).is_err());
        trace.start(1, 1).unwrap();
        let event = RangeTraceEvent {
            range_index: 0,
            planned_payload_offset: 40,
            planned_length: 80,
            physical_request_id: 1,
            physical_object_offset: 100,
            physical_object_length: 80,
            started_ns: 10,
            first_byte_ns: 20,
            completed_ns: 30,
            monotonic_timing: None,
            physical_requests: 1,
            fetched_bytes: 80,
            consumer_sample_ids: vec![7, 8],
        };
        let window = WindowOutput {
            batches: Vec::new(),
            batch_ready_ns: Vec::new(),
            ordered_delivery_ns: Vec::new(),
            stats: ExecutionStats {
                physical_ranges: 2,
                submitted_access_units: 2,
                ..Default::default()
            },
            range_events: vec![
                event.clone(),
                RangeTraceEvent {
                    range_index: 1,
                    ..event
                },
            ],
            decode_events: vec![
                DecodeTraceEvent {
                    record_id: 4,
                    consumer_sample_ids: vec![7, 8],
                },
                DecodeTraceEvent {
                    record_id: 5,
                    consumer_sample_ids: vec![8],
                },
            ],
        };
        trace.observe(&window);
        let result = trace.drain();
        assert_eq!(
            (
                result.range_events.len(),
                result.expected_requests,
                result.dropped_range_events
            ),
            (1, 2, 1)
        );
        assert_eq!(
            (
                result.decode_events.len(),
                result.expected_submitted_aus,
                result.dropped_decode_events
            ),
            (1, 2, 1)
        );
        assert_eq!(result.range_events[0].range.consumer_sample_ids, vec![7, 8]);
        assert_eq!(
            result.decode_events[0].decode.consumer_sample_ids,
            vec![7, 8]
        );
        assert!(trace.enabled.load(super::Ordering::Relaxed));
        assert_eq!(trace.drain().expected_requests, 0);
        trace.observe(&window);
        let final_result = trace.take();
        assert_eq!(final_result.expected_requests, 2);
        assert_eq!(final_result.expected_submitted_aus, 2);
        assert_eq!(final_result.range_events[0].window_id, 1);
        assert!(!trace.enabled.load(super::Ordering::Relaxed));
        assert_eq!(trace.take().expected_requests, 0);
    }

    #[derive(Default)]
    struct FakeSubmitter {
        responses: Mutex<Vec<(u64, mpsc::SyncSender<BatchResult>)>>,
    }

    impl FakeSubmitter {
        fn complete(&self, sample_id: u64) {
            let response = {
                let mut responses = self.responses.lock().unwrap();
                let index = responses
                    .iter()
                    .position(|(id, _)| *id == sample_id)
                    .unwrap();
                responses.remove(index).1
            };
            response
                .send(Ok((
                    vec![output(sample_id)],
                    ExecutionStats {
                        logical_targets: 1,
                        ..Default::default()
                    },
                )))
                .unwrap();
        }
    }

    impl PipelineSubmitter for FakeSubmitter {
        fn submit_batch(&self, targets: Vec<Target>) -> Result<PendingBatch, String> {
            let (response, receiver) = mpsc::sync_channel(1);
            self.responses
                .lock()
                .unwrap()
                .push((targets[0].sample_id, response));
            Ok(PendingBatch { receiver })
        }
    }

    fn output(sample_id: u64) -> FrameOutput {
        FrameOutput {
            sample_id,
            frame: DecodedRgbFrame {
                data: vec![sample_id as u8],
                width: 1,
                height: 1,
            },
        }
    }

    fn target(sample_id: u64, video_id: &str) -> Target {
        Target {
            sample_id,
            video_id: video_id.to_string(),
            frame_idx: sample_id as i32,
        }
    }

    #[test]
    fn pipeline_delivers_submission_order_after_out_of_order_completion() {
        let submitter = Arc::new(FakeSubmitter::default());
        let pipeline = VClaspPipeline::from_submitter(
            submitter.clone(),
            PipelineConfig {
                max_outstanding_batches: 2,
                max_outstanding_targets: Some(2),
            },
        )
        .unwrap();
        assert_eq!(pipeline.submit(vec![target(10, "video-a")]).unwrap(), 0);
        assert_eq!(pipeline.submit(vec![target(20, "video-b")]).unwrap(), 1);

        submitter.complete(20);
        submitter.complete(10);
        pipeline.close();

        let first = pipeline.take().unwrap();
        let second = pipeline.take().unwrap();
        assert_eq!(first.sequence, 0);
        assert_eq!(first.result.unwrap().0[0].sample_id, 10);
        assert_eq!(second.sequence, 1);
        assert_eq!(second.result.unwrap().0[0].sample_id, 20);
        assert!(pipeline.take().is_none());
        let metrics = pipeline.metrics_snapshot();
        assert_eq!(metrics.peak_outstanding_batches, 2);
        assert_eq!(metrics.outstanding_batches, 0);
    }

    #[test]
    fn pipeline_backpressure_releases_only_after_ordered_take() {
        let submitter = Arc::new(FakeSubmitter::default());
        let pipeline = VClaspPipeline::from_submitter(
            submitter.clone(),
            PipelineConfig {
                max_outstanding_batches: 1,
                max_outstanding_targets: None,
            },
        )
        .unwrap();
        pipeline.submit(vec![target(10, "video-a")]).unwrap();

        let producer_pipeline = pipeline.clone();
        let producer =
            std::thread::spawn(move || producer_pipeline.submit(vec![target(20, "video-b")]));
        std::thread::sleep(Duration::from_millis(10));
        assert!(!producer.is_finished());

        submitter.complete(10);
        assert_eq!(pipeline.take().unwrap().sequence, 0);
        assert_eq!(producer.join().unwrap().unwrap(), 1);
        submitter.complete(20);
        pipeline.close();
        assert_eq!(pipeline.take().unwrap().sequence, 1);
        assert!(pipeline.take().is_none());
    }

    #[test]
    fn closed_pipeline_rejects_new_work_but_drains_admitted_batches() {
        let submitter = Arc::new(FakeSubmitter::default());
        let pipeline = VClaspPipeline::from_submitter(
            submitter.clone(),
            PipelineConfig {
                max_outstanding_batches: 1,
                max_outstanding_targets: Some(1),
            },
        )
        .unwrap();
        pipeline.submit(vec![target(10, "video-a")]).unwrap();
        pipeline.close();
        assert!(pipeline.submit(vec![target(20, "video-b")]).is_err());
        submitter.complete(10);
        assert!(pipeline.take().is_some());
        assert!(pipeline.take().is_none());
    }

    #[test]
    fn one_stream_routes_to_one_stable_lane_without_workload_labels() {
        let (response, _) = mpsc::sync_channel(1);
        let submissions = vec![SessionSubmission::Window {
            batches: vec![
                vec![target(1, "video-a"), target(2, "video-a")],
                vec![target(3, "video-a")],
            ],
            response,
        }];
        assert_eq!(common_stream_key(&submissions).as_deref(), Some("video-a"));
        assert_eq!(
            stable_stream_lane("video-a", 4),
            stable_stream_lane("video-a", 4)
        );
    }

    #[test]
    fn mixed_stream_window_is_not_forced_into_a_stateful_route() {
        let (response, _) = mpsc::sync_channel(1);
        let submissions = vec![SessionSubmission::Batch {
            targets: vec![target(1, "video-a"), target(2, "video-b")],
            response,
        }];
        assert_eq!(common_stream_key(&submissions), None);
    }

    #[test]
    fn physical_work_is_attributed_exactly_once() {
        let global = ExecutionStats {
            logical_targets: 96,
            physical_ranges: 17,
            fetched_bytes: 123_456,
            submitted_access_units: 44,
            decoded_access_units: 44,
            ..Default::default()
        };
        let attributed = attribute_joint_stats(&global, &[32, 32, 32]);
        assert_eq!(
            attributed
                .iter()
                .map(|stats| stats.logical_targets)
                .sum::<usize>(),
            96
        );
        assert_eq!(
            attributed
                .iter()
                .map(|stats| stats.physical_ranges)
                .sum::<usize>(),
            17
        );
        assert_eq!(
            attributed
                .iter()
                .map(|stats| stats.fetched_bytes)
                .sum::<u64>(),
            123_456
        );
    }

    #[test]
    fn mixed_batch_and_window_submissions_restore_each_api_boundary() {
        let (batch_response, batch_receiver) = mpsc::sync_channel(1);
        let (window_response, window_receiver) = mpsc::sync_channel(1);
        let submissions = vec![
            SessionSubmission::Batch {
                targets: Vec::new(),
                response: batch_response,
            },
            SessionSubmission::Window {
                batches: vec![Vec::new(), Vec::new()],
                response: window_response,
            },
        ];
        let stats = ExecutionStats {
            logical_targets: 3,
            physical_ranges: 7,
            fetched_bytes: 4096,
            ..Default::default()
        };
        distribute_joint_result(
            Ok(WindowOutput {
                batches: vec![vec![output(10)], vec![output(20)], vec![output(30)]],
                batch_ready_ns: vec![10, 30, 20],
                ordered_delivery_ns: vec![10, 30, 30],
                stats,
                #[cfg(feature = "experiment-controls")]
                range_events: Vec::new(),
                #[cfg(feature = "experiment-controls")]
                decode_events: Vec::new(),
            }),
            submissions,
            5,
        );

        let (batch, batch_stats) = batch_receiver.recv().unwrap().unwrap();
        assert_eq!(batch[0].sample_id, 10);
        assert_eq!(batch_stats.physical_ranges, 7);
        assert_eq!(batch_stats.session_joint_submissions, 2);

        let window = window_receiver.recv().unwrap().unwrap();
        assert_eq!(
            window
                .batches
                .iter()
                .flatten()
                .map(|frame| frame.sample_id)
                .collect::<Vec<_>>(),
            vec![20, 30]
        );
        assert_eq!(window.batch_ready_ns, vec![30, 20]);
        assert_eq!(window.ordered_delivery_ns, vec![30, 30]);
        assert_eq!(window.stats.physical_ranges, 0);
        assert_eq!(window.stats.session_joint_submissions, 2);
        assert_eq!(window.stats.session_admission_ns, 5);
    }

    #[test]
    fn early_batch_completion_is_not_resent_or_double_counted() {
        let (owner_response, owner_receiver) = mpsc::sync_channel(1);
        let (early_response, early_receiver) = mpsc::sync_channel(1);
        let submissions = vec![
            SessionSubmission::Batch {
                targets: Vec::new(),
                response: owner_response,
            },
            SessionSubmission::Batch {
                targets: Vec::new(),
                response: early_response,
            },
        ];

        assert!(
            submissions[1]
                .send_ready_batch(vec![output(20)], 20, 5, 2)
                .is_ok(),
            "early response channel should accept the completed batch"
        );
        distribute_joint_result_with_sent(
            Ok(WindowOutput {
                batches: vec![vec![output(10)], Vec::new()],
                batch_ready_ns: vec![30, 20],
                ordered_delivery_ns: vec![30, 20],
                stats: ExecutionStats {
                    logical_targets: 2,
                    physical_ranges: 7,
                    fetched_bytes: 4096,
                    ..Default::default()
                },
                #[cfg(feature = "experiment-controls")]
                range_events: Vec::new(),
                #[cfg(feature = "experiment-controls")]
                decode_events: Vec::new(),
            }),
            submissions,
            5,
            &[false, true],
        );

        let (early, early_stats) = early_receiver.recv().unwrap().unwrap();
        assert_eq!(early[0].sample_id, 20);
        assert_eq!(early_stats.mode, "global_session_pipeline_secondary");
        assert_eq!(early_stats.physical_ranges, 0);
        assert_eq!(early_stats.fetched_bytes, 0);
        assert_eq!(early_stats.total_ns, 25);

        let (owner, owner_stats) = owner_receiver.recv().unwrap().unwrap();
        assert_eq!(owner[0].sample_id, 10);
        assert_eq!(owner_stats.physical_ranges, 7);
        assert_eq!(owner_stats.fetched_bytes, 4096);
        assert_eq!(owner_stats.session_joint_submissions, 2);
    }

    #[test]
    fn registered_s3_object_size_must_match_local_chunk() {
        assert!(validate_registered_object_size("dataset/chunk.vclasp", 4096, 4096).is_ok());

        let error =
            validate_registered_object_size("dataset/chunk.vclasp", 4096, 2048).unwrap_err();
        assert!(error.contains("dataset/chunk.vclasp"));
        assert!(error.contains("2048 bytes"));
        assert!(error.contains("4096 bytes"));
    }
}
