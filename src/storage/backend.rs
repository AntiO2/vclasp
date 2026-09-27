//! Storage backend abstraction: local mmap vs object store (MinIO/S3).
//!
//! The scheduler resolves record offsets via the local chunk index, then
//! delegates payload byte reads to a StorageBackend. This allows the same
//! scheduler to run against local SSD (zero network cost) or MinIO (real
//! Range GET latency and bytes-transferred measurement).

use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
pub use vclasp_object_store::{
    CompletedRange, ObjectRange, ObjectStorePressure, S3ObjectStoreClient,
};

#[derive(Debug)]
pub struct ProfiledRangeBatch {
    pub buffers: Vec<Vec<u8>>,
    pub physical_requests: usize,
    pub physical_fetched_bytes: u64,
    pub dispatch_ns_sum: u64,
    pub ttfb_ns_sum: u64,
    pub service_ns_sum: u64,
    pub max_in_flight: usize,
    pub timing_samples: usize,
}

/// Abstract byte-range read from a storage backend.
pub trait StorageBackend: Send + Sync {
    fn read_byte_range(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>>;

    fn for_each_byte_range(
        &self,
        ranges: &[(u64, u64)],
        callback: &mut dyn FnMut(CompletedRange) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let started = Instant::now();
        for (index, &(offset, length)) in ranges.iter().enumerate() {
            let range_started = started.elapsed().as_nanos() as u64;
            let bytes = self.read_byte_range(offset, length)?;
            callback(CompletedRange {
                index,
                physical_requests: 1,
                physical_fetched_bytes: bytes.len() as u64,
                bytes,
                started_ns: range_started,
                first_byte_ns: started.elapsed().as_nanos() as u64,
                completed_ns: started.elapsed().as_nanos() as u64,
            })?;
        }
        Ok(())
    }

    fn read_byte_ranges(
        &self,
        ranges: &[(u64, u64)],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        Ok(self.read_byte_ranges_profiled(ranges)?.buffers)
    }

    fn read_byte_ranges_profiled(
        &self,
        ranges: &[(u64, u64)],
    ) -> Result<ProfiledRangeBatch, Box<dyn std::error::Error>> {
        let mut result: Vec<Option<Vec<u8>>> =
            std::iter::repeat_with(|| None).take(ranges.len()).collect();
        let mut dispatch_ns_sum = 0u64;
        let mut ttfb_ns_sum = 0u64;
        let mut service_ns_sum = 0u64;
        let mut physical_requests = 0usize;
        let mut physical_fetched_bytes = 0u64;
        let mut intervals = Vec::with_capacity(ranges.len());
        let mut timing_samples = 0usize;
        self.for_each_byte_range(ranges, &mut |completed| {
            physical_requests += completed.physical_requests;
            physical_fetched_bytes =
                physical_fetched_bytes.saturating_add(completed.physical_fetched_bytes);
            if completed.index >= result.len() {
                return Err(format!(
                    "backend returned range index {} for {} requests",
                    completed.index,
                    result.len()
                )
                .into());
            }
            if completed.physical_requests > 0
                && completed.completed_ns > completed.started_ns
                && completed.first_byte_ns >= completed.started_ns
            {
                dispatch_ns_sum = dispatch_ns_sum.saturating_add(completed.started_ns);
                ttfb_ns_sum =
                    ttfb_ns_sum.saturating_add(completed.first_byte_ns - completed.started_ns);
                service_ns_sum =
                    service_ns_sum.saturating_add(completed.completed_ns - completed.started_ns);
                intervals.push((completed.started_ns, completed.completed_ns));
                timing_samples += 1;
            }
            result[completed.index] = Some(completed.bytes);
            Ok(())
        })?;
        let buffers = result
            .into_iter()
            .map(|value| value.ok_or_else(|| "backend omitted a requested range".into()))
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        if self.executes_planned_ranges_exactly() {
            let planned_bytes = ranges.iter().map(|(_, length)| *length).sum::<u64>();
            let declared_requests = self.physical_ranges_for_ranges(ranges);
            let declared_bytes = self
                .physical_fetched_bytes_for_ranges(ranges)
                .unwrap_or(physical_fetched_bytes);
            if physical_requests != ranges.len()
                || physical_fetched_bytes != planned_bytes
                || declared_requests != physical_requests
                || declared_bytes != physical_fetched_bytes
            {
                return Err(format!(
                    "backend changed an immutable physical plan: planned {} ranges/{} bytes, observed {} ranges/{} bytes, declared {} ranges/{} bytes",
                    ranges.len(),
                    planned_bytes,
                    physical_requests,
                    physical_fetched_bytes,
                    declared_requests,
                    declared_bytes,
                )
                .into());
            }
        }
        let mut events = intervals
            .into_iter()
            .flat_map(|(start, end)| [(start, 1i32), (end, -1i32)])
            .collect::<Vec<_>>();
        // At an equal timestamp, complete the old request before starting the
        // next one so adjacent intervals are not counted as overlapping.
        events.sort_unstable_by_key(|(timestamp, delta)| (*timestamp, *delta));
        let mut in_flight = 0i32;
        let mut max_in_flight = 0usize;
        for (_, delta) in events {
            in_flight += delta;
            max_in_flight = max_in_flight.max(in_flight.max(0) as usize);
        }
        Ok(ProfiledRangeBatch {
            buffers,
            physical_requests,
            physical_fetched_bytes,
            dispatch_ns_sum,
            ttfb_ns_sum,
            service_ns_sum,
            max_in_flight,
            timing_samples,
        })
    }

    /// Number of client-side API calls used to execute one planned range batch.
    fn client_requests_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
        ranges.len()
    }

    /// Number of range entries submitted to the storage service.
    fn server_entries_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
        ranges.len()
    }

    /// Physical requests issued after backend-native coalescing.
    fn physical_ranges_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
        ranges.len()
    }

    /// Actual bytes fetched before a vectored backend slices logical ranges.
    /// `None` means callback buffer lengths are already physical bytes.
    fn physical_fetched_bytes_for_ranges(&self, _ranges: &[(u64, u64)]) -> Option<u64> {
        None
    }

    /// True when the adapter promises to execute every submitted span exactly
    /// once without backend-side merging or expansion.
    fn executes_planned_ranges_exactly(&self) -> bool {
        false
    }

    /// Whether callbacks can arrive while other range I/O is still in flight.
    /// Backends that materialize the complete response before callbacks must
    /// keep this false; completion-driven decode would only fragment batches.
    fn supports_streaming_range_completion(&self) -> bool {
        false
    }

    /// Snapshot shared object-store pressure before planning or dispatch.
    ///
    /// Local and backend adapters without a shared request budget return
    /// `None`; callers must not invent zero pressure for those paths.
    fn object_store_pressure(&self) -> Option<ObjectStorePressure> {
        None
    }
}

// ── Local mmap backend ──────────────────────────────────────────────

/// Reads payload bytes directly from a local mmap'd chunk file.
/// Offset semantics: relative to the chunk payload start.
pub struct LocalBackend {
    mmap: memmap2::Mmap,
    payload_start: usize,
}

impl LocalBackend {
    pub fn new(mmap: memmap2::Mmap, payload_start: usize) -> Self {
        LocalBackend {
            mmap,
            payload_start,
        }
    }
}

impl StorageBackend for LocalBackend {
    fn read_byte_range(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let start = self.payload_start + offset as usize;
        let end = start + length as usize;
        if end > self.mmap.len() {
            return Err(format!("range out of bounds: offset={}, len={}", offset, length).into());
        }
        Ok(self.mmap[start..end].to_vec())
    }

    fn executes_planned_ranges_exactly(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod local_tests {
    use super::{LocalBackend, StorageBackend};
    use std::fs::{self, File};

    fn temporary_backend() -> (tempfile::TempDir, LocalBackend) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("payload.bin");
        fs::write(&path, b"HEADabcdefgh").unwrap();
        let file = File::open(path).unwrap();
        // The fixture is immutable and lives until after its mapping is dropped.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        (directory, LocalBackend::new(mmap, 4))
    }

    #[test]
    fn reads_payload_relative_ranges_from_temporary_storage() {
        let (_directory, backend) = temporary_backend();
        assert_eq!(backend.read_byte_range(2, 3).unwrap(), b"cde");
        assert_eq!(backend.read_byte_range(7, 1).unwrap(), b"h");
        assert!(backend.read_byte_range(8, 1).is_err());
    }

    #[test]
    fn temporary_storage_preserves_range_order_duplicates_and_byte_accounting() {
        let (_directory, backend) = temporary_backend();
        let result = backend
            .read_byte_ranges_profiled(&[(5, 2), (0, 2), (5, 2)])
            .unwrap();
        assert_eq!(
            result.buffers,
            vec![b"fg".to_vec(), b"ab".to_vec(), b"fg".to_vec()]
        );
        assert_eq!(result.physical_requests, 3);
        assert_eq!(result.physical_fetched_bytes, 6);
    }
}

// ── MinIO / S3 backend (via object_store) ─────────────────────────

/// Single-object view over the shared S3 transport used by VClasp layouts.
pub struct S3Backend {
    client: S3ObjectStoreClient,
    object_key: String,
    payload_start: u64,
    read_mode: S3ReadMode,
    range_broker: Option<SharedS3RangeBroker>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum S3ReadMode {
    Explicit,
    #[cfg(feature = "experiment-controls")]
    NativeVectored,
}

#[cfg(any(test, feature = "experiment-controls"))]
const OBJECT_STORE_COALESCE_DEFAULT: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct GlobalRangeBrokerConfig {
    pub expected_concurrent_callers: usize,
    pub collection_delay: Duration,
    /// Maximum time to collect the next physical wave while all I/O slots are
    /// occupied. This comes from backend calibration, not a workload label.
    pub saturation_collection_budget: Duration,
    pub merge_gap_bytes: u64,
    pub max_range_bytes: u64,
}

impl GlobalRangeBrokerConfig {
    pub fn validate(self) -> Result<Self, Box<dyn std::error::Error>> {
        if self.expected_concurrent_callers == 0 {
            return Err("range broker expected callers must be positive".into());
        }
        if self.max_range_bytes == 0 {
            return Err("range broker max range bytes must be positive".into());
        }
        Ok(self)
    }
}

#[derive(Debug)]
struct BrokerRequest {
    ranges: Vec<ObjectRange>,
    response: mpsc::Sender<Result<CompletedRange, String>>,
}

#[derive(Debug, Clone)]
struct BrokerSpanMember {
    request_index: usize,
    range_index: usize,
    offset: u64,
    length: u64,
}

#[derive(Debug)]
struct BrokerPhysicalSpan {
    object_key: String,
    offset: u64,
    end: u64,
    members: Vec<BrokerSpanMember>,
}

#[derive(Clone)]
pub struct SharedS3RangeBroker {
    sender: mpsc::Sender<BrokerRequest>,
}

impl SharedS3RangeBroker {
    pub fn new(
        client: S3ObjectStoreClient,
        config: GlobalRangeBrokerConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = config.validate()?;
        let (sender, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("vclasp-s3-range-broker".to_string())
            .spawn(move || run_s3_range_broker(client, config, receiver))?;
        Ok(Self { sender })
    }

    fn for_each(
        &self,
        ranges: Vec<ObjectRange>,
        callback: &mut dyn FnMut(CompletedRange) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if ranges.is_empty() {
            return Ok(());
        }
        let expected = ranges.len();
        let (response, receiver) = mpsc::channel();
        self.sender
            .send(BrokerRequest { ranges, response })
            .map_err(|_| "global S3 range broker stopped")?;
        for _ in 0..expected {
            let completed = receiver
                .recv()
                .map_err(|_| "global S3 range broker omitted a logical range")?
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            callback(completed)?;
        }
        Ok(())
    }
}

fn collect_broker_window(
    first: BrokerRequest,
    client: &S3ObjectStoreClient,
    receiver: &mpsc::Receiver<BrokerRequest>,
    config: GlobalRangeBrokerConfig,
) -> Vec<BrokerRequest> {
    let mut requests = vec![first];
    if config.expected_concurrent_callers == 1 {
        return requests;
    }
    let started = Instant::now();
    let quiet_deadline = started + config.collection_delay;
    while requests.len() < config.expected_concurrent_callers {
        let now = Instant::now();
        let pressure = client.pressure_snapshot();
        let saturated = pressure.max_concurrency > 0
            && (pressure.active_requests >= pressure.max_concurrency
                || pressure.queued_requests > 0);
        let deadline = if saturated {
            started
                + config
                    .saturation_collection_budget
                    .max(Duration::from_nanos(pressure.service_time_ns_ewma))
                    .max(config.collection_delay)
        } else {
            quiet_deadline
        };
        if now >= deadline {
            break;
        }
        match receiver.recv_timeout(deadline - now) {
            Ok(request) => requests.push(request),
            Err(mpsc::RecvTimeoutError::Timeout) => break,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    requests
}

fn plan_broker_spans(
    requests: &[BrokerRequest],
    config: GlobalRangeBrokerConfig,
) -> Result<Vec<BrokerPhysicalSpan>, String> {
    let mut ranges = requests
        .iter()
        .enumerate()
        .flat_map(|(request_index, request)| {
            request
                .ranges
                .iter()
                .enumerate()
                .map(move |(range_index, range)| {
                    let end = range
                        .offset
                        .checked_add(range.length)
                        .ok_or_else(|| "broker range overflow".to_string())?;
                    Ok((
                        range.object_key.clone(),
                        range.offset,
                        end,
                        BrokerSpanMember {
                            request_index,
                            range_index,
                            offset: range.offset,
                            length: range.length,
                        },
                    ))
                })
        })
        .collect::<Result<Vec<_>, String>>()?;
    ranges.sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.request_index.cmp(&right.3.request_index))
            .then_with(|| left.3.range_index.cmp(&right.3.range_index))
    });

    let mut spans = Vec::<BrokerPhysicalSpan>::new();
    for (object_key, offset, end, member) in ranges {
        let merge = spans.last().is_some_and(|span| {
            span.object_key == object_key
                && offset <= span.end.saturating_add(config.merge_gap_bytes)
                && end.max(span.end).saturating_sub(span.offset) <= config.max_range_bytes
        });
        if merge {
            let span = spans.last_mut().expect("checked broker span");
            span.end = span.end.max(end);
            span.members.push(member);
        } else {
            spans.push(BrokerPhysicalSpan {
                object_key,
                offset,
                end,
                members: vec![member],
            });
        }
    }
    Ok(spans)
}

fn execute_broker_window(
    client: &S3ObjectStoreClient,
    config: GlobalRangeBrokerConfig,
    requests: Vec<BrokerRequest>,
) {
    let planning_config = if requests.len() == 1 {
        // Cross-call planning has no additional information in this case.
        // Preserve the caller's already calibrated range plan.
        GlobalRangeBrokerConfig {
            merge_gap_bytes: 0,
            ..config
        }
    } else {
        config
    };
    let spans = match plan_broker_spans(&requests, planning_config) {
        Ok(spans) => spans,
        Err(error) => {
            for request in requests {
                let _ = request.response.send(Err(error.clone()));
            }
            return;
        }
    };
    let physical = spans
        .iter()
        .map(|span| ObjectRange {
            object_key: span.object_key.clone(),
            offset: span.offset,
            length: span.end - span.offset,
        })
        .collect::<Vec<_>>();

    let fetch_result = client.for_each_object_range(&physical, &mut |completed| {
        let span_index = completed.index;
        let span = spans
            .get(span_index)
            .ok_or_else(|| format!("broker returned invalid physical span {span_index}"))?;
        let owner = span
            .members
            .iter()
            .map(|member| (member.request_index, member.range_index))
            .min()
            .expect("broker spans have members");

        for member in &span.members {
            let relative = member
                .offset
                .checked_sub(span.offset)
                .ok_or_else(|| "broker member precedes its physical span".to_string())?
                as usize;
            let end = relative
                .checked_add(member.length as usize)
                .ok_or_else(|| "broker member slice overflow".to_string())?;
            if end > completed.bytes.len() {
                return Err(format!(
                    "broker member {}:{} exceeds physical span {}",
                    member.request_index, member.range_index, span_index
                )
                .into());
            }
            let owns_physical_request = owner == (member.request_index, member.range_index);
            requests[member.request_index]
                .response
                .send(Ok(CompletedRange {
                    index: member.range_index,
                    bytes: completed.bytes[relative..end].to_vec(),
                    started_ns: completed.started_ns,
                    first_byte_ns: completed.first_byte_ns,
                    completed_ns: completed.completed_ns,
                    physical_requests: usize::from(owns_physical_request),
                    physical_fetched_bytes: if owns_physical_request {
                        completed.bytes.len() as u64
                    } else {
                        0
                    },
                }))
                .map_err(|_| "global S3 range broker consumer stopped".to_string())?;
        }
        Ok(())
    });

    if let Err(error) = fetch_result {
        let error = error.to_string();
        for request in requests {
            let _ = request.response.send(Err(error.clone()));
        }
    }
}

fn run_s3_range_broker(
    client: S3ObjectStoreClient,
    config: GlobalRangeBrokerConfig,
    receiver: mpsc::Receiver<BrokerRequest>,
) {
    while let Ok(first) = receiver.recv() {
        let requests = collect_broker_window(first, &client, &receiver, config);
        let client = client.clone();
        // Execution is detached from collection so requests that become ready
        // early can immediately submit their next intents. The shared S3
        // semaphore remains the single global physical-I/O budget.
        thread::spawn(move || execute_broker_window(&client, config, requests));
    }
}

#[cfg(any(test, feature = "experiment-controls"))]
fn coalesced_span_stats(ranges: &[(u64, u64)], threshold: u64) -> (usize, u64) {
    if ranges.is_empty() {
        return (0, 0);
    }
    let mut ordered = ranges
        .iter()
        .map(|&(offset, length)| (offset, offset.saturating_add(length)))
        .collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|&(start, end)| (start, end));
    let mut count = 0;
    let mut bytes = 0u64;
    let (mut start, mut end) = ordered[0];
    for &(next_start, next_end) in ordered.iter().skip(1) {
        if next_start <= end.saturating_add(threshold) {
            end = end.max(next_end);
        } else {
            count += 1;
            bytes = bytes.saturating_add(end.saturating_sub(start));
            start = next_start;
            end = next_end;
        }
    }
    (count + 1, bytes.saturating_add(end.saturating_sub(start)))
}

impl S3Backend {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        endpoint: String,
        bucket: String,
        object_key: String,
        payload_start: usize,
        access_key_id: String,
        secret_access_key: String,
        region: String,
        max_concurrency: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if object_key.is_empty() {
            return Err("S3 object key must be non-empty".into());
        }
        Ok(Self {
            client: S3ObjectStoreClient::new(
                endpoint,
                bucket,
                access_key_id,
                secret_access_key,
                region,
                max_concurrency,
            )?,
            object_key,
            payload_start: payload_start as u64,
            read_mode: S3ReadMode::Explicit,
            range_broker: None,
        })
    }

    pub fn from_shared_client(
        client: S3ObjectStoreClient,
        object_key: String,
        payload_start: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if object_key.is_empty() {
            return Err("S3 object key must be non-empty".into());
        }
        Ok(Self {
            client,
            object_key,
            payload_start: payload_start as u64,
            read_mode: S3ReadMode::Explicit,
            range_broker: None,
        })
    }

    pub fn with_global_range_broker(
        mut self,
        config: GlobalRangeBrokerConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        self.range_broker = Some(SharedS3RangeBroker::new(self.client.clone(), config)?);
        Ok(self)
    }

    #[cfg(feature = "experiment-controls")]
    pub fn from_shared_client_vectored(
        client: S3ObjectStoreClient,
        object_key: String,
        payload_start: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut backend = Self::from_shared_client(client, object_key, payload_start)?;
        backend.read_mode = S3ReadMode::NativeVectored;
        Ok(backend)
    }
}

impl StorageBackend for S3Backend {
    fn read_byte_range(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut result = self.read_byte_ranges(&[(offset, length)])?;
        Ok(result.pop().expect("one requested S3 range"))
    }

    fn for_each_byte_range(
        &self,
        ranges: &[(u64, u64)],
        callback: &mut dyn FnMut(CompletedRange) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let object_ranges = ranges
            .iter()
            .map(|&(offset, length)| {
                let offset = self
                    .payload_start
                    .checked_add(offset)
                    .ok_or("S3 absolute range offset overflow")?;
                Ok(ObjectRange {
                    object_key: self.object_key.clone(),
                    offset,
                    length,
                })
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        match self.read_mode {
            S3ReadMode::Explicit => {
                let Some(broker) = &self.range_broker else {
                    return self.client.for_each_object_range(&object_ranges, callback);
                };
                broker.for_each(object_ranges, callback)
            }
            #[cfg(feature = "experiment-controls")]
            S3ReadMode::NativeVectored => {
                let started = Instant::now();
                let buffers = self.client.fetch_object_ranges_vectored(&object_ranges)?;
                let fetched_bytes = buffers.iter().map(|buffer| buffer.len() as u64).sum();
                let completed_ns = started.elapsed().as_nanos() as u64;
                for (index, bytes) in buffers.into_iter().enumerate() {
                    callback(CompletedRange {
                        index,
                        physical_requests: usize::from(index == 0),
                        physical_fetched_bytes: if index == 0 { fetched_bytes } else { 0 },
                        bytes,
                        started_ns: 0,
                        first_byte_ns: if index == 0 { completed_ns } else { 0 },
                        completed_ns: if index == 0 { completed_ns } else { 0 },
                    })?;
                }
                Ok(())
            }
        }
    }

    fn client_requests_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
        #[cfg(feature = "experiment-controls")]
        if self.read_mode == S3ReadMode::NativeVectored {
            return coalesced_span_stats(ranges, OBJECT_STORE_COALESCE_DEFAULT).0;
        }
        ranges.len()
    }

    fn server_entries_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
        self.client_requests_for_ranges(ranges)
    }

    fn physical_ranges_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
        self.client_requests_for_ranges(ranges)
    }

    fn physical_fetched_bytes_for_ranges(&self, ranges: &[(u64, u64)]) -> Option<u64> {
        #[cfg(feature = "experiment-controls")]
        if self.read_mode == S3ReadMode::NativeVectored {
            return Some(coalesced_span_stats(ranges, OBJECT_STORE_COALESCE_DEFAULT).1);
        }
        let _ = ranges;
        None
    }

    fn supports_streaming_range_completion(&self) -> bool {
        if self.range_broker.is_some() {
            return false;
        }
        #[cfg(feature = "experiment-controls")]
        if self.read_mode == S3ReadMode::NativeVectored {
            return false;
        }
        true
    }

    fn object_store_pressure(&self) -> Option<ObjectStorePressure> {
        Some(self.client.pressure_snapshot())
    }

    fn executes_planned_ranges_exactly(&self) -> bool {
        if self.range_broker.is_some() {
            return false;
        }
        #[cfg(feature = "experiment-controls")]
        if self.read_mode == S3ReadMode::NativeVectored {
            return false;
        }
        true
    }
}

// -- AIStore GetBatch backend ---------------------------------------

/// One range entry in an AIStore MOSS/GetBatch request.
///
/// The object name is part of the physical address. This matters for strong
/// baselines whose native layout uses one object per fragment, video, or shard.
pub type AIStoreObjectRange = ObjectRange;

/// Multi-object AIStore MOSS/GetBatch transport.
///
/// This layer knows nothing about chunks, codec dependencies, or decoding. It
/// preserves request order and returns exactly one byte buffer per entry.
#[derive(Clone)]
pub struct AIStoreGetBatchClient {
    endpoint: String,
    bucket: String,
    provider: String,
    client: reqwest::blocking::Client,
}

struct InFlightGetBatchPermit {
    state: Arc<(Mutex<usize>, Condvar)>,
}

impl Drop for InFlightGetBatchPermit {
    fn drop(&mut self) {
        let (lock, condition) = &*self.state;
        let mut in_flight = lock.lock().unwrap_or_else(|error| error.into_inner());
        *in_flight = in_flight.saturating_sub(1);
        condition.notify_one();
    }
}

/// Cloneable AIStore client that shares one HTTP pool and one global request
/// budget across all saturation clients.
#[derive(Clone)]
pub struct SharedAIStoreGetBatchClient {
    inner: AIStoreGetBatchClient,
    state: Arc<(Mutex<usize>, Condvar)>,
    max_in_flight: usize,
}

impl SharedAIStoreGetBatchClient {
    pub fn new(
        endpoint: String,
        bucket: String,
        provider: String,
        max_in_flight: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if max_in_flight == 0 {
            return Err("AIStore max_in_flight must be positive".into());
        }
        Ok(Self {
            inner: AIStoreGetBatchClient::new(endpoint, bucket, provider)?,
            state: Arc::new((Mutex::new(0), Condvar::new())),
            max_in_flight,
        })
    }

    fn acquire(&self) -> InFlightGetBatchPermit {
        let (lock, condition) = &*self.state;
        let mut in_flight = lock.lock().unwrap_or_else(|error| error.into_inner());
        while *in_flight >= self.max_in_flight {
            in_flight = condition
                .wait(in_flight)
                .unwrap_or_else(|error| error.into_inner());
        }
        *in_flight += 1;
        InFlightGetBatchPermit {
            state: self.state.clone(),
        }
    }
}

pub trait MultiObjectStorageBackend {
    fn fetch_object_ranges(
        &self,
        ranges: &[AIStoreObjectRange],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>>;

    fn client_requests_for_ranges(&self, ranges: &[AIStoreObjectRange]) -> usize {
        usize::from(!ranges.is_empty())
    }

    fn server_entries_for_ranges(&self, ranges: &[AIStoreObjectRange]) -> usize {
        ranges.len()
    }
}

impl MultiObjectStorageBackend for S3ObjectStoreClient {
    fn fetch_object_ranges(
        &self,
        ranges: &[AIStoreObjectRange],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        S3ObjectStoreClient::fetch_object_ranges(self, ranges)
    }

    fn client_requests_for_ranges(&self, ranges: &[AIStoreObjectRange]) -> usize {
        ranges.len()
    }
}

impl AIStoreGetBatchClient {
    pub fn new(
        endpoint: String,
        bucket: String,
        provider: String,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if endpoint.is_empty() || bucket.is_empty() {
            return Err("AIStore endpoint and bucket must be non-empty".into());
        }
        let client = reqwest::blocking::Client::builder()
            .pool_idle_timeout(Duration::from_secs(90))
            .redirect(reqwest::redirect::Policy::limited(4))
            .build()?;
        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            bucket,
            provider,
            client,
        })
    }

    fn decode_tar_ranges(
        archive: &[u8],
        expected_lengths: &[u64],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let mut result = Vec::with_capacity(expected_lengths.len());
        let mut tar = tar::Archive::new(std::io::Cursor::new(archive));
        for (index, entry) in tar.entries()?.enumerate() {
            if index >= expected_lengths.len() {
                return Err("AIStore GetBatch returned extra archive entries".into());
            }
            let mut entry = entry?;
            let mut bytes = Vec::with_capacity(expected_lengths[index] as usize);
            std::io::Read::read_to_end(&mut entry, &mut bytes)?;
            if bytes.len() != expected_lengths[index] as usize {
                return Err(format!(
                    "AIStore GetBatch range {index} length mismatch: {} != {}",
                    bytes.len(),
                    expected_lengths[index]
                )
                .into());
            }
            result.push(bytes);
        }
        if result.len() != expected_lengths.len() {
            return Err(format!(
                "AIStore GetBatch returned {} entries for {} ranges",
                result.len(),
                expected_lengths.len()
            )
            .into());
        }
        Ok(result)
    }

    fn request_body(
        ranges: &[AIStoreObjectRange],
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut entries = Vec::with_capacity(ranges.len());
        for range in ranges {
            if range.object_key.is_empty()
                || range.length == 0
                || range.offset.checked_add(range.length).is_none()
            {
                return Err(format!(
                    "invalid AIStore object range: object={:?}, offset={}, length={}",
                    range.object_key, range.offset, range.length
                )
                .into());
            }
            entries.push(serde_json::json!({
                "objname": range.object_key,
                "start": range.offset,
                "length": range.length,
            }));
        }
        Ok(serde_json::json!({
            "in": entries,
            "mime": ".tar",
            "coer": false,
            "onob": false,
            "strm": true,
        }))
    }

    pub fn fetch_object_ranges(
        &self,
        ranges: &[AIStoreObjectRange],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        let request = Self::request_body(ranges)?;
        let expected_lengths = ranges.iter().map(|range| range.length).collect::<Vec<_>>();
        let url = format!("{}/v1/ml/moss/{}", self.endpoint, self.bucket);
        let response = self
            .client
            .get(url)
            .query(&[("provider", self.provider.as_str())])
            .json(&request)
            .send()?
            .error_for_status()?;
        let archive = response.bytes()?;
        Self::decode_tar_ranges(&archive, &expected_lengths)
    }
}

impl MultiObjectStorageBackend for AIStoreGetBatchClient {
    fn fetch_object_ranges(
        &self,
        ranges: &[AIStoreObjectRange],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        AIStoreGetBatchClient::fetch_object_ranges(self, ranges)
    }
}

impl MultiObjectStorageBackend for SharedAIStoreGetBatchClient {
    fn fetch_object_ranges(
        &self,
        ranges: &[AIStoreObjectRange],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        let _permit = self.acquire();
        self.inner.fetch_object_ranges(ranges)
    }
}

/// Executes ranges within one object as one AIStore MOSS/GetBatch request.
///
/// The scheduler still owns dependency resolution and decoding. This adapter
/// maps its single-object relative address space onto the multi-object client.
pub struct AIStoreGetBatchBackend {
    client: Box<dyn MultiObjectStorageBackend + Send + Sync>,
    object_key: String,
    payload_start: u64,
}

impl AIStoreGetBatchBackend {
    pub fn new(
        endpoint: String,
        bucket: String,
        object_key: String,
        provider: String,
        payload_start: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if object_key.is_empty() {
            return Err("AIStore object key must be non-empty".into());
        }
        Ok(Self {
            client: Box::new(AIStoreGetBatchClient::new(endpoint, bucket, provider)?),
            object_key,
            payload_start: payload_start as u64,
        })
    }

    pub fn from_shared_client(
        client: SharedAIStoreGetBatchClient,
        object_key: String,
        payload_start: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if object_key.is_empty() {
            return Err("AIStore object key must be non-empty".into());
        }
        Ok(Self {
            client: Box::new(client),
            object_key,
            payload_start: payload_start as u64,
        })
    }

    pub fn client(&self) -> &(dyn MultiObjectStorageBackend + Send + Sync) {
        self.client.as_ref()
    }
}

impl StorageBackend for AIStoreGetBatchBackend {
    fn read_byte_range(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut result = self.read_byte_ranges(&[(offset, length)])?;
        Ok(result.pop().expect("one requested AIStore range"))
    }

    fn read_byte_ranges(
        &self,
        ranges: &[(u64, u64)],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        let mut object_ranges = Vec::with_capacity(ranges.len());
        for &(offset, length) in ranges {
            if length == 0 || offset.checked_add(length).is_none() {
                return Err(
                    format!("invalid AIStore range: offset={offset}, length={length}").into(),
                );
            }
            let start = self
                .payload_start
                .checked_add(offset)
                .ok_or("AIStore absolute range offset overflow")?;
            object_ranges.push(AIStoreObjectRange {
                object_key: self.object_key.clone(),
                offset: start,
                length,
            });
        }
        self.client.fetch_object_ranges(&object_ranges)
    }

    fn for_each_byte_range(
        &self,
        ranges: &[(u64, u64)],
        callback: &mut dyn FnMut(CompletedRange) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let started = Instant::now();
        let buffers = self.read_byte_ranges(ranges)?;
        let fetched_bytes = buffers.iter().map(|buffer| buffer.len() as u64).sum();
        let completed_ns = started.elapsed().as_nanos() as u64;
        for (index, bytes) in buffers.into_iter().enumerate() {
            callback(CompletedRange {
                index,
                physical_requests: usize::from(index == 0),
                physical_fetched_bytes: if index == 0 { fetched_bytes } else { 0 },
                bytes,
                // GetBatch is one client request. Attribute request-level
                // timing once; later callbacks carry only returned entries.
                started_ns: 0,
                first_byte_ns: if index == 0 { completed_ns } else { 0 },
                completed_ns: if index == 0 { completed_ns } else { 0 },
            })?;
        }
        Ok(())
    }

    fn client_requests_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
        usize::from(!ranges.is_empty())
    }

    fn executes_planned_ranges_exactly(&self) -> bool {
        true
    }
}

// ── No-op backend ───────────────────────────────────────────────────

/// Returns an error when called; usable as "plan-only" mode placeholder.
#[cfg(feature = "experiment-controls")]
pub struct NoopBackend;

#[cfg(feature = "experiment-controls")]
impl StorageBackend for NoopBackend {
    fn read_byte_range(
        &self,
        _offset: u64,
        _length: u64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Err("NoopBackend: read_byte_range not available".into())
    }
}

#[cfg(test)]
mod aistore_tests {
    use super::{
        coalesced_span_stats, plan_broker_spans, AIStoreGetBatchBackend, AIStoreGetBatchClient,
        AIStoreObjectRange, BrokerRequest, CompletedRange, GlobalRangeBrokerConfig, ObjectRange,
        S3Backend, S3ObjectStoreClient, SharedAIStoreGetBatchClient, StorageBackend,
    };
    use std::io::Cursor;
    use std::sync::mpsc;
    use std::time::Duration;

    struct ProfiledBackend;

    impl StorageBackend for ProfiledBackend {
        fn read_byte_range(
            &self,
            _offset: u64,
            _length: u64,
        ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            unreachable!("profile test uses the completion callback")
        }

        fn for_each_byte_range(
            &self,
            _ranges: &[(u64, u64)],
            callback: &mut dyn FnMut(CompletedRange) -> Result<(), Box<dyn std::error::Error>>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            for completed in [
                CompletedRange {
                    index: 1,
                    physical_requests: 1,
                    physical_fetched_bytes: 1,
                    bytes: vec![1],
                    started_ns: 5,
                    first_byte_ns: 15,
                    completed_ns: 50,
                },
                CompletedRange {
                    index: 2,
                    physical_requests: 1,
                    physical_fetched_bytes: 1,
                    bytes: vec![2],
                    started_ns: 50,
                    first_byte_ns: 60,
                    completed_ns: 80,
                },
                CompletedRange {
                    index: 0,
                    physical_requests: 1,
                    physical_fetched_bytes: 1,
                    bytes: vec![0],
                    started_ns: 0,
                    first_byte_ns: 10,
                    completed_ns: 100,
                },
            ] {
                callback(completed)?;
            }
            Ok(())
        }
    }

    struct PlanChangingBackend;

    impl StorageBackend for PlanChangingBackend {
        fn read_byte_range(
            &self,
            _offset: u64,
            length: u64,
        ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            Ok(vec![0; length as usize])
        }

        fn physical_ranges_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
            usize::from(!ranges.is_empty())
        }

        fn executes_planned_ranges_exactly(&self) -> bool {
            true
        }
    }

    #[test]
    fn exact_backend_contract_rejects_a_second_physical_plan() {
        let error = PlanChangingBackend
            .read_byte_ranges_profiled(&[(0, 8), (32, 8)])
            .unwrap_err()
            .to_string();
        assert!(error.contains("backend changed an immutable physical plan"));
    }

    #[test]
    fn default_s3_backend_is_an_exact_span_executor() {
        let client = S3ObjectStoreClient::new(
            "http://127.0.0.1:1".to_string(),
            "test".to_string(),
            "access".to_string(),
            "secret".to_string(),
            "us-east-1".to_string(),
            2,
        )
        .unwrap();
        let backend = S3Backend::from_shared_client(client, "layout".to_string(), 64).unwrap();
        assert!(backend.executes_planned_ranges_exactly());
        assert!(backend.supports_streaming_range_completion());
        assert_eq!(backend.client_requests_for_ranges(&[(0, 8), (32, 8)]), 2);
        assert_eq!(backend.physical_ranges_for_ranges(&[(0, 8), (32, 8)]), 2);
        assert_eq!(
            backend.physical_fetched_bytes_for_ranges(&[(0, 8), (32, 8)]),
            None
        );
    }

    fn tar_payload(values: &[&[u8]]) -> Vec<u8> {
        let mut output = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut output);
            for (index, value) in values.iter().enumerate() {
                let mut header = tar::Header::new_gnu();
                header.set_size(value.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder
                    .append_data(&mut header, format!("entry-{index}"), Cursor::new(value))
                    .unwrap();
            }
            builder.finish().unwrap();
        }
        output
    }

    #[test]
    fn object_store_span_stats_merge_within_default_gap() {
        let (count, bytes) = coalesced_span_stats(&[(0, 100), (600, 100), (2_000_000, 50)], 1_000);
        assert_eq!(count, 2);
        assert_eq!(bytes, 750);
    }

    #[test]
    fn object_store_span_stats_restore_order_independently() {
        let forward = coalesced_span_stats(&[(0, 10), (20, 10), (40, 10)], 10);
        let reverse = coalesced_span_stats(&[(40, 10), (0, 10), (20, 10)], 10);
        assert_eq!(forward, (1, 50));
        assert_eq!(reverse, forward);
    }

    fn broker_request(ranges: &[(&str, u64, u64)]) -> BrokerRequest {
        let (response, _receiver) = mpsc::channel();
        BrokerRequest {
            ranges: ranges
                .iter()
                .map(|(object_key, offset, length)| ObjectRange {
                    object_key: (*object_key).to_string(),
                    offset: *offset,
                    length: *length,
                })
                .collect(),
            response,
        }
    }

    fn broker_config(merge_gap_bytes: u64, max_range_bytes: u64) -> GlobalRangeBrokerConfig {
        GlobalRangeBrokerConfig {
            expected_concurrent_callers: 8,
            collection_delay: Duration::ZERO,
            saturation_collection_budget: Duration::ZERO,
            merge_gap_bytes,
            max_range_bytes,
        }
    }

    #[test]
    fn global_broker_merges_ranges_across_concurrent_callers() {
        let requests = vec![
            broker_request(&[("layout", 100, 20), ("layout", 300, 10)]),
            broker_request(&[("layout", 125, 15), ("layout", 280, 10)]),
        ];

        let spans = plan_broker_spans(&requests, broker_config(20, 1024)).unwrap();

        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].offset, spans[0].end), (100, 140));
        assert_eq!((spans[1].offset, spans[1].end), (280, 310));
        assert_eq!(spans[0].members.len(), 2);
        assert_eq!(spans[1].members.len(), 2);
    }

    #[test]
    fn global_broker_deduplicates_identical_ranges_but_preserves_consumers() {
        let requests = vec![
            broker_request(&[("layout", 100, 20)]),
            broker_request(&[("layout", 100, 20)]),
        ];

        let spans = plan_broker_spans(&requests, broker_config(0, 1024)).unwrap();

        assert_eq!(spans.len(), 1);
        assert_eq!((spans[0].offset, spans[0].end), (100, 120));
        assert_eq!(spans[0].members.len(), 2);
        assert_eq!(spans[0].members[0].request_index, 0);
        assert_eq!(spans[0].members[1].request_index, 1);
    }

    #[test]
    fn global_broker_never_merges_different_objects() {
        let requests = vec![
            broker_request(&[("layout-a", 100, 20)]),
            broker_request(&[("layout-b", 100, 20)]),
        ];

        let spans = plan_broker_spans(&requests, broker_config(1024, 4096)).unwrap();

        assert_eq!(spans.len(), 2);
        assert_ne!(spans[0].object_key, spans[1].object_key);
    }

    #[test]
    fn global_broker_respects_maximum_physical_range_size() {
        let requests = vec![
            broker_request(&[("layout", 0, 64)]),
            broker_request(&[("layout", 80, 64)]),
        ];

        let spans = plan_broker_spans(&requests, broker_config(1024, 128)).unwrap();

        assert_eq!(spans.len(), 2);
        assert!(spans.iter().all(|span| span.end - span.offset <= 128));
    }

    #[test]
    fn global_broker_preserves_original_request_and_range_coordinates() {
        let requests = vec![
            broker_request(&[("layout", 200, 10), ("layout", 100, 10)]),
            broker_request(&[("layout", 115, 10)]),
        ];

        let spans = plan_broker_spans(&requests, broker_config(5, 1024)).unwrap();

        assert_eq!(spans.len(), 2);
        let first = &spans[0];
        assert_eq!((first.offset, first.end), (100, 125));
        assert_eq!(
            first
                .members
                .iter()
                .map(|member| (member.request_index, member.range_index))
                .collect::<Vec<_>>(),
            vec![(0, 1), (1, 0)]
        );
        assert_eq!(
            spans[1]
                .members
                .iter()
                .map(|member| (member.request_index, member.range_index))
                .collect::<Vec<_>>(),
            vec![(0, 0)]
        );
    }

    #[test]
    fn profiled_range_batch_restores_order_and_measures_overlap() {
        let profile = ProfiledBackend
            .read_byte_ranges_profiled(&[(0, 1), (1, 1), (2, 1)])
            .unwrap();
        assert_eq!(profile.buffers, vec![vec![0], vec![1], vec![2]]);
        assert_eq!(profile.dispatch_ns_sum, 55);
        assert_eq!(profile.ttfb_ns_sum, 30);
        assert_eq!(profile.service_ns_sum, 175);
        assert_eq!(profile.max_in_flight, 2);
        assert_eq!(profile.timing_samples, 3);
    }

    #[test]
    fn decodes_getbatch_tar_in_request_order() {
        let archive = tar_payload(&[b"abc", b"de"]);
        let result = AIStoreGetBatchClient::decode_tar_ranges(&archive, &[3, 2]).unwrap();
        assert_eq!(result, vec![b"abc".to_vec(), b"de".to_vec()]);
    }

    #[test]
    fn rejects_short_getbatch_entry() {
        let archive = tar_payload(&[b"abc"]);
        let error = AIStoreGetBatchClient::decode_tar_ranges(&archive, &[4]).unwrap_err();
        assert!(error.to_string().contains("length mismatch"));
    }

    #[test]
    fn builds_multi_object_request_in_input_order() {
        let ranges = vec![
            AIStoreObjectRange {
                object_key: "fragments/gop-7.bin".to_string(),
                offset: 11,
                length: 13,
            },
            AIStoreObjectRange {
                object_key: "videos/video-2.mp4".to_string(),
                offset: 17,
                length: 19,
            },
        ];
        let body = AIStoreGetBatchClient::request_body(&ranges).unwrap();
        assert_eq!(body["in"][0]["objname"], "fragments/gop-7.bin");
        assert_eq!(body["in"][0]["start"], 11);
        assert_eq!(body["in"][0]["length"], 13);
        assert_eq!(body["in"][1]["objname"], "videos/video-2.mp4");
        assert_eq!(body["in"][1]["start"], 17);
        assert_eq!(body["in"][1]["length"], 19);
    }

    #[test]
    fn rejects_invalid_multi_object_ranges() {
        for range in [
            AIStoreObjectRange {
                object_key: String::new(),
                offset: 0,
                length: 1,
            },
            AIStoreObjectRange {
                object_key: "object".to_string(),
                offset: 0,
                length: 0,
            },
            AIStoreObjectRange {
                object_key: "object".to_string(),
                offset: u64::MAX,
                length: 1,
            },
        ] {
            assert!(AIStoreGetBatchClient::request_body(&[range]).is_err());
        }
    }

    #[test]
    fn reports_one_client_request_and_one_server_entry_per_range() {
        use super::StorageBackend;

        let backend = AIStoreGetBatchBackend::new(
            "http://localhost:51080".to_string(),
            "bucket".to_string(),
            "object".to_string(),
            "ais".to_string(),
            0,
        )
        .unwrap();
        let ranges = [(0, 4), (8, 2), (16, 1)];
        assert_eq!(backend.client_requests_for_ranges(&ranges), 1);
        assert_eq!(backend.server_entries_for_ranges(&ranges), 3);
        assert_eq!(backend.client_requests_for_ranges(&[]), 0);
        assert!(!backend.supports_streaming_range_completion());
    }

    #[test]
    fn shared_getbatch_requires_a_positive_global_request_budget() {
        assert!(SharedAIStoreGetBatchClient::new(
            "http://localhost:51080".to_string(),
            "bucket".to_string(),
            "ais".to_string(),
            0,
        )
        .is_err());
    }
}
