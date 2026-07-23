//! Storage backend abstraction: local mmap vs object store (MinIO/S3).
//!
//! The scheduler resolves record offsets via the local chunk index, then
//! delegates payload byte reads to a StorageBackend. This allows the same
//! scheduler to run against local SSD (zero network cost) or MinIO (real
//! Range GET latency and bytes-transferred measurement).

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
pub use vclasp_object_store::{CompletedRange, ObjectRange, S3ObjectStoreClient, S3ObjectStream};

/// Abstract byte-range read from a storage backend.
pub trait StorageBackend: Send {
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
        let mut result: Vec<Option<Vec<u8>>> =
            std::iter::repeat_with(|| None).take(ranges.len()).collect();
        self.for_each_byte_range(ranges, &mut |completed| {
            result[completed.index] = Some(completed.bytes);
            Ok(())
        })?;
        result
            .into_iter()
            .map(|value| value.ok_or_else(|| "backend omitted a requested range".into()))
            .collect()
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

    /// Whether callbacks can arrive while other range I/O is still in flight.
    /// Backends that materialize the complete response before callbacks must
    /// keep this false; completion-driven decode would only fragment batches.
    fn supports_streaming_range_completion(&self) -> bool {
        false
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
}

// ── MinIO / S3 backend (via object_store) ─────────────────────────

/// Single-object view over the shared S3 transport used by VClasp layouts.
pub struct S3Backend {
    client: S3ObjectStoreClient,
    object_key: String,
    payload_start: u64,
    read_mode: S3ReadMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum S3ReadMode {
    Explicit,
    NativeVectored,
}

const OBJECT_STORE_COALESCE_DEFAULT: u64 = 1024 * 1024;

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
        })
    }

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
            S3ReadMode::Explicit => self.client.for_each_object_range(&object_ranges, callback),
            S3ReadMode::NativeVectored => {
                let started = Instant::now();
                let buffers = self.client.fetch_object_ranges_vectored(&object_ranges)?;
                let completed_ns = started.elapsed().as_nanos() as u64;
                for (index, bytes) in buffers.into_iter().enumerate() {
                    callback(CompletedRange {
                        index,
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
        match self.read_mode {
            S3ReadMode::Explicit => ranges.len(),
            S3ReadMode::NativeVectored => {
                coalesced_span_stats(ranges, OBJECT_STORE_COALESCE_DEFAULT).0
            }
        }
    }

    fn server_entries_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
        self.client_requests_for_ranges(ranges)
    }

    fn physical_ranges_for_ranges(&self, ranges: &[(u64, u64)]) -> usize {
        self.client_requests_for_ranges(ranges)
    }

    fn physical_fetched_bytes_for_ranges(&self, ranges: &[(u64, u64)]) -> Option<u64> {
        (self.read_mode == S3ReadMode::NativeVectored)
            .then(|| coalesced_span_stats(ranges, OBJECT_STORE_COALESCE_DEFAULT).1)
    }

    fn supports_streaming_range_completion(&self) -> bool {
        self.read_mode == S3ReadMode::Explicit
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
        let completed_ns = started.elapsed().as_nanos() as u64;
        for (index, bytes) in buffers.into_iter().enumerate() {
            callback(CompletedRange {
                index,
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
}

// ── No-op backend ───────────────────────────────────────────────────

/// Returns an error when called; usable as "plan-only" mode placeholder.
pub struct NoopBackend;

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
        coalesced_span_stats, AIStoreGetBatchBackend, AIStoreGetBatchClient, AIStoreObjectRange,
        SharedAIStoreGetBatchClient,
    };
    use std::io::Cursor;

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
