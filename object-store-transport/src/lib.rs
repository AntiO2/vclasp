//! Shared S3-compatible Range GET transport for VClasp and its baselines.

use futures_util::StreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::{ClientOptions, GetOptions, ObjectStore, ObjectStoreExt};
use std::collections::BTreeMap;
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectRange {
    pub object_key: String,
    pub offset: u64,
    pub length: u64,
}

pub struct CompletedRange {
    pub index: usize,
    pub bytes: Vec<u8>,
    pub started_ns: u64,
    pub first_byte_ns: u64,
    pub completed_ns: u64,
    pub physical_requests: usize,
    pub physical_fetched_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ObjectStorePressure {
    pub max_concurrency: usize,
    pub active_requests: usize,
    pub outstanding_requests: usize,
    pub queued_requests: usize,
    pub service_time_ns_ewma: u64,
}

struct OutstandingRequestGuard {
    outstanding: Arc<AtomicUsize>,
}

impl OutstandingRequestGuard {
    fn new(outstanding: Arc<AtomicUsize>) -> Self {
        outstanding.fetch_add(1, Ordering::AcqRel);
        Self { outstanding }
    }
}

impl Drop for OutstandingRequestGuard {
    fn drop(&mut self) {
        self.outstanding.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One persistent object GET with bounded producer/consumer buffering.
///
/// The request holds the same shared semaphore permit as Range GETs until the
/// stream reaches EOF or the consumer drops the receiver.
pub struct S3ObjectStream {
    runtime: Arc<tokio::runtime::Runtime>,
    receiver: tokio::sync::mpsc::Receiver<Result<Vec<u8>, String>>,
    bytes_received: Arc<AtomicU64>,
}

impl S3ObjectStream {
    pub fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        match self.runtime.block_on(self.receiver.recv()) {
            Some(Ok(chunk)) => Ok(Some(chunk)),
            Some(Err(error)) => Err(error.into()),
            None => Ok(None),
        }
    }

    pub fn bytes_received(&self) -> u64 {
        self.bytes_received.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectHead {
    pub object_key: String,
    pub size: u64,
    pub e_tag: Option<String>,
    pub version: Option<String>,
}

/// Authenticated, bounded-concurrency S3 transport shared by all systems.
#[derive(Clone)]
pub struct S3ObjectStoreClient {
    store: Arc<dyn ObjectStore>,
    runtime: Arc<tokio::runtime::Runtime>,
    request_budget: Arc<Semaphore>,
    outstanding_requests: Arc<AtomicUsize>,
    service_time_ns_ewma: Arc<AtomicU64>,
    max_concurrency: usize,
}

impl S3ObjectStoreClient {
    pub fn new(
        endpoint: String,
        bucket: String,
        access_key_id: String,
        secret_access_key: String,
        region: String,
        max_concurrency: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if endpoint.is_empty() || bucket.is_empty() {
            return Err("S3 endpoint and bucket must be non-empty".into());
        }
        if max_concurrency == 0 {
            return Err("S3 max_concurrency must be positive".into());
        }
        let allow_http = endpoint.starts_with("http://");
        let client_options = ClientOptions::new()
            .with_allow_http(allow_http)
            .with_pool_max_idle_per_host(max_concurrency)
            .with_pool_idle_timeout(Duration::from_secs(90));
        let store = AmazonS3Builder::new()
            .with_endpoint(endpoint)
            .with_bucket_name(bucket)
            .with_access_key_id(access_key_id)
            .with_secret_access_key(secret_access_key)
            .with_region(region)
            .with_allow_http(allow_http)
            .with_client_options(client_options)
            .build()?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(max_concurrency)
            .build()?;
        Ok(Self {
            store: Arc::new(store),
            runtime: Arc::new(runtime),
            request_budget: Arc::new(Semaphore::new(max_concurrency)),
            outstanding_requests: Arc::new(AtomicUsize::new(0)),
            service_time_ns_ewma: Arc::new(AtomicU64::new(0)),
            max_concurrency,
        })
    }

    pub fn pressure_snapshot(&self) -> ObjectStorePressure {
        let active_requests = self
            .max_concurrency
            .saturating_sub(self.request_budget.available_permits());
        let outstanding_requests = self.outstanding_requests.load(Ordering::Acquire);
        ObjectStorePressure {
            max_concurrency: self.max_concurrency,
            active_requests,
            outstanding_requests,
            queued_requests: outstanding_requests.saturating_sub(active_requests),
            service_time_ns_ewma: self.service_time_ns_ewma.load(Ordering::Acquire),
        }
    }

    fn validate(ranges: &[ObjectRange]) -> Result<(), Box<dyn std::error::Error>> {
        for range in ranges {
            if range.object_key.is_empty()
                || range.length == 0
                || range.offset.checked_add(range.length).is_none()
            {
                return Err(format!(
                    "invalid S3 object range: object={:?}, offset={}, length={}",
                    range.object_key, range.offset, range.length
                )
                .into());
            }
        }
        Ok(())
    }

    pub fn head_object(&self, object_key: &str) -> Result<ObjectHead, Box<dyn std::error::Error>> {
        if object_key.is_empty() {
            return Err("S3 object key must be non-empty".into());
        }
        let store = Arc::clone(&self.store);
        let key = ObjectPath::from(object_key);
        let meta = self
            .runtime
            .block_on(async move { store.head(&key).await })?;
        Ok(ObjectHead {
            object_key: object_key.to_string(),
            size: meta.size,
            e_tag: meta.e_tag,
            version: meta.version,
        })
    }

    pub fn open_object_stream(
        &self,
        object_key: &str,
        buffered_chunks: usize,
    ) -> Result<S3ObjectStream, Box<dyn std::error::Error>> {
        if object_key.is_empty() {
            return Err("S3 object key must be non-empty".into());
        }
        if buffered_chunks == 0 {
            return Err("stream buffered_chunks must be positive".into());
        }
        let store = Arc::clone(&self.store);
        let request_budget = Arc::clone(&self.request_budget);
        let outstanding = OutstandingRequestGuard::new(Arc::clone(&self.outstanding_requests));
        let runtime = Arc::clone(&self.runtime);
        let key = ObjectPath::from(object_key);
        let (sender, receiver) = tokio::sync::mpsc::channel(buffered_chunks);
        let bytes_received = Arc::new(AtomicU64::new(0));
        let task_bytes_received = Arc::clone(&bytes_received);
        self.runtime.spawn(async move {
            let _outstanding = outstanding;
            let result = async {
                let _permit = request_budget
                    .acquire_owned()
                    .await
                    .map_err(|error| error.to_string())?;
                let response = store.get(&key).await.map_err(|error| error.to_string())?;
                let mut stream = response.into_stream();
                while let Some(chunk) = stream.next().await {
                    let bytes = chunk.map_err(|error| error.to_string())?.to_vec();
                    task_bytes_received.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                    if sender.send(Ok(bytes)).await.is_err() {
                        return Ok::<(), String>(());
                    }
                }
                Ok::<(), String>(())
            }
            .await;
            if let Err(error) = result {
                let _ = sender.send(Err(error)).await;
            }
        });
        Ok(S3ObjectStream {
            runtime,
            receiver,
            bytes_received,
        })
    }

    pub fn for_each_object_range(
        &self,
        ranges: &[ObjectRange],
        callback: &mut dyn FnMut(CompletedRange) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Self::validate(ranges)?;
        let store = Arc::clone(&self.store);
        let request_budget = Arc::clone(&self.request_budget);
        let outstanding_requests = Arc::clone(&self.outstanding_requests);
        let service_time_ns_ewma = Arc::clone(&self.service_time_ns_ewma);
        let requested = ranges.to_vec();
        let max_concurrency = self.max_concurrency;
        let started = Instant::now();
        self.runtime.block_on(async move {
            let mut requests = JoinSet::new();
            let mut pending = requested.into_iter().enumerate();
            let spawn_next = |requests: &mut JoinSet<_>,
                              pending: &mut std::iter::Enumerate<
                std::vec::IntoIter<ObjectRange>,
            >| {
                let Some((index, range)) = pending.next() else {
                    return false;
                };
                let store = Arc::clone(&store);
                let request_budget = Arc::clone(&request_budget);
                let outstanding = OutstandingRequestGuard::new(Arc::clone(&outstanding_requests));
                requests.spawn(async move {
                    let _outstanding = outstanding;
                    let _permit = request_budget.acquire_owned().await.map_err(|error| {
                        object_store::Error::Generic {
                            store: "S3ObjectStoreClient",
                            source: error.into(),
                        }
                    })?;
                    let range_started = started.elapsed().as_nanos() as u64;
                    let key = ObjectPath::from(range.object_key);
                    let end = range.offset + range.length;
                    let options = GetOptions::new().with_range(Some(range.offset..end));
                    let result = store.get_opts(&key, options).await?;
                    let mut stream = result.into_stream();
                    let first = stream.next().await.transpose()?.ok_or_else(|| {
                        object_store::Error::Generic {
                            store: "S3ObjectStoreClient",
                            source: "range response contained no byte chunks".into(),
                        }
                    })?;
                    let first_byte_ns = started.elapsed().as_nanos() as u64;
                    let mut bytes = Vec::with_capacity(range.length as usize);
                    bytes.extend_from_slice(&first);
                    while let Some(chunk) = stream.next().await {
                        bytes.extend_from_slice(&chunk?);
                    }
                    if bytes.len() != range.length as usize {
                        return Err(object_store::Error::Generic {
                            store: "S3ObjectStoreClient",
                            source: format!(
                                "short range read at {}: {} != {}",
                                range.offset,
                                bytes.len(),
                                range.length
                            )
                            .into(),
                        });
                    }
                    Ok::<_, object_store::Error>(CompletedRange {
                        index,
                        physical_requests: 1,
                        physical_fetched_bytes: bytes.len() as u64,
                        bytes,
                        started_ns: range_started,
                        first_byte_ns,
                        completed_ns: started.elapsed().as_nanos() as u64,
                    })
                });
                true
            };
            for _ in 0..max_concurrency {
                if !spawn_next(&mut requests, &mut pending) {
                    break;
                }
            }
            while let Some(completed) = requests.join_next().await {
                let completed = completed??;
                let sample = completed.completed_ns.saturating_sub(completed.started_ns);
                let mut previous = service_time_ns_ewma.load(Ordering::Acquire);
                loop {
                    let next = if previous == 0 {
                        sample
                    } else {
                        previous.saturating_mul(7).saturating_add(sample) / 8
                    };
                    match service_time_ns_ewma.compare_exchange_weak(
                        previous,
                        next,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => break,
                        Err(observed) => previous = observed,
                    }
                }
                // Refill the bounded queue before invoking a potentially
                // expensive decode callback. This preserves transport/decode
                // overlap while honoring the planner's input order.
                spawn_next(&mut requests, &mut pending);
                callback(completed)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    pub fn fetch_object_ranges(
        &self,
        ranges: &[ObjectRange],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let mut result: Vec<Option<Vec<u8>>> =
            std::iter::repeat_with(|| None).take(ranges.len()).collect();
        self.for_each_object_range(ranges, &mut |completed| {
            result[completed.index] = Some(completed.bytes);
            Ok(())
        })?;
        result
            .into_iter()
            .map(|value| value.ok_or_else(|| "S3 client omitted a requested range".into()))
            .collect()
    }

    /// Fetch ranges through the object_store crate's native vectored-read path.
    ///
    /// This is intentionally separate from VClasp's explicit range planner:
    /// `ObjectStore::get_ranges` applies the crate's established 1-MiB
    /// coalescing distance and internal request concurrency. It is used as a
    /// generic I/O baseline, not as a VClasp mechanism.
    pub fn fetch_object_ranges_vectored(
        &self,
        ranges: &[ObjectRange],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        Self::validate(ranges)?;
        if ranges.is_empty() {
            return Ok(Vec::new());
        }

        let mut grouped = BTreeMap::<String, Vec<(usize, std::ops::Range<u64>)>>::new();
        for (index, range) in ranges.iter().enumerate() {
            grouped
                .entry(range.object_key.clone())
                .or_default()
                .push((index, range.offset..range.offset + range.length));
        }

        let store = Arc::clone(&self.store);
        let request_budget = Arc::clone(&self.request_budget);
        let outstanding_requests = Arc::clone(&self.outstanding_requests);
        let fetched = self.runtime.block_on(async move {
            let mut tasks = JoinSet::new();
            for (object_key, entries) in grouped {
                let store = Arc::clone(&store);
                let request_budget = Arc::clone(&request_budget);
                let outstanding = OutstandingRequestGuard::new(Arc::clone(&outstanding_requests));
                tasks.spawn(async move {
                    let _outstanding = outstanding;
                    // Bound independent objects. The vectored-read call retains
                    // object_store's own documented per-object scheduling.
                    let _permit = request_budget.acquire_owned().await.map_err(|error| {
                        object_store::Error::Generic {
                            store: "S3ObjectStoreClient",
                            source: error.into(),
                        }
                    })?;
                    let byte_ranges = entries
                        .iter()
                        .map(|(_, range)| range.clone())
                        .collect::<Vec<_>>();
                    let buffers = store
                        .get_ranges(&ObjectPath::from(object_key), &byte_ranges)
                        .await?;
                    if buffers.len() != entries.len() {
                        return Err(object_store::Error::Generic {
                            store: "S3ObjectStoreClient",
                            source: "vectored read omitted a requested range".into(),
                        });
                    }
                    entries
                        .into_iter()
                        .zip(buffers)
                        .map(|((index, range), bytes)| {
                            let expected = (range.end - range.start) as usize;
                            if bytes.len() != expected {
                                return Err(object_store::Error::Generic {
                                    store: "S3ObjectStoreClient",
                                    source: format!(
                                        "short vectored range read at {}: {} != {}",
                                        range.start,
                                        bytes.len(),
                                        expected
                                    )
                                    .into(),
                                });
                            }
                            Ok((index, bytes.to_vec()))
                        })
                        .collect::<Result<Vec<_>, object_store::Error>>()
                });
            }

            let mut completed = Vec::with_capacity(ranges.len());
            while let Some(result) = tasks.join_next().await {
                completed.extend(result??);
            }
            Ok::<_, Box<dyn std::error::Error>>(completed)
        })?;

        let mut result: Vec<Option<Vec<u8>>> =
            std::iter::repeat_with(|| None).take(ranges.len()).collect();
        for (index, bytes) in fetched {
            result[index] = Some(bytes);
        }
        result
            .into_iter()
            .map(|value| value.ok_or_else(|| "vectored read omitted a requested range".into()))
            .collect()
    }
}

pub struct VClaspS3Client {
    inner: S3ObjectStoreClient,
}

#[repr(C)]
pub struct VClaspObjectRange {
    pub object_key: *const c_char,
    pub offset: u64,
    pub length: u64,
}

#[repr(C)]
pub struct VClaspBuffer {
    pub data: *mut u8,
    pub length: usize,
}

unsafe fn required_string(value: *const c_char, name: &str) -> Result<String, String> {
    if value.is_null() {
        return Err(format!("{name} must not be null"));
    }
    CStr::from_ptr(value)
        .to_str()
        .map(str::to_owned)
        .map_err(|error| format!("{name} is not UTF-8: {error}"))
}

unsafe fn clear_error(error_out: *mut *mut c_char) {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
}

unsafe fn set_error(error_out: *mut *mut c_char, message: impl Into<String>) {
    if !error_out.is_null() {
        let message = message.into().replace('\0', " ");
        *error_out = CString::new(message).expect("NUL bytes removed").into_raw();
    }
}

#[no_mangle]
pub unsafe extern "C" fn vclasp_s3_client_new(
    endpoint: *const c_char,
    bucket: *const c_char,
    access_key_id: *const c_char,
    secret_access_key: *const c_char,
    region: *const c_char,
    max_concurrency: usize,
    error_out: *mut *mut c_char,
) -> *mut VClaspS3Client {
    clear_error(error_out);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let inner = S3ObjectStoreClient::new(
            required_string(endpoint, "endpoint")?,
            required_string(bucket, "bucket")?,
            required_string(access_key_id, "access_key_id")?,
            required_string(secret_access_key, "secret_access_key")?,
            required_string(region, "region")?,
            max_concurrency,
        )
        .map_err(|error| error.to_string())?;
        Ok::<_, String>(Box::into_raw(Box::new(VClaspS3Client { inner })))
    }));
    match result {
        Ok(Ok(client)) => client,
        Ok(Err(error)) => {
            set_error(error_out, error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error(error_out, "panic while constructing Rust S3 client");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vclasp_s3_client_free(client: *mut VClaspS3Client) {
    if !client.is_null() {
        drop(Box::from_raw(client));
    }
}

#[no_mangle]
pub unsafe extern "C" fn vclasp_s3_fetch_ranges(
    client: *mut VClaspS3Client,
    ranges: *const VClaspObjectRange,
    range_count: usize,
    buffers_out: *mut *mut VClaspBuffer,
    wall_ns_out: *mut u64,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    if !buffers_out.is_null() {
        *buffers_out = ptr::null_mut();
    }
    if !wall_ns_out.is_null() {
        *wall_ns_out = 0;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if client.is_null() || buffers_out.is_null() {
            return Err("client and buffers_out must not be null".to_string());
        }
        if range_count > 0 && ranges.is_null() {
            return Err("ranges must not be null when range_count is positive".to_string());
        }
        let input = if range_count == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(ranges, range_count)
        };
        let owned = input
            .iter()
            .map(|range| {
                Ok(ObjectRange {
                    object_key: required_string(range.object_key, "object_key")?,
                    offset: range.offset,
                    length: range.length,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let started = Instant::now();
        let fetched = (*client)
            .inner
            .fetch_object_ranges(&owned)
            .map_err(|error| error.to_string())?;
        if !wall_ns_out.is_null() {
            *wall_ns_out = started.elapsed().as_nanos() as u64;
        }
        let mut output = fetched
            .into_iter()
            .map(|bytes| {
                let mut bytes = bytes.into_boxed_slice();
                let buffer = VClaspBuffer {
                    data: bytes.as_mut_ptr(),
                    length: bytes.len(),
                };
                std::mem::forget(bytes);
                buffer
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        if !output.is_empty() {
            *buffers_out = output.as_mut_ptr();
            std::mem::forget(output);
        }
        Ok::<_, String>(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_error(error_out, error);
            1
        }
        Err(_) => {
            set_error(error_out, "panic while fetching Rust S3 ranges");
            2
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vclasp_buffers_free(buffers: *mut VClaspBuffer, count: usize) {
    if buffers.is_null() {
        return;
    }
    let buffers = Box::from_raw(ptr::slice_from_raw_parts_mut(buffers, count));
    for buffer in buffers.iter() {
        if !buffer.data.is_null() {
            drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
                buffer.data,
                buffer.length,
            )));
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vclasp_error_free(error: *mut c_char) {
    if !error.is_null() {
        drop(CString::from_raw(error));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_ranges_before_io() {
        for range in [
            ObjectRange {
                object_key: String::new(),
                offset: 0,
                length: 1,
            },
            ObjectRange {
                object_key: "object".to_string(),
                offset: 0,
                length: 0,
            },
            ObjectRange {
                object_key: "object".to_string(),
                offset: u64::MAX,
                length: 1,
            },
        ] {
            assert!(S3ObjectStoreClient::validate(&[range]).is_err());
        }
    }

    #[test]
    fn cloned_clients_share_transport_and_request_budget() {
        let client = S3ObjectStoreClient::new(
            "http://127.0.0.1:1".to_string(),
            "test-bucket".to_string(),
            "access".to_string(),
            "secret".to_string(),
            "us-east-1".to_string(),
            3,
        )
        .expect("client construction must not contact the endpoint");
        let clone = client.clone();

        assert!(Arc::ptr_eq(&client.store, &clone.store));
        assert!(Arc::ptr_eq(&client.runtime, &clone.runtime));
        assert!(Arc::ptr_eq(&client.request_budget, &clone.request_budget));
        assert!(Arc::ptr_eq(
            &client.outstanding_requests,
            &clone.outstanding_requests
        ));
        assert!(Arc::ptr_eq(
            &client.service_time_ns_ewma,
            &clone.service_time_ns_ewma
        ));
        assert_eq!(client.request_budget.available_permits(), 3);
    }

    #[test]
    fn pressure_snapshot_separates_active_and_queued_requests() {
        let client = S3ObjectStoreClient::new(
            "http://127.0.0.1:1".to_string(),
            "test-bucket".to_string(),
            "access".to_string(),
            "secret".to_string(),
            "us-east-1".to_string(),
            1,
        )
        .expect("client construction must not contact the endpoint");
        let active = OutstandingRequestGuard::new(Arc::clone(&client.outstanding_requests));
        let permit = client
            .runtime
            .block_on(Arc::clone(&client.request_budget).acquire_owned())
            .unwrap();
        let queued = OutstandingRequestGuard::new(Arc::clone(&client.outstanding_requests));

        assert_eq!(
            client.pressure_snapshot(),
            ObjectStorePressure {
                max_concurrency: 1,
                active_requests: 1,
                outstanding_requests: 2,
                queued_requests: 1,
                service_time_ns_ewma: 0,
            }
        );

        drop(queued);
        drop(permit);
        drop(active);
        assert_eq!(
            client.pressure_snapshot(),
            ObjectStorePressure {
                max_concurrency: 1,
                active_requests: 0,
                outstanding_requests: 0,
                queued_requests: 0,
                service_time_ns_ewma: 0,
            }
        );
    }

    #[test]
    fn head_rejects_an_empty_object_key_before_io() {
        let client = S3ObjectStoreClient::new(
            "http://127.0.0.1:1".to_string(),
            "test-bucket".to_string(),
            "access".to_string(),
            "secret".to_string(),
            "us-east-1".to_string(),
            1,
        )
        .expect("client construction must not contact the endpoint");

        assert!(client.head_object("").is_err());
    }
}
