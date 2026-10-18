use super::*;

#[derive(Clone)]
pub(super) struct IncrementalDecodeJob {
    pub(super) consumer_sample_ids: Vec<u64>,
    pub(super) record_ids: Vec<u64>,
    pub(super) requested: Vec<(u64, usize)>,
    pub(super) codec_config: Vec<u8>,
    pub(super) encoded_access_units: Vec<bytes::Bytes>,
    pub(super) access_units: usize,
    pub(super) memory: Option<Arc<crate::byte_budget::BytePermit>>,
}

pub(super) struct IncrementalDecodedJob {
    pub(super) frames: Vec<(u64, DecodedRgbFrame)>,
    pub(super) completed_ns: u64,
    pub(super) access_units: usize,
    pub(super) metrics: IncrementalExecutionMetrics,
    _memory: Option<Arc<crate::byte_budget::BytePermit>>,
}

pub(super) fn requested_display_ordinals(
    mut records: Vec<&HierarchicalRecordMeta>,
    target_ids: &HashSet<u64>,
) -> Vec<(u64, usize)> {
    records.sort_unstable_by_key(|record| record.frame_idx);
    records
        .into_iter()
        .enumerate()
        .filter_map(|(ordinal, record)| {
            target_ids
                .contains(&record.record_id)
                .then_some((record.record_id, ordinal))
        })
        .collect()
}

pub(super) fn build_incremental_decode_job<F>(
    codec_config: &[u8],
    mut records: Vec<&HierarchicalRecordMeta>,
    target_ids: &HashSet<u64>,
    mut load_annex_b: F,
) -> Result<IncrementalDecodeJob, String>
where
    F: FnMut(&HierarchicalRecordMeta) -> Result<bytes::Bytes, String>,
{
    records.sort_unstable_by_key(|record| record.decode_ordinal);
    let encoded_access_units = records
        .iter()
        .map(|record| load_annex_b(record))
        .collect::<Result<Vec<_>, _>>()?;
    let requested = requested_display_ordinals(records.clone(), target_ids);
    if requested.is_empty() {
        return Err("decode job does not contain a requested target".to_string());
    }
    Ok(IncrementalDecodeJob {
        consumer_sample_ids: Vec::new(),
        record_ids: records.iter().map(|record| record.record_id).collect(),
        requested,
        codec_config: codec_config.to_vec(),
        encoded_access_units,
        access_units: records.len(),
        memory: None,
    })
}

pub(super) enum IncrementalPipelineEvent {
    Range(crate::backend::CompletedRange),
    FetchFinished {
        result: Result<(), String>,
        blocked_sends: u64,
        blocked_ns: u64,
    },
    Decoded(Result<(IncrementalDecodedJob, u64), String>),
}

pub(super) struct IncrementalDecodeWork {
    pub(super) submitted_ns: u64,
    pub(super) job: IncrementalDecodeJob,
    pub(super) total_started: Instant,
    pub(super) completion: mpsc::SyncSender<IncrementalPipelineEvent>,
    pub(super) submitted_at: Instant,
}

pub(super) struct IncrementalDecodePool {
    pub(super) sender: Option<mpsc::Sender<IncrementalDecodeWork>>,
    pub(super) threads: Vec<JoinHandle<()>>,
}

impl IncrementalDecodePool {
    pub(super) fn new(
        slots: SharedDecoderSlots,
        decode_budget: Arc<DecodeBudget>,
        decoder_threads: usize,
        stage_profiling: Arc<decoder::profile::CodecProfiler>,
    ) -> Self {
        let (sender, receiver) = mpsc::channel::<IncrementalDecodeWork>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut threads = Vec::with_capacity(slots.len());
        for slot_index in 0..slots.len() {
            let receiver = Arc::clone(&receiver);
            let slots = Arc::clone(&slots);
            let decode_budget = Arc::clone(&decode_budget);
            let stage_profiling = Arc::clone(&stage_profiling);
            threads.push(std::thread::spawn(move || loop {
                let work = receiver
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .recv();
                let Ok(work) = work else {
                    break;
                };
                let queue_ns = work.submitted_at.elapsed().as_nanos() as u64;
                let profile_enabled = stage_profiling.enabled();
                let worker_started_ns = if profile_enabled {
                    decoder::profile::monotonic_ns()
                } else {
                    0
                };
                let decode_started = Instant::now();
                let result = (|| {
                    let _permit = decode_budget.acquire(decoder_threads)?;
                    let mut pool = slots[slot_index]
                        .lock()
                        .map_err(|_| "hierarchical decoder slot lock poisoned".to_string())?;
                    let resource_wait_ns = decode_started.elapsed().as_nanos() as u64;
                    let ordinals = work
                        .job
                        .requested
                        .iter()
                        .map(|(_, ordinal)| *ordinal)
                        .collect::<Vec<_>>();
                    let service_started = Instant::now();
                    let resource_acquired_ns = if profile_enabled {
                        decoder::profile::monotonic_ns()
                    } else {
                        0
                    };
                    let profile = decoder::profile::ProfileScope::start(profile_enabled);
                    let frames = decoder::decode_access_units_selected_rgb24(
                        &work.job.codec_config,
                        &work.job.encoded_access_units,
                        &mut pool,
                        &ordinals,
                        false,
                    )
                    .map_err(|error| error.to_string())?;
                    let service_ns = service_started.elapsed().as_nanos() as u64;
                    if profile_enabled {
                        stage_profiling.record(decoder::profile::DecodeJobEvent {
                            job_id: 0,
                            record_ids: work.job.record_ids.clone(),
                            target_record_ids: work
                                .job
                                .requested
                                .iter()
                                .map(|(id, _)| *id)
                                .collect(),
                            consumer_sample_ids: work.job.consumer_sample_ids.clone(),
                            execution_path: "stateless_decoder",
                            submitted_ns: work.submitted_ns,
                            worker_started_ns,
                            resource_acquired_ns,
                            completed_ns: decoder::profile::monotonic_ns(),
                            profile: profile.snapshot(),
                        });
                    }
                    if frames.len() != work.job.requested.len() {
                        return Err(format!(
                            "decoded {} hierarchical targets from {} requests",
                            frames.len(),
                            work.job.requested.len()
                        ));
                    }
                    let decoded_rgb_bytes =
                        frames.iter().map(|frame| frame.data.len() as u64).sum();
                    Ok(IncrementalDecodedJob {
                        _memory: work.job.memory,
                        frames: work
                            .job
                            .requested
                            .into_iter()
                            .zip(frames)
                            .map(|((record_id, _), frame)| (record_id, frame))
                            .collect(),
                        completed_ns: work.total_started.elapsed().as_nanos() as u64,
                        access_units: work.job.access_units,
                        metrics: IncrementalExecutionMetrics {
                            codec: profile.snapshot(),
                            decode_queue_ns: queue_ns,
                            decode_resource_wait_ns: resource_wait_ns,
                            decode_service_ns: service_ns,
                            decode_timing_samples: 1,
                            decoded_rgb_bytes,
                            ..Default::default()
                        },
                    })
                })();
                let elapsed_ns = decode_started.elapsed().as_nanos() as u64;
                if work
                    .completion
                    .send(IncrementalPipelineEvent::Decoded(
                        result.map(|decoded| (decoded, elapsed_ns)),
                    ))
                    .is_err()
                {
                    continue;
                }
            }));
        }
        Self {
            sender: Some(sender),
            threads,
        }
    }

    pub(super) fn submit(
        &self,
        job: IncrementalDecodeJob,
        total_started: Instant,
        completion: mpsc::SyncSender<IncrementalPipelineEvent>,
    ) -> Result<(), String> {
        self.sender
            .as_ref()
            .ok_or("incremental decoder pool is closed")?
            .send(IncrementalDecodeWork {
                submitted_ns: decoder::profile::monotonic_ns(),
                job,
                total_started,
                completion,
                submitted_at: Instant::now(),
            })
            .map_err(|_| "incremental decoder worker stopped early".to_string())
    }

    pub(super) fn decode_all(
        &self,
        jobs: Vec<IncrementalDecodeJob>,
        total_started: Instant,
    ) -> Result<Vec<IncrementalDecodedJob>, String> {
        let job_count = jobs.len();
        let (sender, receiver) = mpsc::sync_channel(job_count.max(1));
        for job in jobs {
            self.submit(job, total_started, sender.clone())?;
        }
        drop(sender);

        let mut decoded = Vec::with_capacity(job_count);
        while decoded.len() < job_count {
            match receiver
                .recv()
                .map_err(|_| "parallel decoder workers stopped early".to_string())?
            {
                IncrementalPipelineEvent::Decoded(Ok((job, _))) => decoded.push(job),
                IncrementalPipelineEvent::Decoded(Err(error)) => return Err(error),
                IncrementalPipelineEvent::Range(_)
                | IncrementalPipelineEvent::FetchFinished { .. } => {
                    return Err("decode-only pool returned an unexpected I/O event".to_string())
                }
            }
        }
        Ok(decoded)
    }
}

impl Drop for IncrementalDecodePool {
    fn drop(&mut self) {
        self.sender.take();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_workers_pull_jobs_while_another_slot_is_blocked() {
        let mut encoder = crate::encoder::X264Encoder::new(32, 32, 23, 16, false);
        let encoded = encoder.encode_frame(&vec![128; 32 * 32 * 3 / 2], true);
        let mut encoded = [encoded, encoder.flush()].concat();
        let (config, record, _) = decoder::extract_closed_record_parts(&encoded).unwrap();
        encoded.clear();
        let slots = decoder::shared_decoder_slots_with_threads(2, 1);
        let pool = IncrementalDecodePool::new(
            Arc::clone(&slots),
            DecodeBudget::new(2),
            1,
            Arc::new(decoder::profile::CodecProfiler::default()),
        );
        // Declared after pool so unwinding unlocks the slot before pool joins.
        let blocked = slots[0].lock().unwrap();
        let job = IncrementalDecodeJob {
            consumer_sample_ids: vec![0],
            record_ids: vec![0],
            requested: vec![(0, 0)],
            codec_config: config,
            encoded_access_units: vec![bytes::Bytes::from(record)],
            access_units: 1,
            memory: None,
        };
        let (sender, receiver) = mpsc::sync_channel(3);
        for _ in 0..3 {
            pool.submit(job.clone(), Instant::now(), sender.clone())
                .unwrap();
        }
        for _ in 0..2 {
            assert!(matches!(
                receiver
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .unwrap(),
                IncrementalPipelineEvent::Decoded(Ok(_))
            ));
        }
        drop(blocked);
        assert!(matches!(
            receiver
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap(),
            IncrementalPipelineEvent::Decoded(Ok(_))
        ));
    }
}
