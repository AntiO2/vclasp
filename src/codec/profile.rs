//! Optional, thread-local codec-call accounting. CPU time excludes FFmpeg's
//! internal worker threads; process CPU accounting must be reported separately.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Mutex,
};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DecodeJobEvent {
    pub job_id: u64,
    pub record_ids: Vec<u64>,
    pub target_record_ids: Vec<u64>,
    pub consumer_sample_ids: Vec<u64>,
    pub execution_path: &'static str,
    pub submitted_ns: u64,
    pub worker_started_ns: u64,
    pub resource_acquired_ns: u64,
    pub completed_ns: u64,
    pub profile: DecodeProfile,
}

pub struct CodecProfiler {
    enabled: AtomicBool,
    next_job: AtomicU64,
    buffer: Mutex<(VecDeque<DecodeJobEvent>, usize, u64)>,
}

impl Default for CodecProfiler {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            next_job: AtomicU64::new(0),
            buffer: Mutex::new((VecDeque::new(), 8192, 0)),
        }
    }
}

impl CodecProfiler {
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub(crate) fn record(&self, mut event: DecodeJobEvent) {
        if !self.enabled() {
            return;
        }
        event.job_id = self.next_job.fetch_add(1, Ordering::Relaxed);
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if buffer.0.len() < buffer.1 {
            buffer.0.push_back(event);
        } else {
            buffer.2 += 1;
        }
    }

    pub fn drain(&self) -> (Vec<DecodeJobEvent>, u64) {
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        (buffer.0.drain(..).collect(), std::mem::take(&mut buffer.2))
    }
}

pub(crate) fn monotonic_ns() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let mut time = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } == 0 {
            return time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64;
        }
    }
    0 // Unsupported clocks are explicitly reported as unavailable, not aligned.
}

pub const STAGES: [&str; 7] = [
    "decoder_setup",
    "packet_copy",
    "send",
    "receive",
    "flush",
    "rgb_convert",
    "rgb_copy",
];

#[derive(Clone, Copy)]
pub(crate) enum Stage {
    DecoderSetup,
    PacketCopy,
    CodecSend,
    CodecReceive,
    CodecFlush,
    RgbConvert,
    RgbCopy,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StageTotals {
    pub wall_ns: u64,
    pub thread_cpu_ns: u64,
    pub calls: u64,
    pub cpu_samples: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeProfile {
    pub stages: [StageTotals; 7],
    pub jobs: u64,
    pub accepted_packets: u64,
    pub received_frames: u64,
    pub decoder_opens: u64,
    pub context_address: u64,
}

impl DecodeProfile {
    pub fn accumulate(&mut self, other: &Self) {
        self.jobs += other.jobs;
        self.accepted_packets += other.accepted_packets;
        self.received_frames += other.received_frames;
        self.decoder_opens += other.decoder_opens;
        for (left, right) in self.stages.iter_mut().zip(&other.stages) {
            left.wall_ns += right.wall_ns;
            left.thread_cpu_ns += right.thread_cpu_ns;
            left.calls += right.calls;
            left.cpu_samples += right.cpu_samples;
            left.bytes += right.bytes;
        }
    }

    pub fn metric_pairs(&self) -> Vec<(String, u64)> {
        let mut pairs = vec![
            ("codec_profile_jobs".into(), self.jobs),
            ("codec_accepted_packets".into(), self.accepted_packets),
            ("codec_received_frames".into(), self.received_frames),
            ("codec_decoder_opens".into(), self.decoder_opens),
        ];
        for (name, totals) in STAGES.iter().zip(&self.stages) {
            for (suffix, value) in [
                ("wall_ns", totals.wall_ns),
                ("thread_cpu_ns", totals.thread_cpu_ns),
                ("calls", totals.calls),
                ("cpu_samples", totals.cpu_samples),
                ("bytes", totals.bytes),
            ] {
                pairs.push((format!("codec_{name}_{suffix}"), value));
            }
        }
        pairs
    }
}

thread_local! {
    static CURRENT: RefCell<Option<DecodeProfile>> = const { RefCell::new(None) };
}

pub(crate) fn accepted_packet() {
    CURRENT.with(|current| {
        if let Some(profile) = current.borrow_mut().as_mut() {
            profile.accepted_packets += 1;
        }
    });
}

pub(crate) fn received_frame() {
    CURRENT.with(|current| {
        if let Some(profile) = current.borrow_mut().as_mut() {
            profile.received_frames += 1;
        }
    });
}

pub(crate) fn decoder_opened() {
    CURRENT.with(|current| {
        if let Some(profile) = current.borrow_mut().as_mut() {
            profile.decoder_opens += 1;
        }
    });
}

pub(crate) fn set_context_address(address: u64) {
    CURRENT.with(|current| {
        if let Some(profile) = current.borrow_mut().as_mut() {
            profile.context_address = address;
        }
    });
}

#[cfg(target_os = "linux")]
pub(crate) fn thread_cpu_ns() -> Option<u64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // CLOCK_THREAD_CPUTIME_ID is local to this thread and never blocks on I/O.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) } == 0 {
        Some(time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64)
    } else {
        None
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn thread_cpu_ns() -> Option<u64> {
    None
}

pub(crate) struct ProfileScope(Option<DecodeProfile>);

impl ProfileScope {
    pub(crate) fn start(enabled: bool) -> Self {
        Self(CURRENT.with(|current| {
            current.replace(enabled.then(|| DecodeProfile {
                jobs: 1,
                ..Default::default()
            }))
        }))
    }

    pub(crate) fn snapshot(&self) -> DecodeProfile {
        CURRENT.with(|current| current.borrow().unwrap_or_default())
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        CURRENT.with(|current| current.replace(self.0.take()));
    }
}

pub(crate) struct StageScope {
    stage: Stage,
    started: Option<Instant>,
    cpu: Option<u64>,
    bytes: u64,
}

impl StageScope {
    pub(crate) fn new(stage: Stage, bytes: usize) -> Self {
        let enabled = CURRENT.with(|current| current.borrow().is_some());
        Self {
            stage,
            started: enabled.then(Instant::now),
            cpu: enabled.then(thread_cpu_ns).flatten(),
            bytes: bytes as u64,
        }
    }
}

impl Drop for StageScope {
    fn drop(&mut self) {
        let Some(started) = self.started else { return };
        let elapsed = started.elapsed().as_nanos() as u64;
        let cpu = self
            .cpu
            .zip(thread_cpu_ns())
            .map(|(a, b)| b.saturating_sub(a));
        CURRENT.with(|current| {
            if let Some(profile) = current.borrow_mut().as_mut() {
                let totals = &mut profile.stages[self.stage as usize];
                totals.wall_ns += elapsed;
                totals.calls += 1;
                totals.bytes += self.bytes;
                if let Some(cpu) = cpu {
                    totals.thread_cpu_ns += cpu;
                    totals.cpu_samples += 1;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_and_nested_profiles_do_not_leak() {
        let outer = ProfileScope::start(true);
        {
            let _stage = StageScope::new(Stage::PacketCopy, 17);
        }
        {
            let disabled = ProfileScope::start(false);
            {
                let _stage = StageScope::new(Stage::PacketCopy, 99);
            }
            assert_eq!(disabled.snapshot().jobs, 0);
        }
        assert_eq!(
            outer.snapshot().stages[Stage::PacketCopy as usize].bytes,
            17
        );
        assert_eq!(outer.snapshot().stages[Stage::PacketCopy as usize].calls, 1);
        drop(outer);
        assert!(CURRENT.with(|current| current.borrow().is_none()));
    }

    #[test]
    fn unwind_restores_thread_local_state() {
        let _ = std::panic::catch_unwind(|| {
            let _profile = ProfileScope::start(true);
            panic!("test unwind");
        });
        assert!(CURRENT.with(|current| current.borrow().is_none()));
    }

    #[test]
    fn job_trace_is_bounded_and_reports_loss() {
        let profiler = CodecProfiler::default();
        profiler.set_enabled(true);
        for _ in 0..8193 {
            profiler.record(DecodeJobEvent {
                job_id: 0,
                record_ids: vec![7],
                target_record_ids: vec![7],
                consumer_sample_ids: vec![1],
                execution_path: "stateless_decoder",
                submitted_ns: 1,
                worker_started_ns: 2,
                resource_acquired_ns: 3,
                completed_ns: 4,
                profile: DecodeProfile::default(),
            });
        }
        let (events, dropped) = profiler.drain();
        assert_eq!(events.len(), 8192);
        assert_eq!(dropped, 1);
        assert_eq!(events[0].job_id, 0);
        assert_eq!(events[8191].job_id, 8191);
        assert_eq!(profiler.drain().1, 0);
    }
}
