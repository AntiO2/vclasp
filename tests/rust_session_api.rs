#![cfg(feature = "ffmpeg")]

use std::time::Duration;

use vclasp::session::{
    CostModel, PipelineConfig, ResidentStateBudget, RuntimeFeedbackConfig, SessionConfig, Target,
};

fn valid_config() -> SessionConfig {
    SessionConfig {
        cost_model: CostModel {
            request_latency_ns: 1_000_000.0,
            bandwidth_bytes_per_ns: 1.0,
            io_concurrency: 8,
            wave_request_overhead_ns: vec![1_000_000.0; 8],
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 100_000.0,
            decode_access_unit_ns: 10_000.0,
            fetch_decode_overlap: 0.5,
        },
        runtime_feedback: RuntimeFeedbackConfig::default(),
        max_callers: 8,
        max_pending_calls: 32,
        max_inflight_windows: 2,
        admission_quiet: Duration::from_micros(250),
        max_merge_gap_bytes: Some(16 * 1024),
        max_range_bytes: None,
        decoder_threads: 1,
        global_decode_threads: 8,
        cursor_decoder_threads: 1,
        resident_state: ResidentStateBudget {
            encoded_bytes: 1024 * 1024,
            read_ahead_bytes: 64 * 1024,
            live_cursors: 8,
        },
    }
}

#[test]
fn rust_session_contract_is_public_and_backend_neutral() {
    let config = valid_config();
    config.validate().unwrap();
    let target = Target {
        sample_id: 7,
        video_id: "video-1".to_string(),
        frame_idx: 12,
    };
    assert_eq!(target.sample_id, 7);
}

#[test]
fn session_rejects_resources_that_cannot_be_enforced() {
    let mut config = valid_config();
    config.cursor_decoder_threads = 9;
    assert!(config.validate().is_err());

    let mut config = valid_config();
    config.resident_state.live_cursors = 64;
    assert!(config.validate().is_ok());
}

#[test]
fn pipeline_contract_rejects_unbounded_zero_capacity() {
    assert!(PipelineConfig {
        max_outstanding_batches: 0,
        max_outstanding_targets: None,
    }
    .validate()
    .is_err());
    assert!(PipelineConfig {
        max_outstanding_batches: 8,
        max_outstanding_targets: Some(256),
    }
    .validate()
    .is_ok());
}
