use std::collections::VecDeque;

use crate::hierarchical_layout::HierarchicalCostModel;

const IO_BYTE_SCALE: f64 = 64.0 * 1024.0;
const TAIL_HISTORY: usize = 128;
const TAIL_QUANTILE: f64 = 0.95;
const COLD_START_RISK_MULTIPLIER: f64 = 16.0;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeFeedbackConfig {
    pub enabled: bool,
    pub min_observations: usize,
    pub forgetting_factor: f64,
    pub huber_multiplier: f64,
    pub max_relative_update: f64,
    pub activation_ape_threshold: f64,
    pub learning_rate: f64,
    pub activation_stable_observations: usize,
}

impl Default for RuntimeFeedbackConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_observations: 16,
            forgetting_factor: 0.98,
            huber_multiplier: 3.0,
            max_relative_update: 0.5,
            activation_ape_threshold: 0.5,
            learning_rate: 0.25,
            activation_stable_observations: 4,
        }
    }
}

impl RuntimeFeedbackConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.min_observations == 0 {
            return Err("runtime feedback min_observations must be positive".to_string());
        }
        if !self.forgetting_factor.is_finite()
            || !(0.0..=1.0).contains(&self.forgetting_factor)
            || self.forgetting_factor == 0.0
        {
            return Err("runtime feedback forgetting_factor must be in (0, 1]".to_string());
        }
        if !self.huber_multiplier.is_finite() || self.huber_multiplier <= 0.0 {
            return Err("runtime feedback huber_multiplier must be positive".to_string());
        }
        if !self.max_relative_update.is_finite()
            || !(0.0..=1.0).contains(&self.max_relative_update)
            || self.max_relative_update == 0.0
        {
            return Err("runtime feedback max_relative_update must be in (0, 1]".to_string());
        }
        if !self.activation_ape_threshold.is_finite() || self.activation_ape_threshold <= 0.0 {
            return Err("runtime feedback activation_ape_threshold must be positive".to_string());
        }
        if !self.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&self.learning_rate)
            || self.learning_rate == 0.0
        {
            return Err("runtime feedback learning_rate must be in (0, 1]".to_string());
        }
        if self.activation_stable_observations == 0 {
            return Err(
                "runtime feedback activation_stable_observations must be positive".to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeFeedbackSnapshot {
    pub io_observations: usize,
    pub decode_observations: usize,
    pub rejected_observations: usize,
    pub active: bool,
    pub io_absolute_percentage_error_ppm: u64,
    pub decode_absolute_percentage_error_ppm: u64,
    pub io_tail_multiplier_ppm: u64,
    pub decode_tail_multiplier_ppm: u64,
}

#[derive(Debug, Clone)]
struct RobustOnlineRegression {
    theta: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    residual_scale: f64,
    observations: usize,
    absolute_percentage_error: f64,
    last_absolute_percentage_error: f64,
    accurate_streak: usize,
    latency_ratios: VecDeque<f64>,
}

impl RobustOnlineRegression {
    fn new(initial: Vec<f64>, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        debug_assert_eq!(initial.len(), lower.len());
        debug_assert_eq!(initial.len(), upper.len());
        Self {
            theta: initial,
            lower,
            upper,
            residual_scale: 0.0,
            observations: 0,
            absolute_percentage_error: 0.0,
            last_absolute_percentage_error: f64::INFINITY,
            accurate_streak: 0,
            latency_ratios: VecDeque::with_capacity(TAIL_HISTORY),
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        self.theta
            .iter()
            .zip(features)
            .map(|(coefficient, feature)| coefficient * feature)
            .sum::<f64>()
            .max(0.0)
    }

    fn observe(&mut self, features: &[f64], actual: f64, config: RuntimeFeedbackConfig) -> bool {
        if features.len() != self.theta.len()
            || features.iter().any(|value| !value.is_finite())
            || !actual.is_finite()
            || actual <= 0.0
        {
            return false;
        }

        let predicted = self.predict(features);
        if predicted > 0.0 {
            if self.latency_ratios.len() == TAIL_HISTORY {
                self.latency_ratios.pop_front();
            }
            // Bound one pathological observation without erasing ordinary
            // object-store tail behavior.
            self.latency_ratios
                .push_back((actual / predicted).clamp(0.25, 16.0));
        }
        let residual = actual - predicted;
        let seed_scale = (predicted.abs() * 0.25).max(1_000.0);
        let scale = if self.residual_scale > 0.0 {
            self.residual_scale
        } else {
            seed_scale
        };
        let limit = config.huber_multiplier * scale;
        let clipped_residual = residual.clamp(-limit, limit);
        let feature_norm = features.iter().map(|value| value * value).sum::<f64>() + 1e-9;
        let shared_scale = (predicted / feature_norm.sqrt()).abs().max(1_000.0);
        for (index, feature) in features.iter().copied().enumerate() {
            let current = self.theta[index];
            let proposed =
                current + config.learning_rate * clipped_residual * feature / feature_norm;
            let step_limit = config.max_relative_update * current.abs().max(shared_scale);
            self.theta[index] = proposed
                .clamp(current - step_limit, current + step_limit)
                .clamp(self.lower[index], self.upper[index]);
        }

        self.observations += 1;
        self.residual_scale = if self.observations == 1 {
            residual.abs().max(1_000.0)
        } else {
            config.forgetting_factor * self.residual_scale
                + (1.0 - config.forgetting_factor) * residual.abs()
        };
        let absolute_percentage_error = residual.abs() / actual.max(1.0);
        self.last_absolute_percentage_error = absolute_percentage_error;
        if absolute_percentage_error <= config.activation_ape_threshold {
            self.accurate_streak += 1;
        } else {
            self.accurate_streak = 0;
        }
        self.absolute_percentage_error = if self.observations == 1 {
            absolute_percentage_error
        } else {
            config.forgetting_factor * self.absolute_percentage_error
                + (1.0 - config.forgetting_factor) * absolute_percentage_error
        };
        true
    }

    fn tail_multiplier(&self, min_observations: usize) -> Option<f64> {
        if self.latency_ratios.len() < min_observations {
            return None;
        }
        let mut ratios = self.latency_ratios.iter().copied().collect::<Vec<_>>();
        ratios.sort_by(f64::total_cmp);
        let rank = ((ratios.len() as f64 * TAIL_QUANTILE).ceil() as usize)
            .saturating_sub(1)
            .min(ratios.len() - 1);
        Some(ratios[rank].max(1.0))
    }

    fn clear_tail_history(&mut self) {
        self.latency_ratios.clear();
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeCostFeedback {
    config: RuntimeFeedbackConfig,
    io_concurrency: usize,
    io: RobustOnlineRegression,
    decode: RobustOnlineRegression,
    rejected_observations: usize,
    activated: bool,
}

impl RuntimeCostFeedback {
    pub fn new(
        bootstrap: &HierarchicalCostModel,
        config: RuntimeFeedbackConfig,
    ) -> Result<Self, String> {
        bootstrap.validate()?;
        config.validate()?;

        let bootstrap_wave = if bootstrap.wave_request_overhead_ns.is_empty() {
            vec![bootstrap.request_latency_ns; bootstrap.io_concurrency]
        } else {
            bootstrap.wave_request_overhead_ns.clone()
        };
        let incremental_request_ns = if bootstrap_wave.len() > 1 {
            (bootstrap_wave[bootstrap_wave.len() - 1] - bootstrap_wave[0])
                * bootstrap_wave.len() as f64
                / (bootstrap_wave.len() - 1) as f64
        } else {
            0.0
        };
        let io_initial = vec![
            bootstrap_wave[0].max(1.0),
            incremental_request_ns.max(1.0),
            IO_BYTE_SCALE / bootstrap.bandwidth_bytes_per_ns,
        ];
        let io_lower = vec![1.0; io_initial.len()];
        let io_upper = io_initial
            .iter()
            .map(|value| (value * 50.0).max(1_000_000.0))
            .collect::<Vec<_>>();

        let decode_initial = vec![
            bootstrap.decode_fixed_ns.max(1.0),
            (bootstrap.decode_access_unit_ns * 16.0).max(1.0),
        ];
        let decode_lower = vec![1.0; decode_initial.len()];
        let decode_upper = decode_initial
            .iter()
            .map(|value| (value * 50.0).max(1_000_000.0))
            .collect::<Vec<_>>();

        Ok(Self {
            config,
            io_concurrency: bootstrap.io_concurrency,
            io: RobustOnlineRegression::new(io_initial, io_lower, io_upper),
            decode: RobustOnlineRegression::new(decode_initial, decode_lower, decode_upper),
            rejected_observations: 0,
            activated: false,
        })
    }

    pub fn observe(
        &mut self,
        physical_ranges: usize,
        fetched_bytes: u64,
        fetch_ns: u64,
        submitted_access_units: usize,
        decode_ns: u64,
    ) {
        if !self.config.enabled {
            return;
        }

        if physical_ranges > 0 && fetched_bytes > 0 && fetch_ns > 0 {
            let features = io_features(physical_ranges, fetched_bytes, self.io_concurrency);
            if !self.io.observe(&features, fetch_ns as f64, self.config) {
                self.rejected_observations += 1;
            }
        }
        if submitted_access_units > 0 && decode_ns > 0 {
            let features = [1.0, submitted_access_units as f64 / 16.0];
            if !self
                .decode
                .observe(&features, decode_ns as f64, self.config)
            {
                self.rejected_observations += 1;
            }
        }
        if !self.activated
            && self.io.observations >= self.config.min_observations
            && self.decode.observations >= self.config.min_observations
            && self.io.accurate_streak >= self.config.activation_stable_observations
            && self.decode.accurate_streak >= self.config.activation_stable_observations
        {
            self.activated = true;
            // Bootstrap residuals describe model convergence, not steady-state
            // object-store tail behavior.
            self.io.clear_tail_history();
            self.decode.clear_tail_history();
        }
    }

    pub fn apply_to(&self, bootstrap: &HierarchicalCostModel) -> HierarchicalCostModel {
        if !self.active() {
            return bootstrap.clone();
        }

        let wave_request_overhead_ns = (0..self.io_concurrency)
            .map(|index| {
                self.io.theta[0] + index as f64 / self.io_concurrency as f64 * self.io.theta[1]
            })
            .collect::<Vec<_>>();
        let bytes_per_ns = IO_BYTE_SCALE / self.io.theta[2].max(1.0);
        let mut model = bootstrap.clone();
        model.request_latency_ns = wave_request_overhead_ns[0];
        model.wave_request_overhead_ns = wave_request_overhead_ns;
        model.bandwidth_bytes_per_ns = bytes_per_ns;
        model.decode_fixed_ns = self.decode.theta[0];
        model.decode_access_unit_ns = self.decode.theta[1] / 16.0;
        model
    }

    /// Return a conservative model for latency-SLO admission. Expected costs
    /// continue to rank throughput; this model is used only to reject horizons
    /// whose first-batch latency is exposed to observed tail amplification.
    pub fn apply_tail_to(&self, bootstrap: &HierarchicalCostModel) -> HierarchicalCostModel {
        let mut model = self.apply_to(bootstrap);
        if !self.config.enabled {
            return model;
        }
        let io_multiplier = if self.active() {
            self.io
                .tail_multiplier(self.config.min_observations)
                .unwrap_or(COLD_START_RISK_MULTIPLIER)
        } else {
            COLD_START_RISK_MULTIPLIER
        };
        let decode_multiplier = if self.active() {
            self.decode
                .tail_multiplier(self.config.min_observations)
                .unwrap_or(COLD_START_RISK_MULTIPLIER)
        } else {
            COLD_START_RISK_MULTIPLIER
        };
        model.request_latency_ns *= io_multiplier;
        for overhead in &mut model.wave_request_overhead_ns {
            *overhead *= io_multiplier;
        }
        model.bandwidth_bytes_per_ns /= io_multiplier;
        model.decode_fixed_ns *= decode_multiplier;
        model.decode_access_unit_ns *= decode_multiplier;
        model
    }

    pub fn snapshot(&self) -> RuntimeFeedbackSnapshot {
        RuntimeFeedbackSnapshot {
            io_observations: self.io.observations,
            decode_observations: self.decode.observations,
            rejected_observations: self.rejected_observations,
            active: self.active(),
            io_absolute_percentage_error_ppm: ratio_to_ppm(self.io.last_absolute_percentage_error),
            decode_absolute_percentage_error_ppm: ratio_to_ppm(
                self.decode.last_absolute_percentage_error,
            ),
            io_tail_multiplier_ppm: ratio_to_ppm(
                self.io
                    .tail_multiplier(self.config.min_observations)
                    .unwrap_or_else(|| {
                        if self.config.enabled {
                            COLD_START_RISK_MULTIPLIER
                        } else {
                            1.0
                        }
                    }),
            ),
            decode_tail_multiplier_ppm: ratio_to_ppm(
                self.decode
                    .tail_multiplier(self.config.min_observations)
                    .unwrap_or_else(|| {
                        if self.config.enabled {
                            COLD_START_RISK_MULTIPLIER
                        } else {
                            1.0
                        }
                    }),
            ),
        }
    }

    fn active(&self) -> bool {
        self.config.enabled && self.activated
    }
}

fn io_features(range_count: usize, fetched_bytes: u64, io_concurrency: usize) -> Vec<f64> {
    let full_waves = range_count / io_concurrency;
    let remainder = range_count % io_concurrency;
    let wave_count = full_waves + usize::from(remainder > 0);
    let extra_requests = range_count.saturating_sub(wave_count);
    vec![
        wave_count as f64,
        extra_requests as f64 / io_concurrency as f64,
        fetched_bytes as f64 / IO_BYTE_SCALE,
    ]
}

fn ratio_to_ppm(value: f64) -> u64 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else {
        (value * 1_000_000.0).min(u64::MAX as f64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bootstrap() -> HierarchicalCostModel {
        HierarchicalCostModel {
            request_latency_ns: 1_000_000.0,
            bandwidth_bytes_per_ns: 0.1,
            io_concurrency: 4,
            wave_request_overhead_ns: Vec::new(),
            selection_tolerance_ns: 0.0,
            decode_fixed_ns: 100_000.0,
            decode_access_unit_ns: 20_000.0,
            fetch_decode_overlap: 0.0,
        }
    }

    #[test]
    fn feedback_activates_only_after_minimum_observations() {
        let config = RuntimeFeedbackConfig {
            min_observations: 8,
            activation_ape_threshold: 10.0,
            ..RuntimeFeedbackConfig::default()
        };
        let mut feedback = RuntimeCostFeedback::new(&bootstrap(), config).unwrap();
        for _ in 0..7 {
            feedback.observe(4, 64 * 1024, 1_200_000, 8, 500_000);
        }
        assert!(!feedback.snapshot().active);
        feedback.observe(4, 64 * 1024, 1_200_000, 8, 500_000);
        assert!(feedback.snapshot().active);
    }

    #[test]
    fn feedback_learns_runtime_costs_without_backend_labels() {
        let mut feedback =
            RuntimeCostFeedback::new(&bootstrap(), RuntimeFeedbackConfig::default()).unwrap();
        for round in 0..160 {
            let ranges = 1 + round % 12;
            let bytes = (32 * 1024 + (round % 7) * 48 * 1024) as u64;
            let features = io_features(ranges, bytes, 4);
            let actual_io =
                features[0] * 250_000.0 + features[1] * 70_000.0 + features[2] * 120_000.0;
            let access_units = 1 + round % 16;
            let actual_decode = 80_000 + access_units * 12_000;
            feedback.observe(
                ranges,
                bytes,
                actual_io as u64,
                access_units,
                actual_decode as u64,
            );
        }

        let learned = feedback.apply_to(&bootstrap());
        assert!(feedback.snapshot().active);
        assert!(learned.wave_request_overhead_ns[0] < 600_000.0);
        assert!(learned.bandwidth_bytes_per_ns > bootstrap().bandwidth_bytes_per_ns);
        assert!(learned.decode_access_unit_ns < bootstrap().decode_access_unit_ns);
        learned.validate().unwrap();
    }

    #[test]
    fn one_outlier_cannot_explode_the_model() {
        let mut feedback =
            RuntimeCostFeedback::new(&bootstrap(), RuntimeFeedbackConfig::default()).unwrap();
        for _ in 0..32 {
            feedback.observe(4, 64 * 1024, 1_000_000, 8, 300_000);
        }
        let before = feedback.apply_to(&bootstrap());
        feedback.observe(4, 64 * 1024, 30_000_000_000, 8, 30_000_000_000);
        let after = feedback.apply_to(&bootstrap());
        assert!(feedback.snapshot().active);
        assert!(after.request_latency_ns <= before.request_latency_ns * 1.5 + 1.0);
        assert!(after.decode_access_unit_ns <= before.decode_access_unit_ns * 1.5 + 1.0);
        after.validate().unwrap();
    }

    #[test]
    fn activation_is_not_revoked_by_one_noisy_observation() {
        let config = RuntimeFeedbackConfig {
            min_observations: 8,
            activation_ape_threshold: 10.0,
            ..RuntimeFeedbackConfig::default()
        };
        let mut feedback = RuntimeCostFeedback::new(&bootstrap(), config).unwrap();
        for _ in 0..8 {
            feedback.observe(4, 64 * 1024, 1_200_000, 8, 500_000);
        }
        assert!(feedback.snapshot().active);
        feedback.observe(4, 64 * 1024, 30_000_000_000, 8, 30_000_000_000);
        assert!(feedback.snapshot().active);
    }

    #[test]
    fn invalid_or_cache_only_observations_are_ignored() {
        let mut feedback =
            RuntimeCostFeedback::new(&bootstrap(), RuntimeFeedbackConfig::default()).unwrap();
        feedback.observe(0, 0, 0, 0, 0);
        assert_eq!(feedback.snapshot().io_observations, 0);
        assert_eq!(feedback.snapshot().decode_observations, 0);
    }

    #[test]
    fn tail_model_is_conservative_without_changing_the_expected_model() {
        let config = RuntimeFeedbackConfig {
            min_observations: 8,
            activation_ape_threshold: 10.0,
            activation_stable_observations: 1,
            ..RuntimeFeedbackConfig::default()
        };
        let mut feedback = RuntimeCostFeedback::new(&bootstrap(), config).unwrap();
        for round in 0..32 {
            let tail = if round % 10 == 0 { 4 } else { 1 };
            feedback.observe(4, 64 * 1024, 1_200_000 * tail, 8, 500_000 * tail);
        }

        let expected = feedback.apply_to(&bootstrap());
        let tail = feedback.apply_tail_to(&bootstrap());
        assert!(feedback.snapshot().active);
        assert!(tail.request_latency_ns >= expected.request_latency_ns);
        assert!(tail.decode_access_unit_ns >= expected.decode_access_unit_ns);
        assert!(feedback.snapshot().io_tail_multiplier_ppm >= 1_000_000);
        assert!(feedback.snapshot().decode_tail_multiplier_ppm >= 1_000_000);
    }
}
