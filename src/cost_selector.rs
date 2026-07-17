use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: String,
    pub portfolio_storage_bytes: f64,
    pub metadata_bytes: f64,
    pub ingestion_ns: f64,
    pub requests_per_sample: f64,
    pub fetched_bytes_per_sample: f64,
    pub decode_ns_per_sample: f64,
    pub p95_ns: f64,
    pub p99_ns: f64,
    pub throughput_samples_s: f64,
    pub cache_bytes: f64,
    pub model_utility: f64,
    pub switch_cost_usd: f64,
}

#[derive(Debug, Clone)]
pub struct PriceVector {
    pub storage_usd_per_byte_month: f64,
    pub retention_months: f64,
    pub request_usd_per_request: f64,
    pub transfer_usd_per_byte: f64,
    pub decode_compute_usd_per_ns: f64,
    pub ingestion_compute_usd_per_ns: f64,
    pub workload_samples: f64,
    pub amortization_runs: f64,
    pub slo_penalty_usd_per_ns: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Constraints {
    pub max_p95_ns: Option<f64>,
    pub max_p99_ns: Option<f64>,
    pub min_throughput_samples_s: Option<f64>,
    pub max_portfolio_storage_bytes: Option<f64>,
    pub max_cache_bytes: Option<f64>,
    pub min_model_utility: Option<f64>,
    pub target_p95_ns: Option<f64>,
    pub target_p99_ns: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub id: String,
    pub eligible: bool,
    pub rejection_reasons: Vec<String>,
    pub total_cost_usd: f64,
    pub breakdown: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct Selection {
    pub selected_id: String,
    pub candidates: Vec<ScoredCandidate>,
}

fn nonnegative(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be finite and non-negative"));
    }
    Ok(())
}

fn optional(name: &str, value: Option<f64>) -> Result<(), String> {
    value.map_or(Ok(()), |value| nonnegative(name, value))
}

impl Candidate {
    fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("candidate id must not be empty".to_string());
        }
        for (name, value) in [
            ("portfolio_storage_bytes", self.portfolio_storage_bytes),
            ("metadata_bytes", self.metadata_bytes),
            ("ingestion_ns", self.ingestion_ns),
            ("requests_per_sample", self.requests_per_sample),
            ("fetched_bytes_per_sample", self.fetched_bytes_per_sample),
            ("decode_ns_per_sample", self.decode_ns_per_sample),
            ("p95_ns", self.p95_ns),
            ("p99_ns", self.p99_ns),
            ("throughput_samples_s", self.throughput_samples_s),
            ("cache_bytes", self.cache_bytes),
            ("model_utility", self.model_utility),
            ("switch_cost_usd", self.switch_cost_usd),
        ] {
            nonnegative(name, value)?;
        }
        Ok(())
    }
}

impl PriceVector {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            (
                "storage_usd_per_byte_month",
                self.storage_usd_per_byte_month,
            ),
            ("retention_months", self.retention_months),
            ("request_usd_per_request", self.request_usd_per_request),
            ("transfer_usd_per_byte", self.transfer_usd_per_byte),
            ("decode_compute_usd_per_ns", self.decode_compute_usd_per_ns),
            (
                "ingestion_compute_usd_per_ns",
                self.ingestion_compute_usd_per_ns,
            ),
            ("workload_samples", self.workload_samples),
            ("slo_penalty_usd_per_ns", self.slo_penalty_usd_per_ns),
        ] {
            nonnegative(name, value)?;
        }
        if !self.amortization_runs.is_finite() || self.amortization_runs <= 0.0 {
            return Err("amortization_runs must be finite and positive".to_string());
        }
        Ok(())
    }
}

impl Constraints {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("max_p95_ns", self.max_p95_ns),
            ("max_p99_ns", self.max_p99_ns),
            ("min_throughput_samples_s", self.min_throughput_samples_s),
            (
                "max_portfolio_storage_bytes",
                self.max_portfolio_storage_bytes,
            ),
            ("max_cache_bytes", self.max_cache_bytes),
            ("min_model_utility", self.min_model_utility),
            ("target_p95_ns", self.target_p95_ns),
            ("target_p99_ns", self.target_p99_ns),
        ] {
            optional(name, value)?;
        }
        Ok(())
    }
}

pub fn select(
    candidates: Vec<Candidate>,
    prices: PriceVector,
    constraints: Constraints,
) -> Result<Selection, String> {
    if candidates.is_empty() {
        return Err("portfolio selector requires at least one candidate".to_string());
    }
    prices.validate()?;
    constraints.validate()?;
    let mut ids = HashSet::new();
    let mut scored = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        candidate.validate()?;
        if !ids.insert(candidate.id.clone()) {
            return Err(format!("duplicate candidate id {}", candidate.id));
        }
        let mut reasons = Vec::new();
        let checks = [
            (
                constraints.max_p95_ns.is_some_and(|v| candidate.p95_ns > v),
                "p95_slo",
            ),
            (
                constraints.max_p99_ns.is_some_and(|v| candidate.p99_ns > v),
                "p99_slo",
            ),
            (
                constraints
                    .min_throughput_samples_s
                    .is_some_and(|v| candidate.throughput_samples_s < v),
                "throughput",
            ),
            (
                constraints.max_portfolio_storage_bytes.is_some_and(|v| {
                    candidate.portfolio_storage_bytes + candidate.metadata_bytes > v
                }),
                "portfolio_storage",
            ),
            (
                constraints
                    .max_cache_bytes
                    .is_some_and(|v| candidate.cache_bytes > v),
                "cache",
            ),
            (
                constraints
                    .min_model_utility
                    .is_some_and(|v| candidate.model_utility < v),
                "model_utility",
            ),
        ];
        for (failed, reason) in checks {
            if failed {
                reasons.push(reason.to_string());
            }
        }
        let mut breakdown = HashMap::from([
            (
                "storage_usd".to_string(),
                (candidate.portfolio_storage_bytes + candidate.metadata_bytes)
                    * prices.storage_usd_per_byte_month
                    * prices.retention_months,
            ),
            (
                "request_usd".to_string(),
                candidate.requests_per_sample
                    * prices.workload_samples
                    * prices.request_usd_per_request,
            ),
            (
                "transfer_usd".to_string(),
                candidate.fetched_bytes_per_sample
                    * prices.workload_samples
                    * prices.transfer_usd_per_byte,
            ),
            (
                "decode_usd".to_string(),
                candidate.decode_ns_per_sample
                    * prices.workload_samples
                    * prices.decode_compute_usd_per_ns,
            ),
            (
                "amortized_ingestion_usd".to_string(),
                candidate.ingestion_ns * prices.ingestion_compute_usd_per_ns
                    / prices.amortization_runs,
            ),
            ("switch_usd".to_string(), candidate.switch_cost_usd),
        ]);
        let p95_penalty = constraints
            .target_p95_ns
            .map(|target| (candidate.p95_ns - target).max(0.0))
            .unwrap_or(0.0);
        let p99_penalty = constraints
            .target_p99_ns
            .map(|target| (candidate.p99_ns - target).max(0.0))
            .unwrap_or(0.0);
        breakdown.insert(
            "slo_penalty_usd".to_string(),
            (p95_penalty + p99_penalty) * prices.slo_penalty_usd_per_ns,
        );
        scored.push(ScoredCandidate {
            id: candidate.id,
            eligible: reasons.is_empty(),
            rejection_reasons: reasons,
            total_cost_usd: breakdown.values().sum(),
            breakdown,
        });
    }
    let selected = scored
        .iter()
        .filter(|x| x.eligible)
        .min_by(|a, b| {
            a.total_cost_usd
                .total_cmp(&b.total_cost_usd)
                .then_with(|| a.id.cmp(&b.id))
        })
        .ok_or_else(|| "no candidate satisfies the portfolio constraints".to_string())?;
    Ok(Selection {
        selected_id: selected.id.clone(),
        candidates: scored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, storage: f64, requests: f64) -> Candidate {
        Candidate {
            id: id.into(),
            portfolio_storage_bytes: storage,
            metadata_bytes: 10.0,
            ingestion_ns: 100.0,
            requests_per_sample: requests,
            fetched_bytes_per_sample: 2.0,
            decode_ns_per_sample: 3.0,
            p95_ns: 10.0,
            p99_ns: 20.0,
            throughput_samples_s: 100.0,
            cache_bytes: 50.0,
            model_utility: 0.9,
            switch_cost_usd: 0.0,
        }
    }
    fn prices() -> PriceVector {
        PriceVector {
            storage_usd_per_byte_month: 1.0,
            retention_months: 1.0,
            request_usd_per_request: 1.0,
            transfer_usd_per_byte: 1.0,
            decode_compute_usd_per_ns: 1.0,
            ingestion_compute_usd_per_ns: 1.0,
            workload_samples: 10.0,
            amortization_runs: 10.0,
            slo_penalty_usd_per_ns: 1.0,
        }
    }

    #[test]
    fn charges_complete_portfolio_and_selects_minimum_total() {
        let result = select(
            vec![
                candidate("small", 10.0, 2.0),
                candidate("large", 100.0, 0.0),
            ],
            prices(),
            Constraints::default(),
        )
        .unwrap();
        assert_eq!(result.selected_id, "small");
        let small = result.candidates.iter().find(|x| x.id == "small").unwrap();
        assert_eq!(small.breakdown["storage_usd"], 20.0);
        assert!((small.total_cost_usd - small.breakdown.values().sum::<f64>()).abs() < 1e-9);
    }

    #[test]
    fn filters_constraints_and_applies_soft_slo_penalty() {
        let mut rejected = candidate("cheap", 1.0, 0.0);
        rejected.model_utility = 0.5;
        let mut slow = candidate("slow", 10.0, 1.0);
        slow.p95_ns = 30.0;
        let result = select(
            vec![rejected, slow, candidate("fast", 10.0, 1.0)],
            prices(),
            Constraints {
                min_model_utility: Some(0.8),
                target_p95_ns: Some(10.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.selected_id, "fast");
        assert_eq!(
            result.candidates[0].rejection_reasons,
            vec!["model_utility"]
        );
    }

    #[test]
    fn rejects_invalid_and_infeasible_inputs() {
        let mut bad = candidate("bad", 1.0, 1.0);
        bad.p95_ns = f64::NAN;
        assert!(select(vec![bad], prices(), Constraints::default()).is_err());
        let error = select(
            vec![candidate("too_slow", 1.0, 1.0)],
            prices(),
            Constraints {
                max_p95_ns: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(error.contains("no candidate"));
    }
}
