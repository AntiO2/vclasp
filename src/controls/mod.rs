//! Reproducibility-only representation and mechanism controls.
//!
//! Production requests must enter through `hierarchical_scheduler`.  Modules
//! here remain compiled because registered ablations depend on them, but they
//! are not alternative public VClasp readers.

pub(crate) mod adaptive;
pub(crate) mod closed_record;
pub(crate) mod normalized;
pub(crate) mod portfolio;
