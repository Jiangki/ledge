//! OSQP / Clarabel comparison adapters implementing the Ledge benchmark
//! protocol (`benchmarks/README.md`).
//!
//! External solvers hide behind non-default cargo features so that the
//! default workspace build carries no extra native dependency:
//!
//! ```bash
//! cargo run --release -p ledge-bench-adapters --features osqp,clarabel --bin compare
//! ```
//!
//! Protocol compliance lives in [`protocol`]: shared instance data, shared
//! primal starts, verbatim native statuses, independent `check_kkt`
//! re-verification, phase-split timing (setup / cold / rolling), and at
//! least ten repeats with all samples published.

#![forbid(unsafe_code)]

pub mod convert;
pub mod ledge_adapter;
pub mod protocol;

#[cfg(feature = "clarabel")]
pub mod clarabel_adapter;
#[cfg(feature = "osqp")]
pub mod osqp_adapter;

pub use convert::{ConvertedQp, Formulation};
pub use ledge_adapter::LedgeAdapter;
pub use protocol::{
    run_workload, AdapterSolve, PhasedSolver, RollingWorkload, Sample, WarmStartSupport,
};

#[cfg(feature = "clarabel")]
pub use clarabel_adapter::ClarabelAdapter;
#[cfg(feature = "osqp")]
pub use osqp_adapter::OsqpAdapter;
