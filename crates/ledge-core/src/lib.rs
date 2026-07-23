//! Numerical kernel for Ledge.
//!
//! Ledge solves convex QPs whose Hessian is a factor covariance plus a
//! diagonal. Its ADMM x-update uses the Sherman-Morrison-Woodbury identity, so
//! it factors a system sized by factors plus explicit linear constraints
//! rather than by the number of assets.

#![forbid(unsafe_code)]

mod batch;
mod benchmark;
mod certificate;
mod generator;
mod kkt;
mod linalg;
mod matrix;
mod polish;
mod portfolio;
mod problem;
mod scaling;
mod sequence;
#[cfg(feature = "serde")]
mod serde_support;
mod solver;
mod workspace;

pub use batch::{solve_batch, AccountResult, BatchAccount};
pub use benchmark::{BenchmarkRecord, BenchmarkRunner, ComparisonSolver};
pub use certificate::{
    check_dual_certificate, check_primal_certificate, Certificate, DualCertificate,
    DualCertificateResiduals, PrimalCertificate, PrimalCertificateResiduals,
};
pub use generator::{generate_synthetic, GeneratedInstance, GeneratorError, SyntheticConfig};
pub use kkt::{check_kkt, DualVariables, KktError, KktResiduals};
pub use matrix::{Matrix, MatrixError};
pub use portfolio::{solve_mean_variance_factor, PortfolioError, PortfolioProblem};
pub use problem::{
    FactorCovariance, FactorQuad, L1Term, LinearConstraints, ProblemError, QpProblem,
};
pub use sequence::{solve_sequence, PortfolioSequence, RebalanceStep};
#[cfg(feature = "bench-internals")]
pub use solver::bench_internals;
pub use solver::{
    ConvergenceDiagnostics, Solution, SolveStatus, Solver, SolverError, SolverSettings, WarmStart,
};
pub use workspace::Workspace;
