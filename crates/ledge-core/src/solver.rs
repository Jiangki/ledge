//! ADMM solver with a factor-structured linear system and adaptive penalty.
//!
//! The iteration engine and the factorization cache live in
//! [`crate::workspace`]; this module owns the public one-shot entry point,
//! settings, statuses, and diagnostics.

use std::{
    fmt,
    time::{Duration, Instant},
};

use thiserror::Error;

use crate::{
    certificate::Certificate,
    kkt::{DualVariables, KktError, KktResiduals},
    problem::{ProblemError, QpProblem},
    workspace::Workspace,
};

/// Solver termination status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SolveStatus {
    /// KKT residuals met the configured absolute/relative tolerances.
    Solved,
    /// The iteration budget was exhausted.
    MaxIterations,
    /// An iterate became NaN or infinite.
    NumericalFailure,
    /// The constraints admit no common point; a Farkas certificate is
    /// attached to [`Solution::certificate`].
    PrimalInfeasible,
    /// The objective is unbounded below over the constraints; a descent-ray
    /// certificate is attached to [`Solution::certificate`].
    DualInfeasible,
}

impl fmt::Display for SolveStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Solved => "solved",
            Self::MaxIterations => "maximum iterations reached",
            Self::NumericalFailure => "numerical failure",
            Self::PrimalInfeasible => "primal infeasible",
            Self::DualInfeasible => "dual infeasible",
        })
    }
}

/// ADMM settings.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SolverSettings {
    /// Maximum number of ADMM iterations.
    pub max_iterations: usize,
    /// Absolute stopping tolerance.
    pub absolute_tolerance: f64,
    /// Relative stopping tolerance.
    pub relative_tolerance: f64,
    /// Initial augmented-Lagrangian penalty.
    pub rho: f64,
    /// Positive proximal regularization in the x update.
    pub sigma: f64,
    /// Evaluate KKT residuals every this many iterations.
    pub check_termination_every: usize,
    /// Whether residual balancing may update `rho`.
    pub adaptive_rho: bool,
    /// Consider a penalty update every this many iterations.
    pub adaptive_rho_interval: usize,
    /// Update when one ADMM residual exceeds the other by this ratio.
    pub adaptive_rho_tolerance: f64,
    /// Multiplicative increase or decrease applied to `rho`.
    pub adaptive_rho_multiplier: f64,
    /// Over-relaxation coefficient `alpha` blended into every consensus
    /// update: the slack and multiplier steps see
    /// `alpha * Ax + (1 - alpha) * z_prev` instead of `Ax`.
    ///
    /// Must lie in the open interval `(0, 2)`; `1.0` disables relaxation and
    /// recovers plain ADMM. Values around `1.6` typically cut iteration
    /// counts substantially on feasible convex QPs (see Boyd et al. §3.4.3
    /// and the OSQP default).
    pub over_relaxation: f64,
    /// Smallest penalty selected by adaptation.
    pub minimum_rho: f64,
    /// Largest penalty selected by adaptation.
    pub maximum_rho: f64,
    /// Number of Ruiz equilibration passes over the problem data before
    /// iterating; `0` disables automatic scaling.
    ///
    /// Scaling only changes the space ADMM iterates in. Termination checks and
    /// every reported residual are always evaluated on the original data.
    pub scaling_iterations: usize,
    /// Tolerance for declaring the problem primal or dual infeasible from
    /// ADMM iterate differences; `0` disables infeasibility detection.
    ///
    /// Candidate certificates are normalized to unit infinity norm and
    /// evaluated on the original data, so the tolerance is relative: a
    /// direction qualifies when its defining residuals are below the
    /// tolerance and its gap is below the negated tolerance (see
    /// [`crate::check_primal_certificate`] /
    /// [`crate::check_dual_certificate`]). Problems infeasible by a margin
    /// smaller than this tolerance stop at [`SolveStatus::MaxIterations`]
    /// with diagnostics instead.
    pub infeasibility_tolerance: f64,
    /// Whether to refine `Solved` iterates with one direct active-set solve
    /// (polishing).
    ///
    /// The active set is guessed from the final multipliers and the
    /// resulting KKT system is solved through the same SMW reduction the
    /// iterations use, at the cost of one extra reduced factorization. The
    /// polished iterate is adopted only when its worst KKT residual —
    /// re-audited with [`crate::check_kkt`] on the original data — improves
    /// on the ADMM iterate's, so enabling polish never degrades a solution.
    /// [`Solution::polished`] records the outcome.
    pub polish: bool,
    /// Regularization added to the polishing KKT system so it stays
    /// factorable even when the active-set guess is degenerate.
    ///
    /// The regularization error is removed afterwards by
    /// [`SolverSettings::polish_refinement_iterations`] rounds of iterative
    /// refinement against the unregularized system.
    pub polish_regularization: f64,
    /// Iterative-refinement rounds applied to the polishing solve.
    pub polish_refinement_iterations: usize,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            absolute_tolerance: 1.0e-6,
            relative_tolerance: 1.0e-5,
            rho: 1.0,
            sigma: 1.0e-6,
            check_termination_every: 10,
            adaptive_rho: true,
            adaptive_rho_interval: 25,
            adaptive_rho_tolerance: 5.0,
            adaptive_rho_multiplier: 2.0,
            over_relaxation: 1.6,
            minimum_rho: 1.0e-6,
            maximum_rho: 1.0e6,
            scaling_iterations: 10,
            infeasibility_tolerance: 1.0e-5,
            polish: true,
            polish_regularization: 1.0e-6,
            polish_refinement_iterations: 3,
        }
    }
}

/// Optional initial primal and dual iterates.
///
/// Missing multiplier blocks are initialized to zero. Warm starts reuse
/// iterates; to also reuse the equilibration and the reduced factorization
/// across solves, keep a [`Workspace`](crate::Workspace).
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WarmStart {
    /// Initial decision vector.
    pub x: Vec<f64>,
    /// Optional equality multipliers.
    pub equality_dual: Option<Vec<f64>>,
    /// Optional inequality multipliers.
    pub inequality_dual: Option<Vec<f64>>,
    /// Optional box normal-cone multipliers.
    pub bound_dual: Option<Vec<f64>>,
    /// Optional L1 subgradient multipliers (problems with an
    /// [`L1Term`](crate::L1Term) only).
    pub l1_dual: Option<Vec<f64>>,
}

impl WarmStart {
    /// Constructs a primal-only warm start.
    #[must_use]
    pub fn from_primal(x: Vec<f64>) -> Self {
        Self {
            x,
            ..Self::default()
        }
    }
}

/// Heuristic diagnosis of a solve that stopped before convergence.
///
/// Attached to [`Solution::diagnostics`] when the status is not
/// [`SolveStatus::Solved`]. Hints are heuristics ordered by likely relevance,
/// not certificates; they exist to make `MaxIterations` actionable instead of
/// a dead end.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvergenceDiagnostics {
    /// Primal stopping tolerance in force at the final iterate.
    pub primal_tolerance: f64,
    /// Dual stopping tolerance in force at the final iterate.
    pub dual_tolerance: f64,
    /// Base-10 orders of magnitude spanned by nonzero problem coefficients.
    pub coefficient_spread_decades: f64,
    /// Whether the final penalty sat at `minimum_rho` or `maximum_rho`.
    pub rho_at_limit: bool,
    /// Human-readable tuning hints ordered by likely relevance.
    pub hints: Vec<String>,
}

/// Solver result and diagnostics.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Solution {
    /// Termination status.
    pub status: SolveStatus,
    /// Primal decision vector.
    pub x: Vec<f64>,
    /// Constraint multipliers.
    pub dual: DualVariables,
    /// Objective value at the returned iterate.
    pub objective: f64,
    /// KKT residuals at the returned iterate.
    pub residuals: KktResiduals,
    /// Number of completed ADMM iterations.
    pub iterations: usize,
    /// Wall-clock time this solve was charged with: setup plus iteration for
    /// [`Solver::solve`], iteration only for
    /// [`Workspace::solve`](crate::Workspace::solve) (setup was paid at
    /// workspace construction).
    pub solve_time: Duration,
    /// Penalty used for the final iteration.
    pub final_rho: f64,
    /// Number of adaptive penalty changes. Each change needs the matching
    /// reduced factorization: one-shot solves rebuild it, workspace solves
    /// reuse a cached one when the penalty was already visited.
    pub rho_updates: usize,
    /// Whether the returned iterate is the polished one: `true` only when
    /// the status is [`SolveStatus::Solved`], [`SolverSettings::polish`] is
    /// enabled, and the direct active-set solve improved the worst KKT
    /// residual (otherwise the ADMM iterate is kept and this stays `false`).
    pub polished: bool,
    /// Heuristic hints; present only when the status is not `Solved`.
    pub diagnostics: Option<ConvergenceDiagnostics>,
    /// Infeasibility proof; present only when the status is
    /// [`SolveStatus::PrimalInfeasible`] or [`SolveStatus::DualInfeasible`].
    ///
    /// Certificates are normalized to unit infinity norm, reported in the
    /// original data space, and independently auditable with
    /// [`crate::check_primal_certificate`] /
    /// [`crate::check_dual_certificate`].
    pub certificate: Option<Certificate>,
}

impl Solution {
    /// Converts this solution into a full warm start for a related solve.
    #[must_use]
    pub fn warm_start(&self) -> WarmStart {
        WarmStart {
            x: self.x.clone(),
            equality_dual: Some(self.dual.equalities.clone()),
            inequality_dual: Some(self.dual.inequalities.clone()),
            bound_dual: Some(self.dual.bounds.clone()),
            l1_dual: Some(self.dual.l1.clone()),
        }
    }
}

/// Errors that prevent solver setup.
#[derive(Debug, Error)]
pub enum SolverError {
    /// Invalid QP data.
    #[error(transparent)]
    InvalidProblem(#[from] ProblemError),
    /// Invalid scalar solver setting.
    #[error("invalid solver setting: {0}")]
    InvalidSettings(&'static str),
    /// Dense factor covariance is not positive semidefinite.
    #[error("dense factor covariance is not positive semidefinite")]
    NonPositiveSemidefiniteOmega,
    /// A reduced positive-definite system could not be factored.
    #[error("failed to factor the reduced linear system")]
    LinearSystem,
    /// Internal KKT diagnostic dimensions were inconsistent.
    #[error(transparent)]
    Kkt(#[from] KktError),
    /// A warm-start vector has the wrong length.
    #[error("warm-start {field} has length {actual}; expected {expected}")]
    WarmStartDimension {
        /// Warm-start field.
        field: &'static str,
        /// Expected vector length.
        expected: usize,
        /// Supplied vector length.
        actual: usize,
    },
    /// A warm-start vector contains NaN or infinity.
    #[error("warm-start {0} contains a non-finite value")]
    WarmStartNonFinite(&'static str),
}

/// Structure-exploiting convex QP solver.
#[derive(Clone, Debug, Default)]
pub struct Solver {
    settings: SolverSettings,
}

impl Solver {
    /// Creates a solver with explicit settings.
    #[must_use]
    pub const fn new(settings: SolverSettings) -> Self {
        Self { settings }
    }

    /// Returns the active settings.
    #[must_use]
    pub const fn settings(&self) -> &SolverSettings {
        &self.settings
    }

    /// Solves a convex factor-model QP.
    ///
    /// When [`SolverSettings::scaling_iterations`] is positive, ADMM iterates
    /// on a Ruiz-equilibrated copy of the data; termination checks and every
    /// value on the returned [`Solution`] are evaluated on the original data.
    ///
    /// Each call pays equilibration and factorization setup again. For
    /// rolling sequences over a fixed structure, build a [`Workspace`] with
    /// [`Solver::workspace`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError`] when the input, settings, warm start, factor
    /// covariance, or reduced linear system is invalid.
    pub fn solve(
        &self,
        problem: &QpProblem,
        warm_start: Option<&WarmStart>,
    ) -> Result<Solution, SolverError> {
        let started = Instant::now();
        let mut workspace = Workspace::new(&self.settings, problem)?;
        workspace.solve_from(started, warm_start)
    }

    /// Builds a reusable [`Workspace`] that caches the equilibration and the
    /// SMW-reduced factorization across solves (roadmap 2.4).
    ///
    /// Use it when the problem structure — covariance, constraint matrices,
    /// bounds — is fixed and only the linear cost or right-hand sides change
    /// between solves, as in rolling rebalances.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError`] when the input, settings, factor covariance,
    /// or reduced linear system is invalid.
    pub fn workspace(&self, problem: &QpProblem) -> Result<Workspace, SolverError> {
        Workspace::new(&self.settings, problem)
    }
}

/// Validates scalar solver settings shared by every solve path.
pub(crate) fn validate_settings(settings: &SolverSettings) -> Result<(), SolverError> {
    if settings.max_iterations == 0 {
        return Err(SolverError::InvalidSettings(
            "max_iterations must be positive",
        ));
    }
    if !settings.absolute_tolerance.is_finite() || settings.absolute_tolerance <= 0.0 {
        return Err(SolverError::InvalidSettings(
            "absolute_tolerance must be finite and positive",
        ));
    }
    if !settings.relative_tolerance.is_finite() || settings.relative_tolerance < 0.0 {
        return Err(SolverError::InvalidSettings(
            "relative_tolerance must be finite and non-negative",
        ));
    }
    if !settings.rho.is_finite() || settings.rho <= 0.0 {
        return Err(SolverError::InvalidSettings(
            "rho must be finite and positive",
        ));
    }
    if !settings.sigma.is_finite() || settings.sigma <= 0.0 {
        return Err(SolverError::InvalidSettings(
            "sigma must be finite and positive",
        ));
    }
    if settings.check_termination_every == 0 {
        return Err(SolverError::InvalidSettings(
            "check_termination_every must be positive",
        ));
    }
    if settings.adaptive_rho_interval == 0 {
        return Err(SolverError::InvalidSettings(
            "adaptive_rho_interval must be positive",
        ));
    }
    if !settings.adaptive_rho_tolerance.is_finite() || settings.adaptive_rho_tolerance <= 1.0 {
        return Err(SolverError::InvalidSettings(
            "adaptive_rho_tolerance must be finite and greater than one",
        ));
    }
    if !settings.adaptive_rho_multiplier.is_finite() || settings.adaptive_rho_multiplier <= 1.0 {
        return Err(SolverError::InvalidSettings(
            "adaptive_rho_multiplier must be finite and greater than one",
        ));
    }
    if !settings.over_relaxation.is_finite()
        || settings.over_relaxation <= 0.0
        || settings.over_relaxation >= 2.0
    {
        return Err(SolverError::InvalidSettings(
            "over_relaxation must lie strictly between zero and two",
        ));
    }
    if !settings.minimum_rho.is_finite() || settings.minimum_rho <= 0.0 {
        return Err(SolverError::InvalidSettings(
            "minimum_rho must be finite and positive",
        ));
    }
    if !settings.maximum_rho.is_finite() || settings.maximum_rho < settings.minimum_rho {
        return Err(SolverError::InvalidSettings(
            "maximum_rho must be finite and at least minimum_rho",
        ));
    }
    if settings.rho < settings.minimum_rho || settings.rho > settings.maximum_rho {
        return Err(SolverError::InvalidSettings(
            "rho must lie between minimum_rho and maximum_rho",
        ));
    }
    if !settings.infeasibility_tolerance.is_finite() || settings.infeasibility_tolerance < 0.0 {
        return Err(SolverError::InvalidSettings(
            "infeasibility_tolerance must be finite and non-negative",
        ));
    }
    if !settings.polish_regularization.is_finite() || settings.polish_regularization <= 0.0 {
        return Err(SolverError::InvalidSettings(
            "polish_regularization must be finite and positive",
        ));
    }
    Ok(())
}

/// Benchmark-only hooks into solver internals.
///
/// Compiled only with the `bench-internals` feature so criterion benches can
/// time the reduced factorization and the x-update in isolation. Not a public
/// API; may change without notice.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod bench_internals {
    use super::{QpProblem, SolverError};
    use crate::workspace::FactorizedSystem;

    /// Opaque handle over the SMW-reduced factorized system.
    pub struct ReducedSystem(FactorizedSystem);

    /// Builds the reduced factorization used by every x-update.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError`] when the factor covariance or the reduced
    /// system cannot be factored.
    pub fn factorize(
        problem: &QpProblem,
        rho: f64,
        sigma: f64,
    ) -> Result<ReducedSystem, SolverError> {
        FactorizedSystem::new(problem, rho, sigma).map(ReducedSystem)
    }

    /// Applies one x-update linear solve in place.
    pub fn x_update(system: &ReducedSystem, right_hand_side: &mut [f64]) {
        system.0.solve_in_place(right_hand_side);
    }
}
