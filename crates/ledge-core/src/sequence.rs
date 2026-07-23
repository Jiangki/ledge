//! Rolling rebalance sequences over a fixed portfolio structure (roadmap 2.5).
//!
//! A [`PortfolioSequence`] is the high-level entry point for the workload
//! Ledge is built around: solve the same factor portfolio every date with new
//! expected returns, a new turnover anchor, or new right-hand sides. It wraps
//! a [`Workspace`](crate::Workspace) — the equilibration and the SMW-reduced
//! factorizations are built once and reused — and chains full primal/dual
//! warm starts from one solve into the next automatically.
//!
//! Per-date data changes are described by a [`RebalanceStep`]. Only data that
//! preserves the cached factorizations may change: expected returns, the
//! previous-weights turnover anchor, and the tracking benchmark move the
//! linear cost, and budget / right-hand-side updates move constraint
//! targets. Structural changes
//! (covariance, constraint matrices, bounds, the turnover penalty itself)
//! require a new sequence, because they change the factored system.
//!
//! Steps are applied atomically: a rejected step leaves the sequence exactly
//! as it was, so a caller can drop one bad date and keep rolling.
//!
//! ```
//! use ledge_core::{FactorCovariance, Matrix, PortfolioProblem, RebalanceStep};
//!
//! let problem = PortfolioProblem::new(
//!     Matrix::new(3, 1, vec![1.0, -0.5, 0.25])?,
//!     FactorCovariance::Diagonal(vec![0.1]),
//!     vec![0.2, 0.3, 0.25],
//!     vec![0.08, 0.04, 0.06],
//! )?;
//! let mut sequence = problem.sequence()?;
//! let first = sequence.solve_next(&RebalanceStep::default())?;
//! let second = sequence.solve_next(&RebalanceStep {
//!     expected_returns: Some(vec![0.07, 0.05, 0.06]),
//!     ..RebalanceStep::default()
//! })?;
//! assert_eq!(second.status, ledge_core::SolveStatus::Solved);
//! # let _ = first;
//! # Ok::<(), ledge_core::PortfolioError>(())
//! ```

use crate::{
    portfolio::{
        append_certificate_hints, validate_vector, FinitePolicy, PortfolioError, PortfolioProblem,
        PortfolioSemantics,
    },
    problem::{FactorQuad, ProblemError},
    solver::{Solution, SolveStatus, Solver, SolverSettings, WarmStart},
    workspace::Workspace,
};

/// Per-date data changes for one [`PortfolioSequence::solve_next`] call.
///
/// Every field is optional; `None` keeps the current value. The default value
/// changes nothing, so `RebalanceStep::default()` re-solves the current data
/// (useful for the first solve of a sequence).
///
/// Only factorization-preserving updates are expressible. Anything structural
/// — covariance, constraint matrices, bounds, the turnover penalty — needs a
/// new [`PortfolioProblem`] and a new sequence.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RebalanceStep {
    /// New expected returns (length must match the number of assets).
    pub expected_returns: Option<Vec<f64>>,
    /// New turnover anchor. Requires the base problem to have been built
    /// with [`PortfolioProblem::with_quadratic_turnover`] and/or
    /// [`PortfolioProblem::with_l1_turnover`]; both terms share this single
    /// anchor. The L2 penalty and the L1 costs themselves stay fixed for
    /// the whole sequence.
    pub previous_weights: Option<Vec<f64>>,
    /// New tracking benchmark. Requires the base problem to have been built
    /// with [`PortfolioProblem::with_tracking_benchmark`]. The benchmark
    /// only shifts the linear cost, so cached factorizations survive.
    pub benchmark_weights: Option<Vec<f64>>,
    /// New budget. Requires the base problem to have a budget constraint.
    pub budget: Option<f64>,
    /// New right-hand side for the user-supplied equality constraints
    /// (excluding the budget row).
    pub equality_rhs: Option<Vec<f64>>,
    /// New right-hand side for the inequality constraints.
    pub inequality_rhs: Option<Vec<f64>>,
}

impl RebalanceStep {
    const fn is_empty(&self) -> bool {
        self.expected_returns.is_none()
            && self.previous_weights.is_none()
            && self.benchmark_weights.is_none()
            && self.budget.is_none()
            && self.equality_rhs.is_none()
            && self.inequality_rhs.is_none()
    }
}

/// A rolling solve sequence over a fixed portfolio structure.
///
/// Built by [`PortfolioProblem::sequence`] or
/// [`PortfolioProblem::sequence_with`]. Owns a
/// [`Workspace`](crate::Workspace), so equilibration and SMW-reduced
/// factorizations persist across dates, and manages warm starts internally:
/// each solve starts from the previous solution's full primal/dual iterate.
///
/// [`Solution::solve_time`] on sequence solves covers iteration only; the
/// one-time setup was paid when the sequence was constructed.
pub struct PortfolioSequence {
    workspace: Workspace,
    expected_returns: Vec<f64>,
    previous_weights: Option<Vec<f64>>,
    turnover_penalty: f64,
    has_l1_turnover: bool,
    /// Needed to recompute the linear-cost shift `-risk_aversion * Σ b`
    /// when a step moves the tracking benchmark.
    covariance: FactorQuad,
    risk_aversion: f64,
    benchmark_weights: Option<Vec<f64>>,
    budget: Option<f64>,
    user_equality_rhs: Vec<f64>,
    /// Bounds are fixed for the life of the sequence, so budget reachability
    /// reduces to two precomputed sums.
    minimum_budget: f64,
    maximum_budget: f64,
    warm_start: Option<WarmStart>,
}

impl PortfolioSequence {
    pub(crate) fn new(problem: &PortfolioProblem, solver: &Solver) -> Result<Self, PortfolioError> {
        let qp = problem.to_qp()?;
        let workspace = solver.workspace(&qp)?;
        Ok(Self {
            workspace,
            expected_returns: problem.expected_returns().to_vec(),
            previous_weights: problem.previous_weights().map(<[f64]>::to_vec),
            turnover_penalty: problem.turnover_penalty(),
            has_l1_turnover: problem.has_l1_turnover(),
            covariance: problem.covariance().clone(),
            risk_aversion: problem.risk_aversion(),
            benchmark_weights: problem.benchmark_weights().map(<[f64]>::to_vec),
            budget: problem.budget(),
            user_equality_rhs: problem.user_equality_rhs().to_vec(),
            minimum_budget: problem.lower_bounds().iter().sum(),
            maximum_budget: problem.upper_bounds().iter().sum(),
            warm_start: None,
        })
    }

    /// Number of portfolio weights.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.expected_returns.len()
    }

    /// Settings every solve in this sequence iterates with.
    #[must_use]
    pub const fn settings(&self) -> &SolverSettings {
        self.workspace.settings()
    }

    /// Number of reduced factorizations built since construction (including
    /// the initial one). A stable count across rolling dates demonstrates
    /// factorization reuse.
    #[must_use]
    pub const fn factorizations(&self) -> usize {
        self.workspace.factorizations()
    }

    /// Applies one step's data changes, then solves, warm-started from the
    /// previous solution.
    ///
    /// The step is validated in full before any state changes, so an `Err`
    /// leaves the sequence exactly as it was. The first solve of a sequence
    /// is cold; every later solve chains the previous full primal/dual
    /// iterate (unless the previous solve failed numerically, which would
    /// make a non-finite start).
    ///
    /// # Errors
    ///
    /// Returns [`PortfolioError`] when the step data has wrong dimensions or
    /// non-finite values, updates a constraint the base problem does not
    /// have (budget / turnover anchor), or moves the budget outside what the
    /// box constraints can reach.
    pub fn solve_next(&mut self, step: &RebalanceStep) -> Result<Solution, PortfolioError> {
        self.apply(step)?;
        let mut solution = self.workspace.solve(self.warm_start.as_ref())?;
        append_certificate_hints(
            &mut solution,
            &PortfolioSemantics {
                budget: self.budget,
                user_equality_count: self.user_equality_rhs.len(),
            },
        );
        // A NumericalFailure iterate contains non-finite values and would be
        // rejected as a warm start; an infeasible date's duals diverge along
        // the certificate ray and would poison the next date. Both restart
        // the following solve cold.
        self.warm_start = matches!(
            solution.status,
            SolveStatus::Solved | SolveStatus::MaxIterations
        )
        .then(|| solution.warm_start());
        Ok(solution)
    }

    /// Validates every field of `step`, then applies all of them. No state
    /// (sequence bookkeeping or workspace data) changes on any error.
    fn apply(&mut self, step: &RebalanceStep) -> Result<(), PortfolioError> {
        if step.is_empty() {
            return Ok(());
        }
        let dimension = self.dimension();
        if let Some(expected_returns) = &step.expected_returns {
            validate_vector(
                "expected_returns",
                expected_returns,
                dimension,
                FinitePolicy::Finite,
            )?;
        }
        if let Some(previous_weights) = &step.previous_weights {
            if self.previous_weights.is_none() {
                return Err(PortfolioError::InvalidParameter(
                    "previous_weights updates require a base problem built with \
                     with_quadratic_turnover and/or with_l1_turnover; the L2 penalty \
                     and the L1 costs are part of the problem structure and stay \
                     fixed for the whole sequence",
                ));
            }
            validate_vector(
                "previous_weights",
                previous_weights,
                dimension,
                FinitePolicy::Finite,
            )?;
        }
        if let Some(benchmark_weights) = &step.benchmark_weights {
            if self.benchmark_weights.is_none() {
                return Err(PortfolioError::InvalidParameter(
                    "benchmark_weights updates require a base problem built with \
                     with_tracking_benchmark; switching between absolute-risk and \
                     tracking objectives changes what the sequence is solving",
                ));
            }
            validate_vector(
                "benchmark_weights",
                benchmark_weights,
                dimension,
                FinitePolicy::Finite,
            )?;
        }
        if let Some(budget) = step.budget {
            if self.budget.is_none() {
                return Err(PortfolioError::InvalidParameter(
                    "budget updates require a base problem with a budget constraint; \
                     adding or removing the budget row changes the factored system",
                ));
            }
            if !budget.is_finite() {
                return Err(PortfolioError::InvalidParameter(
                    "budget must be finite when provided",
                ));
            }
            if budget < self.minimum_budget || budget > self.maximum_budget {
                return Err(PortfolioError::BudgetOutsideBounds {
                    budget,
                    minimum: self.minimum_budget,
                    maximum: self.maximum_budget,
                });
            }
        }
        if let Some(equality_rhs) = &step.equality_rhs {
            validate_vector(
                "equality_rhs",
                equality_rhs,
                self.user_equality_rhs.len(),
                FinitePolicy::Finite,
            )?;
        }
        if let Some(inequality_rhs) = &step.inequality_rhs {
            validate_vector(
                "inequality_rhs",
                inequality_rhs,
                self.workspace.problem().inequalities.len(),
                FinitePolicy::Finite,
            )?;
        }

        // Prebuild the derived vectors so their (pathological) overflow is
        // caught before the workspace mutates.
        let linear = if step.expected_returns.is_some()
            || step.previous_weights.is_some()
            || step.benchmark_weights.is_some()
        {
            let expected_returns = step
                .expected_returns
                .as_deref()
                .unwrap_or(&self.expected_returns);
            let mut linear: Vec<f64> = expected_returns.iter().map(|value| -value).collect();
            if let Some(previous_weights) = step
                .previous_weights
                .as_deref()
                .or(self.previous_weights.as_deref())
            {
                for (value, previous) in linear.iter_mut().zip(previous_weights) {
                    *value -= self.turnover_penalty * previous;
                }
            }
            if let Some(benchmark_weights) = step
                .benchmark_weights
                .as_deref()
                .or(self.benchmark_weights.as_deref())
            {
                let covariance_times_benchmark = self.covariance.apply(benchmark_weights);
                for (value, product) in linear.iter_mut().zip(&covariance_times_benchmark) {
                    *value -= self.risk_aversion * product;
                }
            }
            if linear.iter().any(|value| !value.is_finite()) {
                return Err(ProblemError::NonFinite("linear").into());
            }
            Some(linear)
        } else {
            None
        };
        let combined_equality_rhs = if step.budget.is_some() || step.equality_rhs.is_some() {
            let mut combined = Vec::with_capacity(
                usize::from(self.budget.is_some()) + self.user_equality_rhs.len(),
            );
            if let Some(budget) = step.budget.or(self.budget) {
                combined.push(budget);
            }
            combined.extend_from_slice(
                step.equality_rhs
                    .as_deref()
                    .unwrap_or(&self.user_equality_rhs),
            );
            Some(combined)
        } else {
            None
        };

        // Everything is validated; apply. The workspace re-validates each
        // vector but can no longer fail, keeping the step atomic.
        if let Some(linear) = &linear {
            self.workspace.update_linear(linear)?;
        }
        if let (Some(previous_weights), true) = (&step.previous_weights, self.has_l1_turnover) {
            self.workspace.update_l1_anchor(previous_weights)?;
        }
        if let Some(combined) = &combined_equality_rhs {
            self.workspace.update_equality_rhs(combined)?;
        }
        if let Some(inequality_rhs) = &step.inequality_rhs {
            self.workspace.update_inequality_rhs(inequality_rhs)?;
        }
        if let Some(expected_returns) = &step.expected_returns {
            self.expected_returns.clone_from(expected_returns);
        }
        if let Some(previous_weights) = &step.previous_weights {
            self.previous_weights = Some(previous_weights.clone());
        }
        if let Some(benchmark_weights) = &step.benchmark_weights {
            self.benchmark_weights = Some(benchmark_weights.clone());
        }
        if let Some(budget) = step.budget {
            self.budget = Some(budget);
        }
        if let Some(equality_rhs) = &step.equality_rhs {
            self.user_equality_rhs.clone_from(equality_rhs);
        }
        Ok(())
    }
}

/// Solves a whole rolling sequence in one call: one [`Solution`] per step,
/// in order (roadmap 2.5).
///
/// The first step is applied to the base problem before its solve, so pass
/// [`RebalanceStep::default()`] first to solve the base data as-is. Warm
/// starts and factorization reuse are managed internally; for streaming
/// workloads where dates arrive one at a time, hold a [`PortfolioSequence`]
/// (via [`PortfolioProblem::sequence`]) and call
/// [`PortfolioSequence::solve_next`] instead.
///
/// A solve that stops at `MaxIterations` does not abort the sequence; inspect
/// each returned [`SolveStatus`](crate::SolveStatus).
///
/// # Errors
///
/// Returns [`PortfolioError`] on invalid base data or on the first invalid
/// step, together with the index-free context of that step's validation
/// error. Solutions for the steps before the failure are dropped.
pub fn solve_sequence(
    problem: &PortfolioProblem,
    settings: Option<SolverSettings>,
    steps: &[RebalanceStep],
) -> Result<Vec<Solution>, PortfolioError> {
    let solver = Solver::new(settings.unwrap_or_default());
    let mut sequence = problem.sequence_with(&solver)?;
    steps.iter().map(|step| sequence.solve_next(step)).collect()
}
