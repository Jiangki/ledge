//! Multi-thread batch rebalancing over the account axis (roadmap 3.2).
//!
//! The persona-C workload is "one factor model, many accounts": every account
//! shares the risk model but carries its own constraints, budget, and
//! turnover anchor, and every account rolls through the same trading dates.
//! Accounts are independent — no state is shared between them — so the batch
//! axis parallelizes embarrassingly: [`solve_batch`] runs one
//! [`PortfolioSequence`](crate::PortfolioSequence) per account, exactly as if
//! the caller had looped over [`solve_sequence`](crate::solve_sequence).
//!
//! Threading is feature-gated: with the non-default `rayon` cargo feature the
//! accounts are distributed over rayon's global thread pool
//! (`RAYON_NUM_THREADS` or a caller-installed pool control the width);
//! without it the same function runs the accounts serially. Results are
//! identical either way — each account's iterate path never depends on the
//! other accounts or on scheduling — so the feature changes wall-clock time,
//! never answers.
//!
//! Failures stay per-account: one account with invalid step data reports its
//! error while every other account still returns its solutions. Within an
//! account the sequence semantics are unchanged (an invalid step aborts that
//! account; an unconverged date does not).
//!
//! ```
//! use ledge_core::{
//!     solve_batch, BatchAccount, FactorCovariance, Matrix, PortfolioProblem, RebalanceStep,
//! };
//!
//! let problem = PortfolioProblem::new(
//!     Matrix::new(3, 1, vec![1.0, -0.5, 0.25])?,
//!     FactorCovariance::Diagonal(vec![0.1]),
//!     vec![0.2, 0.3, 0.25],
//!     vec![0.08, 0.04, 0.06],
//! )?;
//! let accounts = vec![
//!     BatchAccount {
//!         problem: problem.clone(),
//!         steps: vec![RebalanceStep::default()],
//!         chain_previous_weights: false,
//!     },
//!     BatchAccount {
//!         problem,
//!         steps: vec![RebalanceStep {
//!             budget: Some(0.9),
//!             ..RebalanceStep::default()
//!         }],
//!         chain_previous_weights: false,
//!     },
//! ];
//! let results = solve_batch(&accounts, None);
//! assert_eq!(results.len(), 2);
//! for account in &results {
//!     assert_eq!(account.as_ref().unwrap().len(), 1);
//! }
//! # Ok::<(), ledge_core::PortfolioError>(())
//! ```

use crate::{
    portfolio::{PortfolioError, PortfolioProblem},
    sequence::RebalanceStep,
    solver::{Solution, SolveStatus, Solver, SolverSettings},
};

/// One account in a batch: its own portfolio problem plus its per-date steps.
///
/// The problem carries the account's structure and first date's data; `steps`
/// are applied in order exactly like
/// [`solve_sequence`](crate::solve_sequence), so pass
/// [`RebalanceStep::default()`] first to solve the base data as-is.
#[derive(Clone, Debug)]
pub struct BatchAccount {
    /// The account's portfolio problem (structure plus first date's data).
    pub problem: PortfolioProblem,
    /// Per-date data changes, in order. Only factorization-preserving
    /// updates are expressible; see [`RebalanceStep`].
    pub steps: Vec<RebalanceStep>,
    /// When `true`, every date after a `Solved` one anchors the turnover
    /// terms at that date's solved weights — the standard backtest
    /// convention ("previous weights" are what the account actually holds)
    /// — unless the step provides `previous_weights` explicitly, which
    /// wins. Dates that did not reach `Solved` leave the anchor where it
    /// was, because the account did not trade. Requires the problem to have
    /// been built with a turnover term.
    pub chain_previous_weights: bool,
}

/// Per-account outcome of [`solve_batch`]: all step solutions in order, or
/// the error that stopped that account.
pub type AccountResult = Result<Vec<Solution>, PortfolioError>;

/// Solves every account's rolling sequence, in parallel over the account
/// axis when the `rayon` cargo feature is enabled (roadmap 3.2).
///
/// Each account gets its own workspace (equilibration and factorizations
/// built once per account, warm starts chained across its dates), so the
/// batch does the same numerical work as calling
/// [`solve_sequence`](crate::solve_sequence) per account — results are
/// bit-identical to that loop regardless of the feature or thread count.
/// One entry is returned per account, in input order; a failed account never
/// affects the others.
#[must_use]
pub fn solve_batch(
    accounts: &[BatchAccount],
    settings: Option<SolverSettings>,
) -> Vec<AccountResult> {
    let solver = Solver::new(settings.unwrap_or_default());
    map_accounts(accounts, &solver)
}

#[cfg(feature = "rayon")]
fn map_accounts(accounts: &[BatchAccount], solver: &Solver) -> Vec<AccountResult> {
    use rayon::prelude::*;
    accounts
        .par_iter()
        .map(|account| solve_account(account, solver))
        .collect()
}

#[cfg(not(feature = "rayon"))]
fn map_accounts(accounts: &[BatchAccount], solver: &Solver) -> Vec<AccountResult> {
    accounts
        .iter()
        .map(|account| solve_account(account, solver))
        .collect()
}

fn solve_account(account: &BatchAccount, solver: &Solver) -> AccountResult {
    if account.chain_previous_weights && account.problem.previous_weights().is_none() {
        return Err(PortfolioError::InvalidParameter(
            "chain_previous_weights requires a problem built with \
             with_quadratic_turnover and/or with_l1_turnover; without a \
             turnover term there is no anchor to move",
        ));
    }
    let mut sequence = account.problem.sequence_with(solver)?;
    let mut solutions = Vec::with_capacity(account.steps.len());
    let mut held_weights: Option<Vec<f64>> = None;
    for step in &account.steps {
        let chained;
        let step = if account.chain_previous_weights
            && step.previous_weights.is_none()
            && held_weights.is_some()
        {
            chained = RebalanceStep {
                previous_weights: held_weights.clone(),
                ..step.clone()
            };
            &chained
        } else {
            step
        };
        let solution = sequence.solve_next(step)?;
        if account.chain_previous_weights && solution.status == SolveStatus::Solved {
            held_weights = Some(solution.x.clone());
        }
        solutions.push(solution);
    }
    Ok(solutions)
}
