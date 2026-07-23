//! Minimal comparison harness shared by examples and future integrations.

use std::{fmt::Write as _, time::Duration};

use crate::{QpProblem, Solution, SolveStatus, Solver, WarmStart};

/// Solver-neutral adapter for benchmark integrations.
///
/// Optional OSQP or Clarabel adapters can implement this trait without
/// changing the instance generator or report format.
pub trait ComparisonSolver {
    /// Stable label used in reports.
    fn name(&self) -> &'static str;

    /// Solves one problem, optionally from a common primal warm start.
    ///
    /// # Errors
    ///
    /// Returns a human-readable adapter or solver error.
    fn solve(
        &mut self,
        problem: &QpProblem,
        warm_start: Option<&WarmStart>,
    ) -> Result<Solution, String>;
}

impl ComparisonSolver for Solver {
    fn name(&self) -> &'static str {
        "ledge"
    }

    fn solve(
        &mut self,
        problem: &QpProblem,
        warm_start: Option<&WarmStart>,
    ) -> Result<Solution, String> {
        Solver::solve(self, problem, warm_start).map_err(|error| error.to_string())
    }
}

/// One measured solver/instance result.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkRecord {
    /// Instance label.
    pub instance: String,
    /// Solver label.
    pub solver: String,
    /// Termination status.
    pub status: SolveStatus,
    /// Objective at the returned point.
    pub objective: f64,
    /// Primal KKT residual.
    pub primal_residual: f64,
    /// Dual KKT residual.
    pub dual_residual: f64,
    /// ADMM or adapter iteration count.
    pub iterations: usize,
    /// Measured solver-reported duration.
    pub duration: Duration,
}

impl BenchmarkRecord {
    /// Renders one Markdown table row.
    #[must_use]
    pub fn markdown_row(&self) -> String {
        format!(
            "| {} | {} | {:?} | {:.8e} | {:.3e} | {:.3e} | {} | {:.3} |",
            self.instance,
            self.solver,
            self.status,
            self.objective,
            self.primal_residual,
            self.dual_residual,
            self.iterations,
            self.duration.as_secs_f64() * 1_000.0
        )
    }
}

/// Executes the same generated instance through registered adapters.
#[derive(Default)]
pub struct BenchmarkRunner {
    solvers: Vec<Box<dyn ComparisonSolver>>,
}

impl BenchmarkRunner {
    /// Creates an empty runner.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            solvers: Vec::new(),
        }
    }

    /// Registers a solver adapter.
    pub fn add_solver(&mut self, solver: Box<dyn ComparisonSolver>) {
        self.solvers.push(solver);
    }

    /// Runs all adapters and returns records or labeled failures.
    pub fn run(
        &mut self,
        instance_name: &str,
        problem: &QpProblem,
        warm_start: Option<&WarmStart>,
    ) -> Vec<Result<BenchmarkRecord, String>> {
        self.solvers
            .iter_mut()
            .map(|solver| {
                let solver_name = solver.name().to_owned();
                solver
                    .solve(problem, warm_start)
                    .map(|solution| BenchmarkRecord {
                        instance: instance_name.to_owned(),
                        solver: solver_name.clone(),
                        status: solution.status,
                        objective: solution.objective,
                        primal_residual: solution.residuals.primal,
                        dual_residual: solution.residuals.dual,
                        iterations: solution.iterations,
                        duration: solution.solve_time,
                    })
                    .map_err(|error| {
                        let mut message = String::new();
                        let _ = write!(message, "{solver_name}: {error}");
                        message
                    })
            })
            .collect()
    }

    /// Markdown header matching [`BenchmarkRecord::markdown_row`].
    #[must_use]
    pub const fn markdown_header() -> &'static str {
        "| instance | solver | status | objective | primal residual | dual residual | iterations | time (ms) |\n\
         |---|---|---:|---:|---:|---:|---:|---:|"
    }
}
