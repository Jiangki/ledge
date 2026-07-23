//! Ledge driven through the phased comparison protocol.

use ledge_core::{QpProblem, Solution, SolveStatus, Solver, WarmStart, Workspace};

use crate::protocol::{AdapterSolve, PhasedSolver, WarmStartSupport};

/// Ledge adapter.
///
/// `setup` builds a [`Workspace`] (roadmap 2.4): Ruiz equilibration and the
/// first SMW-reduced factorization are charged to the setup phase, matching
/// how OSQP's symbolic + numeric factorization is charged. Cold and rolling
/// solves reuse the cached factorization; rolling steps update the linear
/// cost in place and warm-start from the previous solution with full
/// primal + dual iterates.
pub struct LedgeAdapter {
    solver: Solver,
    workspace: Option<Workspace>,
    primal_start: Vec<f64>,
    last_solution: Option<Solution>,
}

impl LedgeAdapter {
    /// Creates the adapter with explicit solver settings.
    #[must_use]
    pub const fn new(solver: Solver) -> Self {
        Self {
            solver,
            workspace: None,
            primal_start: Vec::new(),
            last_solution: None,
        }
    }

    fn record(solution: &Solution) -> AdapterSolve {
        AdapterSolve {
            native_status: solution.status.to_string(),
            solved: solution.status == SolveStatus::Solved,
            x: solution.x.clone(),
            dual: solution.dual.clone(),
            iterations: solution.iterations,
        }
    }
}

impl PhasedSolver for LedgeAdapter {
    fn name(&self) -> String {
        "ledge".to_owned()
    }

    fn warm_start_support(&self) -> WarmStartSupport {
        WarmStartSupport::Full
    }

    fn setup(&mut self, problem: &QpProblem, primal_start: &[f64]) -> Result<(), String> {
        self.workspace = Some(
            self.solver
                .workspace(problem)
                .map_err(|error| error.to_string())?,
        );
        self.primal_start = primal_start.to_vec();
        self.last_solution = None;
        Ok(())
    }

    fn solve_cold(&mut self) -> Result<AdapterSolve, String> {
        let workspace = self.workspace.as_mut().ok_or("setup was not called")?;
        let warm = WarmStart::from_primal(self.primal_start.clone());
        let solution = workspace
            .solve(Some(&warm))
            .map_err(|error| error.to_string())?;
        let record = Self::record(&solution);
        self.last_solution = Some(solution);
        Ok(record)
    }

    fn resolve_with_linear(&mut self, linear: &[f64]) -> Result<AdapterSolve, String> {
        let workspace = self.workspace.as_mut().ok_or("setup was not called")?;
        workspace
            .update_linear(linear)
            .map_err(|error| error.to_string())?;
        let warm = self.last_solution.as_ref().map(Solution::warm_start);
        let solution = workspace
            .solve(warm.as_ref())
            .map_err(|error| error.to_string())?;
        let record = Self::record(&solution);
        self.last_solution = Some(solution);
        Ok(record)
    }
}
