//! Clarabel adapter (enabled by the non-default `clarabel` feature).
//!
//! Convention conversion (protocol rule 2): Clarabel minimizes
//! `0.5 x' P x + q' x` subject to `A x + s = b`, `s` in a cone product —
//! the same objective convention as Ledge. Ledge's blocks map to:
//!
//! - equalities (plus lifted rows): rows in a zero cone;
//! - upper inequalities (plus L1 epigraph rows): rows in a nonnegative cone;
//! - variable boxes: `+x_i <= u_i` and `-x_i <= -l_i` rows (finite bounds
//!   only) in the same nonnegative cone.
//!
//! Clarabel's dual `z` satisfies `P x + q + A' z = 0`, so equality and
//! inequality multipliers map sign-for-sign onto Ledge's convention, the
//! box multiplier is `z_upper - z_lower`, and epigraph-row multipliers
//! combine into the L1 subgradient dual via
//! [`ConvertedQp::split_inequality_duals`].
//!
//! Clarabel is an interior-point method and does not accept warm starts;
//! per protocol rule 3 this is recorded rather than worked around. Rolling
//! steps reuse the setup (symbolic structure) through `update_q` and re-run
//! the solver.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use ledge_core::{DualVariables, QpProblem};

use crate::{
    convert::{ConvertedQp, Formulation},
    protocol::{AdapterSolve, PhasedSolver, WarmStartSupport},
};

/// Clarabel driven through the phased comparison protocol.
pub struct ClarabelAdapter {
    formulation: Formulation,
    state: Option<State>,
}

struct State {
    converted: ConvertedQp,
    solver: DefaultSolver<f64>,
    upper_variables: Vec<usize>,
    lower_variables: Vec<usize>,
}

impl ClarabelAdapter {
    /// Creates an adapter for the requested formulation.
    #[must_use]
    pub const fn new(formulation: Formulation) -> Self {
        Self {
            formulation,
            state: None,
        }
    }

    fn solve_inner(&mut self) -> Result<AdapterSolve, String> {
        let state = self.state.as_mut().ok_or("setup was not called")?;
        state.solver.solve();
        let solution = &state.solver.solution;
        let native_status = status_label(solution.status).to_owned();
        match solution.status {
            SolverStatus::Solved
            | SolverStatus::AlmostSolved
            | SolverStatus::MaxIterations
            | SolverStatus::MaxTime
            | SolverStatus::InsufficientProgress => {}
            _ => {
                return Err(format!(
                    "terminated without a primal point: {native_status}"
                ))
            }
        }

        let converted = &state.converted;
        let n = converted.original_variables;
        let equality_count = converted.equality_rows.len() - converted.lifted_rows;
        let inequality_offset = converted.equality_rows.len();
        let upper_offset = inequality_offset + converted.inequality_rows.len();
        let lower_offset = upper_offset + state.upper_variables.len();
        let (inequalities, l1) =
            converted.split_inequality_duals(&solution.z[inequality_offset..upper_offset]);
        let mut dual = DualVariables {
            equalities: solution.z[..equality_count].to_vec(),
            inequalities,
            bounds: vec![0.0; n],
            l1,
        };
        for (row, variable) in state.upper_variables.iter().enumerate() {
            dual.bounds[*variable] += solution.z[upper_offset + row];
        }
        for (row, variable) in state.lower_variables.iter().enumerate() {
            dual.bounds[*variable] -= solution.z[lower_offset + row];
        }
        Ok(AdapterSolve {
            native_status,
            solved: solution.status == SolverStatus::Solved,
            x: solution.x[..n].to_vec(),
            dual,
            iterations: solution.iterations as usize,
        })
    }
}

impl PhasedSolver for ClarabelAdapter {
    fn name(&self) -> String {
        format!("clarabel ({})", self.formulation.label())
    }

    fn warm_start_support(&self) -> WarmStartSupport {
        WarmStartSupport::None
    }

    fn setup(&mut self, problem: &QpProblem, _primal_start: &[f64]) -> Result<(), String> {
        let converted = ConvertedQp::new(problem, self.formulation)?;
        let upper_variables: Vec<usize> = (0..converted.original_variables)
            .filter(|variable| converted.upper_bounds[*variable].is_finite())
            .collect();
        let lower_variables: Vec<usize> = (0..converted.original_variables)
            .filter(|variable| converted.lower_bounds[*variable].is_finite())
            .collect();
        let upper_rows: Vec<Vec<(usize, f64)>> = upper_variables
            .iter()
            .map(|variable| vec![(*variable, 1.0)])
            .collect();
        let lower_rows: Vec<Vec<(usize, f64)>> = lower_variables
            .iter()
            .map(|variable| vec![(*variable, -1.0)])
            .collect();
        let stacked = converted.csc_from_rows(&[
            &converted.equality_rows,
            &converted.inequality_rows,
            &upper_rows,
            &lower_rows,
        ]);

        let mut rhs = Vec::with_capacity(stacked.rows);
        rhs.extend_from_slice(&converted.equality_rhs);
        rhs.extend_from_slice(&converted.inequality_rhs);
        for variable in &upper_variables {
            rhs.push(converted.upper_bounds[*variable]);
        }
        for variable in &lower_variables {
            rhs.push(-converted.lower_bounds[*variable]);
        }
        let cones = [
            SupportedConeT::ZeroConeT(converted.equality_rows.len()),
            SupportedConeT::NonnegativeConeT(stacked.rows - converted.equality_rows.len()),
        ];

        let quadratic = CscMatrix::new(
            converted.quadratic_upper.rows,
            converted.quadratic_upper.cols,
            converted.quadratic_upper.indptr.clone(),
            converted.quadratic_upper.indices.clone(),
            converted.quadratic_upper.values.clone(),
        );
        let constraints = CscMatrix::new(
            stacked.rows,
            stacked.cols,
            stacked.indptr,
            stacked.indices,
            stacked.values,
        );
        // Clarabel keeps its own (tighter) default tolerances; presolve and
        // zero dropping are disabled so update_q stays legal during the
        // rolling phase. Both choices are documented in the report.
        let settings = DefaultSettings {
            verbose: false,
            presolve_enable: false,
            input_sparse_dropzeros: false,
            ..DefaultSettings::default()
        };
        let solver = DefaultSolver::new(
            &quadratic,
            &converted.linear,
            &constraints,
            &rhs,
            &cones,
            settings,
        )
        .map_err(|error| error.to_string())?;

        self.state = Some(State {
            converted,
            solver,
            upper_variables,
            lower_variables,
        });
        Ok(())
    }

    fn solve_cold(&mut self) -> Result<AdapterSolve, String> {
        self.solve_inner()
    }

    fn resolve_with_linear(&mut self, linear: &[f64]) -> Result<AdapterSolve, String> {
        let state = self.state.as_mut().ok_or("setup was not called")?;
        let extended = state.converted.extend_linear(linear);
        state
            .solver
            .update_q(&extended)
            .map_err(|error| error.to_string())?;
        self.solve_inner()
    }
}

fn status_label(status: SolverStatus) -> &'static str {
    match status {
        SolverStatus::Unsolved => "unsolved",
        SolverStatus::Solved => "solved",
        SolverStatus::PrimalInfeasible => "primal_infeasible",
        SolverStatus::DualInfeasible => "dual_infeasible",
        SolverStatus::AlmostSolved => "almost_solved",
        SolverStatus::AlmostPrimalInfeasible => "almost_primal_infeasible",
        SolverStatus::AlmostDualInfeasible => "almost_dual_infeasible",
        SolverStatus::MaxIterations => "max_iterations",
        SolverStatus::MaxTime => "max_time",
        SolverStatus::NumericalError => "numerical_error",
        SolverStatus::InsufficientProgress => "insufficient_progress",
        SolverStatus::CallbackTerminated => "callback_terminated",
    }
}
