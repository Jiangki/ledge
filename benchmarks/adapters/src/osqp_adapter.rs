//! OSQP adapter (enabled by the non-default `osqp` feature).
//!
//! Convention conversion (protocol rule 2): OSQP minimizes
//! `0.5 x' P x + q' x` subject to `l <= A x <= u`, the same objective
//! convention as Ledge, so `P` and `q` are passed through unchanged. Ledge's
//! blocks are stacked into one row set:
//!
//! - equalities (plus lifted rows): `l = u = b`;
//! - upper inequalities (plus L1 epigraph rows): `l = -1e30` (OSQP's
//!   infinity), `u = b`;
//! - variable boxes: one identity row per variable with a finite bound
//!   (epigraph variables carry no box row; the epigraph rows bound them).
//!
//! Dual mapping back to Ledge's convention is sign-preserving: both solvers
//! write stationarity as `Q x + q + A' y = 0` with multipliers positive at
//! active upper limits and negative at active lower limits. Epigraph-row
//! multipliers combine into the L1 subgradient dual via
//! [`ConvertedQp::split_inequality_duals`].

use ledge_core::{DualVariables, QpProblem};
use osqp::{CscMatrix, Problem, Settings, Status};

use crate::{
    convert::{ConvertedQp, Formulation, OSQP_INFINITY},
    protocol::{AdapterSolve, PhasedSolver, WarmStartSupport},
};

/// OSQP driven through the phased comparison protocol.
///
/// Setup covers conversion plus `osqp_setup` (symbolic + numeric
/// factorization). Rolling steps call `update_lin_cost` and warm-start both
/// primal and dual vectors from the previous solve.
pub struct OsqpAdapter {
    formulation: Formulation,
    state: Option<State>,
}

struct State {
    converted: ConvertedQp,
    problem: Problem,
    box_variables: Vec<usize>,
    primal_start: Vec<f64>,
    last_iterates: Option<(Vec<f64>, Vec<f64>)>,
}

impl OsqpAdapter {
    /// Creates an adapter for the requested formulation.
    #[must_use]
    pub const fn new(formulation: Formulation) -> Self {
        Self {
            formulation,
            state: None,
        }
    }

    fn solve_inner(&mut self, warm: WarmAction) -> Result<AdapterSolve, String> {
        let state = self.state.as_mut().ok_or("setup was not called")?;
        match warm {
            WarmAction::PrimalStart => {
                state.problem.warm_start_x(&state.primal_start);
            }
            WarmAction::PreviousSolution => {
                if let Some((x, y)) = &state.last_iterates {
                    state.problem.warm_start(x, y);
                }
            }
        }
        let (native_status, solved, full_x, full_y, iterations) = {
            let status = state.problem.solve();
            let label = status_label(&status);
            let (Status::Solved(solution)
            | Status::SolvedInaccurate(solution)
            | Status::MaxIterationsReached(solution)
            | Status::TimeLimitReached(solution)) = &status
            else {
                return Err(format!("terminated without a primal point: {label}"));
            };
            (
                label.to_owned(),
                matches!(status, Status::Solved(_)),
                solution.x().to_vec(),
                solution.y().to_vec(),
                status.iter() as usize,
            )
        };

        let converted = &state.converted;
        let n = converted.original_variables;
        let equality_count = converted.equality_rows.len() - converted.lifted_rows;
        let inequality_count = converted.inequality_rows.len();
        let inequality_offset = converted.equality_rows.len();
        let box_offset = inequality_offset + inequality_count;
        let (inequalities, l1) =
            converted.split_inequality_duals(&full_y[inequality_offset..box_offset]);
        let mut dual = DualVariables {
            equalities: full_y[..equality_count].to_vec(),
            inequalities,
            bounds: vec![0.0; n],
            l1,
        };
        for (row, variable) in state.box_variables.iter().enumerate() {
            dual.bounds[*variable] = full_y[box_offset + row];
        }
        let record = AdapterSolve {
            native_status,
            solved,
            x: full_x[..n].to_vec(),
            dual,
            iterations,
        };
        state.last_iterates = Some((full_x, full_y));
        Ok(record)
    }
}

#[derive(Clone, Copy)]
enum WarmAction {
    PrimalStart,
    PreviousSolution,
}

impl PhasedSolver for OsqpAdapter {
    fn name(&self) -> String {
        format!("osqp ({})", self.formulation.label())
    }

    fn warm_start_support(&self) -> WarmStartSupport {
        WarmStartSupport::Full
    }

    fn setup(&mut self, problem: &QpProblem, primal_start: &[f64]) -> Result<(), String> {
        let converted = ConvertedQp::new(problem, self.formulation)?;
        let box_variables: Vec<usize> = (0..converted.original_variables)
            .filter(|variable| {
                converted.lower_bounds[*variable].is_finite()
                    || converted.upper_bounds[*variable].is_finite()
            })
            .collect();
        let box_rows: Vec<Vec<(usize, f64)>> = box_variables
            .iter()
            .map(|variable| vec![(*variable, 1.0)])
            .collect();
        let stacked = converted.csc_from_rows(&[
            &converted.equality_rows,
            &converted.inequality_rows,
            &box_rows,
        ]);

        let mut lower = Vec::with_capacity(stacked.rows);
        let mut upper = Vec::with_capacity(stacked.rows);
        lower.extend_from_slice(&converted.equality_rhs);
        upper.extend_from_slice(&converted.equality_rhs);
        for rhs in &converted.inequality_rhs {
            lower.push(-OSQP_INFINITY);
            upper.push(*rhs);
        }
        for variable in &box_variables {
            lower.push(converted.lower_bounds[*variable].max(-OSQP_INFINITY));
            upper.push(converted.upper_bounds[*variable].min(OSQP_INFINITY));
        }

        let quadratic = CscMatrix {
            nrows: converted.quadratic_upper.rows,
            ncols: converted.quadratic_upper.cols,
            indptr: converted.quadratic_upper.indptr.clone().into(),
            indices: converted.quadratic_upper.indices.clone().into(),
            data: converted.quadratic_upper.values.clone().into(),
        };
        let constraints = CscMatrix {
            nrows: stacked.rows,
            ncols: stacked.cols,
            indptr: stacked.indptr.into(),
            indices: stacked.indices.into(),
            data: stacked.values.into(),
        };
        // Tolerances matched to Ledge defaults; everything else stays at
        // OSQP's own defaults (scaling, adaptive rho, no polishing).
        let settings = Settings::default()
            .max_iter(10_000)
            .eps_abs(1.0e-6)
            .eps_rel(1.0e-5)
            .verbose(false);
        let osqp_problem = Problem::new(
            quadratic,
            &converted.linear,
            constraints,
            &lower,
            &upper,
            &settings,
        )
        .map_err(|error| error.to_string())?;

        let primal_start = converted.lift_primal(primal_start);
        self.state = Some(State {
            converted,
            problem: osqp_problem,
            box_variables,
            primal_start,
            last_iterates: None,
        });
        Ok(())
    }

    fn solve_cold(&mut self) -> Result<AdapterSolve, String> {
        self.solve_inner(WarmAction::PrimalStart)
    }

    fn resolve_with_linear(&mut self, linear: &[f64]) -> Result<AdapterSolve, String> {
        let state = self.state.as_mut().ok_or("setup was not called")?;
        let extended = state.converted.extend_linear(linear);
        state.problem.update_lin_cost(&extended);
        self.solve_inner(WarmAction::PreviousSolution)
    }
}

fn status_label(status: &Status<'_>) -> &'static str {
    match status {
        Status::Solved(_) => "solved",
        Status::SolvedInaccurate(_) => "solved_inaccurate",
        Status::MaxIterationsReached(_) => "max_iterations",
        Status::TimeLimitReached(_) => "time_limit",
        Status::PrimalInfeasible(_) => "primal_infeasible",
        Status::PrimalInfeasibleInaccurate(_) => "primal_infeasible_inaccurate",
        Status::DualInfeasible(_) => "dual_infeasible",
        Status::DualInfeasibleInaccurate(_) => "dual_infeasible_inaccurate",
        Status::NonConvex(_) => "non_convex",
        _ => "unknown",
    }
}
