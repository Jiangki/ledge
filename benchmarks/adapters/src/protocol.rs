//! Phase-structured comparison runner implementing `benchmarks/README.md`.
//!
//! Every measurement is a uniform wall-clock time around the adapter call so
//! that no solver benefits from a different internal timing definition. All
//! returned points are independently re-checked with `ledge_core::check_kkt`
//! against the *original* problem data (protocol rule 5), and each solver's
//! own termination status is recorded verbatim (rule 4).

use std::time::{Duration, Instant};

use ledge_core::{check_kkt, DualVariables, QpProblem};

/// Warm-start capability an adapter can honestly claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmStartSupport {
    /// Primal and dual warm starts are used.
    Full,
    /// Only a primal warm start is used.
    PrimalOnly,
    /// The solver restarts from scratch on every solve.
    None,
}

impl WarmStartSupport {
    /// Human-readable label for reports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Full => "primal + dual",
            Self::PrimalOnly => "primal only",
            Self::None => "unsupported",
        }
    }
}

/// One adapter solve, in original-problem coordinates.
#[derive(Clone, Debug)]
pub struct AdapterSolve {
    /// The solver's own termination status, verbatim.
    pub native_status: String,
    /// Whether the native status means "optimal at the solver's tolerance".
    pub solved: bool,
    /// Primal point restricted to the original variables.
    pub x: Vec<f64>,
    /// Multipliers mapped into Ledge's dual convention.
    pub dual: DualVariables,
    /// Iteration count reported by the solver.
    pub iterations: usize,
}

/// A comparison adapter driven through setup / cold / rolling phases.
pub trait PhasedSolver {
    /// Stable label, including the formulation when relevant.
    fn name(&self) -> String;

    /// Warm-start capability used during the rolling phase.
    fn warm_start_support(&self) -> WarmStartSupport;

    /// Builds all solver-side state for `problem`.
    ///
    /// # Errors
    ///
    /// Returns a human-readable adapter or solver error.
    fn setup(&mut self, problem: &QpProblem, primal_start: &[f64]) -> Result<(), String>;

    /// First solve after setup, from the shared primal start.
    ///
    /// # Errors
    ///
    /// Returns a human-readable adapter or solver error.
    fn solve_cold(&mut self) -> Result<AdapterSolve, String>;

    /// Re-solves after replacing the linear cost, warm-started from the
    /// adapter's previous solution where supported.
    ///
    /// # Errors
    ///
    /// Returns a human-readable adapter or solver error.
    fn resolve_with_linear(&mut self, linear: &[f64]) -> Result<AdapterSolve, String>;
}

/// Measurement phase labels used in the CSV output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Conversion plus solver construction.
    Setup,
    /// First solve from the shared primal start.
    Cold,
    /// One rolling re-solve step.
    Roll,
}

impl Phase {
    const fn label(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::Cold => "cold",
            Self::Roll => "roll",
        }
    }
}

/// One measured sample row.
#[derive(Clone, Debug)]
pub struct Sample {
    /// Instance label.
    pub instance: String,
    /// Adapter label.
    pub solver: String,
    /// 1-based repeat index.
    pub repeat: usize,
    /// Measurement phase.
    pub phase: Phase,
    /// 1-based rolling step, `0` for setup/cold rows.
    pub step: usize,
    /// Native termination status (empty for setup rows).
    pub native_status: String,
    /// Whether the solver claimed optimality (false for setup rows).
    pub solved: bool,
    /// Objective evaluated by Ledge on the original data (NaN for setup).
    pub objective: f64,
    /// Independent KKT primal residual (NaN for setup rows).
    pub kkt_primal: f64,
    /// Independent KKT dual residual (NaN for setup rows).
    pub kkt_dual: f64,
    /// Solver-reported iterations (0 for setup rows).
    pub iterations: usize,
    /// Wall-clock duration of the phase.
    pub duration: Duration,
}

impl Sample {
    /// CSV header matching [`Sample::csv_row`].
    #[must_use]
    pub const fn csv_header() -> &'static str {
        "instance,solver,repeat,phase,step,native_status,solved,objective,kkt_primal,kkt_dual,iterations,time_ms"
    }

    /// Renders one CSV row.
    #[must_use]
    pub fn csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.10e},{:.3e},{:.3e},{},{:.3}",
            self.instance,
            self.solver,
            self.repeat,
            self.phase.label(),
            self.step,
            self.native_status,
            self.solved,
            self.objective,
            self.kkt_primal,
            self.kkt_dual,
            self.iterations,
            self.duration.as_secs_f64() * 1_000.0
        )
    }
}

/// Rolling-sequence workload shared by every adapter.
///
/// The base instance is solved once cold, then `linear_steps` provides a
/// deterministic sequence of perturbed linear costs (new expected returns)
/// that each adapter re-solves in order, warm-started where supported.
pub struct RollingWorkload<'a> {
    /// Instance label.
    pub instance: String,
    /// Base problem, shared verbatim across adapters (protocol rule 1).
    pub problem: &'a QpProblem,
    /// Shared primal start for the cold solve (protocol rule 3).
    pub primal_start: &'a [f64],
    /// Perturbed linear costs for the rolling phase, one per step.
    pub linear_steps: &'a [Vec<f64>],
}

/// Runs one adapter through the full workload `repeats` times.
///
/// # Errors
///
/// The first adapter failure aborts the remaining repeats for that adapter
/// and is returned as a labeled error string.
pub fn run_workload(
    workload: &RollingWorkload<'_>,
    solver: &mut dyn PhasedSolver,
    repeats: usize,
) -> Result<Vec<Sample>, String> {
    let mut samples = Vec::new();
    let label = solver.name();
    for repeat in 1..=repeats {
        let started = Instant::now();
        solver
            .setup(workload.problem, workload.primal_start)
            .map_err(|error| format!("{label}: setup: {error}"))?;
        samples.push(Sample {
            instance: workload.instance.clone(),
            solver: label.clone(),
            repeat,
            phase: Phase::Setup,
            step: 0,
            native_status: String::new(),
            solved: false,
            objective: f64::NAN,
            kkt_primal: f64::NAN,
            kkt_dual: f64::NAN,
            iterations: 0,
            duration: started.elapsed(),
        });

        let started = Instant::now();
        let cold = solver
            .solve_cold()
            .map_err(|error| format!("{label}: cold solve: {error}"))?;
        let cold_duration = started.elapsed();
        samples.push(checked_sample(
            workload.problem,
            &workload.instance,
            &label,
            repeat,
            Phase::Cold,
            0,
            &cold,
            cold_duration,
        )?);

        let mut rolled = workload.problem.clone();
        for (index, linear) in workload.linear_steps.iter().enumerate() {
            rolled.linear.clone_from(linear);
            let started = Instant::now();
            let step = solver
                .resolve_with_linear(linear)
                .map_err(|error| format!("{label}: roll step {}: {error}", index + 1))?;
            let duration = started.elapsed();
            samples.push(checked_sample(
                &rolled,
                &workload.instance,
                &label,
                repeat,
                Phase::Roll,
                index + 1,
                &step,
                duration,
            )?);
        }
    }
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
fn checked_sample(
    problem: &QpProblem,
    instance: &str,
    solver: &str,
    repeat: usize,
    phase: Phase,
    step: usize,
    solve: &AdapterSolve,
    duration: Duration,
) -> Result<Sample, String> {
    let residuals =
        check_kkt(problem, &solve.x, &solve.dual).map_err(|error| format!("{solver}: {error}"))?;
    Ok(Sample {
        instance: instance.to_owned(),
        solver: solver.to_owned(),
        repeat,
        phase,
        step,
        native_status: solve.native_status.clone(),
        solved: solve.solved,
        objective: problem.objective(&solve.x),
        kkt_primal: residuals.primal,
        kkt_dual: residuals.dual,
        iterations: solve.iterations,
        duration,
    })
}
