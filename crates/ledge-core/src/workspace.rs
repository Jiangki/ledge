//! Reusable solver workspace: factorization cache across solves.
//!
//! Every one-shot [`Solver::solve`](crate::Solver::solve) pays two structural
//! costs before iterating: Ruiz equilibration of the problem data and the
//! SMW-reduced factorization (`O(n r^2 + r^3)` with reduced dimension
//! `r = factors + explicit linear constraints`). Rolling rebalances keep the
//! covariance and constraint structure fixed and only move the linear cost
//! (new expected returns / previous weights) or right-hand sides, so both
//! costs can be paid once and reused.
//!
//! [`Workspace`] owns that reusable state. It is created from a solver and a
//! problem via [`crate::Solver::workspace`], re-solves with
//! [`Workspace::solve`], and accepts cheap in-place data updates
//! ([`Workspace::update_linear`], [`Workspace::update_equality_rhs`],
//! [`Workspace::update_inequality_rhs`]) that never invalidate the cached
//! factorizations.
//!
//! Every solve replays the one-shot penalty policy exactly — it starts at
//! [`SolverSettings::rho`] and adapts as usual — so a workspace changes cost,
//! never iterates: solving through a workspace and through a fresh
//! [`Solver::solve`](crate::Solver::solve) of the same data walk the same
//! path. (Measured on rolling workloads, carrying the previous solve's final
//! penalty over instead *increased* iteration counts; see
//! `docs/DECISIONS.md`.) Factorizations are cached in a small
//! least-recently-used table keyed by the penalty, so the ladder of penalties
//! an adaptive solve revisits is factored once per workspace, not once per
//! solve: warm-started rolling sequences reach a steady state of zero
//! refactorizations per step.
//!
//! Auditability rules are unchanged from the one-shot path: the equilibration
//! computed at construction is an exact transform applied to every update, and
//! termination checks plus every reported residual are always evaluated on the
//! original (updated) data.

use std::time::Instant;

use crate::{
    certificate::{
        check_dual_certificate, check_primal_certificate, detect_dual_infeasibility,
        detect_primal_infeasibility, Certificate,
    },
    check_kkt,
    kkt::{DualVariables, KktResiduals},
    linalg::{covariance_columns, SmwSystem},
    matrix::{norm_inf, Matrix},
    polish::polish,
    problem::{ProblemError, QpProblem},
    scaling::ScaledProblem,
    solver::{
        validate_settings, ConvergenceDiagnostics, Solution, SolveStatus, SolverError,
        SolverSettings, WarmStart,
    },
};

/// Reduced factorizations kept per workspace. The adaptive-ρ ladder of a
/// well-scaled solve visits a handful of penalties; pathological ladders
/// simply fall back to least-recently-used rebuilding.
const FACTORIZATION_CACHE_CAPACITY: usize = 16;

/// Reusable solve state for a fixed problem structure.
///
/// Constructed by [`crate::Solver::workspace`]. Holds the equilibrated
/// problem copy and a penalty-keyed cache of SMW-reduced factorizations, so
/// repeated solves skip both setup costs. See the
/// [module documentation](self) for what is cached and when it is rebuilt.
pub struct Workspace {
    settings: SolverSettings,
    /// Original-space problem data; updated in place by the `update_*`
    /// methods and used for termination checks and reported residuals.
    problem: QpProblem,
    scaling: Option<ScaledProblem>,
    /// Most-recently-used-first factorization cache keyed by exact penalty
    /// bits. The adaptive policy replays the same multiplicative ladder on
    /// every solve, so lookups are bit-exact.
    systems: Vec<(f64, FactorizedSystem)>,
    factorizations: usize,
}

impl Workspace {
    /// Validates the problem and settings, equilibrates once, and builds the
    /// first reduced factorization.
    pub(crate) fn new(settings: &SolverSettings, problem: &QpProblem) -> Result<Self, SolverError> {
        problem.validate()?;
        validate_settings(settings)?;
        let scaling = if settings.scaling_iterations > 0 {
            Some(ScaledProblem::new(problem, settings.scaling_iterations)?)
        } else {
            None
        };
        let work = scaling.as_ref().map_or(problem, |scaled| &scaled.problem);
        let system = FactorizedSystem::new(work, settings.rho, settings.sigma)?;
        Ok(Self {
            settings: settings.clone(),
            problem: problem.clone(),
            scaling,
            systems: vec![(settings.rho, system)],
            factorizations: 1,
        })
    }

    /// Returns the problem data currently held by the workspace (original,
    /// unscaled space, including any updates applied so far).
    #[must_use]
    pub fn problem(&self) -> &QpProblem {
        &self.problem
    }

    /// Returns the settings this workspace iterates with.
    #[must_use]
    pub const fn settings(&self) -> &SolverSettings {
        &self.settings
    }

    /// Number of reduced factorizations built since construction (including
    /// the initial one). Stable counts across rolling solves demonstrate
    /// factorization reuse.
    #[must_use]
    pub const fn factorizations(&self) -> usize {
        self.factorizations
    }

    /// Replaces the linear objective coefficient (for portfolios: new
    /// expected returns and/or previous weights folded into `q`).
    ///
    /// The cached factorizations stay valid; the equilibration computed at
    /// construction is reapplied to the new vector as an exact transform.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::InvalidProblem`] when the vector has the wrong
    /// length or contains a non-finite value.
    pub fn update_linear(&mut self, linear: &[f64]) -> Result<(), SolverError> {
        replace_vector("linear", &mut self.problem.linear, linear)?;
        if let Some(scaled) = &mut self.scaling {
            scaled.set_linear(linear);
        }
        Ok(())
    }

    /// Replaces the anchor of the L1 term (for portfolios: the previous
    /// weights the proportional turnover cost is measured from).
    ///
    /// The anchor enters neither the quadratic nor the constraint matrices,
    /// so the cached factorizations stay valid; the frozen variable scaling
    /// is reapplied as an exact transform. The costs themselves are part of
    /// the problem structure held by this workspace and stay fixed.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::InvalidProblem`] when the problem has no L1
    /// term, or the vector has the wrong length or a non-finite value.
    pub fn update_l1_anchor(&mut self, anchor: &[f64]) -> Result<(), SolverError> {
        let Some(term) = &mut self.problem.l1 else {
            return Err(SolverError::InvalidProblem(ProblemError::Dimension {
                field: "l1.anchor",
                expected: 0,
                actual: anchor.len(),
            }));
        };
        replace_vector("l1.anchor", &mut term.anchor, anchor)?;
        if let Some(scaled) = &mut self.scaling {
            scaled.set_l1_anchor(anchor);
        }
        Ok(())
    }

    /// Replaces the equality right-hand side (for portfolios: a new budget).
    ///
    /// The cached factorizations stay valid.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::InvalidProblem`] when the vector has the wrong
    /// length or contains a non-finite value.
    pub fn update_equality_rhs(&mut self, rhs: &[f64]) -> Result<(), SolverError> {
        replace_vector("equalities.rhs", &mut self.problem.equalities.rhs, rhs)?;
        if let Some(scaled) = &mut self.scaling {
            scaled.set_equality_rhs(rhs);
        }
        Ok(())
    }

    /// Replaces the inequality right-hand side (for portfolios: new exposure
    /// caps with unchanged constraint rows).
    ///
    /// The cached factorizations stay valid.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::InvalidProblem`] when the vector has the wrong
    /// length or contains a non-finite value.
    pub fn update_inequality_rhs(&mut self, rhs: &[f64]) -> Result<(), SolverError> {
        replace_vector("inequalities.rhs", &mut self.problem.inequalities.rhs, rhs)?;
        if let Some(scaled) = &mut self.scaling {
            scaled.set_inequality_rhs(rhs);
        }
        Ok(())
    }

    /// Solves the current problem data, reusing the cached equilibration and
    /// factorizations.
    ///
    /// The iterate path is identical to a fresh
    /// [`Solver::solve`](crate::Solver::solve) of the same data: the penalty
    /// policy restarts from [`SolverSettings::rho`] and adapts as usual, but
    /// every penalty already visited by this workspace reuses its cached
    /// factorization instead of rebuilding it.
    ///
    /// Unlike [`crate::Solver::solve`], the reported
    /// [`Solution::solve_time`] covers only this call: one-time setup was
    /// paid at construction.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError`] when the warm start is invalid or a rebuilt
    /// reduced system cannot be factored.
    pub fn solve(&mut self, warm_start: Option<&WarmStart>) -> Result<Solution, SolverError> {
        self.solve_from(Instant::now(), warm_start)
    }

    /// Shared ADMM engine; `started` controls what the reported solve time
    /// includes so the one-shot path can charge setup to the solve.
    pub(crate) fn solve_from(
        &mut self,
        started: Instant,
        warm_start: Option<&WarmStart>,
    ) -> Result<Solution, SolverError> {
        let settings = self.settings.clone();
        // Field-level borrows: `work` reads `scaling`/`problem` while the
        // factorization cache below is mutated independently.
        let work = self
            .scaling
            .as_ref()
            .map_or(&self.problem, |scaled| &scaled.problem);
        let systems = &mut self.systems;
        let factorizations = &mut self.factorizations;

        let n = self.problem.quadratic.dimension();
        let equality_count = self.problem.equalities.len();
        let inequality_count = self.problem.inequalities.len();
        let l1_count = if self.problem.l1.is_some() { n } else { 0 };
        let mut x = if let Some(warm) = warm_start {
            validate_warm_vector("x", &warm.x, n)?;
            warm.x.clone()
        } else {
            vec![0.0; n]
        };
        let mut dual = DualVariables {
            equalities: warm_vector(
                warm_start.and_then(|warm| warm.equality_dual.as_deref()),
                "equality_dual",
                equality_count,
            )?,
            inequalities: warm_vector(
                warm_start.and_then(|warm| warm.inequality_dual.as_deref()),
                "inequality_dual",
                inequality_count,
            )?,
            bounds: warm_vector(
                warm_start.and_then(|warm| warm.bound_dual.as_deref()),
                "bound_dual",
                n,
            )?,
            l1: warm_vector(
                warm_start.and_then(|warm| warm.l1_dual.as_deref()),
                "l1_dual",
                l1_count,
            )?,
        };
        if let Some(scaled) = &self.scaling {
            scaled.scale_iterates_in_place(&mut x, &mut dual);
        }

        let mut rho = settings.rho;
        ensure_system(systems, factorizations, work, rho, settings.sigma)?;
        let mut equality_slack = vec![0.0; equality_count];
        let initial_inequality = work.inequalities.matrix.mul_vec(&x);
        let mut inequality_slack: Vec<f64> = initial_inequality
            .iter()
            .zip(&dual.inequalities)
            .zip(&work.inequalities.rhs)
            .map(|((value, multiplier), rhs)| (value + multiplier / rho).min(*rhs))
            .collect();
        equality_slack.copy_from_slice(&work.equalities.rhs);
        let mut bound_slack: Vec<f64> = x
            .iter()
            .zip(&dual.bounds)
            .enumerate()
            .map(|(index, (value, multiplier))| {
                project(
                    value + multiplier / rho,
                    work.lower_bounds[index],
                    work.upper_bounds[index],
                )
            })
            .collect();
        // L1 consensus block `x - anchor = z` (roadmap 2.1): its projection
        // is the soft threshold, initialized like the other slacks.
        let mut l1_slack: Vec<f64> = work.l1.as_ref().map_or_else(Vec::new, |term| {
            x.iter()
                .zip(&dual.l1)
                .enumerate()
                .map(|(index, (value, multiplier))| {
                    soft_threshold(
                        value - term.anchor[index] + multiplier / rho,
                        term.costs[index] / rho,
                    )
                })
                .collect()
        });

        let mut status = SolveStatus::MaxIterations;
        let mut completed_iterations = 0;
        let mut rho_updates = 0;
        let mut certificate: Option<Certificate> = None;
        // Infeasibility detection compares original-space iterates between
        // consecutive termination checks: on infeasible problems those
        // differences converge to certificate directions (OSQP-style; see
        // `certificate.rs`), while scaling-space artifacts never enter.
        let detect_infeasibility = settings.infeasibility_tolerance > 0.0;
        let (mut checked_x, mut checked_dual) = if detect_infeasibility {
            unscaled_iterates(self.scaling.as_ref(), &x, &dual)
        } else {
            (Vec::new(), DualVariables::default())
        };

        for iteration in 1..=settings.max_iterations {
            let mut right_hand_side: Vec<f64> = x
                .iter()
                .zip(&work.linear)
                .map(|(value, linear)| settings.sigma * value - linear)
                .collect();

            let equality_weight: Vec<f64> = equality_slack
                .iter()
                .zip(&dual.equalities)
                .map(|(slack, multiplier)| rho * slack - multiplier)
                .collect();
            work.equalities
                .matrix
                .transpose_mul_add(&equality_weight, &mut right_hand_side);
            let inequality_weight: Vec<f64> = inequality_slack
                .iter()
                .zip(&dual.inequalities)
                .map(|(slack, multiplier)| rho * slack - multiplier)
                .collect();
            work.inequalities
                .matrix
                .transpose_mul_add(&inequality_weight, &mut right_hand_side);
            for ((right_hand_side, slack), multiplier) in right_hand_side
                .iter_mut()
                .zip(&bound_slack)
                .zip(&dual.bounds)
            {
                *right_hand_side += rho * slack - multiplier;
            }
            if let Some(term) = &work.l1 {
                // From `rho/2 * ||x - anchor - z + y/rho||^2`:
                // the x-gradient contributes `rho * (anchor + z) - y`.
                for (((right_hand_side, slack), multiplier), anchor) in right_hand_side
                    .iter_mut()
                    .zip(&l1_slack)
                    .zip(&dual.l1)
                    .zip(&term.anchor)
                {
                    *right_hand_side += rho * (anchor + slack) - multiplier;
                }
            }

            // The current factorization is always the most recently used.
            systems[0].1.solve_in_place(&mut right_hand_side);
            x = right_hand_side;
            if x.iter().any(|value| !value.is_finite()) {
                status = SolveStatus::NumericalFailure;
                completed_iterations = iteration;
                break;
            }

            let equality_values = work.equalities.matrix.mul_vec(&x);
            let inequality_values = work.inequalities.matrix.mul_vec(&x);
            let adapt_rho = settings.adaptive_rho
                && iteration % settings.adaptive_rho_interval == 0
                && iteration < settings.max_iterations;
            let old_inequality_slack = if adapt_rho {
                inequality_slack.clone()
            } else {
                Vec::new()
            };
            let old_bound_slack = if adapt_rho {
                bound_slack.clone()
            } else {
                Vec::new()
            };
            let old_l1_slack = if adapt_rho {
                l1_slack.clone()
            } else {
                Vec::new()
            };
            // Over-relaxation: every consensus block sees the blend
            // `alpha * Ax + (1 - alpha) * z_prev` in place of `Ax`
            // (Boyd et al. §3.4.3); `alpha = 1` recovers plain ADMM.
            let alpha = settings.over_relaxation;
            for ((multiplier, value), slack) in dual
                .equalities
                .iter_mut()
                .zip(&equality_values)
                .zip(&equality_slack)
            {
                let relaxed = alpha * value + (1.0 - alpha) * slack;
                *multiplier += rho * (relaxed - slack);
            }
            for (((slack, multiplier), value), rhs) in inequality_slack
                .iter_mut()
                .zip(&mut dual.inequalities)
                .zip(&inequality_values)
                .zip(&work.inequalities.rhs)
            {
                let relaxed = alpha * value + (1.0 - alpha) * *slack;
                *slack = (relaxed + *multiplier / rho).min(*rhs);
                *multiplier += rho * (relaxed - *slack);
            }
            for (index, ((slack, multiplier), value)) in bound_slack
                .iter_mut()
                .zip(&mut dual.bounds)
                .zip(&x)
                .enumerate()
            {
                let relaxed = alpha * value + (1.0 - alpha) * *slack;
                *slack = project(
                    relaxed + *multiplier / rho,
                    work.lower_bounds[index],
                    work.upper_bounds[index],
                );
                *multiplier += rho * (relaxed - *slack);
            }
            if let Some(term) = &work.l1 {
                // The prox of `(c_i / rho) * |z|` is the soft threshold;
                // the consensus value for this block is `x - anchor`.
                for (index, ((slack, multiplier), value)) in
                    l1_slack.iter_mut().zip(&mut dual.l1).zip(&x).enumerate()
                {
                    let consensus = value - term.anchor[index];
                    let relaxed = alpha * consensus + (1.0 - alpha) * *slack;
                    *slack = soft_threshold(relaxed + *multiplier / rho, term.costs[index] / rho);
                    *multiplier += rho * (relaxed - *slack);
                }
            }
            completed_iterations = iteration;

            if iteration % settings.check_termination_every == 0
                || iteration == settings.max_iterations
            {
                // Termination is always decided on the original data so that a
                // reported `Solved` never depends on the equilibration. The
                // complementarity gate keeps `Solved` as strong as feasibility
                // plus stationarity alone would suggest: `check_kkt` scores
                // stray multipliers by their complementarity products rather
                // than folding them into the dual residual.
                let (original_x, original_dual) =
                    unscaled_iterates(self.scaling.as_ref(), &x, &dual);
                let residuals = check_kkt(&self.problem, &original_x, &original_dual)?;
                let (primal_tolerance, dual_tolerance) =
                    stopping_tolerances(&self.problem, &original_x, &original_dual, &settings);
                if residuals.primal <= primal_tolerance
                    && residuals.dual <= dual_tolerance
                    && residuals.complementarity <= primal_tolerance.max(dual_tolerance)
                {
                    status = SolveStatus::Solved;
                    break;
                }
                if detect_infeasibility {
                    let delta_dual = DualVariables {
                        equalities: difference(&original_dual.equalities, &checked_dual.equalities),
                        inequalities: difference(
                            &original_dual.inequalities,
                            &checked_dual.inequalities,
                        ),
                        bounds: difference(&original_dual.bounds, &checked_dual.bounds),
                        // The L1 multiplier is boxed in [-costs, costs], so it
                        // cannot diverge along a Farkas ray; it plays no role
                        // in either certificate.
                        l1: Vec::new(),
                    };
                    if let Some(found) = detect_primal_infeasibility(
                        &self.problem,
                        &delta_dual,
                        settings.infeasibility_tolerance,
                    ) {
                        certificate = Some(Certificate::Primal(found));
                        status = SolveStatus::PrimalInfeasible;
                        break;
                    }
                    let delta_x = difference(&original_x, &checked_x);
                    if let Some(found) = detect_dual_infeasibility(
                        &self.problem,
                        &delta_x,
                        settings.infeasibility_tolerance,
                    ) {
                        certificate = Some(Certificate::Dual(found));
                        status = SolveStatus::DualInfeasible;
                        break;
                    }
                    checked_x = original_x;
                    checked_dual = original_dual;
                }
            }

            if adapt_rho {
                let primal_residual = consensus_primal_residual(
                    work,
                    &equality_values,
                    &equality_slack,
                    &inequality_values,
                    &inequality_slack,
                    &x,
                    &bound_slack,
                    &l1_slack,
                );
                let dual_residual = consensus_dual_residual(
                    work,
                    &old_inequality_slack,
                    &inequality_slack,
                    &old_bound_slack,
                    &bound_slack,
                    &old_l1_slack,
                    &l1_slack,
                    rho,
                );
                let next_rho = balanced_rho(rho, primal_residual, dual_residual, &settings);
                if next_rho.to_bits() != rho.to_bits() {
                    ensure_system(systems, factorizations, work, next_rho, settings.sigma)?;
                    rho = next_rho;
                    rho_updates += 1;
                }
            }
        }

        let (mut x, mut dual) = unscaled_iterates(self.scaling.as_ref(), &x, &dual);
        let mut residuals = if status == SolveStatus::NumericalFailure {
            KktResiduals {
                primal: f64::INFINITY,
                dual: f64::INFINITY,
                complementarity: f64::INFINITY,
            }
        } else {
            check_kkt(&self.problem, &x, &dual)?
        };
        // Polishing refines only Solved iterates (certificates and failure
        // diagnostics stay untouched) and is adopted only when the audited
        // worst KKT residual improves; see `polish.rs`.
        let mut polished = false;
        if status == SolveStatus::Solved && settings.polish {
            if let Some(improved) = polish(&self.problem, &x, &dual, &residuals, &settings) {
                x = improved.x;
                dual = improved.dual;
                residuals = improved.residuals;
                polished = true;
            }
        }
        let diagnostics = if status == SolveStatus::Solved {
            None
        } else {
            Some(diagnose_failure(
                &self.problem,
                &x,
                &dual,
                &residuals,
                status,
                certificate.as_ref(),
                rho,
                rho_updates,
                &settings,
            ))
        };
        Ok(Solution {
            status,
            objective: self.problem.objective(&x),
            x,
            dual,
            residuals,
            iterations: completed_iterations,
            solve_time: started.elapsed(),
            final_rho: rho,
            rho_updates,
            polished,
            diagnostics,
            certificate,
        })
    }
}

/// Elementwise `current - previous`.
fn difference(current: &[f64], previous: &[f64]) -> Vec<f64> {
    current
        .iter()
        .zip(previous)
        .map(|(new, old)| new - old)
        .collect()
}

/// Moves the factorization for `rho` to the front of the cache, building it
/// on a miss and evicting the least recently used entry beyond capacity.
///
/// A free function over disjoint `Workspace` fields so the solve loop can
/// hold the scaled problem borrowed while the cache mutates.
fn ensure_system(
    systems: &mut Vec<(f64, FactorizedSystem)>,
    factorizations: &mut usize,
    work: &QpProblem,
    rho: f64,
    sigma: f64,
) -> Result<(), SolverError> {
    if let Some(position) = systems
        .iter()
        .position(|(cached, _)| cached.to_bits() == rho.to_bits())
    {
        let entry = systems.remove(position);
        systems.insert(0, entry);
        return Ok(());
    }
    let system = FactorizedSystem::new(work, rho, sigma)?;
    *factorizations += 1;
    systems.insert(0, (rho, system));
    systems.truncate(FACTORIZATION_CACHE_CAPACITY);
    Ok(())
}

fn replace_vector(
    field: &'static str,
    target: &mut [f64],
    values: &[f64],
) -> Result<(), SolverError> {
    if values.len() != target.len() {
        return Err(SolverError::InvalidProblem(ProblemError::Dimension {
            field,
            expected: target.len(),
            actual: values.len(),
        }));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::InvalidProblem(ProblemError::NonFinite(field)));
    }
    target.copy_from_slice(values);
    Ok(())
}

pub(crate) struct FactorizedSystem {
    system: SmwSystem,
}

impl FactorizedSystem {
    pub(crate) fn new(problem: &QpProblem, rho: f64, sigma: f64) -> Result<Self, SolverError> {
        let covariance = covariance_columns(&problem.quadratic)?;
        let n = problem.quadratic.dimension();
        let factor_count = problem.quadratic.factor_count();
        let constraint_count = problem.equalities.len() + problem.inequalities.len();
        let reduced_dimension = factor_count + constraint_count;
        let mut columns = Matrix::zeros(n, reduced_dimension);
        for row in 0..n {
            for col in 0..factor_count {
                columns[(row, col)] = covariance[(row, col)];
            }
        }
        let rho_root = rho.sqrt();
        for constraint in 0..problem.equalities.len() {
            for variable in 0..n {
                columns[(variable, factor_count + constraint)] =
                    rho_root * problem.equalities.matrix[(constraint, variable)];
            }
        }
        let inequality_offset = factor_count + problem.equalities.len();
        for constraint in 0..problem.inequalities.len() {
            for variable in 0..n {
                columns[(variable, inequality_offset + constraint)] =
                    rho_root * problem.inequalities.matrix[(constraint, variable)];
            }
        }

        // The box consensus x = z contributes rho*I, so the base remains
        // diagonal even when every variable has finite bounds. The L1
        // soft-threshold consensus x - anchor = z contributes another rho*I:
        // the reduced dimension never grows with the L1 term.
        let l1_rho = if problem.l1.is_some() { rho } else { 0.0 };
        let diagonal: Vec<f64> = problem
            .quadratic
            .diagonal
            .iter()
            .map(|diagonal| diagonal + sigma + rho + l1_rho)
            .collect();
        Ok(Self {
            system: SmwSystem::factor(&diagonal, columns)?,
        })
    }

    pub(crate) fn solve_in_place(&self, right_hand_side: &mut [f64]) {
        self.system.solve_in_place(right_hand_side);
    }
}

fn project(value: f64, lower: f64, upper: f64) -> f64 {
    value.max(lower).min(upper)
}

/// Proximal operator of `threshold * |z|`: shrink toward zero by
/// `threshold`, exactly zero inside the dead zone.
fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

/// Maps scaled-space iterates back to the original space; the identity when
/// scaling is disabled.
fn unscaled_iterates(
    scaling: Option<&ScaledProblem>,
    x: &[f64],
    dual: &DualVariables,
) -> (Vec<f64>, DualVariables) {
    scaling.map_or_else(
        || (x.to_vec(), dual.clone()),
        |scaled| (scaled.unscaled_x(x), scaled.unscaled_dual(dual)),
    )
}

#[allow(clippy::too_many_arguments)]
fn consensus_primal_residual(
    problem: &QpProblem,
    equality_values: &[f64],
    equality_slack: &[f64],
    inequality_values: &[f64],
    inequality_slack: &[f64],
    x: &[f64],
    bound_slack: &[f64],
    l1_slack: &[f64],
) -> f64 {
    let l1_residual = problem.l1.as_ref().map_or(0.0, |term| {
        x.iter()
            .zip(&term.anchor)
            .zip(l1_slack)
            .map(|((value, anchor), slack)| (value - anchor - slack).abs())
            .fold(0.0, f64::max)
    });
    equality_values
        .iter()
        .zip(equality_slack)
        .chain(inequality_values.iter().zip(inequality_slack))
        .map(|(value, slack)| (value - slack).abs())
        .chain(
            x.iter()
                .zip(bound_slack)
                .map(|(value, slack)| (value - slack).abs()),
        )
        .fold(l1_residual, f64::max)
}

#[allow(clippy::too_many_arguments)]
fn consensus_dual_residual(
    problem: &QpProblem,
    old_inequality_slack: &[f64],
    inequality_slack: &[f64],
    old_bound_slack: &[f64],
    bound_slack: &[f64],
    old_l1_slack: &[f64],
    l1_slack: &[f64],
    rho: f64,
) -> f64 {
    let inequality_change: Vec<f64> = inequality_slack
        .iter()
        .zip(old_inequality_slack)
        .map(|(new, old)| new - old)
        .collect();
    let mut dual_change: Vec<f64> = bound_slack
        .iter()
        .zip(old_bound_slack)
        .map(|(new, old)| new - old)
        .collect();
    // The L1 block's constraint matrix is the identity, like the box block.
    for (change, (new, old)) in dual_change
        .iter_mut()
        .zip(l1_slack.iter().zip(old_l1_slack))
    {
        *change += new - old;
    }
    problem
        .inequalities
        .matrix
        .transpose_mul_add(&inequality_change, &mut dual_change);
    rho * norm_inf(&dual_change)
}

fn balanced_rho(
    rho: f64,
    primal_residual: f64,
    dual_residual: f64,
    settings: &SolverSettings,
) -> f64 {
    if primal_residual > settings.adaptive_rho_tolerance * dual_residual {
        (rho * settings.adaptive_rho_multiplier).min(settings.maximum_rho)
    } else if dual_residual > settings.adaptive_rho_tolerance * primal_residual {
        (rho / settings.adaptive_rho_multiplier).max(settings.minimum_rho)
    } else {
        rho
    }
}

fn validate_warm_vector(
    field: &'static str,
    values: &[f64],
    expected: usize,
) -> Result<(), SolverError> {
    if values.len() != expected {
        return Err(SolverError::WarmStartDimension {
            field,
            expected,
            actual: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::WarmStartNonFinite(field));
    }
    Ok(())
}

fn warm_vector(
    values: Option<&[f64]>,
    field: &'static str,
    expected: usize,
) -> Result<Vec<f64>, SolverError> {
    if let Some(values) = values {
        validate_warm_vector(field, values, expected)?;
        Ok(values.to_vec())
    } else {
        Ok(vec![0.0; expected])
    }
}

#[allow(clippy::too_many_arguments)]
fn diagnose_failure(
    problem: &QpProblem,
    x: &[f64],
    dual: &DualVariables,
    residuals: &KktResiduals,
    status: SolveStatus,
    certificate: Option<&Certificate>,
    rho: f64,
    rho_updates: usize,
    settings: &SolverSettings,
) -> ConvergenceDiagnostics {
    let (primal_tolerance, dual_tolerance) = stopping_tolerances(problem, x, dual, settings);
    let coefficient_spread_decades = coefficient_spread_decades(problem);
    let rho_at_limit = rho <= settings.minimum_rho || rho >= settings.maximum_rho;

    let mut hints = Vec::new();
    match certificate {
        Some(Certificate::Primal(primal)) => {
            if let Ok(audited) = check_primal_certificate(problem, primal) {
                hints.push(format!(
                    "a Farkas certificate proves the constraints admit no common point: \
                     the weighted constraint combination in Solution::certificate cancels \
                     every variable (stationarity residual {:.1e}) yet requires a value \
                     below {:.3e}; relax one of the participating constraints \
                     (audit independently with check_primal_certificate)",
                    audited.stationarity, audited.support_gap
                ));
            }
        }
        Some(Certificate::Dual(dual_certificate)) => {
            if let Ok(audited) = check_dual_certificate(problem, dual_certificate) {
                hints.push(format!(
                    "a descent-ray certificate proves the objective is unbounded below: \
                     along Solution::certificate's direction the quadratic cost stays at \
                     {:.1e} while the linear cost falls at slope {:.3e} and no constraint \
                     blocks the ray; add bounds or constraints capping this direction \
                     (audit independently with check_dual_certificate)",
                    audited.curvature, audited.objective_gap
                ));
            }
        }
        None => {}
    }
    if status == SolveStatus::NumericalFailure {
        hints.push(
            "an iterate became non-finite; check the input for extreme coefficient \
             magnitudes or near-singular covariance data"
                .to_owned(),
        );
    }
    if coefficient_spread_decades > 6.0 {
        if settings.scaling_iterations == 0 {
            hints.push(format!(
                "problem coefficients span about {coefficient_spread_decades:.0} orders of \
                 magnitude and automatic scaling is disabled; set scaling_iterations \
                 to its default (10) or rescale returns, covariance, and constraints \
                 toward comparable units"
            ));
        } else {
            hints.push(format!(
                "problem coefficients span about {coefficient_spread_decades:.0} orders of \
                 magnitude even though Ruiz equilibration ran; rescale returns, \
                 covariance, and constraints toward comparable units before solving"
            ));
        }
    }
    if rho_at_limit {
        hints.push(format!(
            "the adaptive penalty finished pinned at {rho:.1e} (allowed range \
             [{:.1e}, {:.1e}]); widen minimum_rho/maximum_rho or start rho closer to \
             this value",
            settings.minimum_rho, settings.maximum_rho
        ));
    }
    let primal_gap = residuals.primal / primal_tolerance.max(f64::MIN_POSITIVE);
    let dual_gap = residuals.dual / dual_tolerance.max(f64::MIN_POSITIVE);
    if status == SolveStatus::MaxIterations {
        if primal_gap.max(dual_gap) < 100.0 {
            hints.push(format!(
                "residuals are within two orders of magnitude of tolerance \
                 (primal {:.1e} vs {:.1e}, dual {:.1e} vs {:.1e}); raising \
                 max_iterations above {} will likely finish the solve",
                residuals.primal,
                primal_tolerance,
                residuals.dual,
                dual_tolerance,
                settings.max_iterations
            ));
        } else if primal_gap > 100.0 * dual_gap.max(1.0) {
            hints.push(
                "the primal residual dominates and no infeasibility certificate was \
                 found within infeasibility_tolerance; the constraints may be feasible \
                 only barely, or infeasible by a margin below that tolerance — verify \
                 that budget, boxes, and linear constraints admit a common point"
                    .to_owned(),
            );
        } else if dual_gap > 100.0 * primal_gap.max(1.0) {
            hints.push(
                "the dual residual dominates; consider loosening relative_tolerance, \
                 increasing rho, or rescaling the objective so risk and return terms \
                 have comparable magnitude"
                    .to_owned(),
            );
        } else {
            hints.push(format!(
                "both residuals remain far from tolerance after {} iterations; \
                 rescale the problem data or relax tolerances before raising the \
                 iteration budget",
                settings.max_iterations
            ));
        }
    }
    if rho_updates > 10 {
        hints.push(format!(
            "rho was re-tuned {rho_updates} times, a sign of conflicting primal/dual \
             scales; data rescaling usually helps more than iteration budget"
        ));
    }

    ConvergenceDiagnostics {
        primal_tolerance,
        dual_tolerance,
        coefficient_spread_decades,
        rho_at_limit,
        hints,
    }
}

/// Base-10 spread between the largest and smallest nonzero magnitudes across
/// objective, constraint, and bound data.
fn coefficient_spread_decades(problem: &QpProblem) -> f64 {
    let mut smallest = f64::INFINITY;
    let mut largest = 0.0_f64;
    let mut observe = |value: f64| {
        let magnitude = value.abs();
        if magnitude.is_finite() && magnitude > 0.0 {
            smallest = smallest.min(magnitude);
            largest = largest.max(magnitude);
        }
    };
    for value in problem.quadratic.factors.as_slice() {
        observe(*value);
    }
    for value in &problem.quadratic.diagonal {
        observe(*value);
    }
    for value in &problem.linear {
        observe(*value);
    }
    if let Some(l1) = &problem.l1 {
        for value in l1.costs.iter().chain(&l1.anchor) {
            observe(*value);
        }
    }
    for constraints in [&problem.equalities, &problem.inequalities] {
        for value in constraints.matrix.as_slice() {
            observe(*value);
        }
        for value in &constraints.rhs {
            observe(*value);
        }
    }
    if largest == 0.0 || !smallest.is_finite() {
        0.0
    } else {
        (largest / smallest).log10()
    }
}

fn stopping_tolerances(
    problem: &QpProblem,
    x: &[f64],
    dual: &DualVariables,
    settings: &SolverSettings,
) -> (f64, f64) {
    let equality_values = problem.equalities.matrix.mul_vec(x);
    let inequality_values = problem.inequalities.matrix.mul_vec(x);
    let primal_scale = 1.0_f64
        .max(norm_inf(x))
        .max(norm_inf(&equality_values))
        .max(norm_inf(&problem.equalities.rhs))
        .max(norm_inf(&inequality_values))
        .max(norm_inf(&problem.inequalities.rhs));

    let qx = problem.quadratic.apply(x);
    let mut transposed_dual = vec![0.0; x.len()];
    problem
        .equalities
        .matrix
        .transpose_mul_add(&dual.equalities, &mut transposed_dual);
    problem
        .inequalities
        .matrix
        .transpose_mul_add(&dual.inequalities, &mut transposed_dual);
    for (value, bound) in transposed_dual.iter_mut().zip(&dual.bounds) {
        *value += bound;
    }
    for (value, l1) in transposed_dual.iter_mut().zip(&dual.l1) {
        *value += l1;
    }
    let l1_cost_scale = problem
        .l1
        .as_ref()
        .map_or(0.0, |term| norm_inf(&term.costs));
    let dual_scale = 1.0_f64
        .max(norm_inf(&qx))
        .max(norm_inf(&problem.linear))
        .max(l1_cost_scale)
        .max(norm_inf(&transposed_dual));

    (
        settings.absolute_tolerance + settings.relative_tolerance * primal_scale,
        settings.absolute_tolerance + settings.relative_tolerance * dual_scale,
    )
}
