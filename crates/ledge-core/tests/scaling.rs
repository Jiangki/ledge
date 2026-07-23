//! Integration tests for automatic Ruiz equilibration.
//!
//! The acceptance criterion from the roadmap: on an ill-conditioned suite,
//! scaling off must fail (`MaxIterations`) while scaling on must reach
//! `Solved` — and every reported residual must hold on the original data.

use ledge_core::{
    check_kkt, generate_synthetic, GeneratedInstance, SolveStatus, Solver, SolverSettings,
    SyntheticConfig,
};

const RESIDUAL_TOLERANCE: f64 = 1.0e-4;

/// Rescales the units of every variable by `10^(±spread)` without changing
/// the underlying optimization problem: substituting `x_j = s_j * x'_j`
/// multiplies factor rows, constraint columns, and the linear term by `s_j`,
/// the idiosyncratic diagonal by `s_j²`, and divides bounds by `s_j`.
fn with_bad_variable_units(instance: &GeneratedInstance, spread_decades: f64) -> GeneratedInstance {
    let mut skewed = instance.clone();
    let problem = &mut skewed.problem;
    let n = problem.quadratic.dimension();
    let scales: Vec<f64> = (0..n)
        .map(|index| {
            let position = index as f64 / (n - 1).max(1) as f64;
            10.0_f64.powf(spread_decades * (position - 0.5))
        })
        .collect();

    let k = problem.quadratic.factor_count();
    for (row, scale) in scales.iter().enumerate() {
        for col in 0..k {
            problem.quadratic.factors[(row, col)] *= scale;
        }
        problem.quadratic.diagonal[row] *= scale * scale;
        problem.linear[row] *= scale;
        problem.lower_bounds[row] /= scale;
        problem.upper_bounds[row] /= scale;
    }
    for constraints in [&mut problem.equalities, &mut problem.inequalities] {
        for row in 0..constraints.matrix.rows() {
            for (col, scale) in scales.iter().enumerate() {
                constraints.matrix[(row, col)] *= scale;
            }
        }
    }
    for (weight, scale) in skewed.feasible_reference.iter_mut().zip(&scales) {
        *weight /= scale;
    }
    skewed
}

fn ill_conditioned_instance() -> GeneratedInstance {
    let base = generate_synthetic(SyntheticConfig {
        assets: 60,
        factors: 6,
        inequalities: 3,
        seed: 2026,
        budget: 1.0,
        max_weight: 0.2,
    })
    .expect("valid config");
    with_bad_variable_units(&base, 6.0)
}

#[test]
fn ill_conditioned_instance_fails_without_scaling_and_solves_with_it() {
    let instance = ill_conditioned_instance();

    let unscaled = Solver::new(SolverSettings {
        scaling_iterations: 0,
        ..SolverSettings::default()
    })
    .solve(&instance.problem, None)
    .expect("setup succeeds; only convergence differs");
    assert_eq!(
        unscaled.status,
        SolveStatus::MaxIterations,
        "the suite must stay hard enough that raw ADMM fails; got {:?} in {} iterations",
        unscaled.status,
        unscaled.iterations
    );

    let scaled = Solver::default()
        .solve(&instance.problem, None)
        .expect("setup succeeds");
    assert_eq!(scaled.status, SolveStatus::Solved);

    // Residuals must hold on the original data via the independent checker.
    // The skewed units make the recovered weights O(100), so the primal gate
    // is relative to the iterate scale, mirroring the solver's declared
    // absolute + relative stopping contract.
    let residuals = check_kkt(&instance.problem, &scaled.x, &scaled.dual).unwrap();
    let primal_scale = scaled
        .x
        .iter()
        .fold(1.0_f64, |largest, value| largest.max(value.abs()));
    assert!(residuals.primal < RESIDUAL_TOLERANCE * primal_scale);
    assert!(residuals.dual < RESIDUAL_TOLERANCE);
    assert!(residuals.complementarity < RESIDUAL_TOLERANCE);

    // The minimizer can be no worse than the known feasible reference.
    let reference_objective = instance.problem.objective(&instance.feasible_reference);
    assert!(scaled.objective <= reference_objective + RESIDUAL_TOLERANCE);
}

#[test]
fn disabled_scaling_reports_an_actionable_hint() {
    let instance = ill_conditioned_instance();
    let solution = Solver::new(SolverSettings {
        scaling_iterations: 0,
        ..SolverSettings::default()
    })
    .solve(&instance.problem, None)
    .unwrap();

    let diagnostics = solution.diagnostics.expect("failed solves attach hints");
    assert!(diagnostics.coefficient_spread_decades > 6.0);
    assert!(diagnostics
        .hints
        .iter()
        .any(|hint| hint.contains("scaling_iterations")));
}

#[test]
fn scaling_on_and_off_agree_on_a_well_conditioned_problem() {
    let instance = generate_synthetic(SyntheticConfig::default()).unwrap();

    let scaled = Solver::default().solve(&instance.problem, None).unwrap();
    let unscaled = Solver::new(SolverSettings {
        scaling_iterations: 0,
        ..SolverSettings::default()
    })
    .solve(&instance.problem, None)
    .unwrap();

    assert_eq!(scaled.status, SolveStatus::Solved);
    assert_eq!(unscaled.status, SolveStatus::Solved);
    let objective_scale = 1.0 + unscaled.objective.abs();
    assert!((scaled.objective - unscaled.objective).abs() <= RESIDUAL_TOLERANCE * objective_scale);
    for (a, b) in scaled.x.iter().zip(&unscaled.x) {
        assert!((a - b).abs() <= 1.0e-3);
    }
}

#[test]
fn warm_start_round_trips_through_scaling() {
    let instance = ill_conditioned_instance();
    let solver = Solver::default();

    let cold = solver.solve(&instance.problem, None).unwrap();
    assert_eq!(cold.status, SolveStatus::Solved);

    let warm = solver
        .solve(&instance.problem, Some(&cold.warm_start()))
        .unwrap();
    assert_eq!(warm.status, SolveStatus::Solved);
    assert!(
        warm.iterations <= cold.iterations,
        "warm start took {} iterations vs cold {}",
        warm.iterations,
        cold.iterations
    );

    let objective_scale = 1.0 + cold.objective.abs();
    assert!((warm.objective - cold.objective).abs() <= RESIDUAL_TOLERANCE * objective_scale);
}
