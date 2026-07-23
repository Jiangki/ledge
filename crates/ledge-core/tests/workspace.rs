//! Factorization-reuse workspace integration tests (roadmap 2.4).
//!
//! The workspace must be an exact drop-in for one-shot solves: identical
//! termination quality on the original data, cheap linear/rhs updates, and
//! observable factorization reuse across a warm-started rolling sequence.

use ledge_core::{
    check_kkt, generate_synthetic, ProblemError, SolveStatus, Solver, SolverError, SolverSettings,
    SyntheticConfig, WarmStart,
};

const OBJECTIVE_TOLERANCE: f64 = 1.0e-4;
const RESIDUAL_TOLERANCE: f64 = 1.0e-4;

fn synthetic(assets: usize, factors: usize, seed: u64) -> ledge_core::GeneratedInstance {
    generate_synthetic(SyntheticConfig {
        assets,
        factors,
        seed,
        max_weight: (10.0 / assets as f64).min(1.0),
        ..SyntheticConfig::default()
    })
    .expect("synthetic instance")
}

/// Deterministic perturbation of the linear cost, mimicking new expected
/// returns on a fixed structure.
fn perturbed_linear(base: &[f64], step: usize) -> Vec<f64> {
    base.iter()
        .enumerate()
        .map(|(index, value)| {
            let wave = ((index + 3 * step) % 11) as f64 - 5.0;
            value + 1.0e-3 * wave
        })
        .collect()
}

#[test]
fn workspace_solve_matches_one_shot_solve() {
    let instance = synthetic(200, 8, 5);
    let solver = Solver::default();

    let one_shot = solver.solve(&instance.problem, None).unwrap();
    let mut workspace = solver.workspace(&instance.problem).unwrap();
    let reused = workspace.solve(None).unwrap();

    assert_eq!(one_shot.status, SolveStatus::Solved);
    assert_eq!(reused.status, SolveStatus::Solved);
    // Identical settings, data, and start must give the identical iterate
    // path: the workspace runs the same engine, only setup timing moves.
    assert_eq!(reused.iterations, one_shot.iterations);
    assert_eq!(reused.rho_updates, one_shot.rho_updates);
    let scale = 1.0 + one_shot.objective.abs();
    assert!((reused.objective - one_shot.objective).abs() <= 1.0e-12 * scale);
}

#[test]
fn rolling_updates_agree_with_fresh_solves() {
    let instance = synthetic(300, 10, 42);
    let solver = Solver::default();
    let mut workspace = solver.workspace(&instance.problem).unwrap();

    let mut previous = workspace.solve(None).unwrap();
    assert_eq!(previous.status, SolveStatus::Solved);

    for step in 1..=5 {
        let linear = perturbed_linear(&instance.problem.linear, step);
        workspace.update_linear(&linear).unwrap();
        let warm = previous.warm_start();
        let rolled = workspace.solve(Some(&warm)).unwrap();
        assert_eq!(rolled.status, SolveStatus::Solved, "step {step}");

        // Fresh one-shot solve of the same perturbed problem is the oracle.
        let mut perturbed = instance.problem.clone();
        perturbed.linear.clone_from(&linear);
        let fresh = solver.solve(&perturbed, None).unwrap();
        assert_eq!(fresh.status, SolveStatus::Solved, "step {step}");
        let scale = 1.0 + fresh.objective.abs();
        assert!(
            (rolled.objective - fresh.objective).abs() <= OBJECTIVE_TOLERANCE * scale,
            "step {step}: rolled {} vs fresh {}",
            rolled.objective,
            fresh.objective
        );

        // Residuals must hold on the *perturbed original* data.
        let residuals = check_kkt(&perturbed, &rolled.x, &rolled.dual).unwrap();
        assert!(residuals.primal <= RESIDUAL_TOLERANCE, "step {step}");
        assert!(residuals.dual <= RESIDUAL_TOLERANCE, "step {step}");
        assert!(
            residuals.complementarity <= RESIDUAL_TOLERANCE,
            "step {step}"
        );

        previous = rolled;
    }
}

#[test]
fn warm_rolling_steps_reuse_the_factorization() {
    let instance = synthetic(300, 10, 42);
    // Fixed penalty isolates the reuse claim from adaptive-rho heuristics;
    // a companion test covers the adaptive path below.
    let solver = Solver::new(SolverSettings {
        adaptive_rho: false,
        ..SolverSettings::default()
    });
    let mut workspace = solver.workspace(&instance.problem).unwrap();
    let mut previous = workspace.solve(None).unwrap();
    assert_eq!(previous.status, SolveStatus::Solved);
    let after_cold = workspace.factorizations();

    for step in 1..=5 {
        let linear = perturbed_linear(&instance.problem.linear, step);
        workspace.update_linear(&linear).unwrap();
        let warm = previous.warm_start();
        previous = workspace.solve(Some(&warm)).unwrap();
        assert_eq!(previous.status, SolveStatus::Solved, "step {step}");
    }
    assert_eq!(
        workspace.factorizations(),
        after_cold,
        "warm rolling steps with a fixed penalty must not refactorize"
    );
}

#[test]
fn adaptive_penalty_ladder_is_factored_once_per_workspace() {
    // A deliberately poor initial penalty forces several adaptive updates in
    // the first solve. Later solves replay the same one-shot policy — same
    // rho_updates, identical iterate path — but every penalty on the ladder
    // hits the cache, so the factorization count must not grow.
    let instance = synthetic(120, 6, 9);
    let solver = Solver::new(SolverSettings {
        rho: 1.0e-5,
        ..SolverSettings::default()
    });
    let mut workspace = solver.workspace(&instance.problem).unwrap();

    let first = workspace.solve(None).unwrap();
    assert_eq!(first.status, SolveStatus::Solved);
    assert!(first.rho_updates > 0, "the poor start must force re-tuning");
    let after_first = workspace.factorizations();
    assert_eq!(after_first, 1 + first.rho_updates);

    let second = workspace.solve(None).unwrap();
    assert_eq!(second.status, SolveStatus::Solved);
    assert_eq!(second.rho_updates, first.rho_updates, "same policy replay");
    assert_eq!(second.iterations, first.iterations, "identical path");
    assert_eq!(
        workspace.factorizations(),
        after_first,
        "the replayed ladder must be served from the cache"
    );
}

#[test]
fn workspace_without_scaling_matches_one_shot() {
    let instance = synthetic(150, 6, 7);
    let solver = Solver::new(SolverSettings {
        scaling_iterations: 0,
        ..SolverSettings::default()
    });
    let one_shot = solver.solve(&instance.problem, None).unwrap();
    let mut workspace = solver.workspace(&instance.problem).unwrap();
    let reused = workspace.solve(None).unwrap();

    assert_eq!(one_shot.status, SolveStatus::Solved);
    assert_eq!(reused.status, SolveStatus::Solved);
    assert_eq!(reused.iterations, one_shot.iterations);
    let scale = 1.0 + one_shot.objective.abs();
    assert!((reused.objective - one_shot.objective).abs() <= 1.0e-12 * scale);
}

#[test]
fn rhs_updates_agree_with_fresh_solves() {
    let instance = synthetic(200, 8, 11);
    let solver = Solver::default();
    let mut workspace = solver.workspace(&instance.problem).unwrap();
    let cold = workspace.solve(None).unwrap();
    assert_eq!(cold.status, SolveStatus::Solved);

    // Rebalance to a slightly different budget and loosened exposure caps.
    let new_equality_rhs: Vec<f64> = instance
        .problem
        .equalities
        .rhs
        .iter()
        .map(|value| value * 0.95)
        .collect();
    let new_inequality_rhs: Vec<f64> = instance
        .problem
        .inequalities
        .rhs
        .iter()
        .map(|value| value + 0.01)
        .collect();
    workspace.update_equality_rhs(&new_equality_rhs).unwrap();
    workspace
        .update_inequality_rhs(&new_inequality_rhs)
        .unwrap();
    let rolled = workspace.solve(Some(&cold.warm_start())).unwrap();
    assert_eq!(rolled.status, SolveStatus::Solved);

    let mut perturbed = instance.problem.clone();
    perturbed.equalities.rhs.clone_from(&new_equality_rhs);
    perturbed.inequalities.rhs.clone_from(&new_inequality_rhs);
    let fresh = solver.solve(&perturbed, None).unwrap();
    assert_eq!(fresh.status, SolveStatus::Solved);
    let scale = 1.0 + fresh.objective.abs();
    assert!((rolled.objective - fresh.objective).abs() <= OBJECTIVE_TOLERANCE * scale);

    let residuals = check_kkt(&perturbed, &rolled.x, &rolled.dual).unwrap();
    assert!(residuals.primal <= RESIDUAL_TOLERANCE);
    assert!(residuals.dual <= RESIDUAL_TOLERANCE);
    assert!(residuals.complementarity <= RESIDUAL_TOLERANCE);
}

#[test]
fn updates_reject_wrong_dimensions_and_non_finite_values() {
    let instance = synthetic(50, 4, 3);
    let solver = Solver::default();
    let mut workspace = solver.workspace(&instance.problem).unwrap();

    let short = vec![0.0; instance.problem.linear.len() - 1];
    assert!(matches!(
        workspace.update_linear(&short),
        Err(SolverError::InvalidProblem(ProblemError::Dimension { .. }))
    ));

    let mut poisoned = instance.problem.linear.clone();
    poisoned[0] = f64::NAN;
    assert!(matches!(
        workspace.update_linear(&poisoned),
        Err(SolverError::InvalidProblem(ProblemError::NonFinite(_)))
    ));

    assert!(matches!(
        workspace.update_equality_rhs(&[]),
        Err(SolverError::InvalidProblem(ProblemError::Dimension { .. }))
    ));
    assert!(matches!(
        workspace.update_inequality_rhs(&vec![f64::INFINITY; instance.problem.inequalities.len()]),
        Err(SolverError::InvalidProblem(ProblemError::NonFinite(_)))
    ));

    // A failed update must leave the workspace usable.
    let solution = workspace.solve(None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
}

#[test]
fn workspace_accepts_full_warm_start_from_a_previous_solution() {
    let instance = synthetic(200, 8, 5);
    let solver = Solver::new(SolverSettings {
        check_termination_every: 1,
        ..SolverSettings::default()
    });
    let mut workspace = solver.workspace(&instance.problem).unwrap();
    let cold = workspace.solve(None).unwrap();
    assert_eq!(cold.status, SolveStatus::Solved);

    let warm = workspace.solve(Some(&cold.warm_start())).unwrap();
    assert_eq!(warm.status, SolveStatus::Solved);
    assert!(warm.iterations <= cold.iterations);

    let bad = WarmStart::from_primal(vec![0.0; 3]);
    assert!(matches!(
        workspace.solve(Some(&bad)),
        Err(SolverError::WarmStartDimension { .. })
    ));
}
