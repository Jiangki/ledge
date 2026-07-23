//! Property-based invariants on randomly generated convex factor QPs.
//!
//! Every generated instance is feasible by construction (the synthetic
//! generator anchors constraints around a uniform reference portfolio), so
//! the solver must report `Solved` and the independent KKT checker must
//! confirm small residuals. Warm-started re-solves must agree with cold
//! solves and never take more iterations.

use ledge_core::{
    check_kkt, generate_synthetic, SolveStatus, Solver, SolverSettings, SyntheticConfig,
};
use proptest::prelude::*;

const RESIDUAL_TOLERANCE: f64 = 1.0e-4;

fn synthetic_config() -> impl Strategy<Value = SyntheticConfig> {
    (
        2_usize..=24,
        1_usize..=4,
        0_usize..=3,
        any::<u64>(),
        0.5_f64..2.0,
    )
        .prop_map(
            |(assets, factors, inequalities, seed, budget)| SyntheticConfig {
                assets,
                factors: factors.min(assets),
                inequalities,
                seed,
                budget,
                // Loose boxes that always contain the uniform reference.
                max_weight: budget,
            },
        )
}

fn test_solver() -> Solver {
    Solver::new(SolverSettings {
        max_iterations: 20_000,
        ..SolverSettings::default()
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    #[test]
    fn random_feasible_qp_solves_and_passes_independent_kkt(config in synthetic_config()) {
        let instance = generate_synthetic(config).expect("strategy only builds valid configs");
        let solution = test_solver()
            .solve(&instance.problem, None)
            .expect("setup must succeed on generated instances");

        prop_assert_eq!(solution.status, SolveStatus::Solved);
        prop_assert!(solution.diagnostics.is_none());

        let residuals = check_kkt(&instance.problem, &solution.x, &solution.dual)
            .expect("dimensions match by construction");
        prop_assert!(residuals.primal < RESIDUAL_TOLERANCE);
        prop_assert!(residuals.dual < RESIDUAL_TOLERANCE);
        prop_assert!(residuals.complementarity < RESIDUAL_TOLERANCE);

        // The minimizer can be no worse than the known feasible reference.
        let reference_objective = instance.problem.objective(&instance.feasible_reference);
        prop_assert!(solution.objective <= reference_objective + RESIDUAL_TOLERANCE);
    }

    #[test]
    fn workspace_rolling_agrees_with_fresh_solves(config in synthetic_config()) {
        let instance = generate_synthetic(config).expect("strategy only builds valid configs");
        let solver = test_solver();
        let mut workspace = solver
            .workspace(&instance.problem)
            .expect("workspace setup must succeed on generated instances");

        let cold = workspace.solve(None).expect("workspace cold solve");
        prop_assert_eq!(cold.status, SolveStatus::Solved);

        // Perturb the linear cost (new expected returns, fixed structure) and
        // compare the cached-factorization path against a fresh solver.
        let linear: Vec<f64> = instance
            .problem
            .linear
            .iter()
            .enumerate()
            .map(|(index, value)| value + 1.0e-3 * (((index % 5) as f64) - 2.0))
            .collect();
        workspace.update_linear(&linear).expect("dimension matches");
        let rolled = workspace
            .solve(Some(&cold.warm_start()))
            .expect("workspace rolling solve");
        prop_assert_eq!(rolled.status, SolveStatus::Solved);

        let mut perturbed = instance.problem.clone();
        perturbed.linear = linear;
        let fresh = solver
            .solve(&perturbed, None)
            .expect("fresh solve of the perturbed problem");
        prop_assert_eq!(fresh.status, SolveStatus::Solved);

        let scale = 1.0 + fresh.objective.abs();
        prop_assert!((rolled.objective - fresh.objective).abs() <= RESIDUAL_TOLERANCE * scale);

        let residuals = check_kkt(&perturbed, &rolled.x, &rolled.dual)
            .expect("dimensions match by construction");
        prop_assert!(residuals.primal < RESIDUAL_TOLERANCE);
        prop_assert!(residuals.dual < RESIDUAL_TOLERANCE);
        prop_assert!(residuals.complementarity < RESIDUAL_TOLERANCE);
    }

    #[test]
    fn warm_start_agrees_with_cold_solve(config in synthetic_config()) {
        let instance = generate_synthetic(config).expect("strategy only builds valid configs");
        let solver = test_solver();

        let cold = solver
            .solve(&instance.problem, None)
            .expect("setup must succeed on generated instances");
        prop_assert_eq!(cold.status, SolveStatus::Solved);

        let warm = solver
            .solve(&instance.problem, Some(&cold.warm_start()))
            .expect("warm start built from a solution is always valid");
        prop_assert_eq!(warm.status, SolveStatus::Solved);
        prop_assert!(warm.iterations <= cold.iterations);

        let scale = 1.0 + cold.objective.abs();
        prop_assert!((warm.objective - cold.objective).abs() <= RESIDUAL_TOLERANCE * scale);
    }
}
