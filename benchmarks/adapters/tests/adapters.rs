//! Cross-checks external adapters against Ledge on small instances.
//!
//! Compiled only when at least one external-solver feature is enabled, so the
//! default workspace test run stays free of native dependencies.

#![cfg(any(feature = "osqp", feature = "clarabel"))]

use ledge_bench_adapters::{
    protocol::{AdapterSolve, PhasedSolver},
    Formulation, LedgeAdapter,
};
use ledge_core::{check_kkt, generate_synthetic, L1Term, QpProblem, Solver, SyntheticConfig};

const KKT_TOLERANCE: f64 = 5.0e-5;
const OBJECTIVE_TOLERANCE: f64 = 1.0e-6;

fn instance(assets: usize, factors: usize, seed: u64) -> (QpProblem, Vec<f64>) {
    let generated = generate_synthetic(SyntheticConfig {
        assets,
        factors,
        seed,
        max_weight: (10.0 / assets as f64).min(1.0),
        ..SyntheticConfig::default()
    })
    .expect("synthetic instance");
    (generated.problem, generated.feasible_reference)
}

/// Same synthetic instance plus 10 bps proportional turnover costs anchored
/// at the feasible reference, so the optimum keeps a genuine no-trade region
/// and the external epigraph encoding must reproduce kinked multipliers.
fn l1_instance(assets: usize, factors: usize, seed: u64) -> (QpProblem, Vec<f64>) {
    let (mut problem, start) = instance(assets, factors, seed);
    problem.l1 = Some(L1Term {
        costs: vec![1.0e-3; assets],
        anchor: start.clone(),
    });
    (problem, start)
}

fn reference_objective(problem: &QpProblem, primal_start: &[f64]) -> f64 {
    let mut ledge = LedgeAdapter::new(Solver::default());
    ledge.setup(problem, primal_start).expect("ledge setup");
    let solve = ledge.solve_cold().expect("ledge solve");
    assert!(solve.solved, "ledge must solve the reference instance");
    problem.objective(&solve.x)
}

fn assert_matches_ledge(problem: &QpProblem, solve: &AdapterSolve, reference: f64, label: &str) {
    assert!(solve.solved, "{label} did not report optimality");
    let residuals = check_kkt(problem, &solve.x, &solve.dual).expect("kkt dimensions");
    assert!(
        residuals.primal <= KKT_TOLERANCE,
        "{label}: independent primal residual {} above tolerance",
        residuals.primal
    );
    assert!(
        residuals.dual <= KKT_TOLERANCE,
        "{label}: independent dual residual {} above tolerance",
        residuals.dual
    );
    assert!(
        residuals.complementarity <= KKT_TOLERANCE,
        "{label}: independent complementarity {} above tolerance",
        residuals.complementarity
    );
    let objective = problem.objective(&solve.x);
    let scale = 1.0_f64.max(reference.abs());
    assert!(
        (objective - reference).abs() <= OBJECTIVE_TOLERANCE * scale,
        "{label}: objective {objective} differs from ledge {reference}"
    );
}

fn exercise(adapter: &mut dyn PhasedSolver) {
    exercise_instances(adapter, instance);
}

fn exercise_l1(adapter: &mut dyn PhasedSolver) {
    exercise_instances(adapter, l1_instance);
}

fn exercise_instances(
    adapter: &mut dyn PhasedSolver,
    build: fn(usize, usize, u64) -> (QpProblem, Vec<f64>),
) {
    for (assets, factors, seed) in [(40, 4, 3_u64), (120, 8, 9)] {
        let (problem, start) = build(assets, factors, seed);
        let reference = reference_objective(&problem, &start);
        let label = adapter.name();

        adapter.setup(&problem, &start).expect("setup");
        let cold = adapter.solve_cold().expect("cold solve");
        assert_matches_ledge(&problem, &cold, reference, &format!("{label} cold"));

        // A rolling step with a shifted return vector must stay verifiable
        // against a fresh Ledge solve of the same perturbed problem.
        let mut perturbed = problem.clone();
        for (index, value) in perturbed.linear.iter_mut().enumerate() {
            *value += 1.0e-3 * ((index % 7) as f64 - 3.0);
        }
        let rolled = adapter
            .resolve_with_linear(&perturbed.linear)
            .expect("rolling solve");
        let rolled_reference = reference_objective(&perturbed, &start);
        assert_matches_ledge(
            &perturbed,
            &rolled,
            rolled_reference,
            &format!("{label} roll"),
        );
    }
}

#[cfg(feature = "osqp")]
#[test]
fn osqp_dense_matches_ledge() {
    exercise(&mut ledge_bench_adapters::OsqpAdapter::new(
        Formulation::DenseQ,
    ));
}

#[cfg(feature = "osqp")]
#[test]
fn osqp_lifted_matches_ledge() {
    exercise(&mut ledge_bench_adapters::OsqpAdapter::new(
        Formulation::Lifted,
    ));
}

#[cfg(feature = "clarabel")]
#[test]
fn clarabel_dense_matches_ledge() {
    exercise(&mut ledge_bench_adapters::ClarabelAdapter::new(
        Formulation::DenseQ,
    ));
}

#[cfg(feature = "clarabel")]
#[test]
fn clarabel_lifted_matches_ledge() {
    exercise(&mut ledge_bench_adapters::ClarabelAdapter::new(
        Formulation::Lifted,
    ));
}

#[cfg(feature = "osqp")]
#[test]
fn osqp_dense_matches_ledge_with_l1() {
    exercise_l1(&mut ledge_bench_adapters::OsqpAdapter::new(
        Formulation::DenseQ,
    ));
}

#[cfg(feature = "osqp")]
#[test]
fn osqp_lifted_matches_ledge_with_l1() {
    exercise_l1(&mut ledge_bench_adapters::OsqpAdapter::new(
        Formulation::Lifted,
    ));
}

#[cfg(feature = "clarabel")]
#[test]
fn clarabel_dense_matches_ledge_with_l1() {
    exercise_l1(&mut ledge_bench_adapters::ClarabelAdapter::new(
        Formulation::DenseQ,
    ));
}

#[cfg(feature = "clarabel")]
#[test]
fn clarabel_lifted_matches_ledge_with_l1() {
    exercise_l1(&mut ledge_bench_adapters::ClarabelAdapter::new(
        Formulation::Lifted,
    ));
}
