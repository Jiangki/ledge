//! Serialization round-trip tests (roadmap 3.3, `serde` feature).
//!
//! The serialization story is bug reproduction: a problem (plus settings,
//! warm start, and the returned solution) dumped on one machine must replay
//! identically on another. JSON is the human-readable interchange format —
//! including problems with infinite bounds, which JSON cannot represent
//! natively — and `postcard` stands in for "any self-describing binary serde
//! format".

use ledge_core::{
    FactorCovariance, Matrix, PortfolioProblem, QpProblem, Solution, SolveStatus, Solver,
    SolverSettings, WarmStart,
};

/// A small long-short QP with unbounded shorts on some names, an L1 term,
/// and one of each constraint block.
fn qp_fixture() -> QpProblem {
    let assets = 8;
    let factors = Matrix::new(
        assets,
        2,
        (0..assets * 2)
            .map(|index| 0.3 * ((1 + index) as f64 * 12.9898).sin())
            .collect(),
    )
    .unwrap();
    let mut lower = vec![-0.5; assets];
    let mut upper = vec![0.5; assets];
    lower[2] = f64::NEG_INFINITY;
    upper[5] = f64::INFINITY;
    QpProblem {
        quadratic: ledge_core::FactorQuad::new(
            factors,
            FactorCovariance::Diagonal(vec![0.05, 0.08]),
            vec![0.1; assets],
        )
        .unwrap(),
        linear: (0..assets).map(|index| -0.01 * index as f64).collect(),
        l1: Some(ledge_core::L1Term {
            costs: vec![2.0e-3; assets],
            anchor: vec![0.0; assets],
        }),
        equalities: ledge_core::LinearConstraints::new(
            Matrix::new(1, assets, vec![1.0; assets]).unwrap(),
            vec![1.0],
        )
        .unwrap(),
        inequalities: ledge_core::LinearConstraints::new(
            Matrix::new(1, assets, vec![1.0 / assets as f64; assets]).unwrap(),
            vec![0.4],
        )
        .unwrap(),
        lower_bounds: lower,
        upper_bounds: upper,
    }
}

fn portfolio_fixture() -> PortfolioProblem {
    let assets = 12;
    let factors = Matrix::new(
        assets,
        3,
        (0..assets * 3)
            .map(|index| 0.3 * ((1 + index) as f64 * 7.13).sin())
            .collect(),
    )
    .unwrap();
    let benchmark = vec![1.0 / assets as f64; assets];
    PortfolioProblem::new(
        factors,
        FactorCovariance::Diagonal(vec![0.05, 0.06, 0.07]),
        vec![0.1; assets],
        (0..assets)
            .map(|index| 0.05 + 0.001 * index as f64)
            .collect(),
    )
    .unwrap()
    .with_risk_aversion(4.0)
    .unwrap()
    .with_tracking_benchmark(benchmark.clone())
    .unwrap()
    .with_industry_neutrality(&(0..assets).map(|asset| asset % 3).collect::<Vec<_>>())
    .unwrap()
    .with_l1_turnover(benchmark, vec![1.0e-3; assets])
    .unwrap()
    .with_concentration_limit(0.3)
    .unwrap()
}

/// Solutions carry a wall-clock `solve_time`, so bit-equality is asserted
/// field-by-field on everything that must replay.
#[allow(clippy::float_cmp)] // bit-exact replay is the property under test
fn assert_same_solve(left: &Solution, right: &Solution) {
    assert_eq!(left.status, right.status);
    assert_eq!(left.x, right.x, "iterates must replay bit-identically");
    assert_eq!(left.dual.equalities, right.dual.equalities);
    assert_eq!(left.dual.inequalities, right.dual.inequalities);
    assert_eq!(left.dual.bounds, right.dual.bounds);
    assert_eq!(left.dual.l1, right.dual.l1);
    assert_eq!(left.objective, right.objective);
    assert_eq!(left.iterations, right.iterations);
    assert_eq!(left.polished, right.polished);
}

#[test]
fn qp_json_round_trip_preserves_infinite_bounds_and_replays_the_solve() {
    let problem = qp_fixture();
    let json = serde_json::to_string_pretty(&problem).unwrap();
    let restored: QpProblem = serde_json::from_str(&json).unwrap();
    assert_eq!(problem, restored);
    assert!(restored.lower_bounds[2].is_infinite());
    assert!(restored.upper_bounds[5].is_infinite());

    let solver = Solver::default();
    let original = solver.solve(&problem, None).unwrap();
    let replayed = solver.solve(&restored, None).unwrap();
    assert_eq!(original.status, SolveStatus::Solved);
    assert_same_solve(&original, &replayed);
}

#[test]
fn solution_settings_and_warm_start_round_trip_through_json() {
    let problem = qp_fixture();
    let settings = SolverSettings {
        max_iterations: 500,
        polish: false,
        ..SolverSettings::default()
    };
    let solution = Solver::new(settings.clone()).solve(&problem, None).unwrap();

    let settings_restored: SolverSettings =
        serde_json::from_str(&serde_json::to_string(&settings).unwrap()).unwrap();
    assert_eq!(settings, settings_restored);

    let solution_restored: Solution =
        serde_json::from_str(&serde_json::to_string(&solution).unwrap()).unwrap();
    assert_eq!(solution, solution_restored);

    let warm = solution.warm_start();
    let warm_restored: WarmStart =
        serde_json::from_str(&serde_json::to_string(&warm).unwrap()).unwrap();
    assert_eq!(warm, warm_restored);
}

#[test]
fn infeasible_solves_round_trip_their_certificates() {
    // Budget forces the sum to 1 while every weight is capped at 0.05:
    // primal infeasible with a Farkas certificate.
    let mut problem = qp_fixture();
    problem.l1 = None;
    problem.lower_bounds = vec![0.0; 8];
    problem.upper_bounds = vec![0.05; 8];
    let solution = Solver::default().solve(&problem, None).unwrap();
    assert_eq!(solution.status, SolveStatus::PrimalInfeasible);
    assert!(solution.certificate.is_some());

    let restored: Solution =
        serde_json::from_str(&serde_json::to_string(&solution).unwrap()).unwrap();
    assert_eq!(solution.certificate, restored.certificate);
    assert_eq!(solution.status, restored.status);
}

#[test]
fn portfolio_json_round_trip_rebuilds_the_same_qp() {
    let problem = portfolio_fixture();
    let json = serde_json::to_string(&problem).unwrap();
    let restored: PortfolioProblem = serde_json::from_str(&json).unwrap();
    assert_eq!(problem, restored);
    assert_eq!(problem.to_qp().unwrap(), restored.to_qp().unwrap());
}

#[test]
fn binary_round_trip_via_postcard() {
    let problem = qp_fixture();
    let portfolio = portfolio_fixture();
    let solution = Solver::default().solve(&problem, None).unwrap();

    let restored: QpProblem =
        postcard::from_bytes(&postcard::to_allocvec(&problem).unwrap()).unwrap();
    assert_eq!(problem, restored);
    let restored: PortfolioProblem =
        postcard::from_bytes(&postcard::to_allocvec(&portfolio).unwrap()).unwrap();
    assert_eq!(portfolio, restored);
    let restored: Solution =
        postcard::from_bytes(&postcard::to_allocvec(&solution).unwrap()).unwrap();
    assert_eq!(solution, restored);
}

#[test]
fn corrupted_dumps_are_rejected_by_the_same_validation_as_construction() {
    // Matrix whose data length disagrees with its shape.
    let bad_matrix = r#"{"rows": 2, "cols": 3, "data": [1.0, 2.0]}"#;
    assert!(serde_json::from_str::<Matrix>(bad_matrix)
        .unwrap_err()
        .to_string()
        .contains("requires 6 values"));

    // Portfolio dump edited to a negative risk aversion.
    let json = serde_json::to_string(&portfolio_fixture()).unwrap();
    let tampered = json.replace("\"risk_aversion\":4.0", "\"risk_aversion\":-4.0");
    assert_ne!(json, tampered);
    assert!(serde_json::from_str::<PortfolioProblem>(&tampered)
        .unwrap_err()
        .to_string()
        .contains("risk_aversion"));

    // Turnover terms without their anchor.
    let tampered = json.replace("\"previous_weights\":[", "\"previous_weights_ignored\":[");
    assert_ne!(json, tampered);
    assert!(serde_json::from_str::<PortfolioProblem>(&tampered).is_err());
}
