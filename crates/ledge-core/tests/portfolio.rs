//! High-level portfolio API integration tests.

use ledge_core::{
    solve_mean_variance_factor, FactorCovariance, Matrix, PortfolioError, PortfolioProblem,
    SolveStatus,
};

fn two_asset_problem(expected_returns: Vec<f64>) -> PortfolioProblem {
    PortfolioProblem::new(
        Matrix::new(2, 0, Vec::new()).unwrap(),
        FactorCovariance::Diagonal(Vec::new()),
        vec![1.0, 1.0],
        expected_returns,
    )
    .unwrap()
}

#[test]
fn high_level_api_builds_budget_and_box_constraints() {
    let problem = two_asset_problem(vec![0.2, 0.0])
        .with_bounds(vec![0.1, 0.1], vec![0.8, 0.8])
        .unwrap();

    let solution = solve_mean_variance_factor(&problem, None, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!((solution.x.iter().sum::<f64>() - 1.0).abs() < 2.0e-5);
    assert!(solution.x.iter().all(|weight| (0.1..=0.8).contains(weight)));
    assert!(solution.x[0] > solution.x[1]);
}

#[test]
fn quadratic_turnover_penalty_keeps_weights_near_previous_portfolio() {
    let unpenalized = two_asset_problem(vec![1.0, 0.0]).solve(None).unwrap();
    let penalized = two_asset_problem(vec![1.0, 0.0])
        .with_quadratic_turnover(vec![0.0, 1.0], 10.0)
        .unwrap()
        .solve(None)
        .unwrap();

    assert_eq!(unpenalized.status, SolveStatus::Solved);
    assert_eq!(penalized.status, SolveStatus::Solved);
    assert!(penalized.x[0] < unpenalized.x[0]);
    assert!(penalized.x[0] < 0.2);
}

#[test]
fn rejects_a_budget_outside_box_reach() {
    let problem = two_asset_problem(vec![0.0, 0.0])
        .with_bounds(vec![0.0, 0.0], vec![0.4, 0.4])
        .unwrap();

    let error = problem.to_qp().unwrap_err();

    assert!(matches!(error, PortfolioError::BudgetOutsideBounds { .. }));
    assert!(error.to_string().contains("reachable sum is [0, 0.8]"));
}

#[test]
fn solution_produces_a_full_reusable_warm_start() {
    let problem = two_asset_problem(vec![0.1, 0.0]);
    let first = problem.solve(None).unwrap();
    let warm = first.warm_start();
    let second = problem.solve(Some(&warm)).unwrap();

    assert_eq!(second.status, SolveStatus::Solved);
    assert!(second.iterations <= first.iterations);
}
