//! Rolling `solve_sequence` / `PortfolioSequence` integration tests
//! (roadmap 2.5).
//!
//! The sequence API must produce the same answers as building a fresh
//! `PortfolioProblem` per date, while reusing the workspace factorizations
//! and chaining warm starts internally. Failed steps must be atomic.

use ledge_core::{
    check_kkt, solve_sequence, FactorCovariance, Matrix, PortfolioError, PortfolioProblem,
    RebalanceStep, SolveStatus, SolverSettings,
};

const OBJECTIVE_TOLERANCE: f64 = 1.0e-4;
const RESIDUAL_TOLERANCE: f64 = 1.0e-4;

/// Deterministic factor portfolio without external RNG dependencies.
struct Fixture {
    factors: Matrix,
    omega: FactorCovariance,
    specific: Vec<f64>,
    expected: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    inequality_matrix: Matrix,
    inequality_rhs: Vec<f64>,
}

fn fixture(assets: usize, factor_count: usize) -> Fixture {
    let mut factors = Vec::with_capacity(assets * factor_count);
    for row in 0..assets {
        for col in 0..factor_count {
            let angle = (1 + row * factor_count + col) as f64;
            factors.push(0.3 * (angle * 12.9898).sin());
        }
    }
    let omega = FactorCovariance::Diagonal(
        (0..factor_count)
            .map(|index| 0.05 + 0.01 * index as f64)
            .collect(),
    );
    let specific: Vec<f64> = (0..assets)
        .map(|index| 0.08 + 0.04 * ((index * 7 % 13) as f64) / 13.0)
        .collect();
    let expected: Vec<f64> = (0..assets)
        .map(|index| 0.05 + 0.03 * ((index as f64) * 0.7).cos())
        .collect();
    let max_weight = (8.0 / assets as f64).min(1.0);

    // One sector-exposure style inequality row with slack at the uniform
    // portfolio, so rhs updates have room to move.
    let inequality_matrix = Matrix::new(
        1,
        assets,
        (0..assets)
            .map(|index| if index % 3 == 0 { 1.0 } else { 0.0 })
            .collect(),
    )
    .unwrap();
    let exposed = (0..assets).filter(|index| index % 3 == 0).count() as f64;
    let inequality_rhs = vec![1.5 * exposed * max_weight / 2.0];

    Fixture {
        factors: Matrix::new(assets, factor_count, factors).unwrap(),
        omega,
        specific,
        expected,
        lower: vec![0.0; assets],
        upper: vec![max_weight; assets],
        inequality_matrix,
        inequality_rhs,
    }
}

impl Fixture {
    fn problem(&self) -> PortfolioProblem {
        self.problem_with_returns(self.expected.clone())
    }

    fn problem_with_returns(&self, expected_returns: Vec<f64>) -> PortfolioProblem {
        PortfolioProblem::new(
            self.factors.clone(),
            self.omega.clone(),
            self.specific.clone(),
            expected_returns,
        )
        .unwrap()
        .with_risk_aversion(6.0)
        .unwrap()
        .with_bounds(self.lower.clone(), self.upper.clone())
        .unwrap()
        .with_inequalities(self.inequality_matrix.clone(), self.inequality_rhs.clone())
        .unwrap()
    }

    fn perturbed_returns(&self, step: usize) -> Vec<f64> {
        self.expected
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let wave = ((index + 5 * step) % 9) as f64 - 4.0;
                value + 2.0e-3 * wave
            })
            .collect()
    }
}

#[test]
fn sequence_matches_fresh_solves_date_by_date() {
    let fixture = fixture(80, 5);
    let mut sequence = fixture.problem().sequence().unwrap();

    let first = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(first.status, SolveStatus::Solved);

    for step in 1..=4 {
        let returns = fixture.perturbed_returns(step);
        let rolled = sequence
            .solve_next(&RebalanceStep {
                expected_returns: Some(returns.clone()),
                ..RebalanceStep::default()
            })
            .unwrap();
        assert_eq!(rolled.status, SolveStatus::Solved, "step {step}");

        // Fresh construction with the same data is the oracle.
        let fresh_problem = fixture.problem_with_returns(returns);
        let fresh = fresh_problem.solve(None).unwrap();
        assert_eq!(fresh.status, SolveStatus::Solved, "step {step}");
        let scale = 1.0 + fresh.objective.abs();
        assert!(
            (rolled.objective - fresh.objective).abs() <= OBJECTIVE_TOLERANCE * scale,
            "step {step}: rolled {} vs fresh {}",
            rolled.objective,
            fresh.objective
        );

        // Residuals must hold on the updated original data.
        let qp = fresh_problem.to_qp().unwrap();
        let residuals = check_kkt(&qp, &rolled.x, &rolled.dual).unwrap();
        assert!(residuals.primal <= RESIDUAL_TOLERANCE, "step {step}");
        assert!(residuals.dual <= RESIDUAL_TOLERANCE, "step {step}");
        assert!(
            residuals.complementarity <= RESIDUAL_TOLERANCE,
            "step {step}"
        );
    }
}

#[test]
fn warm_rolling_dates_reach_zero_new_factorizations() {
    let fixture = fixture(80, 5);
    let mut sequence = fixture.problem().sequence().unwrap();
    let cold = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(cold.status, SolveStatus::Solved);
    let after_cold = sequence.factorizations();

    let mut total_warm_iterations = 0;
    for step in 1..=5 {
        let solution = sequence
            .solve_next(&RebalanceStep {
                expected_returns: Some(fixture.perturbed_returns(step)),
                ..RebalanceStep::default()
            })
            .unwrap();
        assert_eq!(solution.status, SolveStatus::Solved, "step {step}");
        total_warm_iterations += solution.iterations;
    }
    assert_eq!(
        sequence.factorizations(),
        after_cold,
        "warm rolling dates must be served from the factorization cache"
    );
    assert!(
        total_warm_iterations / 5 <= cold.iterations,
        "warm-started dates must not iterate more than the cold solve on average"
    );
}

#[test]
fn turnover_anchor_updates_agree_with_fresh_solves() {
    let fixture = fixture(60, 4);
    let anchored = fixture
        .problem()
        .with_quadratic_turnover(vec![1.0 / 60.0; 60], 0.5)
        .unwrap();
    let mut sequence = anchored.sequence().unwrap();
    let first = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(first.status, SolveStatus::Solved);

    // Next date anchors on the previous date's weights.
    let rolled = sequence
        .solve_next(&RebalanceStep {
            expected_returns: Some(fixture.perturbed_returns(1)),
            previous_weights: Some(first.x.clone()),
            ..RebalanceStep::default()
        })
        .unwrap();
    assert_eq!(rolled.status, SolveStatus::Solved);

    let fresh_problem = fixture
        .problem_with_returns(fixture.perturbed_returns(1))
        .with_quadratic_turnover(first.x.clone(), 0.5)
        .unwrap();
    let fresh = fresh_problem.solve(None).unwrap();
    assert_eq!(fresh.status, SolveStatus::Solved);
    let scale = 1.0 + fresh.objective.abs();
    assert!(
        (rolled.objective - fresh.objective).abs() <= OBJECTIVE_TOLERANCE * scale,
        "rolled {} vs fresh {}",
        rolled.objective,
        fresh.objective
    );
}

#[test]
fn budget_and_rhs_updates_agree_with_fresh_solves() {
    let fixture = fixture(60, 4);
    let mut sequence = fixture.problem().sequence().unwrap();
    let first = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(first.status, SolveStatus::Solved);

    let new_budget = 0.9;
    let new_inequality_rhs: Vec<f64> = fixture
        .inequality_rhs
        .iter()
        .map(|rhs| rhs + 0.02)
        .collect();
    let rolled = sequence
        .solve_next(&RebalanceStep {
            budget: Some(new_budget),
            inequality_rhs: Some(new_inequality_rhs.clone()),
            ..RebalanceStep::default()
        })
        .unwrap();
    assert_eq!(rolled.status, SolveStatus::Solved);
    let budget_sum: f64 = rolled.x.iter().sum();
    assert!(
        (budget_sum - new_budget).abs() <= 1.0e-4,
        "budget {budget_sum} vs {new_budget}"
    );

    let fresh_problem = fixture
        .problem()
        .with_budget(Some(new_budget))
        .unwrap()
        .with_inequalities(fixture.inequality_matrix.clone(), new_inequality_rhs)
        .unwrap();
    let fresh = fresh_problem.solve(None).unwrap();
    assert_eq!(fresh.status, SolveStatus::Solved);
    let scale = 1.0 + fresh.objective.abs();
    assert!(
        (rolled.objective - fresh.objective).abs() <= OBJECTIVE_TOLERANCE * scale,
        "rolled {} vs fresh {}",
        rolled.objective,
        fresh.objective
    );
}

#[test]
fn invalid_steps_are_rejected_atomically() {
    let fixture = fixture(40, 3);
    let mut sequence = fixture.problem().sequence().unwrap();
    let baseline = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(baseline.status, SolveStatus::Solved);

    // Wrong dimension.
    assert!(sequence
        .solve_next(&RebalanceStep {
            expected_returns: Some(vec![0.0; 3]),
            ..RebalanceStep::default()
        })
        .is_err());
    // Non-finite value.
    assert!(sequence
        .solve_next(&RebalanceStep {
            expected_returns: Some(vec![f64::NAN; 40]),
            ..RebalanceStep::default()
        })
        .is_err());
    // Anchor update without a turnover term in the base problem.
    assert!(matches!(
        sequence.solve_next(&RebalanceStep {
            previous_weights: Some(vec![0.0; 40]),
            ..RebalanceStep::default()
        }),
        Err(PortfolioError::InvalidParameter(_))
    ));
    // Budget outside what the boxes can reach.
    assert!(matches!(
        sequence.solve_next(&RebalanceStep {
            budget: Some(1.0e3),
            ..RebalanceStep::default()
        }),
        Err(PortfolioError::BudgetOutsideBounds { .. })
    ));
    // A partially-valid step must not leak its valid half into the state:
    // the returns here are fine, the budget is not.
    assert!(sequence
        .solve_next(&RebalanceStep {
            expected_returns: Some(fixture.perturbed_returns(9)),
            budget: Some(1.0e3),
            ..RebalanceStep::default()
        })
        .is_err());

    // After every rejection the sequence still solves its original data.
    let after = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(after.status, SolveStatus::Solved);
    let scale = 1.0 + baseline.objective.abs();
    assert!(
        (after.objective - baseline.objective).abs() <= OBJECTIVE_TOLERANCE * scale,
        "state must be unchanged after rejected steps: {} vs {}",
        after.objective,
        baseline.objective
    );
}

#[test]
fn budget_updates_require_a_budget_row() {
    let fixture = fixture(30, 3);
    let unbudgeted = fixture.problem().with_budget(None).unwrap();
    let mut sequence = unbudgeted.sequence().unwrap();
    assert!(matches!(
        sequence.solve_next(&RebalanceStep {
            budget: Some(1.0),
            ..RebalanceStep::default()
        }),
        Err(PortfolioError::InvalidParameter(_))
    ));
}

#[test]
fn solve_sequence_returns_one_solution_per_step() {
    let fixture = fixture(50, 4);
    let steps: Vec<RebalanceStep> = std::iter::once(RebalanceStep::default())
        .chain((1..=3).map(|step| RebalanceStep {
            expected_returns: Some(fixture.perturbed_returns(step)),
            ..RebalanceStep::default()
        }))
        .collect();

    let solutions = solve_sequence(&fixture.problem(), None, &steps).unwrap();
    assert_eq!(solutions.len(), steps.len());
    for (index, solution) in solutions.iter().enumerate() {
        assert_eq!(solution.status, SolveStatus::Solved, "step {index}");
    }
    // Warm-started later dates should not iterate more than the cold first.
    assert!(solutions[1].iterations <= solutions[0].iterations);
}

#[test]
fn solve_sequence_respects_custom_settings() {
    let fixture = fixture(30, 3);
    let error = solve_sequence(
        &fixture.problem(),
        Some(SolverSettings {
            max_iterations: 0,
            ..SolverSettings::default()
        }),
        &[RebalanceStep::default()],
    )
    .unwrap_err();
    assert!(error.to_string().contains("max_iterations"));
}
