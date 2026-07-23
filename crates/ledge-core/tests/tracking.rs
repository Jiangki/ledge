//! Tracking-error objective integration tests (roadmap 2.6).
//!
//! `with_tracking_benchmark` must be pure sugar: the same QP with the linear
//! cost shifted by `-risk_aversion * Σ b`. These tests verify the shift
//! against an explicitly adjusted plain problem, the exact benchmark
//! reproduction property, and rolling benchmark updates in sequences.

use ledge_core::{
    check_kkt, FactorCovariance, Matrix, PortfolioError, PortfolioProblem, RebalanceStep,
    SolveStatus,
};

const OBJECTIVE_TOLERANCE: f64 = 1.0e-4;
const RESIDUAL_TOLERANCE: f64 = 1.0e-4;

/// Deterministic factor portfolio without external RNG dependencies.
struct Fixture {
    factors: Matrix,
    omega: FactorCovariance,
    specific: Vec<f64>,
    expected: Vec<f64>,
    upper: Vec<f64>,
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
    Fixture {
        factors: Matrix::new(assets, factor_count, factors).unwrap(),
        omega,
        specific,
        expected,
        upper: vec![max_weight; assets],
    }
}

impl Fixture {
    fn assets(&self) -> usize {
        self.upper.len()
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
        .with_bounds(vec![0.0; self.assets()], self.upper.clone())
        .unwrap()
    }

    fn problem(&self) -> PortfolioProblem {
        self.problem_with_returns(self.expected.clone())
    }

    /// A benchmark inside the box constraints: uniform weights tilted by a
    /// small deterministic wave, renormalized to the unit budget.
    fn benchmark(&self, step: usize) -> Vec<f64> {
        let assets = self.assets();
        let uniform = 1.0 / assets as f64;
        let mut benchmark: Vec<f64> = (0..assets)
            .map(|index| uniform * (1.0 + 0.2 * (((index + 3 * step) % 7) as f64 - 3.0) / 3.0))
            .collect();
        let total: f64 = benchmark.iter().sum();
        for value in &mut benchmark {
            *value /= total;
        }
        benchmark
    }

    /// `Σ b` on the raw (risk-aversion-free) covariance.
    fn covariance_times(&self, weights: &[f64]) -> Vec<f64> {
        let assets = self.assets();
        let factor_count = match &self.omega {
            FactorCovariance::Diagonal(values) => values.len(),
            FactorCovariance::Dense(matrix) => matrix.rows(),
        };
        let mut loadings = vec![0.0; factor_count];
        for (row, weight) in weights.iter().enumerate() {
            for (col, loading) in loadings.iter_mut().enumerate() {
                *loading += self.factors[(row, col)] * weight;
            }
        }
        let scaled: Vec<f64> = match &self.omega {
            FactorCovariance::Diagonal(values) => loadings
                .iter()
                .zip(values)
                .map(|(loading, omega)| loading * omega)
                .collect(),
            FactorCovariance::Dense(matrix) => (0..factor_count)
                .map(|row| {
                    (0..factor_count)
                        .map(|col| matrix[(row, col)] * loadings[col])
                        .sum()
                })
                .collect(),
        };
        (0..assets)
            .map(|row| {
                let systematic: f64 = (0..factor_count)
                    .map(|col| self.factors[(row, col)] * scaled[col])
                    .sum();
                systematic + self.specific[row] * weights[row]
            })
            .collect()
    }
}

#[test]
fn benchmark_shift_matches_manually_adjusted_returns() {
    let fixture = fixture(60, 4);
    let benchmark = fixture.benchmark(0);
    let risk_aversion = 6.0;

    let tracking = fixture
        .problem()
        .with_tracking_benchmark(benchmark.clone())
        .unwrap();

    // (w-b)'Σ(w-b) expanded is the same QP as absolute risk with expected
    // returns raised by risk_aversion * Σ b.
    let sigma_b = fixture.covariance_times(&benchmark);
    let adjusted_returns: Vec<f64> = fixture
        .expected
        .iter()
        .zip(&sigma_b)
        .map(|(alpha, product)| alpha + risk_aversion * product)
        .collect();
    let adjusted = fixture.problem_with_returns(adjusted_returns);

    let tracking_qp = tracking.to_qp().unwrap();
    let adjusted_qp = adjusted.to_qp().unwrap();
    for (index, (left, right)) in tracking_qp
        .linear
        .iter()
        .zip(&adjusted_qp.linear)
        .enumerate()
    {
        assert!(
            (left - right).abs() <= 1.0e-12 * (1.0 + right.abs()),
            "linear[{index}]: {left} vs {right}"
        );
    }

    let tracking_solution = tracking.solve(None).unwrap();
    let adjusted_solution = adjusted.solve(None).unwrap();
    assert_eq!(tracking_solution.status, SolveStatus::Solved);
    assert_eq!(adjusted_solution.status, SolveStatus::Solved);
    for (index, (left, right)) in tracking_solution
        .x
        .iter()
        .zip(&adjusted_solution.x)
        .enumerate()
    {
        assert!(
            (left - right).abs() <= 1.0e-5,
            "weights[{index}]: {left} vs {right}"
        );
    }

    let residuals = check_kkt(&tracking_qp, &tracking_solution.x, &tracking_solution.dual).unwrap();
    assert!(residuals.primal <= RESIDUAL_TOLERANCE);
    assert!(residuals.dual <= RESIDUAL_TOLERANCE);
    assert!(residuals.complementarity <= RESIDUAL_TOLERANCE);
}

#[test]
fn feasible_benchmark_with_no_alpha_is_reproduced_exactly() {
    let fixture = fixture(50, 4);
    let benchmark = fixture.benchmark(1);

    // Pure tracking: no expected returns, benchmark satisfies budget and
    // boxes, so zero active risk is attainable and optimal at w = b.
    let problem = fixture
        .problem_with_returns(vec![0.0; fixture.assets()])
        .with_tracking_benchmark(benchmark.clone())
        .unwrap();
    let solution = problem.solve(None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
    for (index, (weight, target)) in solution.x.iter().zip(&benchmark).enumerate() {
        assert!(
            (weight - target).abs() <= 1.0e-5,
            "asset {index}: weight {weight} vs benchmark {target}"
        );
    }
}

#[test]
fn tracking_combines_with_turnover_terms() {
    let fixture = fixture(40, 3);
    let assets = fixture.assets();
    let benchmark = fixture.benchmark(2);
    let previous = vec![1.0 / assets as f64; assets];

    let problem = fixture
        .problem()
        .with_tracking_benchmark(benchmark)
        .unwrap()
        .with_quadratic_turnover(previous.clone(), 0.5)
        .unwrap()
        .with_l1_turnover(previous, vec![2.0e-3; assets])
        .unwrap();
    let solution = problem.solve(None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);

    let qp = problem.to_qp().unwrap();
    let residuals = check_kkt(&qp, &solution.x, &solution.dual).unwrap();
    assert!(residuals.primal <= RESIDUAL_TOLERANCE);
    assert!(residuals.dual <= RESIDUAL_TOLERANCE);
    assert!(residuals.complementarity <= RESIDUAL_TOLERANCE);
}

#[test]
fn invalid_benchmarks_are_rejected() {
    let fixture = fixture(30, 3);
    assert!(matches!(
        fixture.problem().with_tracking_benchmark(vec![0.1; 3]),
        Err(PortfolioError::Problem(_))
    ));
    assert!(matches!(
        fixture
            .problem()
            .with_tracking_benchmark(vec![f64::NAN; 30]),
        Err(PortfolioError::Problem(_))
    ));
}

#[test]
fn sequence_benchmark_updates_agree_with_fresh_solves() {
    let fixture = fixture(60, 4);
    let mut sequence = fixture
        .problem()
        .with_tracking_benchmark(fixture.benchmark(0))
        .unwrap()
        .sequence()
        .unwrap();
    let first = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(first.status, SolveStatus::Solved);
    let after_cold = sequence.factorizations();

    for step in 1..=3 {
        let benchmark = fixture.benchmark(step);
        let rolled = sequence
            .solve_next(&RebalanceStep {
                benchmark_weights: Some(benchmark.clone()),
                ..RebalanceStep::default()
            })
            .unwrap();
        assert_eq!(rolled.status, SolveStatus::Solved, "step {step}");

        let fresh_problem = fixture
            .problem()
            .with_tracking_benchmark(benchmark)
            .unwrap();
        let fresh = fresh_problem.solve(None).unwrap();
        assert_eq!(fresh.status, SolveStatus::Solved, "step {step}");
        let scale = 1.0 + fresh.objective.abs();
        assert!(
            (rolled.objective - fresh.objective).abs() <= OBJECTIVE_TOLERANCE * scale,
            "step {step}: rolled {} vs fresh {}",
            rolled.objective,
            fresh.objective
        );

        let qp = fresh_problem.to_qp().unwrap();
        let residuals = check_kkt(&qp, &rolled.x, &rolled.dual).unwrap();
        assert!(residuals.primal <= RESIDUAL_TOLERANCE, "step {step}");
        assert!(residuals.dual <= RESIDUAL_TOLERANCE, "step {step}");
    }
    assert_eq!(
        sequence.factorizations(),
        after_cold,
        "benchmark updates must be served from the factorization cache"
    );
}

#[test]
fn benchmark_updates_require_a_tracking_base() {
    let fixture = fixture(30, 3);
    let mut sequence = fixture.problem().sequence().unwrap();
    assert!(matches!(
        sequence.solve_next(&RebalanceStep {
            benchmark_weights: Some(vec![1.0 / 30.0; 30]),
            ..RebalanceStep::default()
        }),
        Err(PortfolioError::InvalidParameter(_))
    ));
}
