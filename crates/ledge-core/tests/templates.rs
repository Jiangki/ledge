//! Constraint template builder integration tests (roadmap 3.1).
//!
//! Templates must be pure sugar over the existing linear-constraint and box
//! machinery: each builder appends the documented rows (or tightens bounds)
//! and nothing else. These tests verify the emitted QP against hand-built
//! constraints, the portfolio-level semantics on solved weights, template
//! stacking with user constraints, and rolling-sequence target updates.

use ledge_core::{
    check_kkt, FactorCovariance, Matrix, PortfolioError, PortfolioProblem, RebalanceStep,
    SolveStatus,
};

const CONSTRAINT_TOLERANCE: f64 = 1.0e-5;
const RESIDUAL_TOLERANCE: f64 = 1.0e-4;

/// Deterministic factor portfolio without external RNG dependencies.
struct Fixture {
    factors: Matrix,
    omega: FactorCovariance,
    specific: Vec<f64>,
    expected: Vec<f64>,
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
    Fixture {
        factors: Matrix::new(assets, factor_count, factors).unwrap(),
        omega,
        specific,
        expected,
    }
}

impl Fixture {
    fn assets(&self) -> usize {
        self.expected.len()
    }

    fn problem(&self) -> PortfolioProblem {
        PortfolioProblem::new(
            self.factors.clone(),
            self.omega.clone(),
            self.specific.clone(),
            self.expected.clone(),
        )
        .unwrap()
        .with_risk_aversion(6.0)
        .unwrap()
    }

    /// Round-robin industry ids over `count` industries.
    fn industries(&self, count: usize) -> Vec<usize> {
        (0..self.assets()).map(|asset| asset % count).collect()
    }

    /// A benchmark inside the default `[0, 1]` boxes summing to one.
    fn benchmark(&self) -> Vec<f64> {
        let assets = self.assets();
        let uniform = 1.0 / assets as f64;
        let mut benchmark: Vec<f64> = (0..assets)
            .map(|index| uniform * (1.0 + 0.2 * (((index % 7) as f64 - 3.0) / 3.0)))
            .collect();
        let total: f64 = benchmark.iter().sum();
        for value in &mut benchmark {
            *value /= total;
        }
        benchmark
    }

    /// One deterministic style loading row per style.
    fn style_exposures(&self, styles: usize) -> Matrix {
        let assets = self.assets();
        let mut data = Vec::with_capacity(styles * assets);
        for style in 0..styles {
            for asset in 0..assets {
                let angle = (1 + style * assets + asset) as f64;
                data.push((angle * 0.37).sin());
            }
        }
        Matrix::new(styles, assets, data).unwrap()
    }
}

fn group_weight(weights: &[f64], groups: &[usize], group: usize) -> f64 {
    weights
        .iter()
        .zip(groups)
        .filter(|(_, id)| **id == group)
        .map(|(weight, _)| weight)
        .sum()
}

#[test]
fn industry_neutrality_matches_hand_built_equalities_and_pins_group_weights() {
    let fixture = fixture(48, 4);
    let industries = fixture.industries(6);
    let benchmark = fixture.benchmark();

    let templated = fixture
        .problem()
        .with_tracking_benchmark(benchmark.clone())
        .unwrap()
        .with_industry_neutrality(&industries)
        .unwrap();

    // Hand-built equivalent: one indicator row per industry with the
    // benchmark's industry weight as target.
    let mut rows = vec![vec![0.0; fixture.assets()]; 6];
    let mut targets = vec![0.0; 6];
    for (asset, industry) in industries.iter().enumerate() {
        rows[*industry][asset] = 1.0;
        targets[*industry] += benchmark[asset];
    }
    let manual = fixture
        .problem()
        .with_tracking_benchmark(benchmark.clone())
        .unwrap()
        .with_equalities(Matrix::from_rows(rows).unwrap(), targets.clone())
        .unwrap();

    assert_eq!(
        templated.to_qp().unwrap(),
        manual.to_qp().unwrap(),
        "the template must emit exactly the hand-built rows"
    );

    let solution = templated.solve(None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
    for (industry, target) in targets.iter().enumerate() {
        let held = group_weight(&solution.x, &industries, industry);
        assert!(
            (held - target).abs() <= CONSTRAINT_TOLERANCE,
            "industry {industry}: held {held} vs benchmark {target}"
        );
    }
}

#[test]
fn group_targets_hold_on_solved_weights() {
    let fixture = fixture(40, 3);
    let groups = fixture.industries(4);
    let targets = vec![0.4, 0.3, 0.2, 0.1];

    let problem = fixture
        .problem()
        .with_group_targets(&groups, &targets)
        .unwrap();
    let solution = problem.solve(None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
    for (group, target) in targets.iter().enumerate() {
        let held = group_weight(&solution.x, &groups, group);
        assert!(
            (held - target).abs() <= CONSTRAINT_TOLERANCE,
            "group {group}: held {held} vs target {target}"
        );
    }

    let qp = problem.to_qp().unwrap();
    let residuals = check_kkt(&qp, &solution.x, &solution.dual).unwrap();
    assert!(residuals.primal <= RESIDUAL_TOLERANCE);
    assert!(residuals.dual <= RESIDUAL_TOLERANCE);
}

#[test]
fn style_bounds_emit_documented_rows_and_hold_on_solved_weights() {
    let fixture = fixture(36, 3);
    let exposures = fixture.style_exposures(3);
    // Style 0: two-sided band; style 1: upper only; style 2: exact.
    let lower = vec![-0.2, f64::NEG_INFINITY, 0.05];
    let upper = vec![0.2, 0.3, 0.05];

    let templated = fixture
        .problem()
        .with_style_bounds(&exposures, &lower, &upper)
        .unwrap();
    let qp = templated.to_qp().unwrap();

    // Equalities: budget row + the one exact style row.
    assert_eq!(qp.equalities.len(), 2);
    assert_eq!(qp.equalities.matrix.row(1), exposures.row(2));
    assert!((qp.equalities.rhs[1] - 0.05).abs() <= f64::EPSILON);
    // Inequalities: upper+lower rows for style 0, upper row for style 1.
    assert_eq!(qp.inequalities.len(), 3);
    assert_eq!(qp.inequalities.matrix.row(0), exposures.row(0));
    assert!((qp.inequalities.rhs[0] - 0.2).abs() <= f64::EPSILON);
    let negated: Vec<f64> = exposures.row(0).iter().map(|value| -value).collect();
    assert_eq!(qp.inequalities.matrix.row(1), negated.as_slice());
    assert!((qp.inequalities.rhs[1] - 0.2).abs() <= f64::EPSILON);
    assert_eq!(qp.inequalities.matrix.row(2), exposures.row(1));
    assert!((qp.inequalities.rhs[2] - 0.3).abs() <= f64::EPSILON);

    let solution = templated.solve(None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
    for style in 0..3 {
        let exposure: f64 = exposures
            .row(style)
            .iter()
            .zip(&solution.x)
            .map(|(loading, weight)| loading * weight)
            .sum();
        assert!(
            exposure >= lower[style] - CONSTRAINT_TOLERANCE
                && exposure <= upper[style] + CONSTRAINT_TOLERANCE,
            "style {style}: exposure {exposure} outside [{}, {}]",
            lower[style],
            upper[style]
        );
    }
}

#[test]
fn concentration_and_short_limits_tighten_boxes() {
    let fixture = fixture(30, 3);

    // Long-short base book: [-0.1, 0.3] boxes.
    let base = fixture
        .problem()
        .with_bounds(vec![-0.1; 30], vec![0.3; 30])
        .unwrap();

    let capped = base
        .clone()
        .with_concentration_limit(0.05)
        .unwrap()
        .to_qp()
        .unwrap();
    assert_eq!(capped.upper_bounds, vec![0.05; 30]);
    assert_eq!(capped.lower_bounds, vec![-0.05; 30]);

    let long_only = base.clone().with_short_limit(0.0).unwrap().to_qp().unwrap();
    assert_eq!(long_only.lower_bounds, vec![0.0; 30]);
    assert_eq!(long_only.upper_bounds, vec![0.3; 30]);

    let short_capped = base.with_short_limit(0.02).unwrap().to_qp().unwrap();
    assert_eq!(short_capped.lower_bounds, vec![-0.02; 30]);

    // The cap only ever tightens: an already stricter bound is kept.
    let stricter = fixture
        .problem()
        .with_bounds(vec![0.0; 30], vec![0.05; 30])
        .unwrap()
        .with_concentration_limit(0.08)
        .unwrap()
        .to_qp()
        .unwrap();
    assert_eq!(stricter.upper_bounds, vec![0.05; 30]);

    let solution = fixture
        .problem()
        .with_concentration_limit(0.06)
        .unwrap()
        .solve(None)
        .unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
    for (asset, weight) in solution.x.iter().enumerate() {
        assert!(
            *weight <= 0.06 + CONSTRAINT_TOLERANCE,
            "asset {asset}: weight {weight} above the concentration cap"
        );
    }
}

#[test]
fn templates_stack_with_user_constraints_and_each_other() {
    let fixture = fixture(42, 3);
    let assets = fixture.assets();
    let industries = fixture.industries(3);
    let benchmark = fixture.benchmark();
    let exposures = fixture.style_exposures(2);

    // One user inequality row before any template.
    let user_row = Matrix::new(1, assets, vec![1.0 / assets as f64; assets]).unwrap();
    let problem = fixture
        .problem()
        .with_inequalities(user_row.clone(), vec![0.5])
        .unwrap()
        .with_tracking_benchmark(benchmark)
        .unwrap()
        .with_industry_neutrality(&industries)
        .unwrap()
        .with_style_bounds(&exposures, &[-0.25, -0.25], &[0.25, 0.25])
        .unwrap()
        .with_concentration_limit(0.08)
        .unwrap()
        .with_short_limit(0.0)
        .unwrap();

    let qp = problem.to_qp().unwrap();
    // Equalities: budget + 3 industry rows. Inequalities: 1 user row + 2x2
    // style rows. Boxes tightened, no rows added by the box templates.
    assert_eq!(qp.equalities.len(), 4);
    assert_eq!(qp.inequalities.len(), 5);
    assert_eq!(qp.inequalities.matrix.row(0), user_row.row(0));
    assert_eq!(qp.upper_bounds, vec![0.08; assets]);
    assert_eq!(qp.lower_bounds, vec![0.0; assets]);

    let solution = problem.solve(None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
    let residuals = check_kkt(&qp, &solution.x, &solution.dual).unwrap();
    assert!(residuals.primal <= RESIDUAL_TOLERANCE);
    assert!(residuals.dual <= RESIDUAL_TOLERANCE);
}

#[test]
fn sequences_roll_template_targets_through_equality_rhs() {
    let fixture = fixture(36, 3);
    let groups = fixture.industries(3);

    let problem = fixture
        .problem()
        .with_group_targets(&groups, &[0.5, 0.3, 0.2])
        .unwrap();
    let mut sequence = problem.sequence().unwrap();
    let first = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(first.status, SolveStatus::Solved);
    let factorizations = sequence.factorizations();

    // Move the sleeve targets; template rows are ordinary user equality
    // rows, so the whole user RHS (here only the three template rows) rolls.
    let new_targets = vec![0.4, 0.4, 0.2];
    let rolled = sequence
        .solve_next(&RebalanceStep {
            equality_rhs: Some(new_targets.clone()),
            ..RebalanceStep::default()
        })
        .unwrap();
    assert_eq!(rolled.status, SolveStatus::Solved);
    for (group, target) in new_targets.iter().enumerate() {
        let held = group_weight(&rolled.x, &groups, group);
        assert!(
            (held - target).abs() <= CONSTRAINT_TOLERANCE,
            "group {group}: held {held} vs rolled target {target}"
        );
    }
    assert_eq!(
        sequence.factorizations(),
        factorizations,
        "template target updates must be served from the factorization cache"
    );
}

#[test]
fn invalid_template_data_is_rejected() {
    let fixture = fixture(24, 3);
    let assets = fixture.assets();
    let benchmark = fixture.benchmark();

    // Industry neutrality requires a benchmark.
    assert!(matches!(
        fixture
            .problem()
            .with_industry_neutrality(&fixture.industries(3)),
        Err(PortfolioError::Template(_))
    ));
    // Wrong id-vector length.
    assert!(matches!(
        fixture
            .problem()
            .with_tracking_benchmark(benchmark.clone())
            .unwrap()
            .with_industry_neutrality(&vec![0; assets - 1]),
        Err(PortfolioError::Problem(_))
    ));
    // Gap in industry ids leaves industry 1 without members.
    let gappy: Vec<usize> = (0..assets).map(|asset| (asset % 2) * 2).collect();
    assert!(matches!(
        fixture
            .problem()
            .with_tracking_benchmark(benchmark)
            .unwrap()
            .with_industry_neutrality(&gappy),
        Err(PortfolioError::Template(_))
    ));
    // Group id out of the explicit target range.
    assert!(matches!(
        fixture
            .problem()
            .with_group_targets(&fixture.industries(3), &[0.5, 0.5]),
        Err(PortfolioError::Template(_))
    ));
    // Non-finite target.
    assert!(matches!(
        fixture
            .problem()
            .with_group_targets(&fixture.industries(2), &[0.5, f64::NAN]),
        Err(PortfolioError::Problem(_))
    ));

    let exposures = fixture.style_exposures(2);
    // Crossing band.
    assert!(matches!(
        fixture
            .problem()
            .with_style_bounds(&exposures, &[0.3, 0.0], &[0.2, 0.1]),
        Err(PortfolioError::Template(_))
    ));
    // A style with no finite side is a mistake, not a no-op.
    assert!(matches!(
        fixture.problem().with_style_bounds(
            &exposures,
            &[f64::NEG_INFINITY, 0.0],
            &[f64::INFINITY, 0.1]
        ),
        Err(PortfolioError::Template(_))
    ));
    // Wrong exposure width.
    assert!(matches!(
        fixture.problem().with_style_bounds(
            &Matrix::new(1, assets - 1, vec![0.1; assets - 1]).unwrap(),
            &[0.0],
            &[0.1]
        ),
        Err(PortfolioError::Problem(_))
    ));

    // Box templates: invalid caps and contradictions with existing bounds.
    assert!(matches!(
        fixture.problem().with_concentration_limit(0.0),
        Err(PortfolioError::InvalidParameter(_))
    ));
    assert!(matches!(
        fixture.problem().with_short_limit(-0.1),
        Err(PortfolioError::InvalidParameter(_))
    ));
    // Forced minimum position above the concentration cap.
    assert!(matches!(
        fixture
            .problem()
            .with_bounds(vec![0.1; assets], vec![1.0; assets])
            .unwrap()
            .with_concentration_limit(0.05),
        Err(PortfolioError::Problem(_))
    ));
}
