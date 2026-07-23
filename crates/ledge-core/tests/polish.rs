//! Active-set polishing integration tests (roadmap 2.3).

use ledge_core::{
    check_kkt, generate_synthetic, FactorCovariance, FactorQuad, KktResiduals, LinearConstraints,
    Matrix, QpProblem, RebalanceStep, SolveStatus, Solver, SolverSettings, SyntheticConfig,
};

fn worst(residuals: &KktResiduals) -> f64 {
    residuals
        .primal
        .max(residuals.dual)
        .max(residuals.complementarity)
}

fn no_polish() -> SolverSettings {
    SolverSettings {
        polish: false,
        ..SolverSettings::default()
    }
}

#[test]
fn polish_reaches_high_accuracy_on_synthetic_portfolios() {
    for (assets, factors, inequalities) in [(60, 6, 3), (150, 10, 5), (400, 20, 8)] {
        let instance = generate_synthetic(SyntheticConfig {
            assets,
            factors,
            inequalities,
            seed: 7,
            ..SyntheticConfig::default()
        })
        .unwrap();

        let solution = Solver::default().solve(&instance.problem, None).unwrap();

        assert_eq!(solution.status, SolveStatus::Solved);
        assert!(
            solution.polished,
            "expected polish to improve n={assets}, residuals {:?}",
            solution.residuals
        );
        let audited = check_kkt(&instance.problem, &solution.x, &solution.dual).unwrap();
        assert!(
            worst(&audited) <= 1.0e-9,
            "n={assets}: polished residuals should reach 1e-9, got {audited:?}"
        );
    }
}

#[test]
fn polish_never_regresses_the_worst_residual() {
    for seed in [1, 2, 3, 4, 5, 6, 7, 8] {
        let instance = generate_synthetic(SyntheticConfig {
            assets: 80,
            factors: 8,
            inequalities: 4,
            seed,
            ..SyntheticConfig::default()
        })
        .unwrap();

        let polished = Solver::default().solve(&instance.problem, None).unwrap();
        let raw = Solver::new(no_polish())
            .solve(&instance.problem, None)
            .unwrap();

        assert_eq!(polished.status, SolveStatus::Solved);
        assert_eq!(raw.status, SolveStatus::Solved);
        assert_eq!(polished.iterations, raw.iterations);
        assert!(
            worst(&polished.residuals) <= worst(&raw.residuals),
            "seed {seed}: polish must never report worse residuals \
             ({:?} vs {:?})",
            polished.residuals,
            raw.residuals
        );
        for (with, without) in polished.x.iter().zip(&raw.x) {
            assert!(
                (with - without).abs() <= 1.0e-4,
                "seed {seed}: polish moved the solution more than refinement should"
            );
        }
    }
}

#[test]
fn polish_recovers_the_exact_constrained_optimum() {
    // min 0.5 x^2 - 2x  s.t. x <= 1: optimum exactly at the constraint with
    // multiplier 1. The ADMM iterate stops at tolerance; the polished one
    // solves the active system directly.
    let problem = QpProblem {
        quadratic: FactorQuad::new(
            Matrix::new(1, 0, Vec::new()).unwrap(),
            FactorCovariance::Diagonal(Vec::new()),
            vec![1.0],
        )
        .unwrap(),
        linear: vec![-2.0],
        l1: None,
        equalities: LinearConstraints::empty(1),
        inequalities: LinearConstraints::new(Matrix::new(1, 1, vec![1.0]).unwrap(), vec![1.0])
            .unwrap(),
        lower_bounds: vec![f64::NEG_INFINITY],
        upper_bounds: vec![f64::INFINITY],
    };

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(solution.polished);
    assert!((solution.x[0] - 1.0).abs() <= 1.0e-12);
    assert!((solution.dual.inequalities[0] - 1.0).abs() <= 1.0e-12);
    assert!(worst(&solution.residuals) <= 1.0e-12);
}

#[test]
fn polish_handles_active_bounds_without_growing_the_reduced_system() {
    // Long-only budget portfolio where the cheapest assets pin at the 0.1
    // cap, the expensive ones at zero, and the risk term keeps a few
    // interior: both bound sides are active at the optimum.
    let n = 30;
    let linear: Vec<f64> = (0..n).map(|index| -0.05 + 0.01 * index as f64).collect();
    let problem = QpProblem {
        quadratic: FactorQuad::new(
            Matrix::new(n, 0, Vec::new()).unwrap(),
            FactorCovariance::Diagonal(Vec::new()),
            vec![1.0; n],
        )
        .unwrap(),
        linear,
        l1: None,
        equalities: LinearConstraints::new(Matrix::new(1, n, vec![1.0; n]).unwrap(), vec![1.0])
            .unwrap(),
        inequalities: LinearConstraints::empty(n),
        lower_bounds: vec![0.0; n],
        upper_bounds: vec![0.1; n],
    };

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(solution.polished);
    assert!(worst(&solution.residuals) <= 1.0e-10);
    let capped = solution
        .x
        .iter()
        .filter(|&&w| (w - 0.1).abs() < 1e-9)
        .count();
    let floored = solution.x.iter().filter(|&&w| w.abs() < 1e-9).count();
    let interior = solution
        .x
        .iter()
        .filter(|&&w| w > 1e-6 && w < 0.1 - 1e-6)
        .count();
    assert!(capped >= 5, "expected the cheap assets pinned at the cap");
    assert!(floored >= 5, "expected the expensive assets pinned at zero");
    assert!(interior >= 1, "expected at least one interior asset");
}

#[test]
fn fully_degenerate_active_sets_fall_back_to_the_admm_iterate() {
    // Bang-bang portfolio: risk is negligible, so *every* variable lands on
    // a bound and the budget row is linearly dependent on the pins. The
    // multiplier split is then non-unique and the polished candidate cannot
    // be certified; the audit must reject it and keep the ADMM iterate
    // (with its already-checked residuals) rather than adopt uncertain
    // duals.
    let n = 30;
    let linear: Vec<f64> = (0..n).map(|index| -0.05 + 0.01 * index as f64).collect();
    let problem = QpProblem {
        quadratic: FactorQuad::new(
            Matrix::new(n, 0, Vec::new()).unwrap(),
            FactorCovariance::Diagonal(Vec::new()),
            vec![0.01; n],
        )
        .unwrap(),
        linear,
        l1: None,
        equalities: LinearConstraints::new(Matrix::new(1, n, vec![1.0; n]).unwrap(), vec![1.0])
            .unwrap(),
        inequalities: LinearConstraints::empty(n),
        lower_bounds: vec![0.0; n],
        upper_bounds: vec![0.1; n],
    };

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    let audited = check_kkt(&problem, &solution.x, &solution.dual).unwrap();
    assert!(worst(&audited) <= 1.0e-4, "iterate must stay tolerable");
}

#[test]
fn polish_can_be_disabled() {
    let instance = generate_synthetic(SyntheticConfig::default()).unwrap();

    let solution = Solver::new(no_polish())
        .solve(&instance.problem, None)
        .unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(!solution.polished);
}

#[test]
fn infeasible_problems_keep_their_certificate_and_are_never_polished() {
    // `sum(x) = 1` against upper bounds that only admit 0.8 in total.
    let n = 4;
    let problem = QpProblem {
        quadratic: FactorQuad::new(
            Matrix::new(n, 0, Vec::new()).unwrap(),
            FactorCovariance::Diagonal(Vec::new()),
            vec![1.0; n],
        )
        .unwrap(),
        linear: vec![0.0; n],
        l1: None,
        equalities: LinearConstraints::new(Matrix::new(1, n, vec![1.0; n]).unwrap(), vec![1.0])
            .unwrap(),
        inequalities: LinearConstraints::empty(n),
        lower_bounds: vec![0.0; n],
        upper_bounds: vec![0.2; n],
    };

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::PrimalInfeasible);
    assert!(!solution.polished);
    assert!(solution.certificate.is_some());
}

#[test]
fn rolling_sequence_solves_are_polished_every_date() {
    let instance = generate_synthetic(SyntheticConfig {
        assets: 50,
        factors: 5,
        inequalities: 3,
        ..SyntheticConfig::default()
    })
    .unwrap();
    let problem = ledge_core::PortfolioProblem::new(
        instance.problem.quadratic.factors.clone(),
        instance.problem.quadratic.omega.clone(),
        instance.problem.quadratic.diagonal.clone(),
        instance.problem.linear.iter().map(|value| -value).collect(),
    )
    .unwrap()
    .with_bounds(
        instance.problem.lower_bounds.clone(),
        instance.problem.upper_bounds.clone(),
    )
    .unwrap()
    .with_inequalities(
        instance.problem.inequalities.matrix.clone(),
        instance.problem.inequalities.rhs.clone(),
    )
    .unwrap();

    let mut sequence = problem.sequence().unwrap();
    let mut returns: Vec<f64> = instance.problem.linear.iter().map(|v| -v).collect();
    for date in 0..4 {
        let step = if date == 0 {
            RebalanceStep::default()
        } else {
            for (index, value) in returns.iter_mut().enumerate() {
                *value += 1.0e-4 * ((date * 7 + index) % 5) as f64;
            }
            RebalanceStep {
                expected_returns: Some(returns.clone()),
                ..RebalanceStep::default()
            }
        };
        let solution = sequence.solve_next(&step).unwrap();
        assert_eq!(solution.status, SolveStatus::Solved);
        assert!(solution.polished, "date {date} should polish");
        assert!(
            worst(&solution.residuals) <= 1.0e-9,
            "date {date}: rolling polished residuals should reach 1e-9, got {:?}",
            solution.residuals
        );
    }
}

#[test]
fn rejects_non_positive_polish_regularization() {
    let instance = generate_synthetic(SyntheticConfig::default()).unwrap();
    let settings = SolverSettings {
        polish_regularization: 0.0,
        ..SolverSettings::default()
    };

    let error = Solver::new(settings)
        .solve(&instance.problem, None)
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("polish_regularization must be finite and positive"));
}
