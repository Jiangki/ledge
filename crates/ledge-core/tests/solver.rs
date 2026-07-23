//! Numerical solver integration tests.

use ledge_core::{
    check_kkt, generate_synthetic, DualVariables, FactorCovariance, FactorQuad, LinearConstraints,
    Matrix, QpProblem, SolveStatus, Solver, SolverError, SolverSettings, SyntheticConfig,
    WarmStart,
};

fn diagonal_problem(
    diagonal: Vec<f64>,
    linear: Vec<f64>,
    equalities: LinearConstraints,
    inequalities: LinearConstraints,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
) -> QpProblem {
    let n = diagonal.len();
    QpProblem {
        quadratic: FactorQuad::new(
            Matrix::new(n, 0, Vec::new()).unwrap(),
            FactorCovariance::Diagonal(Vec::new()),
            diagonal,
        )
        .unwrap(),
        linear,
        l1: None,
        equalities,
        inequalities,
        lower_bounds,
        upper_bounds,
    }
}

#[test]
fn solves_unconstrained_diagonal_quadratic_with_known_solution() {
    let problem = diagonal_problem(
        vec![2.0, 4.0],
        vec![-2.0, 8.0],
        LinearConstraints::empty(2),
        LinearConstraints::empty(2),
        vec![f64::NEG_INFINITY; 2],
        vec![f64::INFINITY; 2],
    );

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!((solution.x[0] - 1.0).abs() < 2.0e-5);
    assert!((solution.x[1] + 2.0).abs() < 2.0e-5);
    assert!(solution.residuals.primal < 1.0e-8);
    assert!(solution.residuals.dual < 1.0e-4);
}

#[test]
fn solves_budget_equality_and_reports_small_kkt_residuals() {
    let equalities =
        LinearConstraints::new(Matrix::new(1, 2, vec![1.0, 1.0]).unwrap(), vec![1.0]).unwrap();
    let problem = diagonal_problem(
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        equalities,
        LinearConstraints::empty(2),
        vec![0.0; 2],
        vec![1.0; 2],
    );

    let solution = Solver::default().solve(&problem, None).unwrap();
    let independently_checked = check_kkt(&problem, &solution.x, &solution.dual).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!((solution.x[0] - 0.5).abs() < 2.0e-5);
    assert!((solution.x[1] - 0.5).abs() < 2.0e-5);
    assert!(independently_checked.primal < 2.0e-5);
    assert!(independently_checked.dual < 2.0e-5);
    assert!(independently_checked.complementarity < 2.0e-5);
}

#[test]
fn enforces_upper_inequality_and_recovers_multiplier() {
    let inequalities =
        LinearConstraints::new(Matrix::new(1, 1, vec![1.0]).unwrap(), vec![1.0]).unwrap();
    let problem = diagonal_problem(
        vec![1.0],
        vec![-2.0],
        LinearConstraints::empty(1),
        inequalities,
        vec![f64::NEG_INFINITY],
        vec![f64::INFINITY],
    );

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!((solution.x[0] - 1.0).abs() < 3.0e-5);
    assert!((solution.dual.inequalities[0] - 1.0).abs() < 3.0e-5);
    assert!(solution.residuals.primal < 3.0e-5);
    assert!(solution.residuals.dual < 3.0e-5);
}

#[test]
fn accepts_full_warm_start_and_reduces_iterations() {
    let inequalities =
        LinearConstraints::new(Matrix::new(1, 1, vec![1.0]).unwrap(), vec![1.0]).unwrap();
    let problem = diagonal_problem(
        vec![1.0],
        vec![-2.0],
        LinearConstraints::empty(1),
        inequalities,
        vec![f64::NEG_INFINITY],
        vec![f64::INFINITY],
    );
    let settings = SolverSettings {
        check_termination_every: 1,
        ..SolverSettings::default()
    };
    let solver = Solver::new(settings);
    let cold = solver.solve(&problem, None).unwrap();
    let warm = WarmStart {
        x: vec![1.0],
        equality_dual: Some(Vec::new()),
        inequality_dual: Some(vec![1.0]),
        bound_dual: Some(vec![0.0]),
        l1_dual: None,
    };
    let hot = solver.solve(&problem, Some(&warm)).unwrap();

    assert_eq!(hot.status, SolveStatus::Solved);
    assert!(hot.iterations < cold.iterations);
    assert!(hot.iterations <= 2);
}

#[test]
fn adaptive_rho_recovers_from_a_poor_initial_penalty() {
    let equalities =
        LinearConstraints::new(Matrix::new(1, 2, vec![1.0, 1.0]).unwrap(), vec![1.0]).unwrap();
    let problem = diagonal_problem(
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        equalities,
        LinearConstraints::empty(2),
        vec![0.0; 2],
        vec![1.0; 2],
    );
    let solver = Solver::new(SolverSettings {
        rho: 1.0e-6,
        adaptive_rho_interval: 2,
        adaptive_rho_tolerance: 2.0,
        adaptive_rho_multiplier: 10.0,
        check_termination_every: 1,
        ..SolverSettings::default()
    });

    let solution = solver.solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(solution.rho_updates > 0);
    assert!(solution.final_rho > 1.0e-6);
    assert!(solution.residuals.primal < 2.0e-5);
}

#[test]
fn max_iterations_attaches_actionable_diagnostics() {
    let equalities =
        LinearConstraints::new(Matrix::new(1, 2, vec![1.0, 1.0]).unwrap(), vec![1.0]).unwrap();
    let problem = diagonal_problem(
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        equalities,
        LinearConstraints::empty(2),
        vec![0.0; 2],
        vec![1.0; 2],
    );
    let solver = Solver::new(SolverSettings {
        max_iterations: 2,
        ..SolverSettings::default()
    });

    let solution = solver.solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::MaxIterations);
    let diagnostics = solution
        .diagnostics
        .expect("failed solves carry diagnostics");
    assert!(diagnostics.primal_tolerance > 0.0);
    assert!(diagnostics.dual_tolerance > 0.0);
    assert!(!diagnostics.hints.is_empty());
}

#[test]
fn solved_status_carries_no_diagnostics() {
    let problem = diagonal_problem(
        vec![2.0, 4.0],
        vec![-2.0, 8.0],
        LinearConstraints::empty(2),
        LinearConstraints::empty(2),
        vec![f64::NEG_INFINITY; 2],
        vec![f64::INFINITY; 2],
    );

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(solution.diagnostics.is_none());
}

#[test]
fn over_relaxation_reduces_iterations_and_agrees_with_plain_admm() {
    let instance = generate_synthetic(SyntheticConfig {
        assets: 500,
        factors: 10,
        inequalities: 3,
        seed: 42,
        budget: 1.0,
        max_weight: 0.02,
    })
    .unwrap();

    let relaxed = Solver::default().solve(&instance.problem, None).unwrap();
    let plain = Solver::new(SolverSettings {
        over_relaxation: 1.0,
        ..SolverSettings::default()
    })
    .solve(&instance.problem, None)
    .unwrap();

    assert_eq!(relaxed.status, SolveStatus::Solved);
    assert_eq!(plain.status, SolveStatus::Solved);
    assert!(
        relaxed.iterations < plain.iterations,
        "relaxed took {} iterations vs plain {}",
        relaxed.iterations,
        plain.iterations
    );

    let scale = 1.0 + plain.objective.abs();
    assert!((relaxed.objective - plain.objective).abs() <= 1.0e-4 * scale);
    let residuals = check_kkt(&instance.problem, &relaxed.x, &relaxed.dual).unwrap();
    assert!(residuals.primal < 1.0e-4);
    assert!(residuals.dual < 1.0e-4);
    assert!(residuals.complementarity < 1.0e-4);
}

#[test]
fn rejects_out_of_range_over_relaxation() {
    let problem = diagonal_problem(
        vec![1.0],
        vec![0.0],
        LinearConstraints::empty(1),
        LinearConstraints::empty(1),
        vec![f64::NEG_INFINITY],
        vec![f64::INFINITY],
    );
    for alpha in [0.0, -0.5, 2.0, 2.5, f64::NAN] {
        let error = Solver::new(SolverSettings {
            over_relaxation: alpha,
            ..SolverSettings::default()
        })
        .solve(&problem, None)
        .unwrap_err();
        assert!(
            matches!(error, SolverError::InvalidSettings(_)),
            "alpha={alpha} must be rejected"
        );
    }
}

#[test]
fn kkt_checker_detects_wrong_dual_sign() {
    let inequalities =
        LinearConstraints::new(Matrix::new(1, 1, vec![1.0]).unwrap(), vec![1.0]).unwrap();
    let problem = diagonal_problem(
        vec![1.0],
        vec![-2.0],
        LinearConstraints::empty(1),
        inequalities,
        vec![f64::NEG_INFINITY],
        vec![f64::INFINITY],
    );
    let residuals = check_kkt(
        &problem,
        &[1.0],
        &DualVariables {
            equalities: Vec::new(),
            inequalities: vec![-1.0],
            bounds: vec![0.0],
            l1: Vec::new(),
        },
    )
    .unwrap();

    assert!(residuals.dual >= 1.0);
}

#[test]
fn dense_factor_covariance_is_applied_without_materializing_q() {
    let quadratic = FactorQuad::new(
        Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        FactorCovariance::Dense(Matrix::new(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
        vec![0.25, 0.75],
    )
    .unwrap();

    let product = quadratic.apply(&[2.0, -1.0]);

    assert!((product[0] - 4.0).abs() < 1.0e-14);
    assert!((product[1] + 0.75).abs() < 1.0e-14);
}

#[test]
fn rejects_indefinite_dense_factor_covariance_during_setup() {
    let quadratic = FactorQuad::new(
        Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        FactorCovariance::Dense(Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 1.0]).unwrap()),
        vec![1.0, 1.0],
    )
    .unwrap();
    let problem = QpProblem {
        quadratic,
        linear: vec![0.0; 2],
        l1: None,
        equalities: LinearConstraints::empty(2),
        inequalities: LinearConstraints::empty(2),
        lower_bounds: vec![f64::NEG_INFINITY; 2],
        upper_bounds: vec![f64::INFINITY; 2],
    };

    let error = Solver::default().solve(&problem, None).unwrap_err();

    assert!(matches!(error, SolverError::NonPositiveSemidefiniteOmega));
}
