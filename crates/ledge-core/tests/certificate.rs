//! Infeasibility certificate integration tests (roadmap 2.2).

use ledge_core::{
    check_dual_certificate, check_primal_certificate, Certificate, FactorCovariance, FactorQuad,
    LinearConstraints, Matrix, PortfolioProblem, QpProblem, RebalanceStep, SolveStatus, Solver,
    SolverSettings,
};

/// `sum(x) = 1` against upper bounds that only allow `0.8` in total.
fn equality_versus_boxes() -> QpProblem {
    let n = 4;
    QpProblem {
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
    }
}

/// Long-only, budget 1, but the two sector caps only admit 0.4 each.
fn budget_versus_sector_caps() -> PortfolioProblem {
    PortfolioProblem::new(
        Matrix::new(4, 1, vec![0.9, 1.1, 0.8, 1.2]).unwrap(),
        FactorCovariance::Diagonal(vec![0.05]),
        vec![0.1, 0.12, 0.09, 0.11],
        vec![0.08, 0.06, 0.05, 0.07],
    )
    .unwrap()
    .with_inequalities(
        Matrix::new(2, 4, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]).unwrap(),
        vec![0.4, 0.4],
    )
    .unwrap()
}

#[test]
fn equality_versus_boxes_yields_an_audited_farkas_certificate() {
    let problem = equality_versus_boxes();

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::PrimalInfeasible);
    assert!(
        solution.iterations < 1000,
        "detection should preempt the iteration budget, took {}",
        solution.iterations
    );
    let Some(Certificate::Primal(certificate)) = &solution.certificate else {
        panic!("primal infeasible solves must attach a primal certificate");
    };
    let audited = check_primal_certificate(&problem, certificate).unwrap();
    assert!(audited.stationarity <= 1.0e-5);
    assert!(audited.cone_violation <= 1.0e-12);
    assert!(audited.support_gap <= -1.0e-5);
    let diagnostics = solution.diagnostics.expect("failure carries diagnostics");
    assert!(
        diagnostics
            .hints
            .iter()
            .any(|hint| hint.contains("Farkas certificate")),
        "hints were {:?}",
        diagnostics.hints
    );
}

#[test]
fn contradictory_equalities_yield_a_certificate() {
    let problem = QpProblem {
        quadratic: FactorQuad::new(
            Matrix::new(2, 0, Vec::new()).unwrap(),
            FactorCovariance::Diagonal(Vec::new()),
            vec![1.0, 1.0],
        )
        .unwrap(),
        linear: vec![0.0, 0.0],
        l1: None,
        equalities: LinearConstraints::new(
            Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
            vec![1.0, 2.0],
        )
        .unwrap(),
        inequalities: LinearConstraints::empty(2),
        lower_bounds: vec![f64::NEG_INFINITY; 2],
        upper_bounds: vec![f64::INFINITY; 2],
    };

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::PrimalInfeasible);
    let Some(Certificate::Primal(certificate)) = &solution.certificate else {
        panic!("expected a primal certificate");
    };
    let audited = check_primal_certificate(&problem, certificate).unwrap();
    assert!(audited.stationarity <= 1.0e-5);
    assert!(audited.support_gap <= -1.0e-5);
}

#[test]
fn unbounded_objective_yields_a_dual_certificate() {
    // Two assets with identical factor exposure and zero specific variance:
    // the long-short direction (1, -1) carries no risk, and the linear cost
    // strictly rewards it. No budget, no bounds.
    let problem = QpProblem {
        quadratic: FactorQuad::new(
            Matrix::new(2, 1, vec![1.0, 1.0]).unwrap(),
            FactorCovariance::Diagonal(vec![1.0]),
            vec![0.0, 0.0],
        )
        .unwrap(),
        linear: vec![-0.10, -0.05],
        l1: None,
        equalities: LinearConstraints::empty(2),
        inequalities: LinearConstraints::empty(2),
        lower_bounds: vec![f64::NEG_INFINITY; 2],
        upper_bounds: vec![f64::INFINITY; 2],
    };

    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::DualInfeasible);
    let Some(Certificate::Dual(certificate)) = &solution.certificate else {
        panic!("dual infeasible solves must attach a dual certificate");
    };
    let audited = check_dual_certificate(&problem, certificate).unwrap();
    assert!(audited.curvature <= 1.0e-5);
    assert!(audited.recession_violation <= 1.0e-5);
    assert!(audited.objective_gap <= -1.0e-5);
    let diagnostics = solution.diagnostics.expect("failure carries diagnostics");
    assert!(
        diagnostics
            .hints
            .iter()
            .any(|hint| hint.contains("unbounded")),
        "hints were {:?}",
        diagnostics.hints
    );
}

#[test]
fn portfolio_certificate_hints_name_the_budget_and_the_caps() {
    let problem = budget_versus_sector_caps();

    let solution = problem.solve(None).unwrap();

    assert_eq!(solution.status, SolveStatus::PrimalInfeasible);
    assert!(matches!(solution.certificate, Some(Certificate::Primal(_))));
    let diagnostics = solution.diagnostics.expect("failure carries diagnostics");
    let semantic = &diagnostics.hints[0];
    assert!(
        semantic.contains("budget") && semantic.contains("inequality cap"),
        "first hint should name the conflicting portfolio constraints, got {semantic:?}"
    );
}

#[test]
fn detection_disabled_falls_back_to_max_iterations() {
    let problem = equality_versus_boxes();
    let solver = Solver::new(SolverSettings {
        infeasibility_tolerance: 0.0,
        max_iterations: 500,
        ..SolverSettings::default()
    });

    let solution = solver.solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::MaxIterations);
    assert!(solution.certificate.is_none());
}

#[test]
fn boundary_feasible_caps_still_solve() {
    // Caps sum exactly to the budget: feasible with zero slack. The detector
    // must not mistake tightness for infeasibility.
    let problem = PortfolioProblem::new(
        Matrix::new(4, 1, vec![0.9, 1.1, 0.8, 1.2]).unwrap(),
        FactorCovariance::Diagonal(vec![0.05]),
        vec![0.1, 0.12, 0.09, 0.11],
        vec![0.08, 0.06, 0.05, 0.07],
    )
    .unwrap()
    .with_inequalities(
        Matrix::new(2, 4, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]).unwrap(),
        vec![0.5, 0.5],
    )
    .unwrap();

    let solution = problem.solve(None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(solution.certificate.is_none());
}

#[test]
fn sequence_recovers_cold_after_an_infeasible_date() {
    let problem = PortfolioProblem::new(
        Matrix::new(4, 1, vec![0.9, 1.1, 0.8, 1.2]).unwrap(),
        FactorCovariance::Diagonal(vec![0.05]),
        vec![0.1, 0.12, 0.09, 0.11],
        vec![0.08, 0.06, 0.05, 0.07],
    )
    .unwrap()
    .with_inequalities(
        Matrix::new(2, 4, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]).unwrap(),
        vec![0.6, 0.6],
    )
    .unwrap();
    let mut sequence = problem.sequence().unwrap();

    let feasible = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(feasible.status, SolveStatus::Solved);

    let infeasible = sequence
        .solve_next(&RebalanceStep {
            inequality_rhs: Some(vec![0.2, 0.2]),
            ..RebalanceStep::default()
        })
        .unwrap();
    assert_eq!(infeasible.status, SolveStatus::PrimalInfeasible);
    let diagnostics = infeasible
        .diagnostics
        .expect("infeasible dates carry diagnostics");
    assert!(
        diagnostics.hints[0].contains("budget"),
        "sequence hints should speak portfolio vocabulary, got {:?}",
        diagnostics.hints
    );

    // The diverged duals of the infeasible date must not poison the next
    // date: the sequence restarts cold and solves.
    let recovered = sequence
        .solve_next(&RebalanceStep {
            inequality_rhs: Some(vec![0.7, 0.7]),
            ..RebalanceStep::default()
        })
        .unwrap();
    assert_eq!(recovered.status, SolveStatus::Solved);
}
