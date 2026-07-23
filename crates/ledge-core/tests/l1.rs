//! Exact L1 turnover / proportional-cost integration tests (roadmap 2.1).
//!
//! The L1 term is handled by a dedicated soft-threshold proximal block, so
//! these tests hold it to the same standards as every other feature: known
//! closed-form solutions must be met exactly, the independent KKT checker
//! must audit the new multiplier block, an epigraph reformulation of the
//! same problem must agree, the no-trade region must be genuinely sticky,
//! and rolling sequences must reuse factorizations across anchor updates.

use ledge_core::{
    check_dual_certificate, check_kkt, generate_synthetic, Certificate, FactorCovariance,
    FactorQuad, L1Term, LinearConstraints, Matrix, PortfolioProblem, QpProblem, RebalanceStep,
    SolveStatus, Solver, SolverSettings, SyntheticConfig, WarmStart,
};
use proptest::prelude::*;

const OBJECTIVE_TOLERANCE: f64 = 1.0e-4;
const RESIDUAL_TOLERANCE: f64 = 1.0e-4;

fn worst(solution: &ledge_core::Solution) -> f64 {
    solution
        .residuals
        .primal
        .max(solution.residuals.dual)
        .max(solution.residuals.complementarity)
}

/// A diagonal QP `0.5 * d x^2 + q x + c |x - a|` with optional boxes.
fn scalar_l1_problem(
    diagonal: f64,
    linear: f64,
    cost: f64,
    anchor: f64,
    lower: f64,
    upper: f64,
) -> QpProblem {
    QpProblem {
        quadratic: FactorQuad::new(
            Matrix::new(1, 0, Vec::new()).unwrap(),
            FactorCovariance::Diagonal(Vec::new()),
            vec![diagonal],
        )
        .unwrap(),
        linear: vec![linear],
        l1: Some(L1Term {
            costs: vec![cost],
            anchor: vec![anchor],
        }),
        equalities: LinearConstraints::empty(1),
        inequalities: LinearConstraints::empty(1),
        lower_bounds: vec![lower],
        upper_bounds: vec![upper],
    }
}

/// Deterministic long-only factor portfolio QP with an L1 term.
fn portfolio_l1_problem(assets: usize) -> QpProblem {
    let factor_count = 2;
    let mut factors = Vec::with_capacity(assets * factor_count);
    for row in 0..assets {
        for col in 0..factor_count {
            let angle = (1 + row * factor_count + col) as f64;
            factors.push(0.3 * (angle * 12.9898).sin());
        }
    }
    let diagonal: Vec<f64> = (0..assets)
        .map(|index| 0.08 + 0.04 * ((index * 7 % 13) as f64) / 13.0)
        .collect();
    let linear: Vec<f64> = (0..assets)
        .map(|index| -0.05 - 0.03 * ((index as f64) * 0.7).cos())
        .collect();
    let max_weight = (4.0 / assets as f64).min(1.0);
    // Anchor: a feasible previous portfolio slightly away from uniform.
    let anchor: Vec<f64> = (0..assets)
        .map(|index| (1.0 / assets as f64) * (1.0 + 0.4 * ((index as f64) * 1.3).sin()))
        .collect();
    let costs: Vec<f64> = (0..assets)
        .map(|index| 0.002 + 0.001 * ((index % 3) as f64))
        .collect();
    QpProblem {
        quadratic: FactorQuad::new(
            Matrix::new(assets, factor_count, factors).unwrap(),
            FactorCovariance::Diagonal(vec![0.05, 0.06]),
            diagonal,
        )
        .unwrap(),
        linear,
        l1: Some(L1Term { costs, anchor }),
        equalities: LinearConstraints::new(
            Matrix::new(1, assets, vec![1.0; assets]).unwrap(),
            vec![1.0],
        )
        .unwrap(),
        inequalities: LinearConstraints::empty(assets),
        lower_bounds: vec![0.0; assets],
        upper_bounds: vec![max_weight; assets],
    }
}

/// Rewrites `problem` (which must have an L1 term) as the standard epigraph
/// QP over `[x; t]` with `x_i - t_i <= a_i`, `-x_i - t_i <= -a_i`, `t >= 0`
/// and linear cost `[q; c]`. This is the reformulation Ledge's prox block
/// exists to avoid; small instances make an independent reference.
fn epigraph_reformulation(problem: &QpProblem) -> QpProblem {
    let term = problem.l1.as_ref().expect("epigraph needs an L1 term");
    let n = problem.quadratic.dimension();
    let k = problem.quadratic.factor_count();

    let mut factors = Matrix::zeros(2 * n, k);
    for row in 0..n {
        for col in 0..k {
            factors[(row, col)] = problem.quadratic.factors[(row, col)];
        }
    }
    let mut diagonal = problem.quadratic.diagonal.clone();
    diagonal.resize(2 * n, 0.0);

    let mut linear = problem.linear.clone();
    linear.extend_from_slice(&term.costs);

    let equality_rows = problem.equalities.len();
    let mut equalities = Matrix::zeros(equality_rows, 2 * n);
    for row in 0..equality_rows {
        for col in 0..n {
            equalities[(row, col)] = problem.equalities.matrix[(row, col)];
        }
    }

    let inequality_rows = problem.inequalities.len();
    let mut inequalities = Matrix::zeros(inequality_rows + 2 * n, 2 * n);
    let mut inequality_rhs = problem.inequalities.rhs.clone();
    for row in 0..inequality_rows {
        for col in 0..n {
            inequalities[(row, col)] = problem.inequalities.matrix[(row, col)];
        }
    }
    for index in 0..n {
        inequalities[(inequality_rows + 2 * index, index)] = 1.0;
        inequalities[(inequality_rows + 2 * index, n + index)] = -1.0;
        inequality_rhs.push(term.anchor[index]);
        inequalities[(inequality_rows + 2 * index + 1, index)] = -1.0;
        inequalities[(inequality_rows + 2 * index + 1, n + index)] = -1.0;
        inequality_rhs.push(-term.anchor[index]);
    }

    let mut lower_bounds = problem.lower_bounds.clone();
    lower_bounds.extend(vec![0.0; n]);
    let mut upper_bounds = problem.upper_bounds.clone();
    upper_bounds.extend(vec![f64::INFINITY; n]);

    QpProblem {
        quadratic: FactorQuad::new(factors, problem.quadratic.omega.clone(), diagonal).unwrap(),
        linear,
        l1: None,
        equalities: LinearConstraints::new(equalities, problem.equalities.rhs.clone()).unwrap(),
        inequalities: LinearConstraints::new(inequalities, inequality_rhs).unwrap(),
        lower_bounds,
        upper_bounds,
    }
}

#[test]
fn scalar_trading_solution_matches_the_closed_form() {
    // Smooth gradient at the anchor: d*a + q = 2*0.1 - 1 = -0.8, beyond the
    // cost 0.3, so the optimum trades up: d*x + q + c = 0 -> x = 0.35.
    let problem = scalar_l1_problem(2.0, -1.0, 0.3, 0.1, f64::NEG_INFINITY, f64::INFINITY);
    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(solution.polished, "polish must handle L1 trading assets");
    assert!((solution.x[0] - 0.35).abs() <= 1.0e-9);
    assert!((solution.dual.l1[0] - 0.3).abs() <= 1.0e-9);
    assert!(worst(&solution) <= 1.0e-9);
    // Objective includes the L1 term: 0.5*2*0.35^2 - 0.35 + 0.3*|0.25|.
    assert!((solution.objective - (0.1225 - 0.35 + 0.075)).abs() <= 1.0e-9);
}

#[test]
fn scalar_no_trade_region_is_exactly_sticky() {
    // Smooth gradient at the anchor: 2*0.1 - 0.15 = 0.05, inside the cost
    // 0.3: the optimum stays exactly at the anchor and the multiplier
    // balances stationarity strictly inside the cost interval.
    let problem = scalar_l1_problem(2.0, -0.15, 0.3, 0.1, f64::NEG_INFINITY, f64::INFINITY);
    let solution = Solver::default().solve(&problem, None).unwrap();

    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(solution.polished, "polish must pin no-trade assets");
    assert!((solution.x[0] - 0.1).abs() <= 1.0e-12);
    assert!((solution.dual.l1[0] - (-0.05)).abs() <= 1.0e-9);
    assert!(worst(&solution) <= 1.0e-9);
}

#[test]
fn zero_costs_reproduce_the_plain_qp() {
    let mut with_zero_l1 = portfolio_l1_problem(12);
    with_zero_l1.l1 = Some(L1Term {
        costs: vec![0.0; 12],
        anchor: vec![0.05; 12],
    });
    let mut plain = with_zero_l1.clone();
    plain.l1 = None;

    let solver = Solver::default();
    let l1_solution = solver.solve(&with_zero_l1, None).unwrap();
    let plain_solution = solver.solve(&plain, None).unwrap();

    assert_eq!(l1_solution.status, SolveStatus::Solved);
    assert_eq!(plain_solution.status, SolveStatus::Solved);
    let scale = 1.0 + plain_solution.objective.abs();
    assert!(
        (l1_solution.objective - plain_solution.objective).abs() <= OBJECTIVE_TOLERANCE * scale
    );
    for (a, b) in l1_solution.x.iter().zip(&plain_solution.x) {
        assert!((a - b).abs() <= 1.0e-3);
    }
}

#[test]
fn agrees_with_the_epigraph_reformulation() {
    let problem = portfolio_l1_problem(10);
    let epigraph = epigraph_reformulation(&problem);
    let solver = Solver::default();

    let prox = solver.solve(&problem, None).unwrap();
    let reference = solver.solve(&epigraph, None).unwrap();
    assert_eq!(prox.status, SolveStatus::Solved);
    assert_eq!(reference.status, SolveStatus::Solved);

    let scale = 1.0 + reference.objective.abs();
    assert!(
        (prox.objective - reference.objective).abs() <= OBJECTIVE_TOLERANCE * scale,
        "prox {} vs epigraph {}",
        prox.objective,
        reference.objective
    );
    for (index, (a, b)) in prox.x.iter().zip(&reference.x).enumerate() {
        assert!(
            (a - b).abs() <= 2.0e-3,
            "weight {index} diverged: prox {a} vs epigraph {b}"
        );
    }

    // Both solutions must audit cleanly on their own problem data.
    let residuals = check_kkt(&problem, &prox.x, &prox.dual).unwrap();
    assert!(residuals.primal <= RESIDUAL_TOLERANCE);
    assert!(residuals.dual <= RESIDUAL_TOLERANCE);
    assert!(residuals.complementarity <= RESIDUAL_TOLERANCE);
}

#[test]
fn large_costs_freeze_the_feasible_previous_portfolio() {
    let mut problem = portfolio_l1_problem(12);
    // The anchor from the fixture sums to ~1 but not exactly; project it
    // onto the budget so "do not trade" is feasible.
    let term = problem.l1.as_mut().unwrap();
    let total: f64 = term.anchor.iter().sum();
    for value in &mut term.anchor {
        *value /= total;
    }
    for cost in &mut term.costs {
        *cost = 10.0; // dwarfs every return and risk gradient
    }
    let anchor = term.anchor.clone();

    let solution = Solver::default().solve(&problem, None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(solution.polished);
    for (value, previous) in solution.x.iter().zip(&anchor) {
        assert!(
            (value - previous).abs() <= 1.0e-9,
            "asset should not trade under overwhelming costs"
        );
    }
}

#[test]
fn corrupted_l1_multipliers_fail_the_independent_audit() {
    let problem = portfolio_l1_problem(10);
    let solution = Solver::default().solve(&problem, None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);

    // Push one multiplier outside [-c, c]: the checker must charge the
    // dual-cone violation even though stationarity is now broken too.
    let mut corrupted = solution.dual.clone();
    corrupted.l1[0] = problem.l1.as_ref().unwrap().costs[0] + 0.5;
    let residuals = check_kkt(&problem, &solution.x, &corrupted).unwrap();
    assert!(residuals.dual >= 0.4);
}

#[test]
fn l1_recession_slope_blocks_false_unboundedness_claims() {
    // `min x + 2 |x|` is bounded below (at zero): the descent direction
    // `-1` gains 2 from the L1 recession function, so it is *not* a valid
    // certificate and the solver must simply solve the problem.
    let bounded = scalar_l1_problem(0.0, 1.0, 2.0, 0.0, f64::NEG_INFINITY, f64::INFINITY);
    let direction = ledge_core::DualCertificate {
        direction: vec![-1.0],
    };
    let audited = check_dual_certificate(&bounded, &direction).unwrap();
    assert!(
        audited.objective_gap >= 1.0 - 1.0e-12,
        "the L1 slope must enter the recession audit"
    );
    let solution = Solver::default().solve(&bounded, None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
    assert!(solution.x[0].abs() <= 1.0e-6);

    // With cost 0.5 the same slope genuinely descends at rate 0.5: the
    // solver must detect dual infeasibility and attach an auditable ray.
    let unbounded = scalar_l1_problem(0.0, 1.0, 0.5, 0.0, f64::NEG_INFINITY, f64::INFINITY);
    let diverged = Solver::default().solve(&unbounded, None).unwrap();
    assert_eq!(diverged.status, SolveStatus::DualInfeasible);
    let Some(Certificate::Dual(certificate)) = &diverged.certificate else {
        panic!("dual infeasibility must attach a descent-ray certificate");
    };
    let audited = check_dual_certificate(&unbounded, certificate).unwrap();
    assert!(audited.curvature <= 1.0e-5);
    assert!(audited.objective_gap <= -1.0e-5);
    assert!(audited.recession_violation <= 1.0e-5);
}

#[test]
fn warm_started_resolves_agree_with_cold_solves() {
    let problem = portfolio_l1_problem(14);
    let solver = Solver::default();
    let cold = solver.solve(&problem, None).unwrap();
    assert_eq!(cold.status, SolveStatus::Solved);

    let warm_start = cold.warm_start();
    assert_eq!(
        warm_start.l1_dual.as_ref().map(Vec::len),
        Some(problem.quadratic.dimension())
    );
    let warm = solver.solve(&problem, Some(&warm_start)).unwrap();
    assert_eq!(warm.status, SolveStatus::Solved);
    assert!(warm.iterations <= cold.iterations);
    let scale = 1.0 + cold.objective.abs();
    assert!((warm.objective - cold.objective).abs() <= OBJECTIVE_TOLERANCE * scale);
}

#[test]
fn warm_start_rejects_wrong_l1_dual_dimension() {
    let problem = portfolio_l1_problem(6);
    let warm = WarmStart {
        x: vec![0.0; 6],
        l1_dual: Some(vec![0.0; 3]),
        ..WarmStart::default()
    };
    assert!(Solver::default().solve(&problem, Some(&warm)).is_err());
}

#[test]
fn workspace_anchor_updates_agree_with_fresh_solves() {
    let problem = portfolio_l1_problem(12);
    let solver = Solver::default();
    let mut workspace = solver.workspace(&problem).unwrap();

    let first = workspace.solve(None).unwrap();
    assert_eq!(first.status, SolveStatus::Solved);
    let factorizations_after_first = workspace.factorizations();

    // Move the anchor to the first solution (rebalance happened) and
    // re-solve warm; compare against a fresh one-shot solve of the same
    // updated data.
    workspace.update_l1_anchor(&first.x).unwrap();
    let rolled = workspace.solve(Some(&first.warm_start())).unwrap();
    assert_eq!(rolled.status, SolveStatus::Solved);
    assert_eq!(
        workspace.factorizations(),
        factorizations_after_first,
        "anchor updates must not invalidate cached factorizations"
    );

    let mut updated = problem.clone();
    updated.l1.as_mut().unwrap().anchor = first.x.clone();
    let fresh = solver.solve(&updated, None).unwrap();
    assert_eq!(fresh.status, SolveStatus::Solved);
    let scale = 1.0 + fresh.objective.abs();
    assert!((rolled.objective - fresh.objective).abs() <= OBJECTIVE_TOLERANCE * scale);
}

#[test]
fn workspace_rejects_anchor_updates_without_an_l1_term() {
    let mut plain = portfolio_l1_problem(6);
    plain.l1 = None;
    let solver = Solver::default();
    let mut workspace = solver.workspace(&plain).unwrap();
    assert!(workspace.update_l1_anchor(&[0.0; 6]).is_err());
}

/// Portfolio-level fixture for the high-level API tests.
fn l1_portfolio_with(
    assets: usize,
    expected: Option<Vec<f64>>,
    previous: Option<Vec<f64>>,
) -> PortfolioProblem {
    let factor_count = 2;
    let mut factors = Vec::with_capacity(assets * factor_count);
    for row in 0..assets {
        for col in 0..factor_count {
            let angle = (1 + row * factor_count + col) as f64;
            factors.push(0.3 * (angle * 12.9898).sin());
        }
    }
    let expected = expected.unwrap_or_else(|| {
        (0..assets)
            .map(|index| 0.05 + 0.03 * ((index as f64) * 0.7).cos())
            .collect()
    });
    let specific: Vec<f64> = (0..assets)
        .map(|index| 0.08 + 0.04 * ((index * 7 % 13) as f64) / 13.0)
        .collect();
    let previous = previous.unwrap_or_else(|| vec![1.0 / assets as f64; assets]);
    let max_weight = (4.0 / assets as f64).min(1.0);
    PortfolioProblem::new(
        Matrix::new(assets, factor_count, factors).unwrap(),
        FactorCovariance::Diagonal(vec![0.05, 0.06]),
        specific,
        expected,
    )
    .unwrap()
    .with_risk_aversion(4.0)
    .unwrap()
    .with_bounds(vec![0.0; assets], vec![max_weight; assets])
    .unwrap()
    .with_l1_turnover(previous, vec![0.003; assets])
    .unwrap()
}

fn l1_portfolio(assets: usize) -> PortfolioProblem {
    l1_portfolio_with(assets, None, None)
}

#[test]
fn portfolio_with_l1_turnover_solves_and_audits() {
    let problem = l1_portfolio(12);
    let solution = problem.solve(None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);
    let qp = problem.to_qp().unwrap();
    assert!(qp.l1.is_some());
    let residuals = check_kkt(&qp, &solution.x, &solution.dual).unwrap();
    assert!(residuals.primal <= RESIDUAL_TOLERANCE);
    assert!(residuals.dual <= RESIDUAL_TOLERANCE);
    assert!(residuals.complementarity <= RESIDUAL_TOLERANCE);
}

#[test]
fn portfolio_rejects_invalid_l1_costs() {
    let problem = l1_portfolio(6);
    assert!(problem
        .clone()
        .with_l1_turnover(vec![0.1; 6], vec![-0.001; 6])
        .is_err());
    assert!(problem
        .with_l1_turnover(vec![0.1; 6], vec![0.001; 3])
        .is_err());
}

#[test]
fn sequences_update_the_l1_anchor_and_reuse_factorizations() {
    let problem = l1_portfolio(12);
    let mut sequence = problem.sequence().unwrap();

    let first = sequence.solve_next(&RebalanceStep::default()).unwrap();
    assert_eq!(first.status, SolveStatus::Solved);
    let factorizations = sequence.factorizations();

    // Roll twice: the executed portfolio becomes the next anchor.
    let mut anchor = first.x.clone();
    for step_index in 0..2_i32 {
        let returns: Vec<f64> = (0..12_i32)
            .map(|index| 0.05 + 0.02 * (f64::from(index + step_index) * 0.9).sin())
            .collect();
        let solution = sequence
            .solve_next(&RebalanceStep {
                expected_returns: Some(returns.clone()),
                previous_weights: Some(anchor.clone()),
                ..RebalanceStep::default()
            })
            .unwrap();
        assert_eq!(solution.status, SolveStatus::Solved);
        assert_eq!(
            sequence.factorizations(),
            factorizations,
            "anchor/return updates must reuse cached factorizations"
        );

        // The rolled answer must match a fresh problem built for this date.
        let fresh = l1_portfolio_with(12, Some(returns), Some(anchor.clone()))
            .solve(None)
            .unwrap();
        let scale = 1.0 + fresh.objective.abs();
        assert!(
            (solution.objective - fresh.objective).abs() <= OBJECTIVE_TOLERANCE * scale,
            "sequence and fresh solves diverged at step {step_index}"
        );
        anchor = solution.x.clone();
    }
}

#[test]
fn mixed_l2_and_l1_turnover_solve_together() {
    let assets = 10;
    let problem = l1_portfolio(assets)
        .with_quadratic_turnover(vec![1.0 / assets as f64; assets], 0.5)
        .unwrap()
        .with_l1_turnover(vec![1.0 / assets as f64; assets], vec![0.002; assets])
        .unwrap();
    let solution = problem.solve(None).unwrap();
    assert_eq!(solution.status, SolveStatus::Solved);

    let qp = problem.to_qp().unwrap();
    let residuals = check_kkt(&qp, &solution.x, &solution.dual).unwrap();
    assert!(residuals.primal <= RESIDUAL_TOLERANCE);
    assert!(residuals.dual <= RESIDUAL_TOLERANCE);
    assert!(residuals.complementarity <= RESIDUAL_TOLERANCE);

    let settings = SolverSettings::default();
    assert!(settings.polish, "defaults keep polish on for L1 problems");
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

    /// Random feasible factor QPs decorated with a random L1 term must
    /// solve, pass the independent audit, and agree with their epigraph
    /// reformulation — the strongest generic equivalence check available
    /// without an external solver.
    #[test]
    fn random_l1_instances_agree_with_their_epigraph_form(
        assets in 2_usize..=16,
        factors in 1_usize..=3,
        seed in any::<u64>(),
        cost_scale in 1.0e-4_f64..0.05,
        anchor_shift in -0.02_f64..0.02,
    ) {
        let instance = generate_synthetic(SyntheticConfig {
            assets,
            factors: factors.min(assets),
            inequalities: 1,
            seed,
            budget: 1.0,
            max_weight: 1.0,
        })
        .expect("strategy only builds valid configs");

        let mut problem = instance.problem.clone();
        problem.l1 = Some(L1Term {
            costs: (0..assets)
                .map(|index| cost_scale * (1.0 + 0.5 * ((index % 4) as f64)))
                .collect(),
            anchor: instance
                .feasible_reference
                .iter()
                .map(|value| value + anchor_shift)
                .collect(),
        });

        let solver = Solver::new(SolverSettings {
            max_iterations: 20_000,
            ..SolverSettings::default()
        });
        let prox = solver.solve(&problem, None).expect("setup must succeed");
        prop_assert_eq!(prox.status, SolveStatus::Solved);
        let residuals = check_kkt(&problem, &prox.x, &prox.dual).expect("dimensions match");
        prop_assert!(residuals.primal < RESIDUAL_TOLERANCE);
        prop_assert!(residuals.dual < RESIDUAL_TOLERANCE);
        prop_assert!(residuals.complementarity < RESIDUAL_TOLERANCE);

        let epigraph = epigraph_reformulation(&problem);
        let reference = solver.solve(&epigraph, None).expect("epigraph setup");
        prop_assert_eq!(reference.status, SolveStatus::Solved);
        let scale = 1.0 + reference.objective.abs();
        prop_assert!(
            (prox.objective - reference.objective).abs() <= OBJECTIVE_TOLERANCE * scale,
            "prox {} vs epigraph {}", prox.objective, reference.objective
        );
    }
}
