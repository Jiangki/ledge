//! Batch-over-accounts integration tests (roadmap 3.2).
//!
//! `solve_batch` must be exactly the per-account `PortfolioSequence` loop:
//! bit-identical solutions in input order, per-account error isolation, and
//! backtest-style anchor chaining. The same file runs with and without the
//! `rayon` feature — CI exercises both — because the parallel and serial
//! paths must return identical answers.

use ledge_core::{
    solve_batch, BatchAccount, FactorCovariance, Matrix, PortfolioError, PortfolioProblem,
    RebalanceStep, SolveStatus,
};

const ASSETS: usize = 40;
const FACTORS: usize = 4;

/// Deterministic shared factor model; accounts differ only in data.
fn base_problem(account: usize) -> PortfolioProblem {
    let exposures: Vec<f64> = (0..ASSETS * FACTORS)
        .map(|index| 0.3 * ((index + 1) as f64 * 12.9898).sin())
        .collect();
    PortfolioProblem::new(
        Matrix::new(ASSETS, FACTORS, exposures).unwrap(),
        FactorCovariance::Diagonal(vec![0.06; FACTORS]),
        (0..ASSETS)
            .map(|index| 0.08 + 0.04 * ((index * 7 % 13) as f64) / 13.0)
            .collect(),
        expected_returns(account, 0),
    )
    .unwrap()
    .with_risk_aversion(6.0)
    .unwrap()
    .with_bounds(vec![0.0; ASSETS], vec![0.2; ASSETS])
    .unwrap()
}

fn turnover_problem(account: usize) -> PortfolioProblem {
    base_problem(account)
        .with_quadratic_turnover(vec![1.0 / ASSETS as f64; ASSETS], 0.5)
        .unwrap()
        .with_l1_turnover(vec![1.0 / ASSETS as f64; ASSETS], vec![0.001; ASSETS])
        .unwrap()
}

fn expected_returns(account: usize, date: usize) -> Vec<f64> {
    (0..ASSETS)
        .map(|asset| 0.05 + 0.02 * ((asset + 3 * date + 17 * account) as f64 * 0.61).sin())
        .collect()
}

fn return_steps(account: usize, dates: usize) -> Vec<RebalanceStep> {
    (0..dates)
        .map(|date| RebalanceStep {
            expected_returns: (date > 0).then(|| expected_returns(account, date)),
            ..RebalanceStep::default()
        })
        .collect()
}

#[test]
#[allow(clippy::float_cmp)] // bit-identical to the per-account loop is the property under test
fn batch_matches_per_account_sequences_bitwise() {
    let accounts: Vec<BatchAccount> = (0..3)
        .map(|account| BatchAccount {
            problem: base_problem(account),
            steps: return_steps(account, 4),
            chain_previous_weights: false,
        })
        .collect();

    let results = solve_batch(&accounts, None);
    assert_eq!(results.len(), accounts.len());

    for (account, result) in accounts.iter().zip(&results) {
        let batch_solutions = result.as_ref().unwrap();
        let mut sequence = account.problem.sequence().unwrap();
        for (step, batch_solution) in account.steps.iter().zip(batch_solutions) {
            let reference = sequence.solve_next(step).unwrap();
            assert_eq!(batch_solution.status, SolveStatus::Solved);
            assert_eq!(batch_solution.status, reference.status);
            assert_eq!(batch_solution.iterations, reference.iterations);
            assert_eq!(batch_solution.x, reference.x);
            assert_eq!(batch_solution.objective, reference.objective);
        }
    }
}

#[test]
fn chained_anchors_follow_solved_weights() {
    let account = BatchAccount {
        problem: turnover_problem(0),
        steps: return_steps(0, 5),
        chain_previous_weights: true,
    };
    let results = solve_batch(std::slice::from_ref(&account), None);
    let batch_solutions = results[0].as_ref().unwrap();

    // Reference: the manual backtest loop, anchoring each date at the
    // previous date's solved weights.
    let mut sequence = account.problem.sequence().unwrap();
    let mut held: Option<Vec<f64>> = None;
    for (step, batch_solution) in account.steps.iter().zip(batch_solutions) {
        let manual_step = RebalanceStep {
            previous_weights: held.clone(),
            ..step.clone()
        };
        let reference = sequence.solve_next(&manual_step).unwrap();
        assert_eq!(batch_solution.status, SolveStatus::Solved);
        assert_eq!(batch_solution.x, reference.x);
        assert_eq!(batch_solution.iterations, reference.iterations);
        held = Some(reference.x.clone());
    }

    // Chaining must actually move the anchor: with 5 dates of moving
    // returns, the last date's weights differ from the first date's.
    assert_ne!(batch_solutions[0].x, batch_solutions[4].x);
}

#[test]
fn explicit_previous_weights_win_over_chaining() {
    let explicit_anchor = vec![0.5 / ASSETS as f64; ASSETS];
    let mut steps = return_steps(0, 3);
    steps[2].previous_weights = Some(explicit_anchor.clone());

    let account = BatchAccount {
        problem: turnover_problem(0),
        steps,
        chain_previous_weights: true,
    };
    let results = solve_batch(std::slice::from_ref(&account), None);
    let batch_solutions = results[0].as_ref().unwrap();

    let mut sequence = account.problem.sequence().unwrap();
    let first = sequence.solve_next(&account.steps[0]).unwrap();
    let second = sequence
        .solve_next(&RebalanceStep {
            previous_weights: Some(first.x.clone()),
            ..account.steps[1].clone()
        })
        .unwrap();
    assert_eq!(batch_solutions[1].x, second.x);
    // Date 2 uses the explicit anchor, not date 1's weights.
    let third = sequence.solve_next(&account.steps[2]).unwrap();
    assert_eq!(batch_solutions[2].x, third.x);
}

#[test]
fn account_failures_are_isolated() {
    let bad_steps = vec![RebalanceStep {
        expected_returns: Some(vec![0.05; ASSETS + 1]),
        ..RebalanceStep::default()
    }];
    let accounts = vec![
        BatchAccount {
            problem: base_problem(0),
            steps: return_steps(0, 2),
            chain_previous_weights: false,
        },
        BatchAccount {
            problem: base_problem(1),
            steps: bad_steps,
            chain_previous_weights: false,
        },
        BatchAccount {
            problem: base_problem(2),
            steps: return_steps(2, 2),
            chain_previous_weights: false,
        },
    ];

    let results = solve_batch(&accounts, None);
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok());
    assert!(matches!(results[1], Err(PortfolioError::Problem(_))));
    assert!(results[2].is_ok());
    for result in [&results[0], &results[2]] {
        for solution in result.as_ref().unwrap() {
            assert_eq!(solution.status, SolveStatus::Solved);
        }
    }
}

#[test]
fn chaining_requires_a_turnover_term() {
    let account = BatchAccount {
        problem: base_problem(0),
        steps: return_steps(0, 2),
        chain_previous_weights: true,
    };
    let results = solve_batch(std::slice::from_ref(&account), None);
    assert!(matches!(
        results[0],
        Err(PortfolioError::InvalidParameter(_))
    ));
}

#[test]
fn empty_batches_and_empty_accounts_are_fine() {
    assert!(solve_batch(&[], None).is_empty());

    let account = BatchAccount {
        problem: base_problem(0),
        steps: Vec::new(),
        chain_previous_weights: false,
    };
    let results = solve_batch(std::slice::from_ref(&account), None);
    assert!(results[0].as_ref().unwrap().is_empty());
}
