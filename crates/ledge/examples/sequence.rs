//! Rolling multi-date rebalance through the `PortfolioSequence` API.
//!
//! One fixed factor structure, ten dates of new expected returns anchored on
//! the previous date's weights. The sequence reuses the equilibration and the
//! reduced factorizations across dates and chains warm starts internally.

use std::error::Error;

use ledge::{FactorCovariance, Matrix, PortfolioProblem, RebalanceStep, SolveStatus};

const ASSETS: usize = 40;
const FACTORS: usize = 4;
const DATES: usize = 10;

/// Deterministic pseudo-returns so the example needs no RNG dependency.
fn expected_returns(date: usize) -> Vec<f64> {
    (0..ASSETS)
        .map(|asset| 0.05 + 0.02 * ((asset + 3 * date) as f64 * 0.61).sin())
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let exposures: Vec<f64> = (0..ASSETS * FACTORS)
        .map(|index| 0.3 * (index as f64 * 12.9898).sin())
        .collect();
    let problem = PortfolioProblem::new(
        Matrix::new(ASSETS, FACTORS, exposures)?,
        FactorCovariance::Diagonal(vec![0.06; FACTORS]),
        vec![0.1; ASSETS],
        expected_returns(0),
    )?
    .with_risk_aversion(6.0)?
    .with_bounds(vec![0.0; ASSETS], vec![0.2; ASSETS])?
    .with_quadratic_turnover(vec![1.0 / ASSETS as f64; ASSETS], 0.5)?;

    let mut sequence = problem.sequence()?;
    let mut previous_weights: Option<Vec<f64>> = None;

    println!("date | status | iterations | new factorizations | one-way turnover");
    println!("---|---|---|---|---");
    for date in 0..DATES {
        let factorizations_before = sequence.factorizations();
        let step = RebalanceStep {
            expected_returns: (date > 0).then(|| expected_returns(date)),
            previous_weights: previous_weights.clone(),
            ..RebalanceStep::default()
        };
        let solution = sequence.solve_next(&step)?;
        if solution.status != SolveStatus::Solved {
            return Err(format!(
                "date {date} stopped with status '{}' after {} iterations",
                solution.status, solution.iterations
            )
            .into());
        }

        let turnover = previous_weights.as_ref().map_or(0.0, |previous| {
            solution
                .x
                .iter()
                .zip(previous)
                .map(|(new, old)| (new - old).abs())
                .sum::<f64>()
                / 2.0
        });
        println!(
            "{date} | {} | {} | {} | {turnover:.6}",
            solution.status,
            solution.iterations,
            sequence.factorizations() - factorizations_before,
        );
        previous_weights = Some(solution.x);
    }
    println!(
        "total reduced factorizations across {DATES} dates: {}",
        sequence.factorizations()
    );
    Ok(())
}
