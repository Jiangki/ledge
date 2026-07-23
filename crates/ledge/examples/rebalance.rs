//! Two-date mean-variance rebalance through the high-level Rust API.

use std::error::Error;

use ledge::{FactorCovariance, Matrix, PortfolioProblem, Solution, SolveStatus};

fn portfolio(expected_returns: Vec<f64>) -> Result<PortfolioProblem, Box<dyn Error>> {
    let factors = Matrix::new(
        6,
        2,
        vec![
            0.8, 0.1, 0.7, -0.2, 0.2, 0.9, -0.1, 0.8, -0.5, 0.3, -0.6, -0.4,
        ],
    )?;
    Ok(PortfolioProblem::new(
        factors,
        FactorCovariance::Diagonal(vec![0.08, 0.05]),
        vec![0.12, 0.10, 0.09, 0.11, 0.08, 0.10],
        expected_returns,
    )?
    .with_risk_aversion(6.0)?
    .with_bounds(vec![0.0; 6], vec![0.35; 6])?)
}

fn require_solved(solution: &Solution) -> Result<(), Box<dyn Error>> {
    if solution.status == SolveStatus::Solved {
        Ok(())
    } else {
        Err(format!(
            "solve stopped with status '{}' after {} iterations (primal {:.3e}, dual {:.3e})",
            solution.status,
            solution.iterations,
            solution.residuals.primal,
            solution.residuals.dual
        )
        .into())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let first = portfolio(vec![0.09, 0.08, 0.07, 0.06, 0.05, 0.04])?.solve(None)?;
    require_solved(&first)?;

    let warm_start = first.warm_start();
    let second = portfolio(vec![0.07, 0.08, 0.10, 0.05, 0.06, 0.04])?
        .with_quadratic_turnover(first.x.clone(), 0.4)?
        .solve(Some(&warm_start))?;
    require_solved(&second)?;

    let budget: f64 = second.x.iter().sum();
    let one_way_turnover: f64 = first
        .x
        .iter()
        .zip(&second.x)
        .map(|(old, new)| (new - old).abs())
        .sum::<f64>()
        / 2.0;
    println!("first weights:  {:.6?}", first.x);
    println!("second weights: {:.6?}", second.x);
    println!("budget: {budget:.10}");
    println!("one-way turnover: {one_way_turnover:.6}");
    println!(
        "status: {}, iterations: {}, primal residual: {:.3e}, dual residual: {:.3e}",
        second.status, second.iterations, second.residuals.primal, second.residuals.dual
    );
    Ok(())
}
