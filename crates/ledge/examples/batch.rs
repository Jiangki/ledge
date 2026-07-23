//! Multi-account batch rebalance throughput driver (roadmap 3.2).
//!
//! One factor model, many accounts, many trading dates — the persona-C
//! workload from `docs/PLAN.md`. Every account shares the risk model but
//! carries its own expected-return tilt and its own turnover anchor (10 bps
//! proportional costs plus an L2 penalty), and rolls through the dates with
//! backtest anchor chaining: each date trades from the weights the previous
//! `Solved` date left behind.
//!
//! Build with `--features rayon` to distribute accounts over a thread pool
//! (`RAYON_NUM_THREADS` controls the width); without the feature the same
//! code runs serially. Defaults reproduce the published
//! "1 model x 500 accounts x 250 dates" number:
//!
//! ```text
//! cargo run -p ledge --release --features rayon --example batch
//! cargo run -p ledge --release --example batch -- --accounts 20 --dates 10
//! ```

use std::{env, error::Error, fs, time::Instant};

use ledge::{
    solve_batch, BatchAccount, FactorCovariance, Matrix, PortfolioProblem, RebalanceStep,
    SolveStatus,
};

struct Config {
    accounts: usize,
    dates: usize,
    assets: usize,
    factors: usize,
    /// Optional per-account CSV path for published artifacts.
    out: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            accounts: 500,
            dates: 250,
            assets: 200,
            factors: 15,
            out: None,
        }
    }
}

/// Deterministic pseudo-returns: a per-account tilt plus a per-date drift,
/// so every account solves a genuinely different sequence.
fn expected_returns(config: &Config, account: usize, date: usize) -> Vec<f64> {
    (0..config.assets)
        .map(|asset| {
            0.05 + 0.03 * (asset as f64 * 0.7 + account as f64 * 0.31).cos()
                + 0.01 * ((asset + 5 * date) as f64 * 0.17).sin()
        })
        .collect()
}

fn build_accounts(config: &Config) -> Result<Vec<BatchAccount>, Box<dyn Error>> {
    // One shared model: exposures, factor covariance, and specific variance
    // are identical across accounts.
    let exposures: Vec<f64> = (0..config.assets * config.factors)
        .map(|index| 0.3 * ((index + 1) as f64 * 12.9898).sin())
        .collect();
    let factors = Matrix::new(config.assets, config.factors, exposures)?;
    let omega = FactorCovariance::Diagonal(
        (0..config.factors)
            .map(|index| 0.05 + 0.01 * index as f64)
            .collect(),
    );
    let specific: Vec<f64> = (0..config.assets)
        .map(|index| 0.08 + 0.04 * ((index * 7 % 13) as f64) / 13.0)
        .collect();
    let max_weight = (10.0 / config.assets as f64).min(1.0);
    let uniform = vec![1.0 / config.assets as f64; config.assets];

    (0..config.accounts)
        .map(|account| {
            let problem = PortfolioProblem::new(
                factors.clone(),
                omega.clone(),
                specific.clone(),
                expected_returns(config, account, 0),
            )?
            .with_risk_aversion(6.0)?
            .with_bounds(vec![0.0; config.assets], vec![max_weight; config.assets])?
            .with_quadratic_turnover(uniform.clone(), 0.5)?
            .with_l1_turnover(uniform.clone(), vec![0.001; config.assets])?;
            let steps = (0..config.dates)
                .map(|date| RebalanceStep {
                    expected_returns: (date > 0).then(|| expected_returns(config, account, date)),
                    ..RebalanceStep::default()
                })
                .collect();
            Ok(BatchAccount {
                problem,
                steps,
                chain_previous_weights: true,
            })
        })
        .collect()
}

fn thread_count() -> usize {
    if cfg!(feature = "rayon") {
        // Mirrors rayon's default: RAYON_NUM_THREADS, else available CPUs.
        env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&threads| threads > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
            })
    } else {
        1
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = Config::default();
    let arguments: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--accounts" => config.accounts = value.parse()?,
            "--dates" => config.dates = value.parse()?,
            "--n" => config.assets = value.parse()?,
            "--k" => config.factors = value.parse()?,
            "--out" => config.out = Some(value.clone()),
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
        index += 2;
    }

    let build_started = Instant::now();
    let accounts = build_accounts(&config)?;
    let build_seconds = build_started.elapsed().as_secs_f64();

    let solve_started = Instant::now();
    let results = solve_batch(&accounts, None);
    let wall_seconds = solve_started.elapsed().as_secs_f64();

    let mut solved = 0_usize;
    let mut unconverged = 0_usize;
    let mut total_iterations = 0_usize;
    let mut solver_seconds = 0.0_f64;
    let mut rows = Vec::with_capacity(config.accounts);
    for (account, result) in results.iter().enumerate() {
        let solutions = result
            .as_ref()
            .map_err(|error| format!("account {account}: {error}"))?;
        let mut account_iterations = 0_usize;
        let mut account_solved = 0_usize;
        let mut account_seconds = 0.0_f64;
        for solution in solutions {
            match solution.status {
                SolveStatus::Solved => account_solved += 1,
                _ => unconverged += 1,
            }
            account_iterations += solution.iterations;
            account_seconds += solution.solve_time.as_secs_f64();
        }
        solved += account_solved;
        total_iterations += account_iterations;
        solver_seconds += account_seconds;
        rows.push(format!(
            "{account},{},{account_solved},{account_iterations},{:.3}",
            solutions.len(),
            1.0e3 * account_seconds
        ));
    }

    let total_solves = config.accounts * config.dates;
    println!(
        "batch: {} accounts x {} dates, n={} assets, k={} factors, threads={} ({})",
        config.accounts,
        config.dates,
        config.assets,
        config.factors,
        thread_count(),
        if cfg!(feature = "rayon") {
            "rayon"
        } else {
            "serial"
        },
    );
    println!("account setup: {build_seconds:.2} s (problems + steps, single-threaded)");
    println!(
        "solve wall time: {wall_seconds:.2} s for {total_solves} account-dates \
         => {:.0} solves/s",
        total_solves as f64 / wall_seconds
    );
    println!(
        "statuses: {solved} solved, {unconverged} other; iterations: {total_iterations} total, \
         {:.1} mean/solve",
        total_iterations as f64 / total_solves as f64
    );
    println!(
        "solver time (iteration-only, summed across threads): {solver_seconds:.2} s, \
         {:.3} ms mean/solve",
        1.0e3 * solver_seconds / total_solves as f64
    );

    if let Some(path) = &config.out {
        let mut csv = String::from("account,dates,solved,iterations,solve_time_ms\n");
        for row in &rows {
            csv.push_str(row);
            csv.push('\n');
        }
        fs::write(path, csv)?;
        println!("per-account samples written to {path}");
    }

    if unconverged > 0 {
        return Err(format!("{unconverged} account-dates did not reach Solved").into());
    }
    Ok(())
}
