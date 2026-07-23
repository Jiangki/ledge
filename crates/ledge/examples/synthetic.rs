//! Reproducible synthetic factor-QP benchmark.

use std::{env, error::Error};

use ledge::{
    generate_synthetic, BenchmarkRunner, Solver, SolverSettings, SyntheticConfig, WarmStart,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = SyntheticConfig::default();
    let mut settings = SolverSettings::default();
    let arguments: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--n" => config.assets = value.parse()?,
            "--k" => config.factors = value.parse()?,
            "--seed" => config.seed = value.parse()?,
            "--inequalities" => config.inequalities = value.parse()?,
            "--alpha" => settings.over_relaxation = value.parse()?,
            "--polish" => settings.polish = value.parse()?,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
        index += 2;
    }
    config.max_weight = (10.0 / config.assets as f64).min(1.0);

    let instance = generate_synthetic(config)?;
    let warm_start = WarmStart::from_primal(instance.feasible_reference.clone());
    let mut runner = BenchmarkRunner::new();
    runner.add_solver(Box::new(Solver::new(settings)));

    println!("{}", BenchmarkRunner::markdown_header());
    for record in runner.run(&instance.name, &instance.problem, Some(&warm_start)) {
        match record {
            Ok(record) => println!("{}", record.markdown_row()),
            Err(error) => eprintln!("{error}"),
        }
    }
    Ok(())
}
