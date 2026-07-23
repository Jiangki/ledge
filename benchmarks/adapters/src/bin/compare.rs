//! Protocol-compliant OSQP / Clarabel / Ledge comparison over the smoke
//! matrix. Writes all raw samples as CSV plus an aggregated Markdown summary.
//!
//! ```bash
//! cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
//!   --bin compare -- --out benchmarks/results/2026-07 --repeats 10
//! ```

use std::{
    env, error::Error, f64::consts::TAU, fmt::Write as _, fs, path::PathBuf, process::Command,
};

use ledge_bench_adapters::{
    protocol::Phase, run_workload, LedgeAdapter, PhasedSolver, RollingWorkload, Sample,
};
use ledge_core::{generate_synthetic, L1Term, QpProblem, Solver, SyntheticConfig};

#[cfg(any(feature = "osqp", feature = "clarabel"))]
use ledge_bench_adapters::Formulation;

struct Options {
    out: PathBuf,
    repeats: usize,
    rolling_steps: usize,
    max_assets: usize,
    dense_max_assets: usize,
    l1_bps: f64,
    commit: String,
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
    let mut options = Options {
        out: PathBuf::from("benchmarks/results/unnamed"),
        repeats: 10,
        rolling_steps: 10,
        max_assets: 5000,
        dense_max_assets: 1000,
        l1_bps: 10.0,
        commit: String::from("unknown"),
    };
    let arguments: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--out" => options.out = PathBuf::from(value),
            "--repeats" => options.repeats = value.parse()?,
            "--rolling-steps" => options.rolling_steps = value.parse()?,
            "--max-n" => options.max_assets = value.parse()?,
            "--dense-max-n" => options.dense_max_assets = value.parse()?,
            "--l1-bps" => options.l1_bps = value.parse()?,
            "--commit" => options.commit.clone_from(value),
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
        index += 2;
    }
    Ok(options)
}

/// Smoke-matrix instances; must stay in sync with `docs/SMOKE_TIMINGS.md`.
const INSTANCES: [(usize, usize, u64); 5] = [
    (100, 5, 1),
    (500, 10, 42),
    (1000, 20, 7),
    (2000, 50, 3),
    (5000, 100, 11),
];

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_options()?;
    fs::create_dir_all(&options.out)?;

    let mut samples: Vec<Sample> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut solver_labels: Vec<(String, &'static str)> = Vec::new();

    for (assets, factors, seed) in INSTANCES {
        if assets > options.max_assets {
            continue;
        }
        let config = SyntheticConfig {
            assets,
            factors,
            seed,
            max_weight: (10.0 / assets as f64).min(1.0),
            ..SyntheticConfig::default()
        };
        let instance = generate_synthetic(config)?;
        let linear_steps =
            perturbed_linear_steps(&instance.problem.linear, options.rolling_steps, seed);

        // Each smoke instance runs twice: the smooth base problem, then a
        // variant with proportional (L1) turnover costs anchored at the shared
        // primal start (the previous holdings). Ledge keeps its prox block;
        // external solvers receive the standard epigraph reformulation
        // (`n` extra variables, `2n` inequality rows) from `convert.rs`.
        let mut variants: Vec<(String, QpProblem)> =
            vec![(instance.name.clone(), instance.problem.clone())];
        if options.l1_bps > 0.0 {
            let mut with_l1 = instance.problem.clone();
            with_l1.l1 = Some(L1Term {
                costs: vec![options.l1_bps * 1.0e-4; assets],
                anchor: instance.feasible_reference.clone(),
            });
            variants.push((format!("{}-l1", instance.name), with_l1));
        }

        for (name, problem) in &variants {
            let workload = RollingWorkload {
                instance: name.clone(),
                problem,
                primal_start: &instance.feasible_reference,
                linear_steps: &linear_steps,
            };

            let mut adapters: Vec<Box<dyn PhasedSolver>> =
                vec![Box::new(LedgeAdapter::new(Solver::default()))];
            #[cfg(feature = "osqp")]
            {
                adapters.push(Box::new(ledge_bench_adapters::OsqpAdapter::new(
                    Formulation::Lifted,
                )));
                if assets <= options.dense_max_assets {
                    adapters.push(Box::new(ledge_bench_adapters::OsqpAdapter::new(
                        Formulation::DenseQ,
                    )));
                }
            }
            #[cfg(feature = "clarabel")]
            {
                adapters.push(Box::new(ledge_bench_adapters::ClarabelAdapter::new(
                    Formulation::Lifted,
                )));
                if assets <= options.dense_max_assets {
                    adapters.push(Box::new(ledge_bench_adapters::ClarabelAdapter::new(
                        Formulation::DenseQ,
                    )));
                }
            }

            for adapter in &mut adapters {
                let label = adapter.name();
                if !solver_labels.iter().any(|(name, _)| *name == label) {
                    solver_labels.push((label.clone(), adapter.warm_start_support().label()));
                }
                eprintln!("running {label} on {name} ...");
                match run_workload(&workload, adapter.as_mut(), options.repeats) {
                    Ok(mut rows) => samples.append(&mut rows),
                    Err(error) => {
                        eprintln!("  FAILED: {error}");
                        failures.push(format!("{name}: {error}"));
                    }
                }
            }
        }
    }

    let csv_path = options.out.join("samples.csv");
    let mut csv = String::from(Sample::csv_header());
    csv.push('\n');
    for sample in &samples {
        csv.push_str(&sample.csv_row());
        csv.push('\n');
    }
    fs::write(&csv_path, csv)?;
    eprintln!("wrote {}", csv_path.display());

    let summary = render_summary(&options, &samples, &failures, &solver_labels)?;
    let summary_path = options.out.join("summary.md");
    fs::write(&summary_path, summary)?;
    eprintln!("wrote {}", summary_path.display());
    Ok(())
}

/// Deterministic rolling expected-return updates shared by every adapter:
/// step `t` perturbs each entry of the base linear cost by white noise with
/// standard deviation `0.1 * mean(|linear|)`, seeded by instance seed and
/// step only, so all solvers and repeats see identical data.
fn perturbed_linear_steps(base: &[f64], steps: usize, seed: u64) -> Vec<Vec<f64>> {
    let mean_magnitude = if base.is_empty() {
        0.0
    } else {
        base.iter().map(|value| value.abs()).sum::<f64>() / base.len() as f64
    };
    let scale = 0.1 * mean_magnitude;
    (0..steps)
        .map(|step| {
            let mut noise = SplitMix64::new(seed ^ ((step as u64 + 1) * 0x9e37_79b9));
            base.iter()
                .map(|value| value + scale * noise.standard_normal())
                .collect()
        })
        .collect()
}

struct Aggregate {
    instance: String,
    solver: String,
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let position = fraction * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let weight = position - low as f64;
    sorted[low] * (1.0 - weight) + sorted[high] * weight
}

fn phase_rows<'a>(samples: &'a [Sample], key: &Aggregate, phase: Phase) -> Vec<&'a Sample> {
    samples
        .iter()
        .filter(|sample| {
            sample.instance == key.instance && sample.solver == key.solver && sample.phase == phase
        })
        .collect()
}

fn sorted_times(rows: &[&Sample]) -> Vec<f64> {
    let mut times: Vec<f64> = rows
        .iter()
        .map(|sample| sample.duration.as_secs_f64() * 1_000.0)
        .collect();
    times.sort_by(f64::total_cmp);
    times
}

fn render_summary(
    options: &Options,
    samples: &[Sample],
    failures: &[String],
    solver_labels: &[(String, &'static str)],
) -> Result<String, Box<dyn Error>> {
    let rustc = Command::new("rustc").arg("--version").output().map_or_else(
        |_| String::from("unknown"),
        |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
    );
    let cpu = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|content| {
            content.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .map(|rest| rest.trim_start_matches([' ', '\t', ':']).to_owned())
            })
        })
        .unwrap_or_else(|| String::from("unknown"));

    let mut keys: Vec<Aggregate> = Vec::new();
    for sample in samples {
        if !keys
            .iter()
            .any(|key| key.instance == sample.instance && key.solver == sample.solver)
        {
            keys.push(Aggregate {
                instance: sample.instance.clone(),
                solver: sample.solver.clone(),
            });
        }
    }

    let mut report = String::new();
    writeln!(
        report,
        "# Cross-solver comparison summary (auto-generated)\n"
    )?;
    writeln!(
        report,
        "Generated by `compare`; raw samples in [`samples.csv`](samples.csv)."
    )?;
    writeln!(report, "\n- CPU: {cpu}")?;
    writeln!(report, "- Compiler: {rustc} (release, thin LTO)")?;
    writeln!(report, "- Commit: {}", options.commit)?;
    writeln!(
        report,
        "- Repeats: {}; rolling steps per repeat: {}; dense-Q cutoff: n <= {}",
        options.repeats, options.rolling_steps, options.dense_max_assets
    )?;
    if options.l1_bps > 0.0 {
        writeln!(
            report,
            "- `-l1` instances: proportional turnover costs of {} bps per asset, \
             anchored at the shared primal start; Ledge uses its prox block, \
             external solvers the epigraph reformulation (n extra variables, \
             2n inequality rows)",
            options.l1_bps
        )?;
    }
    writeln!(report, "\n## Warm-start support (protocol rule 3)\n")?;
    writeln!(report, "| solver | rolling-phase warm start |")?;
    writeln!(report, "|---|---|")?;
    for (label, support) in solver_labels {
        writeln!(report, "| {label} | {support} |")?;
    }

    writeln!(report, "\n## Setup (conversion + solver construction)\n")?;
    writeln!(report, "| instance | solver | median ms | p90 ms |")?;
    writeln!(report, "|---|---|---:|---:|")?;
    for key in &keys {
        let rows = phase_rows(samples, key, Phase::Setup);
        let times = sorted_times(&rows);
        writeln!(
            report,
            "| {} | {} | {:.3} | {:.3} |",
            key.instance,
            key.solver,
            quantile(&times, 0.5),
            quantile(&times, 0.9)
        )?;
    }

    writeln!(report, "\n## Cold solve (shared primal start)\n")?;
    writeln!(
        report,
        "| instance | solver | status | median ms | p10 ms | p90 ms | iters | objective | max KKT primal | max KKT dual |"
    )?;
    writeln!(report, "|---|---|---|---:|---:|---:|---:|---:|---:|---:|")?;
    for key in &keys {
        let rows = phase_rows(samples, key, Phase::Cold);
        if rows.is_empty() {
            continue;
        }
        let times = sorted_times(&rows);
        let worst_primal = rows.iter().map(|row| row.kkt_primal).fold(0.0, f64::max);
        let worst_dual = rows.iter().map(|row| row.kkt_dual).fold(0.0, f64::max);
        writeln!(
            report,
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {} | {:.6e} | {:.1e} | {:.1e} |",
            key.instance,
            key.solver,
            rows[0].native_status,
            quantile(&times, 0.5),
            quantile(&times, 0.1),
            quantile(&times, 0.9),
            rows[0].iterations,
            rows[0].objective,
            worst_primal,
            worst_dual
        )?;
    }

    writeln!(
        report,
        "\n## Rolling re-solves (perturbed expected returns)\n"
    )?;
    writeln!(
        report,
        "Per-step statistics across all repeats and steps; warm starts as declared above.\n"
    )?;
    writeln!(
        report,
        "| instance | solver | solved steps | median ms/step | p10 ms | p90 ms | median iters | max KKT primal | max KKT dual |"
    )?;
    writeln!(report, "|---|---|---|---:|---:|---:|---:|---:|---:|")?;
    for key in &keys {
        let rows = phase_rows(samples, key, Phase::Roll);
        if rows.is_empty() {
            continue;
        }
        let times = sorted_times(&rows);
        let solved = rows.iter().filter(|row| row.solved).count();
        let mut iterations: Vec<f64> = rows.iter().map(|row| row.iterations as f64).collect();
        iterations.sort_by(f64::total_cmp);
        let worst_primal = rows.iter().map(|row| row.kkt_primal).fold(0.0, f64::max);
        let worst_dual = rows.iter().map(|row| row.kkt_dual).fold(0.0, f64::max);
        writeln!(
            report,
            "| {} | {} | {}/{} | {:.3} | {:.3} | {:.3} | {:.0} | {:.1e} | {:.1e} |",
            key.instance,
            key.solver,
            solved,
            rows.len(),
            quantile(&times, 0.5),
            quantile(&times, 0.1),
            quantile(&times, 0.9),
            quantile(&iterations, 0.5),
            worst_primal,
            worst_dual
        )?;
    }

    if !failures.is_empty() {
        writeln!(report, "\n## Failures\n")?;
        for failure in failures {
            writeln!(report, "- {failure}")?;
        }
    }
    Ok(report)
}

/// Deterministic PRNG matching the generator's distribution quality needs;
/// duplicated locally so benchmark data never depends on solver internals.
struct SplitMix64 {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = TAU * self.uniform();
        self.spare_normal = Some(radius * angle.sin());
        radius * angle.cos()
    }
}
