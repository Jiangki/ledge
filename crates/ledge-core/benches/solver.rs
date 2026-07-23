//! Criterion microbenchmarks for the three costs that dominate a solve:
//! the SMW-reduced factorization, one x-update, and an end-to-end solve.
//!
//! Run everything with a single command:
//!
//! ```bash
//! cargo bench -p ledge-core --features bench-internals
//! ```
//!
//! Numbers from cloud CI are noisy; treat committed baselines as trend
//! indicators and publish only fixed-machine measurements (see
//! `benchmarks/README.md`).

// The criterion_group macro expands to an undocumented public function.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ledge_core::{
    bench_internals::{factorize, x_update},
    generate_synthetic, GeneratedInstance, Solver, SyntheticConfig,
};

const RHO: f64 = 1.0;
const SIGMA: f64 = 1.0e-6;

fn instance(assets: usize, factors: usize) -> GeneratedInstance {
    generate_synthetic(SyntheticConfig {
        assets,
        factors,
        inequalities: 4,
        seed: 42,
        budget: 1.0,
        max_weight: 1.0,
    })
    .expect("synthetic generation must succeed for benchmark dimensions")
}

fn bench_factorization(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("factorized_system_new");
    for (assets, factors) in [(100, 10), (500, 10), (1000, 20), (2000, 50)] {
        let generated = instance(assets, factors);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{assets}_k{factors}")),
            &generated,
            |bencher, generated| {
                bencher.iter(|| {
                    factorize(black_box(&generated.problem), RHO, SIGMA)
                        .expect("factorization must succeed")
                });
            },
        );
    }
    group.finish();
}

fn bench_x_update(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("x_update");
    for (assets, factors) in [(100, 10), (500, 10), (1000, 20), (2000, 50)] {
        let generated = instance(assets, factors);
        let system = factorize(&generated.problem, RHO, SIGMA).expect("factorization must succeed");
        let right_hand_side = vec![1.0; assets];
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{assets}_k{factors}")),
            &system,
            |bencher, system| {
                bencher.iter(|| {
                    let mut buffer = right_hand_side.clone();
                    x_update(black_box(system), &mut buffer);
                    buffer
                });
            },
        );
    }
    group.finish();
}

fn bench_end_to_end(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("solve_synthetic");
    group.sample_size(10);
    for (assets, factors) in [(100, 10), (500, 10)] {
        let generated = instance(assets, factors);
        let solver = Solver::default();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{assets}_k{factors}")),
            &generated,
            |bencher, generated| {
                bencher.iter(|| {
                    solver
                        .solve(black_box(&generated.problem), None)
                        .expect("solve must not error on generated instances")
                });
            },
        );
    }
    group.finish();
}

/// One warm-started rolling step (new linear cost, fixed structure), solved
/// through the one-shot path (rebuilds equilibration + factorization) and
/// through a `Workspace` (reuses both; roadmap 2.4).
fn bench_rolling_resolve(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("rolling_resolve");
    group.sample_size(10);
    for (assets, factors) in [(500, 10), (1000, 20), (2000, 50)] {
        let generated = instance(assets, factors);
        let solver = Solver::default();
        let base = solver
            .solve(&generated.problem, None)
            .expect("base solve must succeed");
        let warm = base.warm_start();
        let perturbed_linear: Vec<f64> = generated
            .problem
            .linear
            .iter()
            .enumerate()
            .map(|(index, value)| value + 1.0e-3 * (((index % 7) as f64) - 3.0))
            .collect();
        let mut perturbed = generated.problem.clone();
        perturbed.linear.clone_from(&perturbed_linear);

        group.bench_with_input(
            BenchmarkId::new("one_shot", format!("n{assets}_k{factors}")),
            &perturbed,
            |bencher, problem| {
                bencher.iter(|| {
                    solver
                        .solve(black_box(problem), Some(&warm))
                        .expect("rolling solve must succeed")
                });
            },
        );

        let mut workspace = solver
            .workspace(&generated.problem)
            .expect("workspace setup must succeed");
        // Prime the factorization cache: the first solve visits the
        // adaptive-rho ladder once; the measured step then matches the
        // steady state of a rolling sequence.
        workspace
            .solve(Some(&warm))
            .expect("priming solve must succeed");
        group.bench_function(
            BenchmarkId::new("workspace", format!("n{assets}_k{factors}")),
            |bencher| {
                bencher.iter(|| {
                    workspace
                        .update_linear(black_box(&perturbed_linear))
                        .expect("dimensions match");
                    workspace
                        .solve(Some(&warm))
                        .expect("rolling solve must succeed")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_factorization,
    bench_x_update,
    bench_end_to_end,
    bench_rolling_resolve
);
criterion_main!(benches);
