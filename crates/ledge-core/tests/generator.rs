//! Synthetic instance generator tests.

use ledge_core::{generate_synthetic, SyntheticConfig};

#[test]
fn generator_is_reproducible_for_a_fixed_seed() {
    let config = SyntheticConfig {
        assets: 120,
        factors: 8,
        inequalities: 3,
        seed: 7,
        budget: 1.0,
        max_weight: 0.1,
    };

    let first = generate_synthetic(config.clone()).unwrap();
    let second = generate_synthetic(config).unwrap();

    assert_eq!(first, second);
}

#[test]
fn generated_reference_satisfies_all_constraints() {
    let instance = generate_synthetic(SyntheticConfig {
        assets: 100,
        factors: 5,
        inequalities: 6,
        seed: 1234,
        budget: 1.0,
        max_weight: 0.05,
    })
    .unwrap();
    let x = &instance.feasible_reference;
    let equality_value: f64 = instance
        .problem
        .equalities
        .matrix
        .row(0)
        .iter()
        .sum::<f64>()
        / instance.config.assets as f64;

    assert!((equality_value - instance.config.budget).abs() < 1.0e-12);
    for row in 0..instance.problem.inequalities.len() {
        let value: f64 = instance
            .problem
            .inequalities
            .matrix
            .row(row)
            .iter()
            .zip(x)
            .map(|(coefficient, weight)| coefficient * weight)
            .sum();
        assert!(value <= instance.problem.inequalities.rhs[row]);
    }
    assert!(x
        .iter()
        .all(|weight| { *weight >= 0.0 && *weight <= instance.config.max_weight }));
}
