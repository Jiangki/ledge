//! Deterministic synthetic factor-model portfolio instances.

use std::f64::consts::TAU;

use thiserror::Error;

use crate::{
    matrix::{dot, Matrix, MatrixError},
    problem::{FactorCovariance, FactorQuad, LinearConstraints, ProblemError, QpProblem},
};

/// Configuration for a reproducible synthetic portfolio QP.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticConfig {
    /// Number of assets.
    pub assets: usize,
    /// Number of covariance factors.
    pub factors: usize,
    /// Number of additional upper-inequality rows.
    pub inequalities: usize,
    /// Deterministic random seed.
    pub seed: u64,
    /// Portfolio budget imposed as an equality.
    pub budget: f64,
    /// Per-asset upper bound. Set to infinity for no upper bound.
    pub max_weight: f64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            assets: 100,
            factors: 10,
            inequalities: 4,
            seed: 42,
            budget: 1.0,
            max_weight: 0.1,
        }
    }
}

/// Generated problem plus a known feasible reference point.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedInstance {
    /// Reproducible name containing dimensions and seed.
    pub name: String,
    /// Generated convex QP.
    pub problem: QpProblem,
    /// Uniform feasible portfolio used to place constraints.
    pub feasible_reference: Vec<f64>,
    /// Configuration used to produce this instance.
    pub config: SyntheticConfig,
}

/// Synthetic generator errors.
#[derive(Debug, Error)]
pub enum GeneratorError {
    /// Invalid generator setting.
    #[error("invalid synthetic configuration: {0}")]
    InvalidConfig(&'static str),
    /// Matrix construction failed.
    #[error(transparent)]
    Matrix(#[from] MatrixError),
    /// Generated problem construction failed.
    #[error(transparent)]
    Problem(#[from] ProblemError),
}

/// Generates a deterministic, feasible, long-only factor-model portfolio QP.
///
/// The instance has one budget equality, configurable random exposure caps,
/// diagonal factor covariance, and positive idiosyncratic risk.
///
/// # Errors
///
/// Returns [`GeneratorError`] when dimensions or bounds cannot contain the
/// uniform reference portfolio.
pub fn generate_synthetic(config: SyntheticConfig) -> Result<GeneratedInstance, GeneratorError> {
    if config.assets == 0 {
        return Err(GeneratorError::InvalidConfig("assets must be positive"));
    }
    if config.factors == 0 || config.factors > config.assets {
        return Err(GeneratorError::InvalidConfig(
            "factors must be in 1..=assets",
        ));
    }
    if !config.budget.is_finite() || config.budget <= 0.0 {
        return Err(GeneratorError::InvalidConfig(
            "budget must be finite and positive",
        ));
    }
    let reference_weight = config.budget / config.assets as f64;
    if config.max_weight.is_nan() || config.max_weight < reference_weight {
        return Err(GeneratorError::InvalidConfig(
            "max_weight must contain the uniform feasible portfolio",
        ));
    }

    let n = config.assets;
    let k = config.factors;
    let mut random = SplitMix64::new(config.seed);
    let factor_scale = 1.0 / (k as f64).sqrt();
    let factor_data: Vec<f64> = (0..n * k)
        .map(|_| 0.25 * factor_scale * random.standard_normal())
        .collect();
    let factors = Matrix::new(n, k, factor_data)?;
    let omega: Vec<f64> = (0..k).map(|_| 0.05 + 0.15 * random.uniform()).collect();
    let diagonal: Vec<f64> = (0..n).map(|_| 0.05 + 0.10 * random.uniform()).collect();
    let quadratic = FactorQuad::new(factors, FactorCovariance::Diagonal(omega), diagonal)?;

    // The linear term is -mu: minimization therefore seeks positive expected
    // returns while risk and constraints regularize the portfolio.
    let linear: Vec<f64> = (0..n)
        .map(|_| -0.01 - 0.005 * random.standard_normal())
        .collect();
    let equalities = LinearConstraints::new(Matrix::new(1, n, vec![1.0; n])?, vec![config.budget])?;
    let feasible_reference = vec![reference_weight; n];

    let mut inequality_matrix = Matrix::zeros(config.inequalities, n);
    let mut inequality_rhs = Vec::with_capacity(config.inequalities);
    for row in 0..config.inequalities {
        for col in 0..n {
            inequality_matrix[(row, col)] = random.standard_normal();
        }
        let center = dot(inequality_matrix.row(row), &feasible_reference);
        let row_norm = dot(inequality_matrix.row(row), inequality_matrix.row(row)).sqrt();
        inequality_rhs.push(center + 0.10 * row_norm * reference_weight.sqrt());
    }
    let inequalities = LinearConstraints::new(inequality_matrix, inequality_rhs)?;
    let lower_bounds = vec![0.0; n];
    let upper_bounds = vec![config.max_weight; n];
    let problem = QpProblem {
        quadratic,
        linear,
        l1: None,
        equalities,
        inequalities,
        lower_bounds,
        upper_bounds,
    };
    problem.validate()?;

    Ok(GeneratedInstance {
        name: format!("factor-n{n}-k{k}-s{}", config.seed),
        problem,
        feasible_reference,
        config,
    })
}

/// Tiny deterministic PRNG to avoid making benchmark reproducibility depend on
/// a third-party distribution implementation.
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
        // 53 random bits mapped to the open interval (0, 1).
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
