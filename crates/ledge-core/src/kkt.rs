//! Karush-Kuhn-Tucker residual checks.

use thiserror::Error;

use crate::{matrix::norm_inf, problem::QpProblem};

/// Dual multipliers associated with each constraint block.
///
/// Inequality multipliers use the convention `A_ineq * x <= b_ineq` and are
/// therefore non-negative. A box multiplier is negative at a lower bound and
/// positive at an upper bound.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DualVariables {
    /// Equality multipliers.
    pub equalities: Vec<f64>,
    /// Upper-inequality multipliers.
    pub inequalities: Vec<f64>,
    /// Combined normal-cone multipliers for variable boxes.
    pub bounds: Vec<f64>,
    /// Subgradient multipliers of the [`L1Term`](crate::L1Term); empty when
    /// the problem has none.
    ///
    /// Each entry lies in `[-costs[i], costs[i]]` and equals the signed cost
    /// wherever the variable moved off the anchor (`+costs[i]` above it,
    /// `-costs[i]` below).
    pub l1: Vec<f64>,
}

/// Infinity-norm KKT diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KktResiduals {
    /// Maximum equality, inequality, or bound violation.
    pub primal: f64,
    /// Maximum stationarity or dual-cone violation.
    pub dual: f64,
    /// Maximum absolute complementary-slackness product.
    pub complementarity: f64,
}

/// Errors from the standalone KKT checker.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum KktError {
    /// A primal or dual vector has the wrong length.
    #[error("{field} has length {actual}; expected {expected}")]
    Dimension {
        /// Name of the vector.
        field: &'static str,
        /// Expected length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
}

/// Evaluates primal feasibility, stationarity/dual feasibility, and
/// complementarity.
///
/// # Errors
///
/// Returns [`KktError::Dimension`] if a primal or multiplier vector has the
/// wrong length.
pub fn check_kkt(
    problem: &QpProblem,
    x: &[f64],
    dual: &DualVariables,
) -> Result<KktResiduals, KktError> {
    let n = problem.quadratic.dimension();
    let l1_expected = if problem.l1.is_some() { n } else { 0 };
    for (field, actual, expected) in [
        ("x", x.len(), n),
        (
            "dual.equalities",
            dual.equalities.len(),
            problem.equalities.len(),
        ),
        (
            "dual.inequalities",
            dual.inequalities.len(),
            problem.inequalities.len(),
        ),
        ("dual.bounds", dual.bounds.len(), n),
        ("dual.l1", dual.l1.len(), l1_expected),
    ] {
        if actual != expected {
            return Err(KktError::Dimension {
                field,
                expected,
                actual,
            });
        }
    }

    let equality_values = problem.equalities.matrix.mul_vec(x);
    let inequality_values = problem.inequalities.matrix.mul_vec(x);
    let mut primal = 0.0_f64;
    for (value, rhs) in equality_values.iter().zip(&problem.equalities.rhs) {
        primal = primal.max((value - rhs).abs());
    }
    for (value, rhs) in inequality_values.iter().zip(&problem.inequalities.rhs) {
        primal = primal.max((value - rhs).max(0.0));
    }
    for (index, value) in x.iter().enumerate() {
        primal = primal
            .max((problem.lower_bounds[index] - value).max(0.0))
            .max((value - problem.upper_bounds[index]).max(0.0));
    }

    let mut stationarity = problem.quadratic.apply(x);
    for (value, linear) in stationarity.iter_mut().zip(&problem.linear) {
        *value += linear;
    }
    problem
        .equalities
        .matrix
        .transpose_mul_add(&dual.equalities, &mut stationarity);
    problem
        .inequalities
        .matrix
        .transpose_mul_add(&dual.inequalities, &mut stationarity);
    for (value, bound_dual) in stationarity.iter_mut().zip(&dual.bounds) {
        *value += bound_dual;
    }
    for (value, l1_dual) in stationarity.iter_mut().zip(&dual.l1) {
        *value += l1_dual;
    }

    let mut cone_violation = 0.0_f64;
    let mut complementarity = 0.0_f64;
    for ((value, rhs), multiplier) in inequality_values
        .iter()
        .zip(&problem.inequalities.rhs)
        .zip(&dual.inequalities)
    {
        cone_violation = cone_violation.max((-multiplier).max(0.0));
        complementarity = complementarity.max((multiplier * (value - rhs)).abs());
    }

    // Box multipliers are scored without an activity window: the positive
    // part of a multiplier must pair with a finite upper bound and the
    // negative part with a finite lower bound (else it violates the dual
    // cone), and each part is charged the epsilon-complementarity product
    // `|multiplier| * distance-to-bound`. This keeps the checker fair to
    // near-active solutions from interior-point solvers, whose multipliers
    // decay smoothly with the distance to the bound instead of switching
    // off at machine precision.
    for (index, value) in x.iter().copied().enumerate() {
        let lower = problem.lower_bounds[index];
        let upper = problem.upper_bounds[index];
        let multiplier = dual.bounds[index];
        let toward_upper = multiplier.max(0.0);
        let toward_lower = (-multiplier).max(0.0);
        if upper.is_finite() {
            complementarity = complementarity.max(toward_upper * (upper - value).abs());
        } else {
            cone_violation = cone_violation.max(toward_upper);
        }
        if lower.is_finite() {
            complementarity = complementarity.max(toward_lower * (value - lower).abs());
        } else {
            cone_violation = cone_violation.max(toward_lower);
        }
    }

    // L1 multipliers must lie in the subdifferential of
    // `sum_i costs[i] * |x[i] - anchor[i]|`: inside `[-c_i, c_i]` everywhere
    // (dual-cone violation otherwise), pinned at the signed cost wherever the
    // variable moved off the anchor. The pinning is scored continuously,
    // mirroring the box scoring above: the shortfall from the required signed
    // cost is charged times the distance moved in that direction.
    if let Some(l1) = &problem.l1 {
        for (index, multiplier) in dual.l1.iter().copied().enumerate() {
            let cost = l1.costs[index];
            let offset = x[index] - l1.anchor[index];
            cone_violation = cone_violation.max(multiplier.abs() - cost);
            complementarity = complementarity
                .max(((cost - multiplier) * offset.max(0.0)).abs())
                .max(((cost + multiplier) * (-offset).max(0.0)).abs());
        }
    }

    Ok(KktResiduals {
        primal,
        dual: norm_inf(&stationarity).max(cone_violation),
        complementarity,
    })
}
