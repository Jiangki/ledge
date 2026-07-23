//! Infeasibility certificates (roadmap 2.2).
//!
//! On infeasible problems, ADMM iterate *differences* converge to certificate
//! directions (OSQP-style detection): the dual differences to a Farkas
//! certificate of primal infeasibility, the primal differences to an
//! unbounded descent direction proving dual infeasibility. The solver checks
//! both on the termination cadence and stops with
//! [`SolveStatus::PrimalInfeasible`](crate::SolveStatus::PrimalInfeasible) or
//! [`SolveStatus::DualInfeasible`](crate::SolveStatus::DualInfeasible),
//! attaching the normalized certificate to
//! [`Solution::certificate`](crate::Solution).
//!
//! Certificates follow the same auditability rule as every other reported
//! quantity: they are detected and reported **in the original data space**
//! (never the equilibrated copy), and the standalone checkers
//! [`check_primal_certificate`] / [`check_dual_certificate`] recompute their
//! defining residuals independently, exactly like [`crate::check_kkt`] does
//! for solutions.

use crate::{
    kkt::{DualVariables, KktError},
    matrix::{dot, norm_inf},
    problem::QpProblem,
};

/// Directions smaller than this are considered numerical silence, not
/// candidate certificates.
const DIVISION_GUARD: f64 = 1.0e-12;

/// Proof that a solve stopped because the problem itself is pathological.
///
/// Attached to [`Solution::certificate`](crate::Solution) when the status is
/// [`SolveStatus::PrimalInfeasible`](crate::SolveStatus::PrimalInfeasible) or
/// [`SolveStatus::DualInfeasible`](crate::SolveStatus::DualInfeasible).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Certificate {
    /// The constraints admit no common point.
    Primal(PrimalCertificate),
    /// The objective is unbounded below over the constraints.
    Dual(DualCertificate),
}

/// A Farkas certificate of primal infeasibility, normalized to unit
/// infinity norm.
///
/// The multipliers prove that no `x` can satisfy
/// `A_e x = b_e`, `A_i x <= b_i`, `l <= x <= u` simultaneously: with
/// `y_i >= 0`, positive bound parts supported on finite upper bounds, and
/// negative bound parts supported on finite lower bounds, any feasible `x`
/// would give
///
/// ```text
/// (A_e' y_e + A_i' y_i + y_b)' x <= b_e' y_e + b_i' y_i + u'(y_b)+ + l'(y_b)-
/// ```
///
/// A valid certificate makes the left side (approximately) zero for every `x`
/// while the right side — the *support gap* — is strictly negative:
/// a contradiction. Verify with [`check_primal_certificate`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrimalCertificate {
    /// Weights on the equality rows (free sign).
    pub equality_dual: Vec<f64>,
    /// Weights on the inequality rows (non-negative).
    pub inequality_dual: Vec<f64>,
    /// Weights on the variable boxes: positive parts cite upper bounds,
    /// negative parts cite lower bounds.
    pub bound_dual: Vec<f64>,
}

/// An unbounded-descent certificate of dual infeasibility, normalized to
/// unit infinity norm.
///
/// The direction `v` proves the objective decreases without bound: `Q v ~ 0`
/// (no quadratic cost along the ray), `q' v < 0` (the linear cost strictly
/// decreases), and `v` is a recession direction of the constraints
/// (`A_e v ~ 0`, `A_i v <~ 0`, non-positive where an upper bound is finite,
/// non-negative where a lower bound is finite). Verify with
/// [`check_dual_certificate`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DualCertificate {
    /// The unbounded descent direction in the decision space.
    pub direction: Vec<f64>,
}

/// Independently recomputed residuals of a [`PrimalCertificate`].
///
/// The certificate is valid to tolerance `eps` when `stationarity <= eps`,
/// `cone_violation <= eps`, and `support_gap <= -eps` (for a certificate
/// normalized to unit infinity norm).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrimalCertificateResiduals {
    /// `||A_e' y_e + A_i' y_i + y_b||_inf`; zero for an exact certificate.
    pub stationarity: f64,
    /// `b_e' y_e + b_i' y_i + u'(y_b)+ + l'(y_b)-` over finite bounds;
    /// strictly negative for a valid certificate.
    pub support_gap: f64,
    /// Largest violation of the certificate cone: negative inequality
    /// weights, or bound weights citing an infinite bound.
    pub cone_violation: f64,
}

/// Independently recomputed residuals of a [`DualCertificate`].
///
/// The certificate is valid to tolerance `eps` when `curvature <= eps`,
/// `recession_violation <= eps`, and `objective_gap <= -eps` (for a
/// direction normalized to unit infinity norm).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DualCertificateResiduals {
    /// `||Q v||_inf`; zero when the ray carries no quadratic cost.
    pub curvature: f64,
    /// `q' v` plus, when the problem has an [`L1Term`](crate::L1Term), its
    /// recession slope `sum_i costs[i] * |v_i|`; strictly negative when the
    /// full objective descends along the ray.
    pub objective_gap: f64,
    /// Largest violation of the constraint recession cone: `|A_e v|`,
    /// positive parts of `A_i v`, positive components where an upper bound
    /// is finite, negative components where a lower bound is finite.
    pub recession_violation: f64,
}

/// Recomputes the defining residuals of a primal infeasibility certificate
/// on the original problem data.
///
/// # Errors
///
/// Returns [`KktError::Dimension`] when a multiplier block has the wrong
/// length.
pub fn check_primal_certificate(
    problem: &QpProblem,
    certificate: &PrimalCertificate,
) -> Result<PrimalCertificateResiduals, KktError> {
    let n = problem.quadratic.dimension();
    for (field, actual, expected) in [
        (
            "certificate.equality_dual",
            certificate.equality_dual.len(),
            problem.equalities.len(),
        ),
        (
            "certificate.inequality_dual",
            certificate.inequality_dual.len(),
            problem.inequalities.len(),
        ),
        ("certificate.bound_dual", certificate.bound_dual.len(), n),
    ] {
        if actual != expected {
            return Err(KktError::Dimension {
                field,
                expected,
                actual,
            });
        }
    }

    let mut combination = certificate.bound_dual.clone();
    problem
        .equalities
        .matrix
        .transpose_mul_add(&certificate.equality_dual, &mut combination);
    problem
        .inequalities
        .matrix
        .transpose_mul_add(&certificate.inequality_dual, &mut combination);
    let stationarity = norm_inf(&combination);

    let mut cone_violation = 0.0_f64;
    let mut support_gap = dot(&problem.equalities.rhs, &certificate.equality_dual);
    for (multiplier, rhs) in certificate
        .inequality_dual
        .iter()
        .zip(&problem.inequalities.rhs)
    {
        cone_violation = cone_violation.max(-multiplier);
        support_gap += rhs * multiplier.max(0.0);
    }
    for (index, multiplier) in certificate.bound_dual.iter().enumerate() {
        let toward_upper = multiplier.max(0.0);
        let toward_lower = multiplier.min(0.0);
        let upper = problem.upper_bounds[index];
        let lower = problem.lower_bounds[index];
        if upper.is_finite() {
            support_gap += upper * toward_upper;
        } else {
            cone_violation = cone_violation.max(toward_upper);
        }
        if lower.is_finite() {
            support_gap += lower * toward_lower;
        } else {
            cone_violation = cone_violation.max(-toward_lower);
        }
    }

    Ok(PrimalCertificateResiduals {
        stationarity,
        support_gap,
        cone_violation,
    })
}

/// Recomputes the defining residuals of a dual infeasibility certificate on
/// the original problem data.
///
/// # Errors
///
/// Returns [`KktError::Dimension`] when the direction has the wrong length.
pub fn check_dual_certificate(
    problem: &QpProblem,
    certificate: &DualCertificate,
) -> Result<DualCertificateResiduals, KktError> {
    let n = problem.quadratic.dimension();
    if certificate.direction.len() != n {
        return Err(KktError::Dimension {
            field: "certificate.direction",
            expected: n,
            actual: certificate.direction.len(),
        });
    }

    let curvature = norm_inf(&problem.quadratic.apply(&certificate.direction));
    // Along the ray `t * v`, the L1 term grows like `t * sum_i c_i |v_i|`
    // (its recession function); a valid descent ray must outrun it.
    let l1_slope = problem.l1.as_ref().map_or(0.0, |term| {
        term.costs
            .iter()
            .zip(&certificate.direction)
            .map(|(cost, value)| cost * value.abs())
            .sum()
    });
    let objective_gap = dot(&problem.linear, &certificate.direction) + l1_slope;

    let mut recession_violation =
        norm_inf(&problem.equalities.matrix.mul_vec(&certificate.direction));
    for value in problem.inequalities.matrix.mul_vec(&certificate.direction) {
        recession_violation = recession_violation.max(value);
    }
    for (index, value) in certificate.direction.iter().enumerate() {
        if problem.upper_bounds[index].is_finite() {
            recession_violation = recession_violation.max(*value);
        }
        if problem.lower_bounds[index].is_finite() {
            recession_violation = recession_violation.max(-*value);
        }
    }

    Ok(DualCertificateResiduals {
        curvature,
        objective_gap,
        recession_violation,
    })
}

/// Tests whether a dual-iterate difference is a primal infeasibility
/// certificate to the given tolerance; returns the normalized certificate
/// when it is.
///
/// The difference must be supplied in the original (unscaled) space. The
/// candidate is normalized to unit infinity norm; cone noise up to the
/// tolerance is clamped to zero so the returned certificate satisfies the
/// sign constraints exactly, while larger violations disqualify the
/// direction. Acceptance re-uses [`check_primal_certificate`], so a returned
/// certificate always passes its own audit.
pub(crate) fn detect_primal_infeasibility(
    problem: &QpProblem,
    delta_dual: &DualVariables,
    tolerance: f64,
) -> Option<PrimalCertificate> {
    let magnitude = norm_inf(&delta_dual.equalities)
        .max(norm_inf(&delta_dual.inequalities))
        .max(norm_inf(&delta_dual.bounds));
    if !magnitude.is_finite() || magnitude <= DIVISION_GUARD {
        return None;
    }

    let equality_dual: Vec<f64> = delta_dual
        .equalities
        .iter()
        .map(|value| value / magnitude)
        .collect();
    let mut inequality_dual: Vec<f64> = delta_dual
        .inequalities
        .iter()
        .map(|value| value / magnitude)
        .collect();
    let mut bound_dual: Vec<f64> = delta_dual
        .bounds
        .iter()
        .map(|value| value / magnitude)
        .collect();

    for value in &mut inequality_dual {
        if *value < -tolerance {
            return None;
        }
        *value = value.max(0.0);
    }
    for (index, value) in bound_dual.iter_mut().enumerate() {
        if !problem.upper_bounds[index].is_finite() {
            if *value > tolerance {
                return None;
            }
            *value = value.min(0.0);
        }
        if !problem.lower_bounds[index].is_finite() {
            if *value < -tolerance {
                return None;
            }
            *value = value.max(0.0);
        }
    }

    let certificate = PrimalCertificate {
        equality_dual,
        inequality_dual,
        bound_dual,
    };
    let residuals = check_primal_certificate(problem, &certificate).ok()?;
    (residuals.stationarity <= tolerance && residuals.support_gap <= -tolerance)
        .then_some(certificate)
}

/// Tests whether a primal-iterate difference is a dual infeasibility
/// certificate to the given tolerance; returns the normalized certificate
/// when it is.
///
/// The difference must be supplied in the original (unscaled) space.
/// Acceptance re-uses [`check_dual_certificate`], so a returned certificate
/// always passes its own audit.
pub(crate) fn detect_dual_infeasibility(
    problem: &QpProblem,
    delta_x: &[f64],
    tolerance: f64,
) -> Option<DualCertificate> {
    let magnitude = norm_inf(delta_x);
    if !magnitude.is_finite() || magnitude <= DIVISION_GUARD {
        return None;
    }
    let certificate = DualCertificate {
        direction: delta_x.iter().map(|value| value / magnitude).collect(),
    };
    let residuals = check_dual_certificate(problem, &certificate).ok()?;
    (residuals.curvature <= tolerance
        && residuals.recession_violation <= tolerance
        && residuals.objective_gap <= -tolerance)
        .then_some(certificate)
}

#[cfg(test)]
mod tests {
    use super::{
        check_dual_certificate, check_primal_certificate, detect_dual_infeasibility,
        detect_primal_infeasibility, PrimalCertificate,
    };
    use crate::{
        kkt::DualVariables,
        problem::{FactorCovariance, FactorQuad, LinearConstraints, QpProblem},
        Matrix,
    };

    /// `sum(x) = 1` against upper bounds allowing at most `0.8`.
    fn budget_versus_boxes() -> QpProblem {
        let n = 4;
        QpProblem {
            quadratic: FactorQuad {
                factors: Matrix::zeros(n, 1),
                omega: FactorCovariance::Diagonal(vec![1.0]),
                diagonal: vec![1.0; n],
            },
            linear: vec![0.0; n],
            l1: None,
            equalities: LinearConstraints {
                matrix: Matrix::new(1, n, vec![1.0; n]).unwrap(),
                rhs: vec![1.0],
            },
            inequalities: LinearConstraints::empty(n),
            lower_bounds: vec![0.0; n],
            upper_bounds: vec![0.2; n],
        }
    }

    #[test]
    fn exact_farkas_certificate_audits_clean() {
        let problem = budget_versus_boxes();
        let certificate = PrimalCertificate {
            equality_dual: vec![-1.0],
            inequality_dual: Vec::new(),
            bound_dual: vec![1.0; 4],
        };
        let residuals = check_primal_certificate(&problem, &certificate).unwrap();
        assert!(residuals.stationarity <= 1.0e-12);
        assert!(residuals.cone_violation <= 1.0e-12);
        assert!((residuals.support_gap - (-0.2)).abs() <= 1.0e-12);
    }

    #[test]
    fn detection_accepts_the_farkas_direction_and_rejects_noise() {
        let problem = budget_versus_boxes();
        let farkas = DualVariables {
            equalities: vec![-2.0],
            inequalities: Vec::new(),
            bounds: vec![2.0; 4],
            l1: Vec::new(),
        };
        let certificate = detect_primal_infeasibility(&problem, &farkas, 1.0e-5).unwrap();
        let residuals = check_primal_certificate(&problem, &certificate).unwrap();
        assert!(residuals.support_gap <= -1.0e-5);

        // A direction with a large stationarity residual must be rejected.
        let noise = DualVariables {
            equalities: vec![1.0],
            inequalities: Vec::new(),
            bounds: vec![1.0; 4],
            l1: Vec::new(),
        };
        assert!(detect_primal_infeasibility(&problem, &noise, 1.0e-5).is_none());

        // Numerical silence is never a certificate.
        let silence = DualVariables {
            equalities: vec![0.0],
            inequalities: Vec::new(),
            bounds: vec![0.0; 4],
            l1: Vec::new(),
        };
        assert!(detect_primal_infeasibility(&problem, &silence, 1.0e-5).is_none());
    }

    #[test]
    fn dual_certificate_requires_descent_and_recession() {
        // One riskless unbounded variable with a positive linear cost slope.
        let problem = QpProblem {
            quadratic: FactorQuad {
                factors: Matrix::zeros(2, 1),
                omega: FactorCovariance::Diagonal(vec![1.0]),
                diagonal: vec![0.0, 1.0],
            },
            linear: vec![1.0, 0.0],
            l1: None,
            equalities: LinearConstraints::empty(2),
            inequalities: LinearConstraints::empty(2),
            lower_bounds: vec![f64::NEG_INFINITY, 0.0],
            upper_bounds: vec![f64::INFINITY, 1.0],
        };
        let descent = detect_dual_infeasibility(&problem, &[-3.0, 0.0], 1.0e-5).unwrap();
        let residuals = check_dual_certificate(&problem, &descent).unwrap();
        assert!(residuals.curvature <= 1.0e-12);
        assert!(residuals.objective_gap <= -1.0);
        assert!(residuals.recession_violation <= 1.0e-12);

        // Ascent along the same ray is not a certificate.
        assert!(detect_dual_infeasibility(&problem, &[3.0, 0.0], 1.0e-5).is_none());
        // A direction moving the bounded variable violates recession.
        assert!(detect_dual_infeasibility(&problem, &[-3.0, -1.0], 1.0e-5).is_none());
    }
}
