//! Ruiz equilibration that preserves the factor structure of the quadratic.
//!
//! The solver never forms `Q = F * omega * F' + diag(d)` explicitly, so the
//! scaled quadratic `c * E * Q * E` is represented the same way: the variable
//! scaling `E` acts on the rows of the covariance columns `G` (where
//! `F * omega * F' = G * G'`) and on `d`; the cost scalar `c` multiplies both.
//! The SMW reduction is unchanged.
//!
//! Column norms of `Q` are never computed exactly (that would cost `O(n^2 k)`).
//! Each Ruiz pass instead uses the exact diagonal `Q_jj = ||G_j||^2 + d_j`
//! and the Cauchy-Schwarz bound `|Q_ij| <= ||G_i|| * ||G_j||` on off-diagonal
//! entries. Estimate quality only affects conditioning, never correctness:
//! the scaled problem is always built from exactly scaled data.

use crate::{
    kkt::DualVariables,
    linalg::covariance_columns,
    matrix::{norm_inf, Matrix},
    problem::{FactorCovariance, FactorQuad, L1Term, LinearConstraints, QpProblem},
    solver::SolverError,
};

/// Per-pass scale factors are clamped into this range so a single degenerate
/// row or column cannot blow up the equilibration.
const MIN_SCALING: f64 = 1.0e-6;
const MAX_SCALING: f64 = 1.0e6;

/// An equilibrated copy of a QP plus the diagonal scalings that map iterates
/// between the original and scaled spaces.
///
/// With variable scaling `E = diag(e)`, constraint scalings
/// `D_e = diag(s_e)`, `D_i = diag(s_i)`, and cost scalar `c`, the scaled
/// problem solves for `x_scaled = E^{-1} x` with data
///
/// ```text
/// Q_scaled = c E Q E        q_scaled = c E q
/// A_scaled = D A E          b_scaled = D b
/// bounds_scaled = E^{-1} [l, u]
/// ```
///
/// Multipliers map back as `y = D y_scaled / c` for linear constraints and
/// `y_b = E^{-1} y_b_scaled / c` for the box block.
#[derive(Clone, Debug)]
pub(crate) struct ScaledProblem {
    /// Equilibrated problem consumed by the ADMM loop.
    pub problem: QpProblem,
    variable: Vec<f64>,
    equality: Vec<f64>,
    inequality: Vec<f64>,
    cost: f64,
}

impl ScaledProblem {
    /// Runs `iterations` Ruiz passes over the KKT-stacked data and returns the
    /// scaled problem together with the accumulated scalings.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::NonPositiveSemidefiniteOmega`] when the dense
    /// factor covariance cannot be factored.
    pub(crate) fn new(problem: &QpProblem, iterations: usize) -> Result<Self, SolverError> {
        let n = problem.quadratic.dimension();
        let mut columns = covariance_columns(&problem.quadratic)?;
        let k = columns.cols();
        let mut diagonal = problem.quadratic.diagonal.clone();
        let mut linear = problem.linear.clone();
        let mut l1_costs = problem
            .l1
            .as_ref()
            .map(|term| term.costs.clone())
            .unwrap_or_default();
        let mut equalities = problem.equalities.matrix.clone();
        let mut inequalities = problem.inequalities.matrix.clone();

        let mut variable = vec![1.0; n];
        let mut equality = vec![1.0; equalities.rows()];
        let mut inequality = vec![1.0; inequalities.rows()];
        let mut cost = 1.0;

        for _ in 0..iterations {
            let quadratic_norms = estimated_quadratic_column_norms(&columns, &diagonal);
            let variable_deltas: Vec<f64> = (0..n)
                .map(|col| {
                    let mut norm = quadratic_norms[col];
                    for row in 0..equalities.rows() {
                        norm = norm.max(equalities[(row, col)].abs());
                    }
                    for row in 0..inequalities.rows() {
                        norm = norm.max(inequalities[(row, col)].abs());
                    }
                    balanced_scale(norm)
                })
                .collect();
            let equality_deltas: Vec<f64> = (0..equalities.rows())
                .map(|row| balanced_scale(norm_inf(equalities.row(row))))
                .collect();
            let inequality_deltas: Vec<f64> = (0..inequalities.rows())
                .map(|row| balanced_scale(norm_inf(inequalities.row(row))))
                .collect();

            for (col, delta) in variable_deltas.iter().enumerate() {
                variable[col] *= delta;
                linear[col] *= delta;
                // The L1 cost transforms like the linear cost:
                // `c_i |x_i - a_i| = (c_i e_i) |x̄_i - a_i / e_i|`.
                if let Some(cost) = l1_costs.get_mut(col) {
                    *cost *= delta;
                }
                diagonal[col] *= delta * delta;
                for factor in 0..k {
                    columns[(col, factor)] *= delta;
                }
            }
            scale_constraints(
                &mut equalities,
                &mut equality,
                &equality_deltas,
                &variable_deltas,
            );
            scale_constraints(
                &mut inequalities,
                &mut inequality,
                &inequality_deltas,
                &variable_deltas,
            );

            // OSQP-style objective normalization: bring the average quadratic
            // column norm and the linear-term norm toward one.
            let scaled_norms = estimated_quadratic_column_norms(&columns, &diagonal);
            let mean_quadratic = scaled_norms.iter().sum::<f64>() / n.max(1) as f64;
            let gamma = balanced_reciprocal(
                mean_quadratic
                    .max(norm_inf(&linear))
                    .max(norm_inf(&l1_costs)),
            );
            let gamma_root = gamma.sqrt();
            for value in columns.as_mut_slice() {
                *value *= gamma_root;
            }
            for value in diagonal
                .iter_mut()
                .chain(linear.iter_mut())
                .chain(l1_costs.iter_mut())
            {
                *value *= gamma;
            }
            cost *= gamma;
        }

        // `G_scaled * G_scaled'` already carries the cost and variable
        // scalings, so the scaled quadratic uses an identity factor
        // covariance.
        let quadratic = FactorQuad {
            factors: columns,
            omega: FactorCovariance::Diagonal(vec![1.0; k]),
            diagonal,
        };
        let scaled = QpProblem {
            quadratic,
            linear,
            // The anchor lives in variable space, so it divides by `E`
            // exactly like the bounds; the costs accumulated `c * E` above.
            l1: problem.l1.as_ref().map(|term| L1Term {
                costs: l1_costs.clone(),
                anchor: divided_bounds(&term.anchor, &variable),
            }),
            equalities: LinearConstraints {
                matrix: equalities,
                rhs: scaled_rhs(&problem.equalities.rhs, &equality),
            },
            inequalities: LinearConstraints {
                matrix: inequalities,
                rhs: scaled_rhs(&problem.inequalities.rhs, &inequality),
            },
            lower_bounds: divided_bounds(&problem.lower_bounds, &variable),
            upper_bounds: divided_bounds(&problem.upper_bounds, &variable),
        };
        Ok(Self {
            problem: scaled,
            variable,
            equality,
            inequality,
            cost,
        })
    }

    /// Replaces the linear cost with the exact transform `c * E * linear` of
    /// a new original-space vector; the caller has already validated it.
    ///
    /// The accumulated scalings are frozen at construction, so an updated
    /// cost may be normalized slightly differently than a fresh equilibration
    /// of the new data would be. That affects conditioning only, never
    /// correctness: the scaled problem remains an exact transform.
    pub(crate) fn set_linear(&mut self, linear: &[f64]) {
        for ((scaled, original), scale) in self
            .problem
            .linear
            .iter_mut()
            .zip(linear)
            .zip(&self.variable)
        {
            *scaled = self.cost * scale * original;
        }
    }

    /// Replaces the L1 anchor with the exact transform `E^{-1} * anchor`;
    /// the caller has already validated it and checked the term exists.
    pub(crate) fn set_l1_anchor(&mut self, anchor: &[f64]) {
        if let Some(term) = &mut self.problem.l1 {
            for ((scaled, original), scale) in
                term.anchor.iter_mut().zip(anchor).zip(&self.variable)
            {
                *scaled = original / scale;
            }
        }
    }

    /// Replaces the equality right-hand side with `D_e * rhs`.
    pub(crate) fn set_equality_rhs(&mut self, rhs: &[f64]) {
        for ((scaled, original), scale) in self
            .problem
            .equalities
            .rhs
            .iter_mut()
            .zip(rhs)
            .zip(&self.equality)
        {
            *scaled = scale * original;
        }
    }

    /// Replaces the inequality right-hand side with `D_i * rhs`.
    pub(crate) fn set_inequality_rhs(&mut self, rhs: &[f64]) {
        for ((scaled, original), scale) in self
            .problem
            .inequalities
            .rhs
            .iter_mut()
            .zip(rhs)
            .zip(&self.inequality)
        {
            *scaled = scale * original;
        }
    }

    /// Maps original-space iterates into the scaled space in place.
    pub(crate) fn scale_iterates_in_place(&self, x: &mut [f64], dual: &mut DualVariables) {
        for (value, scale) in x.iter_mut().zip(&self.variable) {
            *value /= scale;
        }
        for (value, scale) in dual.equalities.iter_mut().zip(&self.equality) {
            *value *= self.cost / scale;
        }
        for (value, scale) in dual.inequalities.iter_mut().zip(&self.inequality) {
            *value *= self.cost / scale;
        }
        for (value, scale) in dual.bounds.iter_mut().zip(&self.variable) {
            *value *= self.cost * scale;
        }
        // The L1 multiplier enters stationarity with an identity
        // coefficient, exactly like the box multiplier.
        for (value, scale) in dual.l1.iter_mut().zip(&self.variable) {
            *value *= self.cost * scale;
        }
    }

    /// Returns the original-space decision vector for a scaled iterate.
    pub(crate) fn unscaled_x(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.variable)
            .map(|(value, scale)| value * scale)
            .collect()
    }

    /// Returns original-space multipliers for scaled multipliers.
    pub(crate) fn unscaled_dual(&self, dual: &DualVariables) -> DualVariables {
        DualVariables {
            equalities: dual
                .equalities
                .iter()
                .zip(&self.equality)
                .map(|(value, scale)| value * scale / self.cost)
                .collect(),
            inequalities: dual
                .inequalities
                .iter()
                .zip(&self.inequality)
                .map(|(value, scale)| value * scale / self.cost)
                .collect(),
            bounds: dual
                .bounds
                .iter()
                .zip(&self.variable)
                .map(|(value, scale)| value / (scale * self.cost))
                .collect(),
            l1: dual
                .l1
                .iter()
                .zip(&self.variable)
                .map(|(value, scale)| value / (scale * self.cost))
                .collect(),
        }
    }
}

/// Exact diagonal plus Cauchy-Schwarz off-diagonal bound per column of
/// `G G' + diag(d)`.
fn estimated_quadratic_column_norms(columns: &Matrix, diagonal: &[f64]) -> Vec<f64> {
    let n = columns.rows();
    let row_norms: Vec<f64> = (0..n)
        .map(|row| {
            columns
                .row(row)
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let largest_row = row_norms
        .iter()
        .fold(0.0_f64, |largest, norm| largest.max(*norm));
    row_norms
        .iter()
        .zip(diagonal)
        .map(|(norm, d)| (norm * norm + d).max(norm * largest_row))
        .collect()
}

fn scale_constraints(
    matrix: &mut Matrix,
    cumulative: &mut [f64],
    row_deltas: &[f64],
    column_deltas: &[f64],
) {
    for (row, row_delta) in row_deltas.iter().enumerate() {
        cumulative[row] *= row_delta;
        for (col, column_delta) in column_deltas.iter().enumerate() {
            matrix[(row, col)] *= row_delta * column_delta;
        }
    }
}

fn scaled_rhs(rhs: &[f64], scales: &[f64]) -> Vec<f64> {
    rhs.iter()
        .zip(scales)
        .map(|(value, scale)| value * scale)
        .collect()
}

fn divided_bounds(bounds: &[f64], scales: &[f64]) -> Vec<f64> {
    bounds
        .iter()
        .zip(scales)
        .map(|(value, scale)| value / scale)
        .collect()
}

/// `1 / sqrt(norm)` clamped; degenerate norms leave the row or column alone.
fn balanced_scale(norm: f64) -> f64 {
    if norm <= f64::MIN_POSITIVE {
        1.0
    } else {
        (1.0 / norm.sqrt()).clamp(MIN_SCALING, MAX_SCALING)
    }
}

/// `1 / value` clamped; degenerate values disable the cost update.
fn balanced_reciprocal(value: f64) -> f64 {
    if value <= f64::MIN_POSITIVE {
        1.0
    } else {
        (1.0 / value).clamp(MIN_SCALING, MAX_SCALING)
    }
}

#[cfg(test)]
mod tests {
    use super::ScaledProblem;
    use crate::{generate_synthetic, kkt::DualVariables, matrix::dot, SyntheticConfig};

    /// The scaled problem must be the exact transform `c E Q E` of the
    /// original: objectives agree up to the cost scalar for any point.
    #[test]
    fn scaled_objective_matches_original_up_to_cost_scalar() {
        let instance = generate_synthetic(SyntheticConfig::default()).unwrap();
        let scaled = ScaledProblem::new(&instance.problem, 10).unwrap();

        let reference = &instance.feasible_reference;
        let scaled_point: Vec<f64> = reference
            .iter()
            .zip(&scaled.variable)
            .map(|(value, scale)| value / scale)
            .collect();

        let original_objective = instance.problem.objective(reference);
        let scaled_objective = scaled.problem.objective(&scaled_point);
        assert!(
            (scaled_objective - scaled.cost * original_objective).abs()
                <= 1.0e-12 * (1.0 + scaled_objective.abs())
        );
    }

    /// Constraint rows must describe the same feasible set after scaling.
    #[test]
    fn scaled_constraints_preserve_feasibility_of_the_reference() {
        let instance = generate_synthetic(SyntheticConfig {
            inequalities: 6,
            ..SyntheticConfig::default()
        })
        .unwrap();
        let scaled = ScaledProblem::new(&instance.problem, 10).unwrap();
        let scaled_point: Vec<f64> = instance
            .feasible_reference
            .iter()
            .zip(&scaled.variable)
            .map(|(value, scale)| value / scale)
            .collect();

        for row in 0..scaled.problem.equalities.len() {
            let value = dot(scaled.problem.equalities.matrix.row(row), &scaled_point);
            let rhs = scaled.problem.equalities.rhs[row];
            assert!((value - rhs).abs() <= 1.0e-10 * (1.0 + rhs.abs()));
        }
        for row in 0..scaled.problem.inequalities.len() {
            let value = dot(scaled.problem.inequalities.matrix.row(row), &scaled_point);
            let rhs = scaled.problem.inequalities.rhs[row];
            assert!(value <= rhs + 1.0e-10 * (1.0 + rhs.abs()));
        }
        for (index, value) in scaled_point.iter().enumerate() {
            assert!(*value >= scaled.problem.lower_bounds[index] - 1.0e-12);
            assert!(*value <= scaled.problem.upper_bounds[index] + 1.0e-12);
        }
    }

    /// Scaling then unscaling iterates must be the identity map.
    #[test]
    fn iterate_round_trip_is_identity() {
        let instance = generate_synthetic(SyntheticConfig::default()).unwrap();
        let scaled = ScaledProblem::new(&instance.problem, 10).unwrap();
        let n = instance.problem.quadratic.dimension();

        let original_x: Vec<f64> = (0..n).map(|index| 0.01 * (index as f64 + 1.0)).collect();
        let original_dual = DualVariables {
            equalities: vec![0.3; instance.problem.equalities.len()],
            inequalities: vec![0.7; instance.problem.inequalities.len()],
            bounds: (0..n).map(|index| -0.5 + 0.001 * index as f64).collect(),
            l1: Vec::new(),
        };

        let mut x = original_x.clone();
        let mut dual = original_dual.clone();
        scaled.scale_iterates_in_place(&mut x, &mut dual);
        let x = scaled.unscaled_x(&x);
        let dual = scaled.unscaled_dual(&dual);

        for (roundtrip, original) in x.iter().zip(&original_x) {
            assert!((roundtrip - original).abs() <= 1.0e-12 * (1.0 + original.abs()));
        }
        for (roundtrip, original) in dual
            .equalities
            .iter()
            .chain(&dual.inequalities)
            .chain(&dual.bounds)
            .zip(
                original_dual
                    .equalities
                    .iter()
                    .chain(&original_dual.inequalities)
                    .chain(&original_dual.bounds),
            )
        {
            assert!((roundtrip - original).abs() <= 1.0e-12 * (1.0 + original.abs()));
        }
    }

    /// Zero Ruiz iterations must produce the identity scaling.
    #[test]
    #[allow(clippy::float_cmp)] // exact identity is the property under test
    fn zero_iterations_is_the_identity_scaling() {
        let instance = generate_synthetic(SyntheticConfig::default()).unwrap();
        let scaled = ScaledProblem::new(&instance.problem, 0).unwrap();

        assert!(scaled.variable.iter().all(|scale| *scale == 1.0));
        assert!(scaled.equality.iter().all(|scale| *scale == 1.0));
        assert!(scaled.inequality.iter().all(|scale| *scale == 1.0));
        assert_eq!(scaled.cost, 1.0);
        assert_eq!(scaled.problem.linear, instance.problem.linear);
        assert_eq!(
            scaled.problem.equalities.rhs,
            instance.problem.equalities.rhs
        );
    }
}
