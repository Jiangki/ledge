//! Convex QP data structures.

use thiserror::Error;

use crate::matrix::{dot, Matrix};

/// Factor covariance matrix \(\Omega\).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FactorCovariance {
    /// Diagonal factor covariance.
    Diagonal(Vec<f64>),
    /// Dense symmetric positive-semidefinite factor covariance.
    Dense(Matrix),
}

impl FactorCovariance {
    /// Number of factors represented by this covariance.
    #[must_use]
    pub fn dimension(&self) -> usize {
        match self {
            Self::Diagonal(diagonal) => diagonal.len(),
            Self::Dense(matrix) => matrix.rows(),
        }
    }
}

/// A positive-semidefinite quadratic represented as
/// \(Q = F\Omega F^\mathsf{T} + \operatorname{diag}(d)\).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FactorQuad {
    /// Asset-by-factor exposure matrix.
    pub factors: Matrix,
    /// Factor covariance.
    pub omega: FactorCovariance,
    /// Non-negative idiosyncratic diagonal.
    pub diagonal: Vec<f64>,
}

impl FactorQuad {
    /// Creates a factor quadratic after checking dimensions and convexity data.
    ///
    /// Positive semidefiniteness of a dense `omega` is checked during solver
    /// setup, where its Cholesky-like factor is needed.
    ///
    /// # Errors
    ///
    /// Returns a [`ProblemError`] for inconsistent dimensions, non-finite
    /// values, asymmetry, or negative diagonal entries.
    pub fn new(
        factors: Matrix,
        omega: FactorCovariance,
        diagonal: Vec<f64>,
    ) -> Result<Self, ProblemError> {
        let quadratic = Self {
            factors,
            omega,
            diagonal,
        };
        quadratic.validate()?;
        Ok(quadratic)
    }

    /// Number of decision variables.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.factors.rows()
    }

    /// Number of latent factors.
    #[must_use]
    pub fn factor_count(&self) -> usize {
        self.factors.cols()
    }

    /// Computes `Q * x` without materializing `Q`.
    #[must_use]
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.dimension());
        let n = self.dimension();
        let k = self.factor_count();
        let mut factor_projection = vec![0.0; k];
        self.factors.transpose_mul_add(x, &mut factor_projection);

        let weighted = match &self.omega {
            FactorCovariance::Diagonal(diagonal) => factor_projection
                .iter()
                .zip(diagonal)
                .map(|(value, weight)| value * weight)
                .collect(),
            FactorCovariance::Dense(matrix) => matrix.mul_vec(&factor_projection),
        };

        let mut result: Vec<f64> = self
            .diagonal
            .iter()
            .zip(x)
            .map(|(diagonal, value)| diagonal * value)
            .collect();
        for (row, value) in result.iter_mut().enumerate().take(n) {
            *value += dot(self.factors.row(row), &weighted);
        }
        result
    }

    fn validate(&self) -> Result<(), ProblemError> {
        let n = self.factors.rows();
        let k = self.factors.cols();
        if self.diagonal.len() != n {
            return Err(ProblemError::Dimension {
                field: "quadratic.diagonal",
                expected: n,
                actual: self.diagonal.len(),
            });
        }
        if self.omega.dimension() != k {
            return Err(ProblemError::Dimension {
                field: "quadratic.omega",
                expected: k,
                actual: self.omega.dimension(),
            });
        }
        if self
            .factors
            .as_slice()
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(ProblemError::NonFinite("quadratic.factors"));
        }
        if self
            .diagonal
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ProblemError::NonConvex(
                "quadratic diagonal must be finite and non-negative",
            ));
        }
        match &self.omega {
            FactorCovariance::Diagonal(diagonal) => {
                if diagonal
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                {
                    return Err(ProblemError::NonConvex(
                        "factor covariance diagonal must be finite and non-negative",
                    ));
                }
            }
            FactorCovariance::Dense(matrix) => {
                if matrix.rows() != matrix.cols() {
                    return Err(ProblemError::NotSquare("quadratic.omega"));
                }
                if matrix.as_slice().iter().any(|value| !value.is_finite()) {
                    return Err(ProblemError::NonFinite("quadratic.omega"));
                }
                for row in 0..k {
                    for col in 0..row {
                        let scale = 1.0_f64
                            .max(matrix[(row, col)].abs())
                            .max(matrix[(col, row)].abs());
                        if (matrix[(row, col)] - matrix[(col, row)]).abs() > 1.0e-12 * scale {
                            return Err(ProblemError::NotSymmetric("quadratic.omega"));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// A piecewise-linear proportional cost
/// `sum_i costs[i] * |x[i] - anchor[i]|` added to the smooth objective
/// (for portfolios: exact L1 turnover around the previous weights).
///
/// Costs must be non-negative. The term is handled by a dedicated
/// soft-threshold proximal block inside the solver, so it never grows the
/// SMW-reduced dimension — unlike an epigraph reformulation, which would add
/// \(2n\) general constraint rows (see `docs/algorithm.md` §4).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct L1Term {
    /// Non-negative per-variable cost coefficients \(c\).
    pub costs: Vec<f64>,
    /// Anchor point \(a\) the absolute deviations are measured from.
    pub anchor: Vec<f64>,
}

impl L1Term {
    /// Evaluates `sum_i costs[i] * |x[i] - anchor[i]|`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        self.costs
            .iter()
            .zip(&self.anchor)
            .zip(x)
            .map(|((cost, anchor), value)| cost * (value - anchor).abs())
            .sum()
    }

    fn validate(&self, dimension: usize) -> Result<(), ProblemError> {
        if self.costs.len() != dimension {
            return Err(ProblemError::Dimension {
                field: "l1.costs",
                expected: dimension,
                actual: self.costs.len(),
            });
        }
        if self.anchor.len() != dimension {
            return Err(ProblemError::Dimension {
                field: "l1.anchor",
                expected: dimension,
                actual: self.anchor.len(),
            });
        }
        if self
            .costs
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ProblemError::NonConvex(
                "l1 costs must be finite and non-negative",
            ));
        }
        if self.anchor.iter().any(|value| !value.is_finite()) {
            return Err(ProblemError::NonFinite("l1.anchor"));
        }
        Ok(())
    }
}

/// A block of constraints `matrix * x = rhs` or `matrix * x <= rhs`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearConstraints {
    /// Constraint matrix.
    pub matrix: Matrix,
    /// Right-hand side.
    pub rhs: Vec<f64>,
}

impl LinearConstraints {
    /// Creates a validated constraint block.
    ///
    /// # Errors
    ///
    /// Returns [`ProblemError::Dimension`] if row and RHS counts differ.
    pub fn new(matrix: Matrix, rhs: Vec<f64>) -> Result<Self, ProblemError> {
        if matrix.rows() != rhs.len() {
            return Err(ProblemError::Dimension {
                field: "constraints.rhs",
                expected: matrix.rows(),
                actual: rhs.len(),
            });
        }
        Ok(Self { matrix, rhs })
    }

    /// An empty block with the requested decision dimension.
    #[must_use]
    pub fn empty(dimension: usize) -> Self {
        Self {
            matrix: Matrix::zeros(0, dimension),
            rhs: Vec::new(),
        }
    }

    /// Number of constraint rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rhs.len()
    }

    /// Whether this block contains no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rhs.is_empty()
    }
}

/// Standard-form convex QP accepted by Ledge.
///
/// The objective is `0.5 * x' Q x + linear' x` plus an optional
/// piecewise-linear term `sum_i costs[i] * |x[i] - anchor[i]|`
/// ([`L1Term`]), with equality, upper-inequality, and box constraints.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QpProblem {
    /// Structured positive-semidefinite quadratic.
    pub quadratic: FactorQuad,
    /// Linear objective coefficient.
    pub linear: Vec<f64>,
    /// Optional proportional-cost term `sum_i costs[i] * |x[i] - anchor[i]|`.
    pub l1: Option<L1Term>,
    /// Constraints `A_eq * x = b_eq`.
    pub equalities: LinearConstraints,
    /// Constraints `A_ineq * x <= b_ineq`.
    pub inequalities: LinearConstraints,
    /// Variable lower bounds; `-inf` means unbounded.
    ///
    /// Serialized as `Option<f64>` entries (`null` in JSON) for unbounded
    /// sides, because JSON cannot represent infinities — see
    /// `serde_support`.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::lower_bounds"))]
    pub lower_bounds: Vec<f64>,
    /// Variable upper bounds; `+inf` means unbounded.
    ///
    /// Serialized as `Option<f64>` entries (`null` in JSON) for unbounded
    /// sides, because JSON cannot represent infinities — see
    /// `serde_support`.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::upper_bounds"))]
    pub upper_bounds: Vec<f64>,
}

impl QpProblem {
    /// Validates dimensions, finite coefficients, and bound consistency.
    ///
    /// # Errors
    ///
    /// Returns a [`ProblemError`] when the problem is malformed.
    pub fn validate(&self) -> Result<(), ProblemError> {
        self.quadratic.validate()?;
        let n = self.quadratic.dimension();
        for (field, actual) in [
            ("linear", self.linear.len()),
            ("lower_bounds", self.lower_bounds.len()),
            ("upper_bounds", self.upper_bounds.len()),
        ] {
            if actual != n {
                return Err(ProblemError::Dimension {
                    field,
                    expected: n,
                    actual,
                });
            }
        }
        for (field, constraints) in [
            ("equalities", &self.equalities),
            ("inequalities", &self.inequalities),
        ] {
            if constraints.matrix.cols() != n {
                return Err(ProblemError::Dimension {
                    field,
                    expected: n,
                    actual: constraints.matrix.cols(),
                });
            }
            if constraints.matrix.rows() != constraints.rhs.len() {
                return Err(ProblemError::Dimension {
                    field,
                    expected: constraints.matrix.rows(),
                    actual: constraints.rhs.len(),
                });
            }
            if constraints
                .matrix
                .as_slice()
                .iter()
                .chain(&constraints.rhs)
                .any(|value| !value.is_finite())
            {
                return Err(ProblemError::NonFinite(field));
            }
        }
        if self.linear.iter().any(|value| !value.is_finite()) {
            return Err(ProblemError::NonFinite("linear"));
        }
        if let Some(l1) = &self.l1 {
            l1.validate(n)?;
        }
        for index in 0..n {
            let lower = self.lower_bounds[index];
            let upper = self.upper_bounds[index];
            if lower.is_nan() || upper.is_nan() {
                return Err(ProblemError::NonFinite("bounds"));
            }
            if lower > upper {
                return Err(ProblemError::InvalidBounds {
                    index,
                    lower,
                    upper,
                });
            }
        }
        Ok(())
    }

    /// Evaluates the objective at `x`, including the L1 term when present.
    #[must_use]
    pub fn objective(&self, x: &[f64]) -> f64 {
        let qx = self.quadratic.apply(x);
        let smooth = 0.5 * dot(x, &qx) + dot(&self.linear, x);
        smooth + self.l1.as_ref().map_or(0.0, |term| term.evaluate(x))
    }
}

/// Input validation errors.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ProblemError {
    /// A vector or matrix dimension is inconsistent.
    #[error("{field} has dimension {actual}; expected {expected}")]
    Dimension {
        /// Name of the invalid field.
        field: &'static str,
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// A coefficient that must be finite is NaN or infinite.
    #[error("{0} contains a non-finite coefficient")]
    NonFinite(&'static str),
    /// Convexity data is invalid.
    #[error("non-convex quadratic data: {0}")]
    NonConvex(&'static str),
    /// A matrix expected to be square is not.
    #[error("{0} must be square")]
    NotSquare(&'static str),
    /// A matrix expected to be symmetric is not.
    #[error("{0} must be symmetric")]
    NotSymmetric(&'static str),
    /// Lower and upper bounds are inconsistent.
    #[error("invalid bounds at index {index}: lower {lower} exceeds upper {upper}")]
    InvalidBounds {
        /// Variable index.
        index: usize,
        /// Supplied lower bound.
        lower: f64,
        /// Supplied upper bound.
        upper: f64,
    },
}
