//! Internal dense linear algebra on reduced systems.

use crate::{
    matrix::Matrix,
    problem::{FactorCovariance, FactorQuad},
    SolverError,
};

/// Returns `G` such that `F * omega * F' = G * G'`.
pub(crate) fn covariance_columns(quadratic: &FactorQuad) -> Result<Matrix, SolverError> {
    let n = quadratic.dimension();
    let k = quadratic.factor_count();
    let omega_root = match &quadratic.omega {
        FactorCovariance::Diagonal(diagonal) => {
            let mut root = Matrix::zeros(k, k);
            for index in 0..k {
                root[(index, index)] = diagonal[index].sqrt();
            }
            root
        }
        FactorCovariance::Dense(omega) => semidefinite_cholesky(omega)?,
    };

    let mut columns = Matrix::zeros(n, k);
    for row in 0..n {
        for col in 0..k {
            let mut value = 0.0;
            for inner in col..k {
                value += quadratic.factors[(row, inner)] * omega_root[(inner, col)];
            }
            columns[(row, col)] = value;
        }
    }
    Ok(columns)
}

fn semidefinite_cholesky(matrix: &Matrix) -> Result<Matrix, SolverError> {
    let dimension = matrix.rows();
    let mut lower = Matrix::zeros(dimension, dimension);
    let scale = matrix
        .as_slice()
        .iter()
        .fold(1.0_f64, |largest, value| largest.max(value.abs()));
    let tolerance = 1.0e-12 * scale;

    for col in 0..dimension {
        let diagonal_remainder = matrix[(col, col)]
            - (0..col)
                .map(|inner| lower[(col, inner)].powi(2))
                .sum::<f64>();
        if diagonal_remainder < -tolerance {
            return Err(SolverError::NonPositiveSemidefiniteOmega);
        }
        if diagonal_remainder <= tolerance {
            lower[(col, col)] = 0.0;
            for row in col + 1..dimension {
                let remainder = matrix[(row, col)]
                    - (0..col)
                        .map(|inner| lower[(row, inner)] * lower[(col, inner)])
                        .sum::<f64>();
                if remainder.abs() > tolerance {
                    return Err(SolverError::NonPositiveSemidefiniteOmega);
                }
            }
            continue;
        }

        lower[(col, col)] = diagonal_remainder.sqrt();
        for row in col + 1..dimension {
            let product: f64 = (0..col)
                .map(|inner| lower[(row, inner)] * lower[(col, inner)])
                .sum();
            lower[(row, col)] = (matrix[(row, col)] - product) / lower[(col, col)];
        }
    }
    Ok(lower)
}

/// Sherman-Morrison-Woodbury solver for `H = diag(diagonal) + C * C'`.
///
/// Factors the reduced Gram matrix `S = I + C' diag(diagonal)^{-1} C` once
/// (`O(n r^2 + r^3)` with `r = C.cols()`); every subsequent solve costs
/// `O(n r + r^2)`. This is the reduction behind both the ADMM x-update and
/// the polishing KKT solve.
pub(crate) struct SmwSystem {
    inverse_diagonal: Vec<f64>,
    columns: Matrix,
    reduced_cholesky: Cholesky,
}

impl SmwSystem {
    /// Factors `diag(diagonal) + columns * columns'`; `diagonal` must be
    /// strictly positive.
    pub(crate) fn factor(diagonal: &[f64], columns: Matrix) -> Result<Self, SolverError> {
        debug_assert_eq!(diagonal.len(), columns.rows());
        let n = diagonal.len();
        let reduced_dimension = columns.cols();
        let inverse_diagonal: Vec<f64> = diagonal.iter().map(|value| 1.0 / value).collect();
        let mut reduced = vec![0.0; reduced_dimension * reduced_dimension];
        for row in 0..reduced_dimension {
            for col in 0..=row {
                let mut value = if row == col { 1.0 } else { 0.0 };
                for variable in 0..n {
                    value += columns[(variable, row)]
                        * inverse_diagonal[variable]
                        * columns[(variable, col)];
                }
                reduced[row * reduced_dimension + col] = value;
                reduced[col * reduced_dimension + row] = value;
            }
        }
        let reduced_cholesky = Cholesky::factor(&reduced, reduced_dimension)?;
        Ok(Self {
            inverse_diagonal,
            columns,
            reduced_cholesky,
        })
    }

    /// Applies `H^{-1}` in place via
    /// `H^{-1} = B^{-1} - B^{-1} C S^{-1} C' B^{-1}` with `B = diag(diagonal)`.
    pub(crate) fn solve_in_place(&self, right_hand_side: &mut [f64]) {
        let n = self.inverse_diagonal.len();
        let reduced_dimension = self.columns.cols();
        for (value, inverse) in right_hand_side.iter_mut().zip(&self.inverse_diagonal) {
            *value *= inverse;
        }
        let mut reduced_rhs = vec![0.0; reduced_dimension];
        for (col, reduced_value) in reduced_rhs.iter_mut().enumerate() {
            for (variable, right_value) in right_hand_side.iter().enumerate().take(n) {
                *reduced_value += self.columns[(variable, col)] * right_value;
            }
        }
        self.reduced_cholesky.solve_in_place(&mut reduced_rhs);
        for (variable, right_value) in right_hand_side.iter_mut().enumerate().take(n) {
            let correction: f64 = (0..reduced_dimension)
                .map(|col| self.columns[(variable, col)] * reduced_rhs[col])
                .sum();
            *right_value -= self.inverse_diagonal[variable] * correction;
        }
    }
}

/// Cholesky factorization of a symmetric positive-definite row-major matrix.
pub(crate) struct Cholesky {
    dimension: usize,
    lower: Vec<f64>,
}

impl Cholesky {
    pub(crate) fn factor(matrix: &[f64], dimension: usize) -> Result<Self, SolverError> {
        debug_assert_eq!(matrix.len(), dimension * dimension);
        let mut lower = vec![0.0; matrix.len()];
        for row in 0..dimension {
            for col in 0..=row {
                let product: f64 = (0..col)
                    .map(|inner| lower[row * dimension + inner] * lower[col * dimension + inner])
                    .sum();
                if row == col {
                    let pivot = matrix[row * dimension + row] - product;
                    if !pivot.is_finite() || pivot <= 1.0e-14 {
                        return Err(SolverError::LinearSystem);
                    }
                    lower[row * dimension + col] = pivot.sqrt();
                } else {
                    lower[row * dimension + col] =
                        (matrix[row * dimension + col] - product) / lower[col * dimension + col];
                }
            }
        }
        Ok(Self { dimension, lower })
    }

    pub(crate) fn solve_in_place(&self, right_hand_side: &mut [f64]) {
        debug_assert_eq!(right_hand_side.len(), self.dimension);
        for row in 0..self.dimension {
            let product: f64 = (0..row)
                .map(|col| self.lower[row * self.dimension + col] * right_hand_side[col])
                .sum();
            right_hand_side[row] =
                (right_hand_side[row] - product) / self.lower[row * self.dimension + row];
        }
        for row in (0..self.dimension).rev() {
            let product: f64 = (row + 1..self.dimension)
                .map(|other| self.lower[other * self.dimension + row] * right_hand_side[other])
                .sum();
            right_hand_side[row] =
                (right_hand_side[row] - product) / self.lower[row * self.dimension + row];
        }
    }
}
