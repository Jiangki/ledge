//! A small row-major dense matrix used at the API boundary.
//!
//! Ledge deliberately owns this type instead of exposing a particular linear
//! algebra dependency. The solver only forms matrices whose smaller dimension
//! is the number of factors plus explicit linear constraints.

use std::ops::{Index, IndexMut};

use thiserror::Error;

/// Errors raised while constructing a [`Matrix`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MatrixError {
    /// The requested element count overflows `usize`.
    #[error("matrix dimensions {rows}x{cols} overflow the address space")]
    DimensionsOverflow {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        cols: usize,
    },
    /// The data length does not match `rows * cols`.
    #[error("matrix shape {rows}x{cols} requires {expected} values, got {actual}")]
    InvalidShape {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        cols: usize,
        /// Required number of values.
        expected: usize,
        /// Supplied number of values.
        actual: usize,
    },
    /// Rows supplied to `from_rows` do not have equal lengths.
    #[error("matrix row {row} has {actual} columns; expected {expected}")]
    RaggedRows {
        /// Zero-based row index.
        row: usize,
        /// Expected column count.
        expected: usize,
        /// Actual column count.
        actual: usize,
    },
}

/// A contiguous row-major dense matrix.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "MatrixData", into = "MatrixData")
)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

/// Wire format for [`Matrix`]: deserialization rebuilds through
/// [`Matrix::new`], so shape and storage can never disagree.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct MatrixData {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

#[cfg(feature = "serde")]
impl TryFrom<MatrixData> for Matrix {
    type Error = MatrixError;

    fn try_from(data: MatrixData) -> Result<Self, Self::Error> {
        Self::new(data.rows, data.cols, data.data)
    }
}

#[cfg(feature = "serde")]
impl From<Matrix> for MatrixData {
    fn from(matrix: Matrix) -> Self {
        Self {
            rows: matrix.rows,
            cols: matrix.cols,
            data: matrix.data,
        }
    }
}

impl Matrix {
    /// Builds a matrix from row-major data.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::DimensionsOverflow`] if the dimensions overflow,
    /// or [`MatrixError::InvalidShape`] when the data length is wrong.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(MatrixError::DimensionsOverflow { rows, cols })?;
        if data.len() != expected {
            return Err(MatrixError::InvalidShape {
                rows,
                cols,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// Builds a matrix from nested rows.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::RaggedRows`] if row lengths differ.
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self, MatrixError> {
        let row_count = rows.len();
        let cols = rows.first().map_or(0, Vec::len);
        let mut data = Vec::with_capacity(row_count.saturating_mul(cols));
        for (row_index, row) in rows.into_iter().enumerate() {
            if row.len() != cols {
                return Err(MatrixError::RaggedRows {
                    row: row_index,
                    expected: cols,
                    actual: row.len(),
                });
            }
            data.extend(row);
        }
        Ok(Self {
            rows: row_count,
            cols,
            data,
        })
    }

    /// Returns a zero-filled matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows.saturating_mul(cols)],
        }
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major backing storage.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable row-major backing storage.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Returns one row.
    #[must_use]
    pub fn row(&self, row: usize) -> &[f64] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    pub(crate) fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(self.cols, x.len());
        (0..self.rows).map(|row| dot(self.row(row), x)).collect()
    }

    pub(crate) fn transpose_mul_add(&self, x: &[f64], output: &mut [f64]) {
        debug_assert_eq!(self.rows, x.len());
        debug_assert_eq!(self.cols, output.len());
        for (row, &scale) in x.iter().enumerate() {
            for col in 0..self.cols {
                output[col] += self[(row, col)] * scale;
            }
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}

pub(crate) fn dot(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

pub(crate) fn norm_inf(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |largest, value| {
        if value.is_nan() {
            f64::NAN
        } else {
            largest.max(value.abs())
        }
    })
}
