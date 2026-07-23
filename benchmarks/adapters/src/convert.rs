//! Conversions from Ledge's factor-structured QP to general solver forms.
//!
//! Ledge, OSQP, and Clarabel all minimize `0.5 * x' P x + q' x`, so no
//! objective rescaling is needed anywhere in this module; only the constraint
//! encoding differs. Two formulations are produced:
//!
//! - [`Formulation::DenseQ`]: materializes `Q = F * omega * F' + diag(d)`
//!   as a dense upper triangle. This is what a general QP solver receives
//!   when the user does not exploit factor structure.
//! - [`Formulation::Lifted`]: adds `k` auxiliary variables `y = G' x` with
//!   `G = F * omega^{1/2}`, keeping the objective sparse
//!   (`0.5 x' diag(d) x + 0.5 y' y + q' x`) at the cost of `k` extra equality
//!   rows. This is the standard manual reformulation a sophisticated user
//!   would feed to a general sparse solver.
//!
//! An [`L1Term`](ledge_core::L1Term) (`sum_i c_i |x_i - a_i|`, exact L1
//! turnover) has no native encoding in a general QP solver, so both
//! formulations append the standard epigraph reformulation a user would
//! write by hand: `n` extra variables `t` with linear cost `c` and the `2n`
//! inequality rows `x_i - t_i <= a_i` and `-x_i - t_i <= -a_i`. Ledge itself
//! keeps the term as a proximal block with no extra rows — that asymmetry is
//! the measured claim, not an unfairness: each solver receives the best
//! encoding available to it (see `DECISIONS.md`, 2026-07-21). The epigraph
//! multipliers map back onto Ledge's L1 subgradient dual as
//! `lambda_i = y_upper_i - y_lower_i` (see
//! [`ConvertedQp::split_inequality_duals`]).

use ledge_core::{FactorCovariance, Matrix, QpProblem};

/// Value treated as infinity by OSQP's C core (`OSQP_INFTY`).
pub const OSQP_INFINITY: f64 = 1.0e30;

/// Which encoding external solvers receive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Formulation {
    /// Materialized dense `Q`; no structure exploitation.
    DenseQ,
    /// Factor-lifted sparse form with `k` auxiliary variables.
    Lifted,
}

impl Formulation {
    /// Short label used in report rows.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::DenseQ => "dense-Q",
            Self::Lifted => "lifted",
        }
    }
}

/// Minimal compressed-sparse-column matrix with sorted row indices.
#[derive(Clone, Debug, PartialEq)]
pub struct CscData {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Column offsets into `indices`/`values`; length `cols + 1`.
    pub indptr: Vec<usize>,
    /// Row index of each stored entry.
    pub indices: Vec<usize>,
    /// Value of each stored entry.
    pub values: Vec<f64>,
}

/// One sparse constraint row as `(column, coefficient)` pairs.
pub type SparseRow = Vec<(usize, f64)>;

/// A solver-neutral converted problem.
///
/// Adapters stack `equality_rows`, `inequality_rows`, and per-variable box
/// bounds into their native constraint containers.
#[derive(Clone, Debug)]
pub struct ConvertedQp {
    /// Formulation used to produce this conversion.
    pub formulation: Formulation,
    /// Number of decision variables in the original problem.
    pub original_variables: usize,
    /// Number of variables the external solver sees (`n` or `n + k`).
    pub variables: usize,
    /// Upper triangle of the quadratic cost in CSC form.
    pub quadratic_upper: CscData,
    /// Linear cost over `variables` entries (zero on lifted variables).
    pub linear: Vec<f64>,
    /// Equality rows: the original equalities followed by lifted rows.
    pub equality_rows: Vec<SparseRow>,
    /// Right-hand side of `equality_rows`.
    pub equality_rhs: Vec<f64>,
    /// Number of trailing lifted rows inside `equality_rows`.
    pub lifted_rows: usize,
    /// Upper-inequality rows (`row * z <= rhs`), original rows first,
    /// epigraph rows last.
    pub inequality_rows: Vec<SparseRow>,
    /// Right-hand side of `inequality_rows`.
    pub inequality_rhs: Vec<f64>,
    /// Number of trailing epigraph variables (`n` for problems with an L1
    /// term, `0` otherwise).
    pub epigraph_variables: usize,
    /// Number of trailing epigraph rows inside `inequality_rows` (`2n` for
    /// problems with an L1 term): the `n` upper rows `x_i - t_i <= a_i`
    /// followed by the `n` lower rows `-x_i - t_i <= -a_i`.
    pub epigraph_rows: usize,
    /// Box lower bounds on the first `original_variables` variables.
    pub lower_bounds: Vec<f64>,
    /// Box upper bounds on the first `original_variables` variables.
    pub upper_bounds: Vec<f64>,
    /// `G = F * omega^{1/2}` when lifted; used to lift primal warm starts.
    lift_columns: Option<Matrix>,
    /// L1 anchor; used to lift primal starts onto the epigraph variables.
    epigraph_anchor: Option<Vec<f64>>,
    /// Linear cost on the variables beyond the original ones (zeros on
    /// lifted variables, L1 costs on epigraph variables); reappended by
    /// [`Self::extend_linear`] during rolling updates.
    tail_linear: Vec<f64>,
}

impl ConvertedQp {
    /// Converts a validated Ledge problem into the requested formulation.
    ///
    /// # Errors
    ///
    /// Returns a human-readable message when the problem is invalid or a
    /// dense factor covariance is not positive semidefinite.
    pub fn new(problem: &QpProblem, formulation: Formulation) -> Result<Self, String> {
        problem.validate().map_err(|error| error.to_string())?;
        let n = problem.quadratic.dimension();
        let k = problem.quadratic.factor_count();
        let lift = covariance_root_columns(problem)?;

        let mut equality_rows: Vec<SparseRow> = Vec::new();
        let mut equality_rhs = Vec::new();
        for row in 0..problem.equalities.len() {
            equality_rows.push(dense_row(problem.equalities.matrix.row(row)));
            equality_rhs.push(problem.equalities.rhs[row]);
        }
        let mut inequality_rows: Vec<SparseRow> = Vec::new();
        let mut inequality_rhs = Vec::new();
        for row in 0..problem.inequalities.len() {
            inequality_rows.push(dense_row(problem.inequalities.matrix.row(row)));
            inequality_rhs.push(problem.inequalities.rhs[row]);
        }

        let (mut variables, mut quadratic_upper, mut linear, lifted_rows, lift_columns) =
            match formulation {
                Formulation::DenseQ => {
                    let quadratic_upper = dense_q_upper(problem, &lift);
                    (n, quadratic_upper, problem.linear.clone(), 0, None)
                }
                Formulation::Lifted => {
                    // Objective: 0.5 x' diag(d) x + 0.5 y' y + q' x.
                    let mut indptr = Vec::with_capacity(n + k + 1);
                    let mut indices = Vec::with_capacity(n + k);
                    let mut values = Vec::with_capacity(n + k);
                    indptr.push(0);
                    for (col, diagonal) in problem.quadratic.diagonal.iter().enumerate() {
                        indices.push(col);
                        values.push(*diagonal);
                        indptr.push(values.len());
                    }
                    for col in 0..k {
                        indices.push(n + col);
                        values.push(1.0);
                        indptr.push(values.len());
                    }
                    let quadratic_upper = CscData {
                        rows: n + k,
                        cols: n + k,
                        indptr,
                        indices,
                        values,
                    };
                    let mut linear = problem.linear.clone();
                    linear.resize(n + k, 0.0);
                    // Lifted rows: G' x - y = 0, one row per factor column.
                    for factor in 0..k {
                        let mut row: SparseRow = Vec::with_capacity(n + 1);
                        for variable in 0..n {
                            let value = lift[(variable, factor)];
                            if value != 0.0 {
                                row.push((variable, value));
                            }
                        }
                        row.push((n + factor, -1.0));
                        equality_rows.push(row);
                        equality_rhs.push(0.0);
                    }
                    (n + k, quadratic_upper, linear, k, Some(lift))
                }
            };

        let mut epigraph_variables = 0;
        let mut epigraph_rows = 0;
        let mut epigraph_anchor = None;
        if let Some(term) = &problem.l1 {
            // Epigraph block: minimize `c' t` subject to `|x_i - a_i| <= t_i`
            // written as two upper-inequality rows per asset. The `t` block
            // sits after every smooth variable so existing row/column
            // indices are unchanged.
            let t_offset = variables;
            epigraph_variables = n;
            epigraph_rows = 2 * n;
            variables += n;
            quadratic_upper.rows = variables;
            quadratic_upper.cols = variables;
            let filled = quadratic_upper.indptr.last().copied().unwrap_or(0);
            quadratic_upper.indptr.resize(variables + 1, filled);
            linear.extend_from_slice(&term.costs);
            for (asset, anchor) in term.anchor.iter().enumerate() {
                inequality_rows.push(vec![(asset, 1.0), (t_offset + asset, -1.0)]);
                inequality_rhs.push(*anchor);
            }
            for (asset, anchor) in term.anchor.iter().enumerate() {
                inequality_rows.push(vec![(asset, -1.0), (t_offset + asset, -1.0)]);
                inequality_rhs.push(-anchor);
            }
            epigraph_anchor = Some(term.anchor.clone());
        }
        let tail_linear = linear[n..].to_vec();

        Ok(Self {
            formulation,
            original_variables: n,
            variables,
            quadratic_upper,
            linear,
            equality_rows,
            equality_rhs,
            lifted_rows,
            inequality_rows,
            inequality_rhs,
            epigraph_variables,
            epigraph_rows,
            lower_bounds: problem.lower_bounds.clone(),
            upper_bounds: problem.upper_bounds.clone(),
            lift_columns,
            epigraph_anchor,
            tail_linear,
        })
    }

    /// Extends an original-space primal point with lifted and epigraph
    /// variables (`y = G' x`, `t_i = |x_i - a_i|`).
    #[must_use]
    pub fn lift_primal(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.original_variables);
        let mut lifted = x.to_vec();
        if let Some(columns) = &self.lift_columns {
            for factor in 0..columns.cols() {
                let mut value = 0.0;
                for (variable, weight) in x.iter().enumerate() {
                    value += columns[(variable, factor)] * weight;
                }
                lifted.push(value);
            }
        }
        if let Some(anchor) = &self.epigraph_anchor {
            for (value, anchor) in x.iter().zip(anchor) {
                lifted.push((value - anchor).abs());
            }
        }
        lifted
    }

    /// Extends an original-space linear cost with zeros on lifted variables
    /// and the L1 costs on epigraph variables.
    #[must_use]
    pub fn extend_linear(&self, linear: &[f64]) -> Vec<f64> {
        debug_assert_eq!(linear.len(), self.original_variables);
        let mut extended = linear.to_vec();
        extended.extend_from_slice(&self.tail_linear);
        extended
    }

    /// Splits the multipliers of the full inequality block into the
    /// original-row multipliers and Ledge's L1 subgradient multipliers.
    ///
    /// `duals` must cover every row of `inequality_rows` in order. With an
    /// epigraph block, stationarity in `x_i` carries `+y_upper_i` from
    /// `x_i - t_i <= a_i` and `-y_lower_i` from `-x_i - t_i <= -a_i`, so the
    /// L1 multiplier in Ledge's convention is `y_upper_i - y_lower_i`; the
    /// `t_i` row of stationarity (`c_i - y_upper_i - y_lower_i = 0`) keeps it
    /// inside `[-c_i, c_i]`. Returns an empty L1 vector for problems without
    /// an L1 term, matching [`DualVariables`](ledge_core::DualVariables).
    #[must_use]
    pub fn split_inequality_duals(&self, duals: &[f64]) -> (Vec<f64>, Vec<f64>) {
        debug_assert_eq!(duals.len(), self.inequality_rows.len());
        let original = self.inequality_rows.len() - self.epigraph_rows;
        let inequalities = duals[..original].to_vec();
        let l1 = (0..self.epigraph_variables)
            .map(|asset| {
                duals[original + asset] - duals[original + self.epigraph_variables + asset]
            })
            .collect();
        (inequalities, l1)
    }

    /// Builds a CSC matrix from stacked sparse rows over `self.variables`
    /// columns.
    #[must_use]
    pub fn csc_from_rows(&self, row_blocks: &[&[SparseRow]]) -> CscData {
        let rows: usize = row_blocks.iter().map(|block| block.len()).sum();
        let cols = self.variables;
        let mut column_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); cols];
        let mut row_offset = 0;
        for block in row_blocks {
            for (local_row, row) in block.iter().enumerate() {
                for (column, value) in row {
                    column_entries[*column].push((row_offset + local_row, *value));
                }
            }
            row_offset += block.len();
        }
        let mut indptr = Vec::with_capacity(cols + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        indptr.push(0);
        for mut entries in column_entries {
            entries.sort_unstable_by_key(|(row, _)| *row);
            for (row, value) in entries {
                indices.push(row);
                values.push(value);
            }
            indptr.push(values.len());
        }
        CscData {
            rows,
            cols,
            indptr,
            indices,
            values,
        }
    }
}

fn dense_row(coefficients: &[f64]) -> SparseRow {
    coefficients
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0.0)
        .map(|(column, value)| (column, *value))
        .collect()
}

/// Upper triangle of `Q = G G' + diag(d)` in CSC (column-major) order.
fn dense_q_upper(problem: &QpProblem, lift: &Matrix) -> CscData {
    let n = problem.quadratic.dimension();
    let k = lift.cols();
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0);
    for col in 0..n {
        for row in 0..=col {
            let mut value: f64 = (0..k).map(|f| lift[(row, f)] * lift[(col, f)]).sum();
            if row == col {
                value += problem.quadratic.diagonal[row];
            }
            if value != 0.0 || row == col {
                indices.push(row);
                values.push(value);
            }
        }
        indptr.push(values.len());
    }
    CscData {
        rows: n,
        cols: n,
        indptr,
        indices,
        values,
    }
}

/// Returns `G` with `F * omega * F' = G * G'`.
///
/// Mirrors the (crate-private) factor root used inside `ledge-core`; kept
/// local so the benchmark harness never depends on solver internals.
fn covariance_root_columns(problem: &QpProblem) -> Result<Matrix, String> {
    let n = problem.quadratic.dimension();
    let k = problem.quadratic.factor_count();
    let mut columns = Matrix::zeros(n, k);
    match &problem.quadratic.omega {
        FactorCovariance::Diagonal(diagonal) => {
            for factor in 0..k {
                let root = diagonal[factor].sqrt();
                for variable in 0..n {
                    columns[(variable, factor)] =
                        problem.quadratic.factors[(variable, factor)] * root;
                }
            }
        }
        FactorCovariance::Dense(omega) => {
            let lower = semidefinite_cholesky(omega)?;
            for variable in 0..n {
                for factor in 0..k {
                    let mut value = 0.0;
                    for inner in factor..k {
                        value +=
                            problem.quadratic.factors[(variable, inner)] * lower[(inner, factor)];
                    }
                    columns[(variable, factor)] = value;
                }
            }
        }
    }
    Ok(columns)
}

fn semidefinite_cholesky(matrix: &Matrix) -> Result<Matrix, String> {
    let dimension = matrix.rows();
    let mut lower = Matrix::zeros(dimension, dimension);
    let scale = matrix
        .as_slice()
        .iter()
        .fold(1.0_f64, |largest, value| largest.max(value.abs()));
    let tolerance = 1.0e-12 * scale;
    for col in 0..dimension {
        let remainder = matrix[(col, col)]
            - (0..col)
                .map(|inner| lower[(col, inner)].powi(2))
                .sum::<f64>();
        if remainder < -tolerance {
            return Err("dense factor covariance is not positive semidefinite".to_owned());
        }
        if remainder <= tolerance {
            for row in col + 1..dimension {
                let off = matrix[(row, col)]
                    - (0..col)
                        .map(|inner| lower[(row, inner)] * lower[(col, inner)])
                        .sum::<f64>();
                if off.abs() > tolerance {
                    return Err("dense factor covariance is not positive semidefinite".to_owned());
                }
            }
            continue;
        }
        lower[(col, col)] = remainder.sqrt();
        for row in col + 1..dimension {
            let product: f64 = (0..col)
                .map(|inner| lower[(row, inner)] * lower[(col, inner)])
                .sum();
            lower[(row, col)] = (matrix[(row, col)] - product) / lower[(col, col)];
        }
    }
    Ok(lower)
}
