//! Ergonomic factor-model portfolio problem construction.

use thiserror::Error;

use crate::{
    certificate::{Certificate, DualCertificate, PrimalCertificate},
    problem::L1Term,
    sequence::PortfolioSequence,
    FactorCovariance, FactorQuad, LinearConstraints, Matrix, MatrixError, ProblemError, QpProblem,
    Solution, Solver, SolverError, SolverSettings, WarmStart,
};

/// A mean-variance portfolio problem backed by a factor covariance.
///
/// The objective is
///
/// ```text
/// risk_aversion / 2 * (w - b)' Σ (w - b) - expected_returns' w
/// + turnover_penalty / 2 * ||w - previous_weights||²
/// + l1_turnover_costs' |w - previous_weights|
/// ```
///
/// where the benchmark `b` defaults to zero (absolute risk) and is set by
/// [`PortfolioProblem::with_tracking_benchmark`] for tracking-error
/// problems. The benchmark only shifts the linear term — the QP structure
/// is unchanged — and the constant `risk_aversion / 2 * b' Σ b` is dropped
/// from reported objective values, as is conventional for QP solvers.
///
/// The quadratic turnover term is an L2 approximation useful when a smooth
/// preference for stable weights is acceptable
/// ([`PortfolioProblem::with_quadratic_turnover`]). The L1 term models
/// proportional transaction costs exactly
/// ([`PortfolioProblem::with_l1_turnover`]); it is handled by a dedicated
/// proximal block inside the solver, so it never grows the reduced system.
/// Both may be combined; they share one previous portfolio.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "PortfolioProblemData", into = "PortfolioProblemData")
)]
pub struct PortfolioProblem {
    covariance: FactorQuad,
    expected_returns: Vec<f64>,
    risk_aversion: f64,
    budget: Option<f64>,
    equalities: LinearConstraints,
    inequalities: LinearConstraints,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    previous_weights: Option<Vec<f64>>,
    turnover_penalty: f64,
    l1_turnover_costs: Option<Vec<f64>>,
    benchmark_weights: Option<Vec<f64>>,
}

/// Wire format for [`PortfolioProblem`] (roadmap 3.3): the builder inputs,
/// replayed through the builder methods on deserialization so a dump can
/// never smuggle in data that construction would have rejected.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct PortfolioProblemData {
    covariance: FactorQuad,
    expected_returns: Vec<f64>,
    risk_aversion: f64,
    budget: Option<f64>,
    equalities: LinearConstraints,
    inequalities: LinearConstraints,
    #[serde(with = "crate::serde_support::lower_bounds")]
    lower_bounds: Vec<f64>,
    #[serde(with = "crate::serde_support::upper_bounds")]
    upper_bounds: Vec<f64>,
    previous_weights: Option<Vec<f64>>,
    turnover_penalty: f64,
    l1_turnover_costs: Option<Vec<f64>>,
    benchmark_weights: Option<Vec<f64>>,
}

#[cfg(feature = "serde")]
impl TryFrom<PortfolioProblemData> for PortfolioProblem {
    type Error = PortfolioError;

    fn try_from(data: PortfolioProblemData) -> Result<Self, Self::Error> {
        let mut problem = Self::new(
            data.covariance.factors,
            data.covariance.omega,
            data.covariance.diagonal,
            data.expected_returns,
        )?
        .with_risk_aversion(data.risk_aversion)?
        .with_budget(data.budget)?
        .with_bounds(data.lower_bounds, data.upper_bounds)?
        .with_equalities(data.equalities.matrix, data.equalities.rhs)?
        .with_inequalities(data.inequalities.matrix, data.inequalities.rhs)?;
        if let Some(previous_weights) = data.previous_weights {
            problem =
                problem.with_quadratic_turnover(previous_weights.clone(), data.turnover_penalty)?;
            if let Some(costs) = data.l1_turnover_costs {
                problem = problem.with_l1_turnover(previous_weights, costs)?;
            }
        } else if data.turnover_penalty != 0.0 || data.l1_turnover_costs.is_some() {
            return Err(PortfolioError::InvalidParameter(
                "turnover terms require previous_weights",
            ));
        }
        if let Some(benchmark_weights) = data.benchmark_weights {
            problem = problem.with_tracking_benchmark(benchmark_weights)?;
        }
        Ok(problem)
    }
}

#[cfg(feature = "serde")]
impl From<PortfolioProblem> for PortfolioProblemData {
    fn from(problem: PortfolioProblem) -> Self {
        Self {
            covariance: problem.covariance,
            expected_returns: problem.expected_returns,
            risk_aversion: problem.risk_aversion,
            budget: problem.budget,
            equalities: problem.equalities,
            inequalities: problem.inequalities,
            lower_bounds: problem.lower_bounds,
            upper_bounds: problem.upper_bounds,
            previous_weights: problem.previous_weights,
            turnover_penalty: problem.turnover_penalty,
            l1_turnover_costs: problem.l1_turnover_costs,
            benchmark_weights: problem.benchmark_weights,
        }
    }
}

impl PortfolioProblem {
    /// Creates a long-only, fully invested mean-variance problem.
    ///
    /// Defaults are risk aversion `1`, budget `1`, and bounds `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error when covariance data or expected-return dimensions are
    /// invalid.
    pub fn new(
        factors: Matrix,
        omega: FactorCovariance,
        specific_variance: Vec<f64>,
        expected_returns: Vec<f64>,
    ) -> Result<Self, PortfolioError> {
        let covariance = FactorQuad::new(factors, omega, specific_variance)?;
        let dimension = covariance.dimension();
        validate_vector(
            "expected_returns",
            &expected_returns,
            dimension,
            FinitePolicy::Finite,
        )?;
        Ok(Self {
            covariance,
            expected_returns,
            risk_aversion: 1.0,
            budget: Some(1.0),
            equalities: LinearConstraints::empty(dimension),
            inequalities: LinearConstraints::empty(dimension),
            lower_bounds: vec![0.0; dimension],
            upper_bounds: vec![1.0; dimension],
            previous_weights: None,
            turnover_penalty: 0.0,
            l1_turnover_costs: None,
            benchmark_weights: None,
        })
    }

    /// Number of portfolio weights.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.covariance.dimension()
    }

    /// Sets the positive multiplier on portfolio variance.
    ///
    /// # Errors
    ///
    /// Returns an error unless `risk_aversion` is finite and positive.
    pub fn with_risk_aversion(mut self, risk_aversion: f64) -> Result<Self, PortfolioError> {
        if !risk_aversion.is_finite() || risk_aversion <= 0.0 {
            return Err(PortfolioError::InvalidParameter(
                "risk_aversion must be finite and positive",
            ));
        }
        self.risk_aversion = risk_aversion;
        Ok(self)
    }

    /// Sets the budget equality, or removes it with `None`.
    ///
    /// # Errors
    ///
    /// Returns an error when a supplied budget is not finite.
    pub fn with_budget(mut self, budget: Option<f64>) -> Result<Self, PortfolioError> {
        if budget.is_some_and(|value| !value.is_finite()) {
            return Err(PortfolioError::InvalidParameter(
                "budget must be finite when provided",
            ));
        }
        self.budget = budget;
        Ok(self)
    }

    /// Replaces the per-asset box constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for wrong dimensions, NaNs, or lower bounds above
    /// upper bounds.
    pub fn with_bounds(
        mut self,
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
    ) -> Result<Self, PortfolioError> {
        let dimension = self.dimension();
        validate_vector(
            "lower_bounds",
            &lower_bounds,
            dimension,
            FinitePolicy::AllowInfinity,
        )?;
        validate_vector(
            "upper_bounds",
            &upper_bounds,
            dimension,
            FinitePolicy::AllowInfinity,
        )?;
        for index in 0..dimension {
            if lower_bounds[index] > upper_bounds[index] {
                return Err(ProblemError::InvalidBounds {
                    index,
                    lower: lower_bounds[index],
                    upper: upper_bounds[index],
                }
                .into());
            }
        }
        self.lower_bounds = lower_bounds;
        self.upper_bounds = upper_bounds;
        Ok(self)
    }

    /// Adds user-supplied equality constraints alongside the optional budget.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions or coefficients are invalid.
    pub fn with_equalities(
        mut self,
        matrix: Matrix,
        rhs: Vec<f64>,
    ) -> Result<Self, PortfolioError> {
        self.equalities = validated_constraints("equalities", matrix, rhs, self.dimension())?;
        Ok(self)
    }

    /// Adds upper-form linear constraints `matrix * weights <= rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions or coefficients are invalid.
    pub fn with_inequalities(
        mut self,
        matrix: Matrix,
        rhs: Vec<f64>,
    ) -> Result<Self, PortfolioError> {
        self.inequalities = validated_constraints("inequalities", matrix, rhs, self.dimension())?;
        Ok(self)
    }

    /// Adds an L2 penalty around the previous portfolio.
    ///
    /// This is not proportional transaction cost — use
    /// [`PortfolioProblem::with_l1_turnover`] for that. The problem models a
    /// single previous portfolio: calling this after `with_l1_turnover`
    /// replaces the shared anchor.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong-length/non-finite previous portfolio or a
    /// negative/non-finite penalty.
    pub fn with_quadratic_turnover(
        mut self,
        previous_weights: Vec<f64>,
        turnover_penalty: f64,
    ) -> Result<Self, PortfolioError> {
        validate_vector(
            "previous_weights",
            &previous_weights,
            self.dimension(),
            FinitePolicy::Finite,
        )?;
        if !turnover_penalty.is_finite() || turnover_penalty < 0.0 {
            return Err(PortfolioError::InvalidParameter(
                "turnover_penalty must be finite and non-negative",
            ));
        }
        self.previous_weights = Some(previous_weights);
        self.turnover_penalty = turnover_penalty;
        Ok(self)
    }

    /// Adds exact proportional transaction costs
    /// `costs' |w - previous_weights|` around the previous portfolio
    /// (roadmap 2.1).
    ///
    /// Unlike the L2 approximation, this is the real trading-cost model: a
    /// per-asset charge proportional to the traded amount, with a genuine
    /// no-trade region. The term is handled by a dedicated soft-threshold
    /// proximal block, so the reduced factorization keeps its
    /// `factors + constraints` dimension — an epigraph reformulation would
    /// add `2n` constraint rows instead.
    ///
    /// May be combined with [`PortfolioProblem::with_quadratic_turnover`];
    /// the two terms share a single previous portfolio, and calling either
    /// method replaces that shared anchor.
    ///
    /// # Errors
    ///
    /// Returns an error for wrong-length or non-finite previous weights, or
    /// costs that are not finite and non-negative.
    pub fn with_l1_turnover(
        mut self,
        previous_weights: Vec<f64>,
        costs: Vec<f64>,
    ) -> Result<Self, PortfolioError> {
        validate_vector(
            "previous_weights",
            &previous_weights,
            self.dimension(),
            FinitePolicy::Finite,
        )?;
        validate_vector(
            "l1_turnover_costs",
            &costs,
            self.dimension(),
            FinitePolicy::Finite,
        )?;
        if costs.iter().any(|value| *value < 0.0) {
            return Err(PortfolioError::InvalidParameter(
                "l1 turnover costs must be non-negative",
            ));
        }
        self.previous_weights = Some(previous_weights);
        self.l1_turnover_costs = Some(costs);
        Ok(self)
    }

    /// Sets the benchmark for a tracking-error objective (roadmap 2.6).
    ///
    /// The risk term becomes `risk_aversion / 2 * (w - b)' Σ (w - b)`:
    /// active risk against the benchmark instead of absolute risk. It is the
    /// same QP underneath — expanding the square only shifts the linear
    /// objective by `-risk_aversion * Σ b` — so no solver machinery changes
    /// and rolling sequences keep their cached factorizations when the
    /// benchmark moves.
    ///
    /// The constant `risk_aversion / 2 * b' Σ b` is dropped from reported
    /// objective values, as is conventional for QP solvers; add it back if
    /// an absolute tracking-error number is needed.
    ///
    /// The benchmark does not need to satisfy the portfolio's constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for wrong-length or non-finite benchmark weights.
    pub fn with_tracking_benchmark(
        mut self,
        benchmark_weights: Vec<f64>,
    ) -> Result<Self, PortfolioError> {
        validate_vector(
            "benchmark_weights",
            &benchmark_weights,
            self.dimension(),
            FinitePolicy::Finite,
        )?;
        self.benchmark_weights = Some(benchmark_weights);
        Ok(self)
    }

    /// Adds industry-neutrality equalities derived from the tracking
    /// benchmark (roadmap 3.1).
    ///
    /// `industries[i]` is the zero-based industry id of asset `i`; industry
    /// ids run from `0` to `max(industries)`. One equality row is appended
    /// per industry, pinning the portfolio's industry weight to the
    /// benchmark's:
    ///
    /// ```text
    /// sum_{i in industry g} w_i = sum_{i in industry g} b_i
    /// ```
    ///
    /// which is exactly "zero net active industry exposure". Requires
    /// [`PortfolioProblem::with_tracking_benchmark`] to have been called
    /// first; for explicit targets (or without a benchmark) use
    /// [`PortfolioProblem::with_group_targets`].
    ///
    /// The rows are appended after any existing user equality rows, one per
    /// industry in id order, and count as user equality rows from then on:
    /// rolling sequences move the targets through
    /// [`RebalanceStep::equality_rhs`](crate::RebalanceStep) like any other
    /// equality target. Call
    /// [`PortfolioProblem::with_equalities`] *before* this method — it
    /// replaces the user equality block.
    ///
    /// # Errors
    ///
    /// Returns an error when no tracking benchmark is set, `industries` has
    /// the wrong length, or an industry has no member assets.
    pub fn with_industry_neutrality(self, industries: &[usize]) -> Result<Self, PortfolioError> {
        let Some(benchmark_weights) = self.benchmark_weights.clone() else {
            return Err(PortfolioError::Template(
                "industry neutrality derives its targets from the tracking benchmark; \
                 call with_tracking_benchmark first or supply explicit targets with \
                 with_group_targets"
                    .to_owned(),
            ));
        };
        validate_group_ids("industries", industries, self.dimension(), None)?;
        let group_count = industries.iter().copied().max().map_or(0, |max| max + 1);
        let mut targets = vec![0.0; group_count];
        for (asset, group) in industries.iter().enumerate() {
            targets[*group] += benchmark_weights[asset];
        }
        self.append_group_equalities("industries", industries, &targets)
    }

    /// Adds group-weight equalities: one row per group pinning the summed
    /// member weights to an explicit target (roadmap 3.1).
    ///
    /// `groups[i]` is the zero-based group id of asset `i` and must be
    /// smaller than `targets.len()`; group `g` contributes the row
    ///
    /// ```text
    /// sum_{i in group g} w_i = targets[g]
    /// ```
    ///
    /// This is the explicit-target form of
    /// [`PortfolioProblem::with_industry_neutrality`] and also covers
    /// sector/country sleeves or a zero-net-exposure book
    /// (`targets = [0.0, ...]`).
    ///
    /// The rows are appended after any existing user equality rows, one per
    /// group in id order, and count as user equality rows from then on:
    /// rolling sequences move the targets through
    /// [`RebalanceStep::equality_rhs`](crate::RebalanceStep). Call
    /// [`PortfolioProblem::with_equalities`] *before* this method — it
    /// replaces the user equality block.
    ///
    /// # Errors
    ///
    /// Returns an error when `groups` has the wrong length or an id out of
    /// range, a target is non-finite, or a group has no member assets.
    pub fn with_group_targets(
        self,
        groups: &[usize],
        targets: &[f64],
    ) -> Result<Self, PortfolioError> {
        validate_group_ids("groups", groups, self.dimension(), Some(targets.len()))?;
        if targets.iter().any(|value| !value.is_finite()) {
            return Err(ProblemError::NonFinite("group targets").into());
        }
        self.append_group_equalities("groups", groups, targets)
    }

    /// Adds style-exposure bands `lower[s] <= exposures[s] · w <= upper[s]`
    /// as linear constraint rows (roadmap 3.1).
    ///
    /// `exposures` is styles-by-assets: row `s` holds the per-asset loadings
    /// of style `s` (momentum, value, size, ...). Use
    /// `f64::NEG_INFINITY` / `f64::INFINITY` for one-sided bands. A band
    /// with `lower[s] == upper[s]` becomes a single equality row (style
    /// neutrality); other bands append their finite sides as inequality
    /// rows.
    ///
    /// Appended row order, per style in index order: the equality row (when
    /// the band is exact), otherwise the upper row
    /// `exposures[s] · w <= upper[s]` followed by the lower row
    /// `-exposures[s] · w <= -lower[s]`, skipping infinite sides. The rows
    /// count as user constraint rows from then on: rolling sequences move
    /// the bands through
    /// [`RebalanceStep::equality_rhs`](crate::RebalanceStep) /
    /// [`RebalanceStep::inequality_rhs`](crate::RebalanceStep) (mind the
    /// sign convention on lower rows). Call
    /// [`PortfolioProblem::with_equalities`] /
    /// [`PortfolioProblem::with_inequalities`] *before* this method — they
    /// replace their constraint blocks.
    ///
    /// # Errors
    ///
    /// Returns an error for dimension mismatches, non-finite loadings, NaN
    /// or crossing bounds, or a style with neither side finite.
    pub fn with_style_bounds(
        mut self,
        exposures: &Matrix,
        lower: &[f64],
        upper: &[f64],
    ) -> Result<Self, PortfolioError> {
        let dimension = self.dimension();
        let styles = exposures.rows();
        if exposures.cols() != dimension {
            return Err(ProblemError::Dimension {
                field: "style exposures",
                expected: dimension,
                actual: exposures.cols(),
            }
            .into());
        }
        if exposures.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(ProblemError::NonFinite("style exposures").into());
        }
        validate_vector("style lower", lower, styles, FinitePolicy::AllowInfinity)?;
        validate_vector("style upper", upper, styles, FinitePolicy::AllowInfinity)?;

        let mut equality_rows = Vec::new();
        let mut equality_rhs = Vec::new();
        let mut inequality_rows = Vec::new();
        let mut inequality_rhs = Vec::new();
        for style in 0..styles {
            let (low, high) = (lower[style], upper[style]);
            if low > high {
                return Err(PortfolioError::Template(format!(
                    "style {style} bounds cross: lower {low} exceeds upper {high}"
                )));
            }
            // Exact equality is deliberate here: a band collapses to an
            // equality row only when the caller passed identical bounds.
            if low >= high {
                equality_rows.push(exposures.row(style).to_vec());
                equality_rhs.push(high);
                continue;
            }
            if !low.is_finite() && !high.is_finite() {
                return Err(PortfolioError::Template(format!(
                    "style {style} has neither a finite lower nor a finite upper bound; \
                     drop the row instead of leaving it unconstrained"
                )));
            }
            if high.is_finite() {
                inequality_rows.push(exposures.row(style).to_vec());
                inequality_rhs.push(high);
            }
            if low.is_finite() {
                inequality_rows.push(exposures.row(style).iter().map(|value| -value).collect());
                inequality_rhs.push(-low);
            }
        }
        if !equality_rows.is_empty() {
            self.equalities =
                append_constraint_rows(&self.equalities, equality_rows, equality_rhs)?;
        }
        if !inequality_rows.is_empty() {
            self.inequalities =
                append_constraint_rows(&self.inequalities, inequality_rows, inequality_rhs)?;
        }
        Ok(self)
    }

    /// Caps every asset's absolute weight: `|w_i| <= max_weight`
    /// (roadmap 3.1).
    ///
    /// Concentration limits map onto the existing box constraints — upper
    /// bounds are tightened to `min(upper_i, max_weight)` and lower bounds
    /// raised to `max(lower_i, -max_weight)` — so no constraint rows are
    /// added and the reduced factorization keeps its dimension. Long-only
    /// problems (default bounds `[0, 1]`) end up with `w_i <= max_weight`.
    ///
    /// Apply after [`PortfolioProblem::with_bounds`]; the tightening keeps
    /// whichever side is already stricter.
    ///
    /// # Errors
    ///
    /// Returns an error unless `max_weight` is finite and positive, or when
    /// the cap contradicts an existing bound (for example a forced minimum
    /// position above the cap).
    pub fn with_concentration_limit(self, max_weight: f64) -> Result<Self, PortfolioError> {
        if !max_weight.is_finite() || max_weight <= 0.0 {
            return Err(PortfolioError::InvalidParameter(
                "max_weight must be finite and positive",
            ));
        }
        let lower: Vec<f64> = self
            .lower_bounds
            .iter()
            .map(|value| value.max(-max_weight))
            .collect();
        let upper: Vec<f64> = self
            .upper_bounds
            .iter()
            .map(|value| value.min(max_weight))
            .collect();
        self.with_bounds(lower, upper)
    }

    /// Caps every asset's short position: `w_i >= -max_short`
    /// (roadmap 3.1).
    ///
    /// Short limits map onto the existing box constraints — lower bounds are
    /// raised to `max(lower_i, -max_short)` — so no constraint rows are
    /// added. `max_short = 0` forces a long-only portfolio. This is a
    /// per-asset limit; a total short-budget cap
    /// (`sum_i max(-w_i, 0) <= S`) needs a long/short variable split that
    /// the QP form deliberately does not model.
    ///
    /// Apply after [`PortfolioProblem::with_bounds`]; the tightening keeps
    /// whichever lower bound is already stricter.
    ///
    /// # Errors
    ///
    /// Returns an error unless `max_short` is finite and non-negative, or
    /// when the limit contradicts an existing bound (an upper bound forcing
    /// a deeper short).
    pub fn with_short_limit(self, max_short: f64) -> Result<Self, PortfolioError> {
        if !max_short.is_finite() || max_short < 0.0 {
            return Err(PortfolioError::InvalidParameter(
                "max_short must be finite and non-negative",
            ));
        }
        let lower: Vec<f64> = self
            .lower_bounds
            .iter()
            .map(|value| value.max(-max_short))
            .collect();
        let upper = self.upper_bounds.clone();
        self.with_bounds(lower, upper)
    }

    /// Appends one equality row per group; shared by the industry and
    /// explicit-target templates. `targets.len()` is the group count and
    /// every group must have at least one member asset.
    fn append_group_equalities(
        mut self,
        field: &'static str,
        groups: &[usize],
        targets: &[f64],
    ) -> Result<Self, PortfolioError> {
        let dimension = self.dimension();
        let mut member_counts = vec![0_usize; targets.len()];
        let mut rows = vec![vec![0.0; dimension]; targets.len()];
        for (asset, group) in groups.iter().enumerate() {
            rows[*group][asset] = 1.0;
            member_counts[*group] += 1;
        }
        if let Some(empty) = member_counts.iter().position(|count| *count == 0) {
            return Err(PortfolioError::Template(format!(
                "{field} group {empty} has no member assets; its row would read 0 = target"
            )));
        }
        self.equalities = append_constraint_rows(&self.equalities, rows, targets.to_vec())?;
        Ok(self)
    }

    /// Builds the standard-form QP consumed by the numerical kernel.
    ///
    /// # Errors
    ///
    /// Returns an error when the budget is already impossible under the box
    /// constraints or if final QP validation fails.
    pub fn to_qp(&self) -> Result<QpProblem, PortfolioError> {
        if let Some(budget) = self.budget {
            let minimum: f64 = self.lower_bounds.iter().sum();
            let maximum: f64 = self.upper_bounds.iter().sum();
            if budget < minimum || budget > maximum {
                return Err(PortfolioError::BudgetOutsideBounds {
                    budget,
                    minimum,
                    maximum,
                });
            }
        }

        let omega = match &self.covariance.omega {
            FactorCovariance::Diagonal(values) => FactorCovariance::Diagonal(
                values
                    .iter()
                    .map(|value| self.risk_aversion * value)
                    .collect(),
            ),
            FactorCovariance::Dense(matrix) => {
                let mut scaled = matrix.clone();
                for value in scaled.as_mut_slice() {
                    *value *= self.risk_aversion;
                }
                FactorCovariance::Dense(scaled)
            }
        };
        let diagonal: Vec<f64> = self
            .covariance
            .diagonal
            .iter()
            .map(|value| self.risk_aversion * value + self.turnover_penalty)
            .collect();
        let quadratic = FactorQuad::new(self.covariance.factors.clone(), omega, diagonal)?;
        let mut linear: Vec<f64> = self.expected_returns.iter().map(|value| -value).collect();
        if let Some(previous_weights) = &self.previous_weights {
            for (value, previous) in linear.iter_mut().zip(previous_weights) {
                *value -= self.turnover_penalty * previous;
            }
        }
        if let Some(benchmark_weights) = &self.benchmark_weights {
            // (w-b)' Σ (w-b) expanded: the cross term shifts the linear cost
            // by -risk_aversion * Σ b. Uses the raw covariance — the turnover
            // penalty tracks the previous portfolio, not the benchmark.
            let covariance_times_benchmark = self.covariance.apply(benchmark_weights);
            for (value, product) in linear.iter_mut().zip(&covariance_times_benchmark) {
                *value -= self.risk_aversion * product;
            }
        }

        let l1 = match (&self.l1_turnover_costs, &self.previous_weights) {
            (Some(costs), Some(previous_weights)) => Some(L1Term {
                costs: costs.clone(),
                anchor: previous_weights.clone(),
            }),
            _ => None,
        };

        let equalities = self.combined_equalities()?;
        let problem = QpProblem {
            quadratic,
            linear,
            l1,
            equalities,
            inequalities: self.inequalities.clone(),
            lower_bounds: self.lower_bounds.clone(),
            upper_bounds: self.upper_bounds.clone(),
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Solves with default settings.
    ///
    /// # Errors
    ///
    /// Returns setup and validation errors. A completed solve reports
    /// convergence through [`crate::SolveStatus`] on the returned solution.
    pub fn solve(&self, warm_start: Option<&WarmStart>) -> Result<Solution, PortfolioError> {
        self.solve_with(&Solver::default(), warm_start)
    }

    /// Builds a rolling [`PortfolioSequence`] with default settings
    /// (roadmap 2.5).
    ///
    /// The sequence reuses the equilibration and the reduced factorizations
    /// across dates and chains warm starts automatically; per-date data
    /// changes are described by [`crate::RebalanceStep`].
    ///
    /// # Errors
    ///
    /// Returns setup and validation errors for the base problem.
    pub fn sequence(&self) -> Result<PortfolioSequence, PortfolioError> {
        self.sequence_with(&Solver::default())
    }

    /// Builds a rolling [`PortfolioSequence`] iterating with an explicitly
    /// configured solver.
    ///
    /// # Errors
    ///
    /// Returns setup and validation errors for the base problem.
    pub fn sequence_with(&self, solver: &Solver) -> Result<PortfolioSequence, PortfolioError> {
        PortfolioSequence::new(self, solver)
    }

    /// Solves with an explicitly configured solver.
    ///
    /// # Errors
    ///
    /// Returns setup and validation errors. A completed solve reports
    /// convergence through [`crate::SolveStatus`] on the returned solution.
    pub fn solve_with(
        &self,
        solver: &Solver,
        warm_start: Option<&WarmStart>,
    ) -> Result<Solution, PortfolioError> {
        let problem = self.to_qp()?;
        let mut solution = solver.solve(&problem, warm_start)?;
        append_certificate_hints(
            &mut solution,
            &PortfolioSemantics {
                budget: self.budget,
                user_equality_count: self.equalities.len(),
            },
        );
        Ok(solution)
    }

    pub(crate) fn expected_returns(&self) -> &[f64] {
        &self.expected_returns
    }

    pub(crate) fn previous_weights(&self) -> Option<&[f64]> {
        self.previous_weights.as_deref()
    }

    pub(crate) const fn turnover_penalty(&self) -> f64 {
        self.turnover_penalty
    }

    pub(crate) fn has_l1_turnover(&self) -> bool {
        self.l1_turnover_costs.is_some()
    }

    pub(crate) fn benchmark_weights(&self) -> Option<&[f64]> {
        self.benchmark_weights.as_deref()
    }

    pub(crate) const fn covariance(&self) -> &FactorQuad {
        &self.covariance
    }

    pub(crate) const fn risk_aversion(&self) -> f64 {
        self.risk_aversion
    }

    pub(crate) const fn budget(&self) -> Option<f64> {
        self.budget
    }

    pub(crate) fn user_equality_rhs(&self) -> &[f64] {
        &self.equalities.rhs
    }

    pub(crate) fn lower_bounds(&self) -> &[f64] {
        &self.lower_bounds
    }

    pub(crate) fn upper_bounds(&self) -> &[f64] {
        &self.upper_bounds
    }

    fn combined_equalities(&self) -> Result<LinearConstraints, PortfolioError> {
        let budget_rows = usize::from(self.budget.is_some());
        let rows = budget_rows + self.equalities.len();
        let dimension = self.dimension();
        let mut matrix = Matrix::zeros(rows, dimension);
        let mut rhs = Vec::with_capacity(rows);
        if let Some(budget) = self.budget {
            for column in 0..dimension {
                matrix[(0, column)] = 1.0;
            }
            rhs.push(budget);
        }
        for row in 0..self.equalities.len() {
            for column in 0..dimension {
                matrix[(budget_rows + row, column)] = self.equalities.matrix[(row, column)];
            }
            rhs.push(self.equalities.rhs[row]);
        }
        Ok(LinearConstraints::new(matrix, rhs)?)
    }
}

/// Solves a factor mean-variance portfolio with optional custom settings.
///
/// # Errors
///
/// Returns validation or solver setup errors. Inspect the returned
/// [`crate::SolveStatus`] to determine whether iteration converged.
pub fn solve_mean_variance_factor(
    problem: &PortfolioProblem,
    settings: Option<SolverSettings>,
    warm_start: Option<&WarmStart>,
) -> Result<Solution, PortfolioError> {
    let solver = Solver::new(settings.unwrap_or_default());
    problem.solve_with(&solver, warm_start)
}

/// Errors from high-level portfolio construction and solution.
#[derive(Debug, Error)]
pub enum PortfolioError {
    /// Invalid covariance, objective, constraint, or box data.
    #[error(transparent)]
    Problem(#[from] ProblemError),
    /// A matrix could not be constructed.
    #[error(transparent)]
    Matrix(#[from] MatrixError),
    /// Numerical solver setup failed.
    #[error(transparent)]
    Solver(#[from] SolverError),
    /// A high-level scalar parameter is invalid.
    #[error("invalid portfolio parameter: {0}")]
    InvalidParameter(&'static str),
    /// A constraint template's group, style, or bound data is inconsistent.
    #[error("invalid constraint template: {0}")]
    Template(String),
    /// The budget cannot be met under the supplied box constraints.
    #[error(
        "budget {budget} is impossible under the box constraints: reachable sum is [{minimum}, {maximum}]"
    )]
    BudgetOutsideBounds {
        /// Requested budget.
        budget: f64,
        /// Sum of lower bounds.
        minimum: f64,
        /// Sum of upper bounds.
        maximum: f64,
    },
}

/// How the portfolio layer maps rows of the standard-form QP back to the
/// user's vocabulary when explaining an infeasibility certificate.
pub(crate) struct PortfolioSemantics {
    pub budget: Option<f64>,
    pub user_equality_count: usize,
}

/// Weights below this share of the (unit-norm) certificate are treated as
/// non-participating when naming constraints in hints.
const PARTICIPATION_THRESHOLD: f64 = 0.05;

/// Prepends a portfolio-vocabulary explanation of an infeasibility
/// certificate to the solution's hints (roadmap 2.2).
///
/// The certificate itself stays on [`Solution::certificate`] untouched; this
/// only translates "equality row 0" into "the budget" and groups bound
/// participants, so users see which portfolio constraints conflict.
pub(crate) fn append_certificate_hints(solution: &mut Solution, semantics: &PortfolioSemantics) {
    let hint = match &solution.certificate {
        Some(Certificate::Primal(certificate)) => explain_primal(certificate, semantics),
        Some(Certificate::Dual(certificate)) => Some(explain_dual(certificate)),
        None => None,
    };
    if let (Some(hint), Some(diagnostics)) = (hint, solution.diagnostics.as_mut()) {
        diagnostics.hints.insert(0, hint);
    }
}

fn explain_primal(
    certificate: &PrimalCertificate,
    semantics: &PortfolioSemantics,
) -> Option<String> {
    let budget_rows = usize::from(semantics.budget.is_some());
    let mut parts = Vec::new();

    if budget_rows == 1
        && certificate
            .equality_dual
            .first()
            .is_some_and(|weight| weight.abs() >= PARTICIPATION_THRESHOLD)
    {
        let budget = semantics.budget.unwrap_or_default();
        parts.push(format!("the budget (sum of weights = {budget})"));
    }
    let equality_rows: Vec<usize> = (0..semantics.user_equality_count)
        .filter(|row| certificate.equality_dual[budget_rows + row].abs() >= PARTICIPATION_THRESHOLD)
        .collect();
    if !equality_rows.is_empty() {
        parts.push(format!(
            "equality row(s) {}",
            enumerate_indices(&equality_rows)
        ));
    }
    let inequality_rows: Vec<usize> = certificate
        .inequality_dual
        .iter()
        .enumerate()
        .filter(|(_, weight)| **weight >= PARTICIPATION_THRESHOLD)
        .map(|(row, _)| row)
        .collect();
    if !inequality_rows.is_empty() {
        parts.push(format!(
            "inequality cap(s) {}",
            enumerate_indices(&inequality_rows)
        ));
    }
    let upper_assets: Vec<usize> = certificate
        .bound_dual
        .iter()
        .enumerate()
        .filter(|(_, weight)| **weight >= PARTICIPATION_THRESHOLD)
        .map(|(asset, _)| asset)
        .collect();
    if !upper_assets.is_empty() {
        parts.push(format!(
            "upper bounds on asset(s) {}",
            enumerate_indices(&upper_assets)
        ));
    }
    let lower_assets: Vec<usize> = certificate
        .bound_dual
        .iter()
        .enumerate()
        .filter(|(_, weight)| **weight <= -PARTICIPATION_THRESHOLD)
        .map(|(asset, _)| asset)
        .collect();
    if !lower_assets.is_empty() {
        parts.push(format!(
            "lower bounds on asset(s) {}",
            enumerate_indices(&lower_assets)
        ));
    }

    if parts.is_empty() {
        return None;
    }
    Some(format!(
        "no portfolio satisfies these constraints simultaneously: {}; relax at \
         least one of them (Farkas weights are on Solution::certificate)",
        parts.join(", ")
    ))
}

fn explain_dual(certificate: &DualCertificate) -> String {
    let mut assets: Vec<(usize, f64)> = certificate
        .direction
        .iter()
        .enumerate()
        .filter(|(_, value)| value.abs() >= 2.0 * PARTICIPATION_THRESHOLD)
        .map(|(asset, value)| (asset, *value))
        .collect();
    assets.sort_by(|left, right| right.1.abs().total_cmp(&left.1.abs()));
    let indices: Vec<usize> = assets.iter().map(|(asset, _)| *asset).collect();
    format!(
        "the objective is unbounded: expected returns reward a direction the \
         risk model prices at (near) zero risk and no bound or constraint caps \
         it (largest components at asset(s) {}); check factor exposures and \
         specific variance for missing risk, or add bounds \
         (the direction is on Solution::certificate)",
        enumerate_indices(&indices)
    )
}

/// `"0, 3, 7, ... (12 total)"` — at most five indices spelled out.
fn enumerate_indices(indices: &[usize]) -> String {
    const LIMIT: usize = 5;
    let shown: Vec<String> = indices
        .iter()
        .take(LIMIT)
        .map(ToString::to_string)
        .collect();
    if indices.len() > LIMIT {
        format!("{}, ... ({} total)", shown.join(", "), indices.len())
    } else {
        shown.join(", ")
    }
}

#[derive(Clone, Copy)]
pub(crate) enum FinitePolicy {
    Finite,
    AllowInfinity,
}

pub(crate) fn validate_vector(
    field: &'static str,
    values: &[f64],
    expected: usize,
    finite_policy: FinitePolicy,
) -> Result<(), PortfolioError> {
    if values.len() != expected {
        return Err(ProblemError::Dimension {
            field,
            expected,
            actual: values.len(),
        }
        .into());
    }
    let invalid = match finite_policy {
        FinitePolicy::Finite => values.iter().any(|value| !value.is_finite()),
        FinitePolicy::AllowInfinity => values.iter().any(|value| value.is_nan()),
    };
    if invalid {
        return Err(ProblemError::NonFinite(field).into());
    }
    Ok(())
}

/// Checks per-asset group ids: right length and, when the group count is
/// fixed by an explicit target vector, ids within range.
fn validate_group_ids(
    field: &'static str,
    groups: &[usize],
    dimension: usize,
    group_count: Option<usize>,
) -> Result<(), PortfolioError> {
    if groups.len() != dimension {
        return Err(ProblemError::Dimension {
            field,
            expected: dimension,
            actual: groups.len(),
        }
        .into());
    }
    if let Some(count) = group_count {
        if let Some(out_of_range) = groups.iter().find(|group| **group >= count) {
            return Err(PortfolioError::Template(format!(
                "{field} contains id {out_of_range} but only {count} target(s) were supplied"
            )));
        }
    }
    Ok(())
}

/// Returns `existing` with `rows` appended (template builders never replace
/// user-supplied constraint rows).
fn append_constraint_rows(
    existing: &LinearConstraints,
    rows: Vec<Vec<f64>>,
    rhs: Vec<f64>,
) -> Result<LinearConstraints, PortfolioError> {
    let dimension = existing.matrix.cols();
    let total_rows = existing.len() + rows.len();
    let mut data = Vec::with_capacity(total_rows * dimension);
    data.extend_from_slice(existing.matrix.as_slice());
    for row in rows {
        debug_assert_eq!(row.len(), dimension);
        data.extend(row);
    }
    let mut combined_rhs = Vec::with_capacity(total_rows);
    combined_rhs.extend_from_slice(&existing.rhs);
    combined_rhs.extend(rhs);
    Ok(LinearConstraints::new(
        Matrix::new(total_rows, dimension, data)?,
        combined_rhs,
    )?)
}

fn validated_constraints(
    field: &'static str,
    matrix: Matrix,
    rhs: Vec<f64>,
    dimension: usize,
) -> Result<LinearConstraints, PortfolioError> {
    if matrix.cols() != dimension {
        return Err(ProblemError::Dimension {
            field,
            expected: dimension,
            actual: matrix.cols(),
        }
        .into());
    }
    if matrix.as_slice().iter().any(|value| !value.is_finite())
        || rhs.iter().any(|value| !value.is_finite())
    {
        return Err(ProblemError::NonFinite(field).into());
    }
    Ok(LinearConstraints::new(matrix, rhs)?)
}
