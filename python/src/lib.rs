//! Python bindings for Ledge.

use ledge::{
    solve_batch as rust_solve_batch, solve_mean_variance_factor as rust_solve_mean_variance_factor,
    BatchAccount as RustBatchAccount, Certificate, FactorCovariance, Matrix, PortfolioError,
    PortfolioProblem as RustPortfolioProblem, PortfolioSequence as RustPortfolioSequence,
    RebalanceStep, Solution, SolveStatus, Solver, SolverSettings, WarmStart,
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

/// Proof that a solve stopped because the problem itself is pathological
/// (status `'primal infeasible'` or `'dual infeasible'`).
///
/// Certificates are normalized to unit infinity norm and reported in the
/// original data space. For a primal certificate the multiplier arrays are a
/// Farkas combination of the constraints that admits no common portfolio;
/// for a dual certificate `direction` is a ray along which the objective
/// decreases without bound.
#[pyclass(name = "InfeasibilityCertificate", frozen)]
struct PyCertificate {
    /// `"primal"` or `"dual"`.
    #[pyo3(get)]
    kind: String,
    equality_dual: Option<Vec<f64>>,
    inequality_dual: Option<Vec<f64>>,
    bound_dual: Option<Vec<f64>>,
    direction: Option<Vec<f64>>,
}

#[pymethods]
impl PyCertificate {
    /// Farkas weights on the equality rows (budget first, when present);
    /// `None` for dual certificates.
    #[getter]
    fn equality_dual<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.equality_dual
            .as_ref()
            .map(|values| PyArray1::from_vec(py, values.clone()))
    }

    /// Farkas weights on the inequality rows; `None` for dual certificates.
    #[getter]
    fn inequality_dual<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inequality_dual
            .as_ref()
            .map(|values| PyArray1::from_vec(py, values.clone()))
    }

    /// Farkas weights on the boxes (positive parts cite upper bounds,
    /// negative parts lower bounds); `None` for dual certificates.
    #[getter]
    fn bound_dual<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.bound_dual
            .as_ref()
            .map(|values| PyArray1::from_vec(py, values.clone()))
    }

    /// Unbounded descent direction in weight space; `None` for primal
    /// certificates.
    #[getter]
    fn direction<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.direction
            .as_ref()
            .map(|values| PyArray1::from_vec(py, values.clone()))
    }

    fn __repr__(&self) -> String {
        format!("InfeasibilityCertificate(kind='{}')", self.kind)
    }
}

impl From<&Certificate> for PyCertificate {
    fn from(certificate: &Certificate) -> Self {
        match certificate {
            Certificate::Primal(primal) => Self {
                kind: "primal".to_owned(),
                equality_dual: Some(primal.equality_dual.clone()),
                inequality_dual: Some(primal.inequality_dual.clone()),
                bound_dual: Some(primal.bound_dual.clone()),
                direction: None,
            },
            Certificate::Dual(dual) => Self {
                kind: "dual".to_owned(),
                equality_dual: None,
                inequality_dual: None,
                bound_dual: None,
                direction: Some(dual.direction.clone()),
            },
        }
    }
}

#[pyclass(name = "SolveResult", frozen)]
struct PySolveResult {
    /// The full solver result, kept for lossless serialization
    /// (`to_json`); the extracted fields below feed the getters.
    inner: Solution,
    weights: Vec<f64>,
    certificate: Option<Certificate>,
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    objective: f64,
    #[pyo3(get)]
    primal_residual: f64,
    #[pyo3(get)]
    dual_residual: f64,
    #[pyo3(get)]
    complementarity: f64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    solve_time_seconds: f64,
    #[pyo3(get)]
    final_rho: f64,
    #[pyo3(get)]
    rho_updates: usize,
    /// Whether the returned weights are the polished iterate: `True` only
    /// when the solve succeeded and the direct active-set solve improved
    /// the worst KKT residual (see the `polish` solver option).
    #[pyo3(get)]
    polished: bool,
    #[pyo3(get)]
    convergence_hints: Vec<String>,
}

#[pymethods]
impl PySolveResult {
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.weights.clone())
    }

    /// Infeasibility proof; present only when the status is
    /// `'primal infeasible'` or `'dual infeasible'`.
    #[getter]
    fn certificate(&self) -> Option<PyCertificate> {
        self.certificate.as_ref().map(PyCertificate::from)
    }

    /// Serializes the full solver result — weights, every dual multiplier
    /// block, residuals, diagnostics, and any infeasibility certificate —
    /// to a JSON string for bug reports and reproduction (roadmap 3.3).
    ///
    /// Pair it with ``PortfolioProblem.to_json`` so a report carries both
    /// the problem and what Ledge returned on it.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveResult(status='{}', objective={:.6e}, iterations={}, \
             primal_residual={:.3e}, dual_residual={:.3e})",
            self.status, self.objective, self.iterations, self.primal_residual, self.dual_residual
        )
    }
}

impl From<Solution> for PySolveResult {
    fn from(solution: Solution) -> Self {
        Self {
            weights: solution.x.clone(),
            certificate: solution.certificate.clone(),
            status: solution.status.to_string(),
            objective: solution.objective,
            primal_residual: solution.residuals.primal,
            dual_residual: solution.residuals.dual,
            complementarity: solution.residuals.complementarity,
            iterations: solution.iterations,
            solve_time_seconds: solution.solve_time.as_secs_f64(),
            final_rho: solution.final_rho,
            rho_updates: solution.rho_updates,
            polished: solution.polished,
            convergence_hints: solution
                .diagnostics
                .as_ref()
                .map(|diagnostics| diagnostics.hints.clone())
                .unwrap_or_default(),
            inner: solution,
        }
    }
}

/// A reusable factor mean-variance portfolio specification.
#[pyclass(name = "PortfolioProblem", skip_from_py_object)]
#[derive(Clone)]
struct PyPortfolioProblem {
    inner: RustPortfolioProblem,
}

#[pymethods]
impl PyPortfolioProblem {
    #[new]
    #[pyo3(signature = (
        factors,
        omega,
        specific_variance,
        expected_returns,
        *,
        risk_aversion=1.0,
        budget=1.0,
        lower_bounds=None,
        upper_bounds=None,
        equality_matrix=None,
        equality_rhs=None,
        inequality_matrix=None,
        inequality_rhs=None,
        previous_weights=None,
        turnover_penalty=0.0,
        l1_turnover_costs=None,
        benchmark_weights=None,
        industry_ids=None,
        industry_targets=None,
        style_matrix=None,
        style_lower=None,
        style_upper=None,
        max_weight=None,
        max_short=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        factors: PyReadonlyArray2<'_, f64>,
        omega: PyReadonlyArray2<'_, f64>,
        specific_variance: PyReadonlyArray1<'_, f64>,
        expected_returns: PyReadonlyArray1<'_, f64>,
        risk_aversion: f64,
        budget: f64,
        lower_bounds: Option<PyReadonlyArray1<'_, f64>>,
        upper_bounds: Option<PyReadonlyArray1<'_, f64>>,
        equality_matrix: Option<PyReadonlyArray2<'_, f64>>,
        equality_rhs: Option<PyReadonlyArray1<'_, f64>>,
        inequality_matrix: Option<PyReadonlyArray2<'_, f64>>,
        inequality_rhs: Option<PyReadonlyArray1<'_, f64>>,
        previous_weights: Option<PyReadonlyArray1<'_, f64>>,
        turnover_penalty: f64,
        l1_turnover_costs: Option<Bound<'_, PyAny>>,
        benchmark_weights: Option<PyReadonlyArray1<'_, f64>>,
        industry_ids: Option<Vec<usize>>,
        industry_targets: Option<PyReadonlyArray1<'_, f64>>,
        style_matrix: Option<PyReadonlyArray2<'_, f64>>,
        style_lower: Option<PyReadonlyArray1<'_, f64>>,
        style_upper: Option<PyReadonlyArray1<'_, f64>>,
        max_weight: Option<f64>,
        max_short: Option<f64>,
    ) -> PyResult<Self> {
        let inner = build_problem(
            factors,
            omega,
            specific_variance,
            expected_returns,
            risk_aversion,
            budget,
            lower_bounds,
            upper_bounds,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            previous_weights,
            turnover_penalty,
            l1_turnover_costs,
            benchmark_weights,
            ConstraintTemplates {
                industry_ids,
                industry_targets,
                style_matrix,
                style_lower,
                style_upper,
                max_weight,
                max_short,
            },
        )?;
        Ok(Self { inner })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Serializes the full problem specification to a JSON string
    /// (roadmap 3.3).
    ///
    /// The dump carries everything needed to rebuild this exact problem —
    /// covariance, returns, constraints (including rows appended by the
    /// constraint templates), bounds, turnover terms, and benchmark — so it
    /// can be attached to a bug report and replayed with
    /// ``PortfolioProblem.from_json``. Unbounded box sides are encoded as
    /// ``null`` (JSON has no representation for infinities).
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Rebuilds a problem from a ``to_json`` dump, re-running the same
    /// validation as construction (roadmap 3.3).
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        serde_json::from_str(text)
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[pyo3(signature = (
        *,
        warm_start=None,
        max_iterations=10_000,
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-5,
        rho=1.0,
        adaptive_rho=true,
        over_relaxation=1.6,
        scaling_iterations=10,
        infeasibility_tolerance=1.0e-5,
        polish=true,
        raise_on_failure=true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        py: Python<'_>,
        warm_start: Option<PyReadonlyArray1<'_, f64>>,
        max_iterations: usize,
        absolute_tolerance: f64,
        relative_tolerance: f64,
        rho: f64,
        adaptive_rho: bool,
        over_relaxation: f64,
        scaling_iterations: usize,
        infeasibility_tolerance: f64,
        polish: bool,
        raise_on_failure: bool,
    ) -> PyResult<PySolveResult> {
        let warm_start = warm_start
            .map(|values| vector_from_array(&values).map(WarmStart::from_primal))
            .transpose()?;
        solve_owned(
            py,
            self.inner.clone(),
            warm_start,
            max_iterations,
            absolute_tolerance,
            relative_tolerance,
            rho,
            adaptive_rho,
            over_relaxation,
            scaling_iterations,
            infeasibility_tolerance,
            polish,
            raise_on_failure,
        )
    }

    /// Builds a rolling solve sequence over this problem's fixed structure.
    ///
    /// Equilibration and reduced factorizations are computed once and reused
    /// for every date; warm starts chain automatically.
    #[pyo3(signature = (
        *,
        max_iterations=10_000,
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-5,
        rho=1.0,
        adaptive_rho=true,
        over_relaxation=1.6,
        scaling_iterations=10,
        infeasibility_tolerance=1.0e-5,
        polish=true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn sequence(
        &self,
        py: Python<'_>,
        max_iterations: usize,
        absolute_tolerance: f64,
        relative_tolerance: f64,
        rho: f64,
        adaptive_rho: bool,
        over_relaxation: f64,
        scaling_iterations: usize,
        infeasibility_tolerance: f64,
        polish: bool,
    ) -> PyResult<PyPortfolioSequence> {
        let settings = SolverSettings {
            max_iterations,
            absolute_tolerance,
            relative_tolerance,
            rho,
            adaptive_rho,
            over_relaxation,
            scaling_iterations,
            infeasibility_tolerance,
            polish,
            ..SolverSettings::default()
        };
        let problem = self.inner.clone();
        let inner = py
            .detach(move || problem.sequence_with(&Solver::new(settings)))
            .map_err(portfolio_value_error)?;
        Ok(PyPortfolioSequence { inner })
    }

    fn __repr__(&self) -> String {
        format!("PortfolioProblem(dimension={})", self.inner.dimension())
    }
}

/// A rolling rebalance sequence with cached factorizations and chained warm
/// starts.
#[pyclass(name = "PortfolioSequence", skip_from_py_object)]
struct PyPortfolioSequence {
    inner: RustPortfolioSequence,
}

#[pymethods]
impl PyPortfolioSequence {
    /// Number of portfolio weights.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Reduced factorizations built since construction (including the
    /// initial one). A stable count across dates demonstrates reuse.
    #[getter]
    fn factorizations(&self) -> usize {
        self.inner.factorizations()
    }

    /// Applies this date's data changes, then solves warm-started from the
    /// previous date's solution.
    ///
    /// Every argument is optional; omitted data keeps its current value, so
    /// calling with no arguments re-solves the current data (the usual first
    /// call). Only factorization-preserving updates are accepted: new
    /// expected returns, a new turnover anchor (`previous_weights`; requires
    /// the problem to have been built with a quadratic turnover penalty
    /// and/or `l1_turnover_costs` — both terms share the anchor, and the
    /// penalty/costs themselves stay fixed), a new tracking benchmark
    /// (`benchmark_weights`; requires the problem to have been built with
    /// one), a new budget, and new equality / inequality right-hand sides.
    /// Structural changes require a new `PortfolioProblem` and a new
    /// sequence.
    ///
    /// A rejected date leaves the sequence unchanged, so the caller can skip
    /// it and keep rolling.
    #[pyo3(signature = (
        *,
        expected_returns=None,
        previous_weights=None,
        benchmark_weights=None,
        budget=None,
        equality_rhs=None,
        inequality_rhs=None,
        raise_on_failure=true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn solve_next(
        &mut self,
        py: Python<'_>,
        expected_returns: Option<PyReadonlyArray1<'_, f64>>,
        previous_weights: Option<PyReadonlyArray1<'_, f64>>,
        benchmark_weights: Option<PyReadonlyArray1<'_, f64>>,
        budget: Option<f64>,
        equality_rhs: Option<PyReadonlyArray1<'_, f64>>,
        inequality_rhs: Option<PyReadonlyArray1<'_, f64>>,
        raise_on_failure: bool,
    ) -> PyResult<PySolveResult> {
        let step = RebalanceStep {
            expected_returns: expected_returns
                .map(|values| vector_from_array(&values))
                .transpose()?,
            previous_weights: previous_weights
                .map(|values| vector_from_array(&values))
                .transpose()?,
            benchmark_weights: benchmark_weights
                .map(|values| vector_from_array(&values))
                .transpose()?,
            budget,
            equality_rhs: equality_rhs
                .map(|values| vector_from_array(&values))
                .transpose()?,
            inequality_rhs: inequality_rhs
                .map(|values| vector_from_array(&values))
                .transpose()?,
        };
        let inner = &mut self.inner;
        let solution = py
            .detach(move || inner.solve_next(&step))
            .map_err(portfolio_value_error)?;
        check_solution_status(&solution, raise_on_failure)?;
        Ok(solution.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "PortfolioSequence(dimension={}, factorizations={})",
            self.inner.dimension(),
            self.inner.factorizations()
        )
    }
}

/// Solves a factor mean-variance problem without constructing a reusable class.
#[pyfunction]
#[pyo3(signature = (
    factors,
    omega,
    specific_variance,
    expected_returns,
    *,
    risk_aversion=1.0,
    budget=1.0,
    lower_bounds=None,
    upper_bounds=None,
    equality_matrix=None,
    equality_rhs=None,
    inequality_matrix=None,
    inequality_rhs=None,
    previous_weights=None,
    turnover_penalty=0.0,
    l1_turnover_costs=None,
    benchmark_weights=None,
    industry_ids=None,
    industry_targets=None,
    style_matrix=None,
    style_lower=None,
    style_upper=None,
    max_weight=None,
    max_short=None,
    warm_start=None,
    max_iterations=10_000,
    absolute_tolerance=1.0e-6,
    relative_tolerance=1.0e-5,
    rho=1.0,
    adaptive_rho=true,
    over_relaxation=1.6,
    scaling_iterations=10,
    infeasibility_tolerance=1.0e-5,
    polish=true,
    raise_on_failure=true
))]
#[allow(clippy::too_many_arguments)]
fn solve_mean_variance_factor(
    py: Python<'_>,
    factors: PyReadonlyArray2<'_, f64>,
    omega: PyReadonlyArray2<'_, f64>,
    specific_variance: PyReadonlyArray1<'_, f64>,
    expected_returns: PyReadonlyArray1<'_, f64>,
    risk_aversion: f64,
    budget: f64,
    lower_bounds: Option<PyReadonlyArray1<'_, f64>>,
    upper_bounds: Option<PyReadonlyArray1<'_, f64>>,
    equality_matrix: Option<PyReadonlyArray2<'_, f64>>,
    equality_rhs: Option<PyReadonlyArray1<'_, f64>>,
    inequality_matrix: Option<PyReadonlyArray2<'_, f64>>,
    inequality_rhs: Option<PyReadonlyArray1<'_, f64>>,
    previous_weights: Option<PyReadonlyArray1<'_, f64>>,
    turnover_penalty: f64,
    l1_turnover_costs: Option<Bound<'_, PyAny>>,
    benchmark_weights: Option<PyReadonlyArray1<'_, f64>>,
    industry_ids: Option<Vec<usize>>,
    industry_targets: Option<PyReadonlyArray1<'_, f64>>,
    style_matrix: Option<PyReadonlyArray2<'_, f64>>,
    style_lower: Option<PyReadonlyArray1<'_, f64>>,
    style_upper: Option<PyReadonlyArray1<'_, f64>>,
    max_weight: Option<f64>,
    max_short: Option<f64>,
    warm_start: Option<PyReadonlyArray1<'_, f64>>,
    max_iterations: usize,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    rho: f64,
    adaptive_rho: bool,
    over_relaxation: f64,
    scaling_iterations: usize,
    infeasibility_tolerance: f64,
    polish: bool,
    raise_on_failure: bool,
) -> PyResult<PySolveResult> {
    let problem = build_problem(
        factors,
        omega,
        specific_variance,
        expected_returns,
        risk_aversion,
        budget,
        lower_bounds,
        upper_bounds,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        previous_weights,
        turnover_penalty,
        l1_turnover_costs,
        benchmark_weights,
        ConstraintTemplates {
            industry_ids,
            industry_targets,
            style_matrix,
            style_lower,
            style_upper,
            max_weight,
            max_short,
        },
    )?;
    let warm_start = warm_start
        .map(|values| vector_from_array(&values).map(WarmStart::from_primal))
        .transpose()?;
    solve_owned(
        py,
        problem,
        warm_start,
        max_iterations,
        absolute_tolerance,
        relative_tolerance,
        rho,
        adaptive_rho,
        over_relaxation,
        scaling_iterations,
        infeasibility_tolerance,
        polish,
        raise_on_failure,
    )
}

/// Solves many accounts' rolling rebalance sequences in one call, in
/// parallel over the account axis (roadmap 3.2).
///
/// ``problems`` is one ``PortfolioProblem`` per account (typically sharing
/// one factor model); ``steps`` is one list of per-date step dicts per
/// account, applied in order exactly like ``PortfolioSequence.solve_next``
/// — dict keys mirror its keyword arguments (``expected_returns``,
/// ``previous_weights``, ``benchmark_weights``, ``budget``,
/// ``equality_rhs``, ``inequality_rhs``), and an empty dict re-solves the
/// current data (the usual first date). Every account gets its own
/// workspace: equilibration and reduced factorizations are built once per
/// account and warm starts chain across its dates.
///
/// With ``chain_previous_weights=True`` every date after a solved one
/// anchors the turnover terms at that date's solved weights — the standard
/// backtest convention — unless the step provides ``previous_weights``
/// explicitly; requires the problems to have a turnover term.
///
/// Accounts are independent, so results are identical to looping
/// ``PortfolioProblem.sequence()`` per account regardless of thread count
/// (threads follow the ``RAYON_NUM_THREADS`` environment variable, default
/// all CPUs). Returns one list of ``SolveResult`` per account, in input
/// order.
#[pyfunction]
#[pyo3(signature = (
    problems,
    steps,
    *,
    chain_previous_weights=false,
    max_iterations=10_000,
    absolute_tolerance=1.0e-6,
    relative_tolerance=1.0e-5,
    rho=1.0,
    adaptive_rho=true,
    over_relaxation=1.6,
    scaling_iterations=10,
    infeasibility_tolerance=1.0e-5,
    polish=true,
    raise_on_failure=true
))]
#[allow(clippy::too_many_arguments)]
fn solve_batch(
    py: Python<'_>,
    problems: Vec<Py<PyPortfolioProblem>>,
    steps: Vec<Vec<Bound<'_, PyDict>>>,
    chain_previous_weights: bool,
    max_iterations: usize,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    rho: f64,
    adaptive_rho: bool,
    over_relaxation: f64,
    scaling_iterations: usize,
    infeasibility_tolerance: f64,
    polish: bool,
    raise_on_failure: bool,
) -> PyResult<Vec<Vec<PySolveResult>>> {
    if problems.len() != steps.len() {
        return Err(PyValueError::new_err(
            "problems and steps must have the same length (one step list per account)",
        ));
    }
    let accounts: Vec<RustBatchAccount> = problems
        .iter()
        .zip(&steps)
        .enumerate()
        .map(|(account, (problem, account_steps))| {
            let steps = account_steps
                .iter()
                .enumerate()
                .map(|(date, dict)| {
                    step_from_dict(dict).map_err(|error| {
                        PyValueError::new_err(format!(
                            "account {account}, step {date}: {}",
                            error.value(py)
                        ))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(RustBatchAccount {
                problem: problem.borrow(py).inner.clone(),
                steps,
                chain_previous_weights,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let settings = SolverSettings {
        max_iterations,
        absolute_tolerance,
        relative_tolerance,
        rho,
        adaptive_rho,
        over_relaxation,
        scaling_iterations,
        infeasibility_tolerance,
        polish,
        ..SolverSettings::default()
    };

    let results = py.detach(move || rust_solve_batch(&accounts, Some(settings)));

    let mut converted = Vec::with_capacity(results.len());
    for (account, result) in results.into_iter().enumerate() {
        let solutions =
            result.map_err(|error| PyValueError::new_err(format!("account {account}: {error}")))?;
        if raise_on_failure {
            if let Some((date, solution)) = solutions
                .iter()
                .enumerate()
                .find(|(_, solution)| solution.status != SolveStatus::Solved)
            {
                return Err(PyRuntimeError::new_err(format!(
                    "account {account}, step {date}: {}",
                    failure_message(solution)
                )));
            }
        }
        converted.push(solutions.into_iter().map(PySolveResult::from).collect());
    }
    Ok(converted)
}

/// Parses one per-date step dict; keys mirror the
/// `PortfolioSequence.solve_next` keyword arguments, and `None` values mean
/// "keep the current data", exactly like an omitted keyword.
fn step_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<RebalanceStep> {
    let mut step = RebalanceStep::default();
    for (key, value) in dict.iter() {
        let key: String = key
            .extract()
            .map_err(|_| PyValueError::new_err("step keys must be strings"))?;
        if value.is_none() {
            continue;
        }
        match key.as_str() {
            "expected_returns" => step.expected_returns = Some(f64_vector(&value)?),
            "previous_weights" => step.previous_weights = Some(f64_vector(&value)?),
            "benchmark_weights" => step.benchmark_weights = Some(f64_vector(&value)?),
            "budget" => step.budget = Some(value.extract::<f64>()?),
            "equality_rhs" => step.equality_rhs = Some(f64_vector(&value)?),
            "inequality_rhs" => step.inequality_rhs = Some(f64_vector(&value)?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown step key '{other}'; valid keys are expected_returns, \
                     previous_weights, benchmark_weights, budget, equality_rhs, \
                     inequality_rhs"
                )));
            }
        }
    }
    Ok(step)
}

/// Accepts a one-dimensional float64 array or any sequence of floats.
fn f64_vector(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return vector_from_array(&array);
    }
    value.extract::<Vec<f64>>().map_err(|_| {
        PyValueError::new_err("step values must be floats or one-dimensional float arrays")
    })
}

/// Constraint template inputs shared by the class constructor and the free
/// function (roadmap 3.1); each maps onto a `PortfolioProblem` builder.
struct ConstraintTemplates<'py> {
    industry_ids: Option<Vec<usize>>,
    industry_targets: Option<PyReadonlyArray1<'py, f64>>,
    style_matrix: Option<PyReadonlyArray2<'py, f64>>,
    style_lower: Option<PyReadonlyArray1<'py, f64>>,
    style_upper: Option<PyReadonlyArray1<'py, f64>>,
    max_weight: Option<f64>,
    max_short: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
fn build_problem(
    factors: PyReadonlyArray2<'_, f64>,
    omega: PyReadonlyArray2<'_, f64>,
    specific_variance: PyReadonlyArray1<'_, f64>,
    expected_returns: PyReadonlyArray1<'_, f64>,
    risk_aversion: f64,
    budget: f64,
    lower_bounds: Option<PyReadonlyArray1<'_, f64>>,
    upper_bounds: Option<PyReadonlyArray1<'_, f64>>,
    equality_matrix: Option<PyReadonlyArray2<'_, f64>>,
    equality_rhs: Option<PyReadonlyArray1<'_, f64>>,
    inequality_matrix: Option<PyReadonlyArray2<'_, f64>>,
    inequality_rhs: Option<PyReadonlyArray1<'_, f64>>,
    previous_weights: Option<PyReadonlyArray1<'_, f64>>,
    turnover_penalty: f64,
    l1_turnover_costs: Option<Bound<'_, PyAny>>,
    benchmark_weights: Option<PyReadonlyArray1<'_, f64>>,
    templates: ConstraintTemplates<'_>,
) -> PyResult<RustPortfolioProblem> {
    let factors = matrix_from_array(&factors)?;
    let omega = FactorCovariance::Dense(matrix_from_array(&omega)?);
    let specific_variance = vector_from_array(&specific_variance)?;
    let expected_returns = vector_from_array(&expected_returns)?;
    let dimension = expected_returns.len();

    let mut problem =
        RustPortfolioProblem::new(factors, omega, specific_variance, expected_returns)
            .map_err(portfolio_value_error)?
            .with_risk_aversion(risk_aversion)
            .map_err(portfolio_value_error)?
            .with_budget(Some(budget))
            .map_err(portfolio_value_error)?;

    match (lower_bounds, upper_bounds) {
        (Some(lower), Some(upper)) => {
            problem = problem
                .with_bounds(vector_from_array(&lower)?, vector_from_array(&upper)?)
                .map_err(portfolio_value_error)?;
        }
        (None, None) => {}
        _ => {
            return Err(PyValueError::new_err(
                "lower_bounds and upper_bounds must be provided together",
            ));
        }
    }

    match (equality_matrix, equality_rhs) {
        (Some(matrix), Some(rhs)) => {
            problem = problem
                .with_equalities(matrix_from_array(&matrix)?, vector_from_array(&rhs)?)
                .map_err(portfolio_value_error)?;
        }
        (None, None) => {}
        _ => {
            return Err(PyValueError::new_err(
                "equality_matrix and equality_rhs must be provided together",
            ));
        }
    }

    match (inequality_matrix, inequality_rhs) {
        (Some(matrix), Some(rhs)) => {
            problem = problem
                .with_inequalities(matrix_from_array(&matrix)?, vector_from_array(&rhs)?)
                .map_err(portfolio_value_error)?;
        }
        (None, None) => {}
        _ => {
            return Err(PyValueError::new_err(
                "inequality_matrix and inequality_rhs must be provided together",
            ));
        }
    }

    let previous_weights = previous_weights
        .map(|values| vector_from_array(&values))
        .transpose()?;
    if let Some(previous_weights) = &previous_weights {
        problem = problem
            .with_quadratic_turnover(previous_weights.clone(), turnover_penalty)
            .map_err(portfolio_value_error)?;
    } else if turnover_penalty != 0.0 {
        return Err(PyValueError::new_err(
            "previous_weights is required when turnover_penalty is non-zero",
        ));
    }

    if let Some(costs) = l1_turnover_costs {
        let Some(previous_weights) = &previous_weights else {
            return Err(PyValueError::new_err(
                "previous_weights is required when l1_turnover_costs is provided",
            ));
        };
        let costs = l1_costs_vector(&costs, dimension)?;
        problem = problem
            .with_l1_turnover(previous_weights.clone(), costs)
            .map_err(portfolio_value_error)?;
    }

    if let Some(benchmark_weights) = benchmark_weights {
        problem = problem
            .with_tracking_benchmark(vector_from_array(&benchmark_weights)?)
            .map_err(portfolio_value_error)?;
    }

    // Constraint templates (roadmap 3.1); applied after the benchmark so
    // industry neutrality can derive its targets from it.
    match (templates.industry_ids, templates.industry_targets) {
        (Some(industry_ids), Some(targets)) => {
            problem = problem
                .with_group_targets(&industry_ids, &vector_from_array(&targets)?)
                .map_err(portfolio_value_error)?;
        }
        (Some(industry_ids), None) => {
            problem = problem
                .with_industry_neutrality(&industry_ids)
                .map_err(portfolio_value_error)?;
        }
        (None, Some(_)) => {
            return Err(PyValueError::new_err(
                "industry_targets requires industry_ids",
            ));
        }
        (None, None) => {}
    }
    match (
        templates.style_matrix,
        templates.style_lower,
        templates.style_upper,
    ) {
        (Some(matrix), Some(lower), Some(upper)) => {
            problem = problem
                .with_style_bounds(
                    &matrix_from_array(&matrix)?,
                    &vector_from_array(&lower)?,
                    &vector_from_array(&upper)?,
                )
                .map_err(portfolio_value_error)?;
        }
        (None, None, None) => {}
        _ => {
            return Err(PyValueError::new_err(
                "style_matrix, style_lower, and style_upper must be provided together",
            ));
        }
    }
    if let Some(max_weight) = templates.max_weight {
        problem = problem
            .with_concentration_limit(max_weight)
            .map_err(portfolio_value_error)?;
    }
    if let Some(max_short) = templates.max_short {
        problem = problem
            .with_short_limit(max_short)
            .map_err(portfolio_value_error)?;
    }

    if dimension != problem.dimension() {
        return Err(PyValueError::new_err(
            "expected_returns length must match the number of factor rows",
        ));
    }
    Ok(problem)
}

#[allow(clippy::too_many_arguments)]
fn solve_owned(
    py: Python<'_>,
    problem: RustPortfolioProblem,
    warm_start: Option<WarmStart>,
    max_iterations: usize,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    rho: f64,
    adaptive_rho: bool,
    over_relaxation: f64,
    scaling_iterations: usize,
    infeasibility_tolerance: f64,
    polish: bool,
    raise_on_failure: bool,
) -> PyResult<PySolveResult> {
    let settings = SolverSettings {
        max_iterations,
        absolute_tolerance,
        relative_tolerance,
        rho,
        adaptive_rho,
        over_relaxation,
        scaling_iterations,
        infeasibility_tolerance,
        polish,
        ..SolverSettings::default()
    };
    let solution = py
        .detach(move || {
            rust_solve_mean_variance_factor(&problem, Some(settings), warm_start.as_ref())
        })
        .map_err(portfolio_value_error)?;
    check_solution_status(&solution, raise_on_failure)?;
    Ok(solution.into())
}

/// Raises `RuntimeError` for a completed but unconverged solve when the
/// caller asked for it.
fn check_solution_status(solution: &Solution, raise_on_failure: bool) -> PyResult<()> {
    if raise_on_failure && solution.status != SolveStatus::Solved {
        return Err(PyRuntimeError::new_err(failure_message(solution)));
    }
    Ok(())
}

/// The `raise_on_failure` message for a solve that did not reach `Solved`.
fn failure_message(solution: &Solution) -> String {
    let hints = solution
        .diagnostics
        .as_ref()
        .map(|diagnostics| diagnostics.hints.join("; "))
        .filter(|joined| !joined.is_empty())
        .map(|joined| format!(" Hints: {joined}."))
        .unwrap_or_default();
    format!(
        "Ledge stopped with status '{}' after {} iterations \
         (primal residual {:.3e}, dual residual {:.3e}).{hints} \
         Set raise_on_failure=False to inspect the returned iterate and \
         its convergence_hints.",
        solution.status, solution.iterations, solution.residuals.primal, solution.residuals.dual
    )
}

fn matrix_from_array(array: &PyReadonlyArray2<'_, f64>) -> PyResult<Matrix> {
    let view = array.as_array();
    let shape = view.shape();
    Matrix::new(shape[0], shape[1], view.iter().copied().collect())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn vector_from_array(array: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    Ok(array.as_array().iter().copied().collect())
}

/// Accepts a scalar (uniform per-asset cost) or a one-dimensional array.
fn l1_costs_vector(value: &Bound<'_, PyAny>, dimension: usize) -> PyResult<Vec<f64>> {
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(vec![scalar; dimension]);
    }
    let array = value.extract::<PyReadonlyArray1<'_, f64>>().map_err(|_| {
        PyValueError::new_err("l1_turnover_costs must be a float or a one-dimensional float array")
    })?;
    vector_from_array(&array)
}

fn portfolio_value_error(error: PortfolioError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn _ledge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPortfolioProblem>()?;
    module.add_class::<PyPortfolioSequence>()?;
    module.add_class::<PySolveResult>()?;
    module.add_class::<PyCertificate>()?;
    module.add_function(wrap_pyfunction!(solve_mean_variance_factor, module)?)?;
    module.add_function(wrap_pyfunction!(solve_batch, module)?)?;
    Ok(())
}
