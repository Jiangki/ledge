"""Executes every cvxpy -> Ledge mapping in docs/cvxpy_migration.md.

Each test mirrors one section of the migration guide: the cvxpy side is
written exactly as the guide shows it, the Ledge side likewise, and both
weight vectors are scored with an independent NumPy objective so solver
conventions (dropped constants, minimize vs maximize) cannot mask a
disagreement. If an API change breaks the guide, these tests break first.

Requires the `test` extra (`pip install -e "python/[test]"`); skipped when
cvxpy is unavailable.
"""

import numpy as np
import pytest

from ledge import PortfolioProblem, solve_mean_variance_factor

cp = pytest.importorskip("cvxpy")

RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-6
WEIGHT_TOLERANCE = 1.0e-4

LEDGE_TIGHT = {
    "absolute_tolerance": 1.0e-8,
    "relative_tolerance": 1.0e-8,
    "max_iterations": 200_000,
}


def base_instance(seed=11, n=30, k=4):
    """Deterministic factor instance shared by every guide section."""
    rng = np.random.default_rng(seed)
    instance = {
        "n": n,
        "F": 0.3 * rng.normal(size=(n, k)) / np.sqrt(k),
        "d": rng.uniform(0.05, 0.2, size=n),
        "mu": 0.01 + 0.005 * rng.normal(size=n),
        "gamma": 2.0,
        "lower": np.zeros(n),
        "upper": np.full(n, 3.0 / n),
    }
    root = rng.normal(size=(k, k))
    instance["Omega"] = root @ root.T / k + 0.05 * np.eye(k)
    instance["Sigma"] = (
        instance["F"] @ instance["Omega"] @ instance["F"].T + np.diag(instance["d"])
    )
    return instance


def assert_agreement(objective, ledge_weights, cvxpy_weights, label):
    """Guide section 8: score both weight vectors with one NumPy objective."""
    ledge_objective = objective(ledge_weights)
    cvxpy_objective = objective(cvxpy_weights)
    scale = max(1.0, abs(cvxpy_objective))
    assert ledge_objective - cvxpy_objective <= RELATIVE_OBJECTIVE_TOLERANCE * scale, (
        f"{label}: ledge objective {ledge_objective:.10e} worse than "
        f"cvxpy {cvxpy_objective:.10e}"
    )
    np.testing.assert_allclose(
        ledge_weights,
        cvxpy_weights,
        atol=WEIGHT_TOLERANCE,
        err_msg=f"{label}: weight vectors disagree",
    )


def test_core_model_section_2():
    """Basic long-only rebalance: budget, boxes, half-scaled risk term."""
    inst = base_instance()

    w = cp.Variable(inst["n"])
    risk = 0.5 * inst["gamma"] * cp.quad_form(w, cp.psd_wrap(inst["Sigma"]))
    prob = cp.Problem(
        cp.Minimize(risk - inst["mu"] @ w),
        [cp.sum(w) == 1.0, w >= inst["lower"], w <= inst["upper"]],
    )
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == cp.OPTIMAL

    result = solve_mean_variance_factor(
        inst["F"], inst["Omega"], inst["d"], inst["mu"],
        risk_aversion=inst["gamma"],
        budget=1.0,
        lower_bounds=inst["lower"],
        upper_bounds=inst["upper"],
        **LEDGE_TIGHT,
    )
    assert result.status == "solved"

    def objective(weights):
        return float(
            0.5 * inst["gamma"] * weights @ inst["Sigma"] @ weights
            - inst["mu"] @ weights
        )

    assert_agreement(objective, np.asarray(result.weights), np.asarray(w.value), "core")


def test_constraints_turnover_tracking_sections_3_to_5():
    """Exposure equalities, floors/caps via negation, L2 + L1 turnover, and a
    tracking benchmark, all in one model."""
    inst = base_instance()
    n = inst["n"]
    rng = np.random.default_rng(23)

    reference = np.full(n, 1.0 / n)
    exposures = rng.normal(size=(2, n))
    exposure_targets = exposures @ reference
    sectors = (rng.uniform(size=(3, n)) < 0.4).astype(float)
    sector_reference = sectors @ reference
    floor = sector_reference - 0.05
    cap = sector_reference + 0.05
    w_prev = reference.copy()
    benchmark = rng.uniform(0.5, 1.5, size=n)
    benchmark /= benchmark.sum()
    eta = 0.5
    kappa = rng.uniform(0.0005, 0.002, size=n)

    w = cp.Variable(n)
    active = w - benchmark
    objective_expr = (
        0.5 * inst["gamma"] * cp.quad_form(active, cp.psd_wrap(inst["Sigma"]))
        - inst["mu"] @ w
        + 0.5 * eta * cp.sum_squares(w - w_prev)
        + kappa @ cp.abs(w - w_prev)
    )
    prob = cp.Problem(
        cp.Minimize(objective_expr),
        [
            cp.sum(w) == 1.0,
            w >= inst["lower"],
            w <= inst["upper"],
            exposures @ w == exposure_targets,
            sectors @ w >= floor,
            sectors @ w <= cap,
        ],
    )
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == cp.OPTIMAL

    # Guide section 3: floors become negated upper-inequality rows.
    inequality_matrix = np.vstack([sectors, -sectors])
    inequality_rhs = np.concatenate([cap, -floor])
    result = solve_mean_variance_factor(
        inst["F"], inst["Omega"], inst["d"], inst["mu"],
        risk_aversion=inst["gamma"],
        budget=1.0,
        lower_bounds=inst["lower"],
        upper_bounds=inst["upper"],
        equality_matrix=exposures,
        equality_rhs=exposure_targets,
        inequality_matrix=inequality_matrix,
        inequality_rhs=inequality_rhs,
        previous_weights=w_prev,
        turnover_penalty=eta,
        l1_turnover_costs=kappa,
        benchmark_weights=benchmark,
        **LEDGE_TIGHT,
    )
    assert result.status == "solved"

    def objective(weights):
        active = weights - benchmark
        delta = weights - w_prev
        return float(
            0.5 * inst["gamma"] * active @ inst["Sigma"] @ active
            - inst["mu"] @ weights
            + 0.5 * eta * delta @ delta
            + kappa @ np.abs(delta)
        )

    assert_agreement(objective, np.asarray(result.weights), np.asarray(w.value), "full")


def test_constraint_templates_section_3():
    """Template kwargs vs the hand-built cvxpy rows they compile onto:
    industry neutrality against the benchmark plus a per-name cap."""
    inst = base_instance()
    n = inst["n"]
    rng = np.random.default_rng(41)

    industry_ids = [asset % 4 for asset in range(n)]
    indicator = np.zeros((4, n))
    for asset, industry in enumerate(industry_ids):
        indicator[industry, asset] = 1.0
    benchmark = rng.uniform(0.5, 1.5, size=n)
    benchmark /= benchmark.sum()

    w = cp.Variable(n)
    active = w - benchmark
    prob = cp.Problem(
        cp.Minimize(
            0.5 * inst["gamma"] * cp.quad_form(active, cp.psd_wrap(inst["Sigma"]))
            - inst["mu"] @ w
        ),
        [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= 0.06,
            indicator @ w == indicator @ benchmark,
        ],
    )
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == cp.OPTIMAL

    result = solve_mean_variance_factor(
        inst["F"], inst["Omega"], inst["d"], inst["mu"],
        risk_aversion=inst["gamma"],
        budget=1.0,
        benchmark_weights=benchmark,
        industry_ids=industry_ids,
        max_weight=0.06,
        **LEDGE_TIGHT,
    )
    assert result.status == "solved"

    def objective(weights):
        active = weights - benchmark
        return float(
            0.5 * inst["gamma"] * active @ inst["Sigma"] @ active
            - inst["mu"] @ weights
        )

    assert_agreement(
        objective, np.asarray(result.weights), np.asarray(w.value), "templates"
    )


def test_rolling_sequence_section_6():
    """cp.Parameter re-solve loop vs PortfolioSequence with a rolling anchor."""
    inst = base_instance()
    n = inst["n"]
    rng = np.random.default_rng(31)
    kappa = 0.001
    w0 = np.full(n, 1.0 / n)
    dates = [inst["mu"] + 0.002 * rng.normal(size=n) for _ in range(4)]

    mu_parameter = cp.Parameter(n)
    anchor_parameter = cp.Parameter(n)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(
            0.5 * inst["gamma"] * cp.quad_form(w, cp.psd_wrap(inst["Sigma"]))
            - mu_parameter @ w
            + kappa * cp.norm1(w - anchor_parameter)
        ),
        [cp.sum(w) == 1.0, w >= inst["lower"], w <= inst["upper"]],
    )

    problem = PortfolioProblem(
        inst["F"], inst["Omega"], inst["d"], inst["mu"],
        risk_aversion=inst["gamma"],
        budget=1.0,
        lower_bounds=inst["lower"],
        upper_bounds=inst["upper"],
        previous_weights=w0,
        l1_turnover_costs=kappa,
    )
    sequence = problem.sequence(**LEDGE_TIGHT)

    cvxpy_held = w0.copy()
    ledge_held = w0.copy()
    for step, mu_t in enumerate(dates):
        mu_parameter.value = mu_t
        anchor_parameter.value = cvxpy_held
        prob.solve(solver=cp.CLARABEL, warm_start=True)
        assert prob.status == cp.OPTIMAL
        cvxpy_held = np.asarray(w.value)

        result = sequence.solve_next(
            expected_returns=mu_t,
            previous_weights=ledge_held,
        )
        assert result.status == "solved"
        ledge_held = np.asarray(result.weights)

        anchor = np.asarray(anchor_parameter.value)

        def objective(weights, mu_t=mu_t, anchor=anchor):
            delta = weights - anchor
            return float(
                0.5 * inst["gamma"] * weights @ inst["Sigma"] @ weights
                - mu_t @ weights
                + kappa * np.abs(delta).sum()
            )

        assert_agreement(objective, ledge_held, cvxpy_held, f"date {step}")

    # One structure, one equilibration: the factorization cache is shared
    # across all dates (a stable count is the reuse the guide advertises).
    assert sequence.factorizations <= 2


def test_statuses_section_7():
    """cvxpy INFEASIBLE maps to 'primal infeasible' plus a certificate.

    The contradiction lives in a general inequality row (gross exposure
    capped below the budget) because obvious box-vs-budget conflicts are
    rejected at build time with a ValueError before any solve happens.
    """
    inst = base_instance()
    n = inst["n"]
    exposure_cap = np.ones((1, n))
    cap_rhs = np.array([0.4])  # conflicts with sum(w) == 1

    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(
            0.5 * inst["gamma"] * cp.quad_form(w, cp.psd_wrap(inst["Sigma"]))
            - inst["mu"] @ w
        ),
        [
            cp.sum(w) == 1.0,
            w >= inst["lower"],
            w <= inst["upper"],
            exposure_cap @ w <= cap_rhs,
        ],
    )
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == cp.INFEASIBLE

    with pytest.raises(RuntimeError, match="primal infeasible"):
        solve_mean_variance_factor(
            inst["F"], inst["Omega"], inst["d"], inst["mu"],
            risk_aversion=inst["gamma"],
            budget=1.0,
            lower_bounds=inst["lower"],
            upper_bounds=inst["upper"],
            inequality_matrix=exposure_cap,
            inequality_rhs=cap_rhs,
        )

    result = solve_mean_variance_factor(
        inst["F"], inst["Omega"], inst["d"], inst["mu"],
        risk_aversion=inst["gamma"],
        budget=1.0,
        lower_bounds=inst["lower"],
        upper_bounds=inst["upper"],
        inequality_matrix=exposure_cap,
        inequality_rhs=cap_rhs,
        raise_on_failure=False,
    )
    assert result.status == "primal infeasible"
    assert result.certificate is not None
    assert result.certificate.kind == "primal"
