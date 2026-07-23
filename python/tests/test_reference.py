"""Gold-standard checks against cvxpy + Clarabel.

Ledge and Clarabel solve the same factor-model QP; both weight vectors are
evaluated with one NumPy objective so solver-internal objective conventions
(dropped constants, sign flips) cannot mask a disagreement.

These tests require the `test` extra (`pip install -e "python/[test]"`). They
are skipped when cvxpy is unavailable so the default test run stays light.
"""

import numpy as np
import pytest

from ledge import solve_mean_variance_factor

cp = pytest.importorskip("cvxpy")

RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-6
WEIGHT_TOLERANCE = 1.0e-4
INSTANCE_SEEDS = list(range(20))


def random_instance(seed):
    """Builds a random feasible long-only factor QP keyed by seed."""
    rng = np.random.default_rng(seed)
    n = int(rng.choice([5, 10, 20, 30]))
    k = int(rng.choice([1, 2, 3, 5]))
    m = int(rng.choice([0, 1, 3]))

    factors = 0.3 * rng.normal(size=(n, k)) / np.sqrt(k)
    root = rng.normal(size=(k, k))
    omega = root @ root.T / k + 0.05 * np.eye(k)
    specific = rng.uniform(0.05, 0.2, size=n)
    expected = 0.01 + 0.005 * rng.normal(size=n)

    lower = np.zeros(n)
    upper = rng.uniform(1.5 / n, 3.0 / n, size=n)
    reference = np.full(n, 1.0 / n)

    instance = {
        "factors": factors,
        "omega": omega,
        "specific": specific,
        "expected": expected,
        "risk_aversion": float(rng.uniform(0.5, 3.0)),
        "lower": lower,
        "upper": upper,
        "inequality_matrix": None,
        "inequality_rhs": None,
        "previous_weights": None,
        "turnover_penalty": 0.0,
    }
    if m > 0:
        matrix = rng.normal(size=(m, n))
        # Anchor the right-hand side around the uniform feasible portfolio so
        # the instance is feasible by construction.
        slack = 0.1 * np.linalg.norm(matrix, axis=1) / np.sqrt(n)
        instance["inequality_matrix"] = matrix
        instance["inequality_rhs"] = matrix @ reference + slack
    if seed % 3 == 0:
        instance["previous_weights"] = reference.copy()
        instance["turnover_penalty"] = float(rng.uniform(0.1, 1.0))
    return instance


def numpy_objective(instance, weights):
    """Evaluates the mean-variance objective independently of any solver."""
    covariance = (
        instance["factors"] @ instance["omega"] @ instance["factors"].T
        + np.diag(instance["specific"])
    )
    value = 0.5 * instance["risk_aversion"] * weights @ covariance @ weights
    value -= instance["expected"] @ weights
    if instance["previous_weights"] is not None:
        delta = weights - instance["previous_weights"]
        value += 0.5 * instance["turnover_penalty"] * delta @ delta
    return float(value)


def solve_with_clarabel(instance):
    n = instance["expected"].shape[0]
    covariance = (
        instance["factors"] @ instance["omega"] @ instance["factors"].T
        + np.diag(instance["specific"])
    )
    weights = cp.Variable(n)
    objective = 0.5 * instance["risk_aversion"] * cp.quad_form(
        weights, cp.psd_wrap(covariance)
    ) - instance["expected"] @ weights
    if instance["previous_weights"] is not None:
        objective += 0.5 * instance["turnover_penalty"] * cp.sum_squares(
            weights - instance["previous_weights"]
        )
    constraints = [
        cp.sum(weights) == 1.0,
        weights >= instance["lower"],
        weights <= instance["upper"],
    ]
    if instance["inequality_matrix"] is not None:
        constraints.append(
            instance["inequality_matrix"] @ weights <= instance["inequality_rhs"]
        )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL, f"Clarabel status: {problem.status}"
    return np.asarray(weights.value)


def solve_with_ledge(instance):
    result = solve_mean_variance_factor(
        instance["factors"],
        instance["omega"],
        instance["specific"],
        instance["expected"],
        risk_aversion=instance["risk_aversion"],
        budget=1.0,
        lower_bounds=instance["lower"],
        upper_bounds=instance["upper"],
        inequality_matrix=instance["inequality_matrix"],
        inequality_rhs=instance["inequality_rhs"],
        previous_weights=instance["previous_weights"],
        turnover_penalty=instance["turnover_penalty"],
        absolute_tolerance=1.0e-8,
        relative_tolerance=1.0e-8,
        max_iterations=200_000,
    )
    assert result.status == "solved"
    return np.asarray(result.weights)


@pytest.mark.parametrize("seed", INSTANCE_SEEDS)
def test_ledge_matches_clarabel_oracle(seed):
    instance = random_instance(seed)

    ledge_weights = solve_with_ledge(instance)
    clarabel_weights = solve_with_clarabel(instance)

    ledge_objective = numpy_objective(instance, ledge_weights)
    clarabel_objective = numpy_objective(instance, clarabel_weights)

    scale = max(1.0, abs(clarabel_objective))
    assert ledge_objective - clarabel_objective <= RELATIVE_OBJECTIVE_TOLERANCE * scale, (
        f"seed {seed}: ledge objective {ledge_objective:.10e} worse than "
        f"Clarabel {clarabel_objective:.10e}"
    )
    np.testing.assert_allclose(
        ledge_weights,
        clarabel_weights,
        atol=WEIGHT_TOLERANCE,
        err_msg=f"seed {seed}: weight vectors disagree",
    )
