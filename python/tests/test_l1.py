"""Exact L1 turnover tests for the Python bindings (roadmap 2.1).

The gold-standard class compares Ledge's dedicated soft-threshold prox
block against cvxpy + Clarabel solving the same objective with an explicit
`norm1`-style term; both weight vectors are evaluated with one NumPy
objective so solver conventions cannot mask a disagreement. The API class
runs without cvxpy.
"""

import unittest

import numpy as np
import pytest

from ledge import PortfolioProblem, solve_mean_variance_factor

RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-5
WEIGHT_TOLERANCE = 5.0e-4


def random_l1_instance(seed):
    """Random feasible long-only factor QP with proportional costs."""
    rng = np.random.default_rng(seed)
    n = int(rng.choice([5, 10, 20]))
    k = int(rng.choice([1, 2, 3]))

    factors = 0.3 * rng.normal(size=(n, k)) / np.sqrt(k)
    root = rng.normal(size=(k, k))
    omega = root @ root.T / k + 0.05 * np.eye(k)
    specific = rng.uniform(0.05, 0.2, size=n)
    expected = 0.01 + 0.005 * rng.normal(size=n)
    previous = np.full(n, 1.0 / n)
    costs = rng.uniform(0.0005, 0.01, size=n)

    return {
        "factors": factors,
        "omega": omega,
        "specific": specific,
        "expected": expected,
        "risk_aversion": float(rng.uniform(0.5, 3.0)),
        "lower": np.zeros(n),
        "upper": rng.uniform(1.5 / n, 3.0 / n, size=n),
        "previous": previous,
        "costs": costs,
    }


def numpy_objective(instance, weights):
    """Mean-variance objective plus the exact L1 cost, solver-independent."""
    covariance = (
        instance["factors"] @ instance["omega"] @ instance["factors"].T
        + np.diag(instance["specific"])
    )
    value = 0.5 * instance["risk_aversion"] * weights @ covariance @ weights
    value -= instance["expected"] @ weights
    value += instance["costs"] @ np.abs(weights - instance["previous"])
    return float(value)


class L1ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.factors = np.array([[1.0], [-0.5], [0.25]], dtype=np.float64)
        self.omega = np.array([[0.1]], dtype=np.float64)
        self.specific = np.array([0.2, 0.3, 0.25], dtype=np.float64)
        self.expected = np.array([0.08, 0.04, 0.06], dtype=np.float64)
        self.previous = np.array([0.4, 0.35, 0.25], dtype=np.float64)

    def test_scalar_and_array_costs_agree(self) -> None:
        scalar = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            previous_weights=self.previous,
            l1_turnover_costs=0.002,
        )
        array = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            previous_weights=self.previous,
            l1_turnover_costs=np.full(3, 0.002),
        )

        self.assertEqual(scalar.status, "solved")
        self.assertEqual(array.status, "solved")
        np.testing.assert_allclose(scalar.weights, array.weights, atol=1.0e-9)

    def test_l1_costs_require_previous_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "previous_weights is required"):
            solve_mean_variance_factor(
                self.factors,
                self.omega,
                self.specific,
                self.expected,
                l1_turnover_costs=0.002,
            )

    def test_negative_costs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            solve_mean_variance_factor(
                self.factors,
                self.omega,
                self.specific,
                self.expected,
                previous_weights=self.previous,
                l1_turnover_costs=np.array([-0.001, 0.001, 0.001]),
            )

    def test_overwhelming_costs_freeze_the_previous_portfolio(self) -> None:
        result = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            previous_weights=self.previous,
            l1_turnover_costs=10.0,
        )

        self.assertEqual(result.status, "solved")
        self.assertTrue(result.polished)
        np.testing.assert_allclose(result.weights, self.previous, atol=1.0e-8)

    def test_l1_and_quadratic_turnover_combine(self) -> None:
        result = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            previous_weights=self.previous,
            turnover_penalty=0.5,
            l1_turnover_costs=0.002,
        )
        self.assertEqual(result.status, "solved")

    def test_sequence_rolls_the_l1_anchor_without_refactorizing(self) -> None:
        problem = PortfolioProblem(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            previous_weights=self.previous,
            l1_turnover_costs=0.002,
        )
        sequence = problem.sequence()
        first = sequence.solve_next()
        self.assertEqual(first.status, "solved")
        factorizations = sequence.factorizations

        second = sequence.solve_next(
            expected_returns=np.array([0.07, 0.05, 0.06]),
            previous_weights=first.weights,
        )
        self.assertEqual(second.status, "solved")
        self.assertEqual(sequence.factorizations, factorizations)

        # The rolled date must agree with a fresh problem carrying the
        # updated anchor and returns.
        fresh = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            np.array([0.07, 0.05, 0.06]),
            previous_weights=first.weights,
            l1_turnover_costs=0.002,
        )
        self.assertAlmostEqual(second.objective, fresh.objective, places=5)


class L1GoldStandardTest(unittest.TestCase):
    """cvxpy + Clarabel oracle on random L1 instances."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cp = pytest.importorskip("cvxpy")

    def solve_with_clarabel(self, instance):
        cp = self.cp
        n = instance["factors"].shape[0]
        weights = cp.Variable(n)
        covariance = (
            instance["factors"] @ instance["omega"] @ instance["factors"].T
            + np.diag(instance["specific"])
        )
        objective = 0.5 * instance["risk_aversion"] * cp.quad_form(
            weights, cp.psd_wrap(covariance)
        )
        objective -= instance["expected"] @ weights
        objective += instance["costs"] @ cp.abs(weights - instance["previous"])
        constraints = [
            cp.sum(weights) == 1.0,
            weights >= instance["lower"],
            weights <= instance["upper"],
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.CLARABEL)
        self.assertEqual(problem.status, "optimal")
        return np.asarray(weights.value)

    def test_agrees_with_cvxpy_clarabel_on_random_instances(self) -> None:
        for seed in range(12):
            with self.subTest(seed=seed):
                instance = random_l1_instance(seed)
                result = solve_mean_variance_factor(
                    instance["factors"],
                    instance["omega"],
                    instance["specific"],
                    instance["expected"],
                    risk_aversion=instance["risk_aversion"],
                    lower_bounds=instance["lower"],
                    upper_bounds=instance["upper"],
                    previous_weights=instance["previous"],
                    l1_turnover_costs=instance["costs"],
                )
                self.assertEqual(result.status, "solved")
                reference = self.solve_with_clarabel(instance)

                ledge_objective = numpy_objective(instance, result.weights)
                clarabel_objective = numpy_objective(instance, reference)
                scale = 1.0 + abs(clarabel_objective)
                self.assertLessEqual(
                    ledge_objective - clarabel_objective,
                    RELATIVE_OBJECTIVE_TOLERANCE * scale,
                    f"Ledge objective {ledge_objective} worse than Clarabel "
                    f"{clarabel_objective} beyond tolerance",
                )
                np.testing.assert_allclose(
                    result.weights, reference, atol=WEIGHT_TOLERANCE
                )


if __name__ == "__main__":
    unittest.main()
