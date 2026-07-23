"""Tracking-error objective tests for the Python bindings (roadmap 2.6).

`benchmark_weights` must be pure sugar for the same QP with the linear cost
shifted by `-risk_aversion * Sigma @ benchmark`. The gold-standard class
compares against cvxpy + Clarabel minimizing the explicit
`(w - b)' Sigma (w - b)` objective; the API class runs without cvxpy.
"""

import unittest

import numpy as np
import pytest

from ledge import PortfolioProblem, solve_mean_variance_factor

RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-5
WEIGHT_TOLERANCE = 5.0e-4


def random_tracking_instance(seed):
    """Random long-only factor QP with a feasible benchmark."""
    rng = np.random.default_rng(seed)
    n = int(rng.choice([5, 10, 20]))
    k = int(rng.choice([1, 2, 3]))

    factors = 0.3 * rng.normal(size=(n, k)) / np.sqrt(k)
    root = rng.normal(size=(k, k))
    omega = root @ root.T / k + 0.05 * np.eye(k)
    specific = rng.uniform(0.05, 0.2, size=n)
    expected = 0.01 + 0.005 * rng.normal(size=n)
    benchmark = rng.uniform(0.5, 1.5, size=n)
    benchmark /= benchmark.sum()

    return {
        "factors": factors,
        "omega": omega,
        "specific": specific,
        "expected": expected,
        "risk_aversion": float(rng.uniform(0.5, 3.0)),
        "lower": np.zeros(n),
        "upper": np.full(n, 3.0 / n),
        "benchmark": benchmark,
    }


def covariance_of(instance):
    return (
        instance["factors"] @ instance["omega"] @ instance["factors"].T
        + np.diag(instance["specific"])
    )


def numpy_active_risk_objective(instance, weights):
    """Tracking objective with the full constant, solver-independent."""
    active = weights - instance["benchmark"]
    value = 0.5 * instance["risk_aversion"] * active @ covariance_of(instance) @ active
    value -= instance["expected"] @ weights
    return float(value)


class TrackingApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.factors = np.array([[1.0], [-0.5], [0.25]], dtype=np.float64)
        self.omega = np.array([[0.1]], dtype=np.float64)
        self.specific = np.array([0.2, 0.3, 0.25], dtype=np.float64)
        self.expected = np.array([0.08, 0.04, 0.06], dtype=np.float64)
        self.benchmark = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    def test_pure_tracking_reproduces_a_feasible_benchmark(self) -> None:
        result = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            np.zeros(3),
            benchmark_weights=self.benchmark,
        )
        self.assertEqual(result.status, "solved")
        np.testing.assert_allclose(result.weights, self.benchmark, atol=1.0e-6)

    def test_benchmark_matches_manually_adjusted_returns(self) -> None:
        risk_aversion = 4.0
        covariance = (
            self.factors @ self.omega @ self.factors.T + np.diag(self.specific)
        )
        adjusted = self.expected + risk_aversion * covariance @ self.benchmark

        tracking = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            risk_aversion=risk_aversion,
            benchmark_weights=self.benchmark,
        )
        absolute = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            adjusted,
            risk_aversion=risk_aversion,
        )
        self.assertEqual(tracking.status, "solved")
        self.assertEqual(absolute.status, "solved")
        np.testing.assert_allclose(tracking.weights, absolute.weights, atol=1.0e-6)

    def test_wrong_length_benchmark_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            solve_mean_variance_factor(
                self.factors,
                self.omega,
                self.specific,
                self.expected,
                benchmark_weights=np.array([0.5, 0.5]),
            )

    def test_sequence_rolls_the_benchmark_without_refactorizing(self) -> None:
        problem = PortfolioProblem(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            benchmark_weights=self.benchmark,
        )
        sequence = problem.sequence()
        first = sequence.solve_next()
        self.assertEqual(first.status, "solved")
        factorizations = sequence.factorizations

        new_benchmark = np.array([0.4, 0.35, 0.25])
        second = sequence.solve_next(benchmark_weights=new_benchmark)
        self.assertEqual(second.status, "solved")
        self.assertEqual(sequence.factorizations, factorizations)

        fresh = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            benchmark_weights=new_benchmark,
        )
        self.assertAlmostEqual(second.objective, fresh.objective, places=5)

    def test_benchmark_updates_require_a_tracking_base(self) -> None:
        problem = PortfolioProblem(
            self.factors, self.omega, self.specific, self.expected
        )
        sequence = problem.sequence()
        with self.assertRaisesRegex(ValueError, "with_tracking_benchmark"):
            sequence.solve_next(benchmark_weights=self.benchmark)


class TrackingGoldStandardTest(unittest.TestCase):
    """cvxpy + Clarabel oracle minimizing the explicit active-risk form."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cp = pytest.importorskip("cvxpy")

    def solve_with_clarabel(self, instance):
        cp = self.cp
        n = instance["factors"].shape[0]
        weights = cp.Variable(n)
        active = weights - instance["benchmark"]
        objective = 0.5 * instance["risk_aversion"] * cp.quad_form(
            active, cp.psd_wrap(covariance_of(instance))
        )
        objective -= instance["expected"] @ weights
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
                instance = random_tracking_instance(seed)
                result = solve_mean_variance_factor(
                    instance["factors"],
                    instance["omega"],
                    instance["specific"],
                    instance["expected"],
                    risk_aversion=instance["risk_aversion"],
                    lower_bounds=instance["lower"],
                    upper_bounds=instance["upper"],
                    benchmark_weights=instance["benchmark"],
                )
                self.assertEqual(result.status, "solved")
                reference = self.solve_with_clarabel(instance)

                ledge_objective = numpy_active_risk_objective(
                    instance, result.weights
                )
                clarabel_objective = numpy_active_risk_objective(
                    instance, reference
                )
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
