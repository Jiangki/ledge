"""Rolling PortfolioSequence API tests (roadmap 2.5)."""

import unittest

import numpy as np

from ledge import PortfolioProblem

ASSETS = 40
FACTORS = 4


def base_arrays():
    rng = np.random.default_rng(11)
    factors = rng.normal(0.0, 0.2, size=(ASSETS, FACTORS))
    omega = np.eye(FACTORS) * 0.05
    specific = rng.uniform(0.05, 0.10, size=ASSETS)
    expected = rng.normal(0.05, 0.01, size=ASSETS)
    return factors, omega, specific, expected


def build_problem(expected, previous_weights=None, turnover_penalty=0.0):
    factors, omega, specific, _ = base_arrays()
    kwargs = dict(
        risk_aversion=6.0,
        lower_bounds=np.zeros(ASSETS),
        upper_bounds=np.full(ASSETS, 0.2),
    )
    if previous_weights is not None:
        kwargs.update(
            previous_weights=previous_weights, turnover_penalty=turnover_penalty
        )
    return PortfolioProblem(factors, omega, specific, expected, **kwargs)


class PortfolioSequenceTest(unittest.TestCase):
    def setUp(self) -> None:
        _, _, _, self.expected = base_arrays()

    def test_rolling_dates_match_fresh_solves(self) -> None:
        sequence = build_problem(self.expected).sequence()
        first = sequence.solve_next()
        self.assertEqual(first.status, "solved")

        rng = np.random.default_rng(3)
        for step in range(1, 4):
            returns = self.expected + rng.normal(0.0, 0.002, size=ASSETS)
            rolled = sequence.solve_next(expected_returns=returns)
            fresh = build_problem(returns).solve()

            self.assertEqual(rolled.status, "solved", f"step {step}")
            scale = 1.0 + abs(fresh.objective)
            self.assertLessEqual(
                abs(rolled.objective - fresh.objective), 1.0e-4 * scale, f"step {step}"
            )
            self.assertAlmostEqual(float(rolled.weights.sum()), 1.0, places=4)

    def test_factorizations_are_reused_across_dates(self) -> None:
        sequence = build_problem(self.expected).sequence()
        cold = sequence.solve_next()
        self.assertEqual(cold.status, "solved")
        after_cold = sequence.factorizations

        rng = np.random.default_rng(5)
        for _ in range(4):
            returns = self.expected + rng.normal(0.0, 0.002, size=ASSETS)
            warm = sequence.solve_next(expected_returns=returns)
            self.assertEqual(warm.status, "solved")
            self.assertLessEqual(warm.iterations, cold.iterations)
        self.assertEqual(sequence.factorizations, after_cold)

    def test_turnover_anchor_updates_roll_forward(self) -> None:
        anchor = np.full(ASSETS, 1.0 / ASSETS)
        sequence = build_problem(
            self.expected, previous_weights=anchor, turnover_penalty=0.5
        ).sequence()
        first = sequence.solve_next()
        rolled = sequence.solve_next(
            expected_returns=self.expected + 0.002,
            previous_weights=first.weights,
        )
        fresh = build_problem(
            self.expected + 0.002,
            previous_weights=first.weights,
            turnover_penalty=0.5,
        ).solve()

        self.assertEqual(rolled.status, "solved")
        scale = 1.0 + abs(fresh.objective)
        self.assertLessEqual(abs(rolled.objective - fresh.objective), 1.0e-4 * scale)

    def test_budget_update_moves_the_invested_sum(self) -> None:
        sequence = build_problem(self.expected).sequence()
        sequence.solve_next()
        rebudgeted = sequence.solve_next(budget=0.9)

        self.assertEqual(rebudgeted.status, "solved")
        self.assertAlmostEqual(float(rebudgeted.weights.sum()), 0.9, places=4)

    def test_invalid_steps_are_rejected_and_atomic(self) -> None:
        sequence = build_problem(self.expected).sequence()
        baseline = sequence.solve_next()

        with self.assertRaises(ValueError):
            sequence.solve_next(expected_returns=np.zeros(3))
        with self.assertRaisesRegex(ValueError, "with_quadratic_turnover"):
            sequence.solve_next(previous_weights=np.zeros(ASSETS))
        with self.assertRaisesRegex(ValueError, "reachable sum"):
            sequence.solve_next(budget=1000.0)
        # Valid returns + invalid budget must not leak the returns in.
        with self.assertRaises(ValueError):
            sequence.solve_next(
                expected_returns=self.expected + 0.01, budget=1000.0
            )

        recovered = sequence.solve_next()
        self.assertEqual(recovered.status, "solved")
        scale = 1.0 + abs(baseline.objective)
        self.assertLessEqual(
            abs(recovered.objective - baseline.objective), 1.0e-4 * scale
        )

    def test_failed_dates_report_hints_or_raise(self) -> None:
        sequence = build_problem(self.expected).sequence(max_iterations=1)
        result = sequence.solve_next(raise_on_failure=False)
        self.assertEqual(result.status, "maximum iterations reached")
        self.assertGreater(len(result.convergence_hints), 0)

        with self.assertRaisesRegex(RuntimeError, "raise_on_failure"):
            sequence.solve_next()

    def test_rolling_dates_are_polished(self) -> None:
        sequence = build_problem(self.expected).sequence()
        first = sequence.solve_next()
        rolled = sequence.solve_next(expected_returns=self.expected + 0.002)

        for result in (first, rolled):
            self.assertEqual(result.status, "solved")
            self.assertTrue(result.polished)
            self.assertLess(
                max(result.primal_residual, result.dual_residual), 1.0e-9
            )

        unpolished = build_problem(self.expected).sequence(polish=False).solve_next()
        self.assertFalse(unpolished.polished)

    def test_repr_reports_dimension_and_factorizations(self) -> None:
        sequence = build_problem(self.expected).sequence()
        sequence.solve_next()
        text = repr(sequence)
        self.assertIn(f"dimension={ASSETS}", text)
        self.assertIn("factorizations=", text)


if __name__ == "__main__":
    unittest.main()
