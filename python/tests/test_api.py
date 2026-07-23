import unittest

import numpy as np

from ledge import PortfolioProblem, solve_mean_variance_factor


class PortfolioApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.factors = np.array([[1.0], [-0.5], [0.25]], dtype=np.float64)
        self.omega = np.array([[0.1]], dtype=np.float64)
        self.specific = np.array([0.2, 0.3, 0.25], dtype=np.float64)
        self.expected = np.array([0.08, 0.04, 0.06], dtype=np.float64)
        self.upper = np.full(3, 0.6, dtype=np.float64)

    def test_class_api_solves_and_warm_starts(self) -> None:
        problem = PortfolioProblem(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            upper_bounds=self.upper,
            lower_bounds=np.zeros(3),
        )
        first = problem.solve()
        second = problem.solve(warm_start=first.weights)

        self.assertEqual(first.status, "solved")
        self.assertAlmostEqual(float(first.weights.sum()), 1.0, places=5)
        self.assertLessEqual(second.iterations, first.iterations)
        self.assertLess(first.primal_residual, 1.0e-5)

    def test_function_api_reports_input_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            solve_mean_variance_factor(
                self.factors,
                self.omega,
                self.specific,
                self.expected,
                lower_bounds=np.zeros(3),
            )

    def test_failed_solve_exposes_convergence_hints(self) -> None:
        result = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            max_iterations=1,
            raise_on_failure=False,
        )

        self.assertEqual(result.status, "maximum iterations reached")
        self.assertGreater(len(result.convergence_hints), 0)

    def test_failed_solve_error_message_includes_hints(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Hints:"):
            solve_mean_variance_factor(
                self.factors,
                self.omega,
                self.specific,
                self.expected,
                max_iterations=1,
            )

    def test_solved_result_has_no_hints(self) -> None:
        result = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
        )

        self.assertEqual(result.status, "solved")
        self.assertEqual(result.convergence_hints, [])

    def test_over_relaxation_agrees_with_plain_admm(self) -> None:
        relaxed = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            upper_bounds=self.upper,
            lower_bounds=np.zeros(3),
        )
        plain = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            upper_bounds=self.upper,
            lower_bounds=np.zeros(3),
            over_relaxation=1.0,
        )

        self.assertEqual(relaxed.status, "solved")
        self.assertEqual(plain.status, "solved")
        self.assertAlmostEqual(relaxed.objective, plain.objective, places=5)

    def test_over_relaxation_out_of_range_is_rejected(self) -> None:
        for alpha in (0.0, 2.0, -1.0):
            with self.assertRaisesRegex(ValueError, "over_relaxation"):
                solve_mean_variance_factor(
                    self.factors,
                    self.omega,
                    self.specific,
                    self.expected,
                    over_relaxation=alpha,
                )

    def test_polish_is_on_by_default_and_reaches_high_accuracy(self) -> None:
        result = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            upper_bounds=self.upper,
            lower_bounds=np.zeros(3),
        )

        self.assertEqual(result.status, "solved")
        self.assertTrue(result.polished)
        self.assertLess(max(result.primal_residual, result.dual_residual), 1.0e-10)
        self.assertLess(result.complementarity, 1.0e-10)

    def test_polish_can_be_disabled(self) -> None:
        result = solve_mean_variance_factor(
            self.factors,
            self.omega,
            self.specific,
            self.expected,
            upper_bounds=self.upper,
            lower_bounds=np.zeros(3),
            polish=False,
        )

        self.assertEqual(result.status, "solved")
        self.assertFalse(result.polished)

    def test_scaling_solves_badly_scaled_units(self) -> None:
        # Express each asset in its own unit (10**±2 spread) so objective and
        # constraint coefficients span many decades. Raw ADMM stalls on this
        # data; the default Ruiz equilibration must absorb the units.
        rng = np.random.default_rng(7)
        n, k = 60, 6
        scales = np.logspace(-2.0, 2.0, n)
        rng.shuffle(scales)
        factors = rng.normal(0.0, 0.2, size=(n, k)) * scales[:, None]
        omega = np.eye(k) * 0.05
        specific = rng.uniform(0.05, 0.10, size=n) * scales**2
        mu = rng.normal(0.01, 0.005, size=n) * scales

        # Reference: the uniform portfolio in original units.
        reference = np.full(n, 1.0 / n) / scales
        exposures = rng.normal(size=(3, n)) * scales[None, :]
        exposure_caps = exposures @ reference + 0.1 * np.linalg.norm(
            exposures, axis=1
        ) * np.sqrt(1.0 / n)

        kwargs = dict(
            risk_aversion=1.0,
            budget=float(reference.sum()),
            lower_bounds=np.zeros(n),
            upper_bounds=0.2 / scales,
            # Full-investment constraint written in original units.
            equality_matrix=scales[None, :],
            equality_rhs=np.array([1.0]),
            inequality_matrix=exposures,
            inequality_rhs=exposure_caps,
            raise_on_failure=False,
        )
        unscaled = solve_mean_variance_factor(
            factors, omega, specific, mu, scaling_iterations=0, **kwargs
        )
        scaled = solve_mean_variance_factor(factors, omega, specific, mu, **kwargs)

        self.assertNotEqual(unscaled.status, "solved")
        self.assertEqual(scaled.status, "solved")
        # The recovered weights are O(100) in original units, so the budget
        # holds to the solver's absolute + relative contract, not to a fixed
        # 1e-5: |sum - 1| <= abs_tol + rel_tol * max|w|.
        budget_error = abs(float((scaled.weights * scales).sum()) - 1.0)
        weight_scale = float(np.abs(scaled.weights).max())
        self.assertLessEqual(budget_error, 1.0e-6 + 1.0e-5 * max(1.0, weight_scale))


if __name__ == "__main__":
    unittest.main()
