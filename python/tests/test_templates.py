"""Constraint template builders (roadmap 3.1).

Templates must be pure sugar over the existing constraint machinery: each
kwarg reproduces exactly what the equivalent hand-built matrices produce,
so every test cross-checks against the explicit form through the public
API.
"""

import unittest

import numpy as np

from ledge import PortfolioProblem

CONSTRAINT_TOLERANCE = 1.0e-5


def fixture(assets: int = 36, factor_count: int = 3):
    rng = np.random.default_rng(20260722)
    factors = 0.3 * rng.standard_normal((assets, factor_count))
    omega = np.diag(np.linspace(0.05, 0.08, factor_count))
    specific = np.linspace(0.08, 0.12, assets)
    expected = 0.05 + 0.03 * np.cos(0.7 * np.arange(assets))
    return factors, omega, specific, expected


def benchmark_weights(assets: int) -> np.ndarray:
    raw = 1.0 + 0.2 * ((np.arange(assets) % 7) - 3.0) / 3.0
    return raw / raw.sum()


class IndustryTemplateTest(unittest.TestCase):
    def test_industry_neutrality_matches_benchmark_industry_weights(self) -> None:
        factors, omega, specific, expected = fixture()
        assets = len(expected)
        industries = np.arange(assets) % 4
        benchmark = benchmark_weights(assets)

        problem = PortfolioProblem(
            factors,
            omega,
            specific,
            expected,
            benchmark_weights=benchmark,
            industry_ids=[int(industry) for industry in industries],
        )
        result = problem.solve()
        self.assertEqual(result.status, "solved")
        for industry in range(4):
            members = industries == industry
            held = float(result.weights[members].sum())
            target = float(benchmark[members].sum())
            self.assertAlmostEqual(held, target, delta=CONSTRAINT_TOLERANCE)

    def test_explicit_targets_match_hand_built_equalities(self) -> None:
        factors, omega, specific, expected = fixture()
        assets = len(expected)
        groups = np.arange(assets) % 3
        targets = np.array([0.5, 0.3, 0.2])

        templated = PortfolioProblem(
            factors,
            omega,
            specific,
            expected,
            industry_ids=[int(group) for group in groups],
            industry_targets=targets,
        ).solve()

        matrix = np.zeros((3, assets))
        for group in range(3):
            matrix[group, groups == group] = 1.0
        manual = PortfolioProblem(
            factors,
            omega,
            specific,
            expected,
            equality_matrix=matrix,
            equality_rhs=targets,
        ).solve()

        np.testing.assert_array_equal(templated.weights, manual.weights)

    def test_targets_without_ids_are_rejected(self) -> None:
        factors, omega, specific, expected = fixture()
        with self.assertRaisesRegex(ValueError, "industry_ids"):
            PortfolioProblem(
                factors,
                omega,
                specific,
                expected,
                industry_targets=np.array([0.5, 0.5]),
            )

    def test_neutrality_without_benchmark_is_rejected(self) -> None:
        factors, omega, specific, expected = fixture()
        with self.assertRaisesRegex(ValueError, "benchmark"):
            PortfolioProblem(
                factors,
                omega,
                specific,
                expected,
                industry_ids=[0] * len(expected),
            )


class StyleTemplateTest(unittest.TestCase):
    def test_style_bounds_match_hand_built_inequalities(self) -> None:
        factors, omega, specific, expected = fixture()
        assets = len(expected)
        rng = np.random.default_rng(7)
        styles = rng.standard_normal((2, assets))
        lower = np.array([-0.15, -np.inf])
        upper = np.array([0.15, 0.2])

        templated = PortfolioProblem(
            factors,
            omega,
            specific,
            expected,
            style_matrix=styles,
            style_lower=lower,
            style_upper=upper,
        ).solve()

        # Documented row order: upper then lower per style, finite sides only.
        matrix = np.vstack([styles[0], -styles[0], styles[1]])
        rhs = np.array([0.15, 0.15, 0.2])
        manual = PortfolioProblem(
            factors,
            omega,
            specific,
            expected,
            inequality_matrix=matrix,
            inequality_rhs=rhs,
        ).solve()

        np.testing.assert_array_equal(templated.weights, manual.weights)
        exposures = styles @ templated.weights
        self.assertLessEqual(exposures[0], 0.15 + CONSTRAINT_TOLERANCE)
        self.assertGreaterEqual(exposures[0], -0.15 - CONSTRAINT_TOLERANCE)
        self.assertLessEqual(exposures[1], 0.2 + CONSTRAINT_TOLERANCE)

    def test_partial_style_kwargs_are_rejected(self) -> None:
        factors, omega, specific, expected = fixture()
        with self.assertRaisesRegex(ValueError, "provided together"):
            PortfolioProblem(
                factors,
                omega,
                specific,
                expected,
                style_matrix=np.ones((1, len(expected))),
                style_upper=np.array([0.1]),
            )


class BoxTemplateTest(unittest.TestCase):
    def test_max_weight_caps_every_position(self) -> None:
        factors, omega, specific, expected = fixture()
        result = PortfolioProblem(
            factors,
            omega,
            specific,
            expected,
            max_weight=0.06,
        ).solve()
        self.assertEqual(result.status, "solved")
        self.assertLessEqual(
            float(result.weights.max()), 0.06 + CONSTRAINT_TOLERANCE
        )

    def test_max_short_zero_keeps_long_short_book_long_only(self) -> None:
        factors, omega, specific, expected = fixture()
        assets = len(expected)
        result = PortfolioProblem(
            factors,
            omega,
            specific,
            expected,
            lower_bounds=np.full(assets, -0.1),
            upper_bounds=np.full(assets, 0.3),
            max_short=0.0,
        ).solve()
        self.assertEqual(result.status, "solved")
        self.assertGreaterEqual(
            float(result.weights.min()), -CONSTRAINT_TOLERANCE
        )

    def test_contradictory_caps_are_rejected(self) -> None:
        factors, omega, specific, expected = fixture()
        assets = len(expected)
        with self.assertRaises(ValueError):
            PortfolioProblem(
                factors,
                omega,
                specific,
                expected,
                lower_bounds=np.full(assets, 0.1),
                upper_bounds=np.ones(assets),
                max_weight=0.05,
            )


class TemplateSequenceTest(unittest.TestCase):
    def test_template_targets_roll_through_equality_rhs(self) -> None:
        factors, omega, specific, expected = fixture()
        assets = len(expected)
        groups = np.arange(assets) % 3

        problem = PortfolioProblem(
            factors,
            omega,
            specific,
            expected,
            industry_ids=[int(group) for group in groups],
            industry_targets=np.array([0.5, 0.3, 0.2]),
        )
        sequence = problem.sequence()
        first = sequence.solve_next()
        self.assertEqual(first.status, "solved")
        factorizations = sequence.factorizations

        rolled = sequence.solve_next(equality_rhs=np.array([0.4, 0.4, 0.2]))
        self.assertEqual(rolled.status, "solved")
        for group, target in enumerate([0.4, 0.4, 0.2]):
            held = float(rolled.weights[groups == group].sum())
            self.assertAlmostEqual(held, target, delta=CONSTRAINT_TOLERANCE)
        self.assertEqual(sequence.factorizations, factorizations)


if __name__ == "__main__":
    unittest.main()
