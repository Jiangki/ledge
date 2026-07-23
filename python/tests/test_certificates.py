import unittest

import numpy as np

from ledge import PortfolioProblem

SECTOR_MATRIX = np.array(
    [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float64
)


def sector_capped_problem(cap: float) -> PortfolioProblem:
    """Long-only, budget 1, two sector caps of `cap` each."""
    return PortfolioProblem(
        np.array([[0.9], [1.1], [0.8], [1.2]], dtype=np.float64),
        np.array([[0.05]], dtype=np.float64),
        np.array([0.1, 0.12, 0.09, 0.11], dtype=np.float64),
        np.array([0.08, 0.06, 0.05, 0.07], dtype=np.float64),
        inequality_matrix=SECTOR_MATRIX,
        inequality_rhs=np.array([cap, cap], dtype=np.float64),
    )


class CertificateTest(unittest.TestCase):
    def test_infeasible_caps_return_an_auditable_farkas_certificate(self) -> None:
        result = sector_capped_problem(0.4).solve(raise_on_failure=False)

        self.assertEqual(result.status, "primal infeasible")
        certificate = result.certificate
        self.assertIsNotNone(certificate)
        self.assertEqual(certificate.kind, "primal")
        self.assertIsNone(certificate.direction)

        # Audit the Farkas conditions independently: the weighted constraint
        # combination cancels every variable yet demands a negative value.
        equality_dual = certificate.equality_dual
        inequality_dual = certificate.inequality_dual
        bound_dual = certificate.bound_dual
        self.assertEqual(equality_dual.shape, (1,))  # budget row
        self.assertEqual(inequality_dual.shape, (2,))  # sector caps
        self.assertEqual(bound_dual.shape, (4,))
        self.assertGreaterEqual(float(inequality_dual.min()), 0.0)

        combination = (
            np.ones(4) * equality_dual[0]
            + SECTOR_MATRIX.T @ inequality_dual
            + bound_dual
        )
        support_gap = (
            1.0 * equality_dual[0]
            + np.array([0.4, 0.4]) @ inequality_dual
            + np.ones(4) @ np.maximum(bound_dual, 0.0)  # upper bounds are 1
            + np.zeros(4) @ np.minimum(bound_dual, 0.0)  # lower bounds are 0
        )
        self.assertLessEqual(float(np.abs(combination).max()), 1.0e-5)
        self.assertLessEqual(float(support_gap), -1.0e-5)

    def test_infeasible_solve_raises_with_portfolio_vocabulary(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "primal infeasible"):
            sector_capped_problem(0.4).solve()

        result = sector_capped_problem(0.4).solve(raise_on_failure=False)
        self.assertTrue(
            any("budget" in hint for hint in result.convergence_hints),
            f"hints were {result.convergence_hints}",
        )

    def test_detection_can_be_disabled(self) -> None:
        result = sector_capped_problem(0.4).solve(
            infeasibility_tolerance=0.0,
            max_iterations=200,
            raise_on_failure=False,
        )

        self.assertEqual(result.status, "maximum iterations reached")
        self.assertIsNone(result.certificate)

    def test_solved_result_has_no_certificate(self) -> None:
        result = sector_capped_problem(0.6).solve()

        self.assertEqual(result.status, "solved")
        self.assertIsNone(result.certificate)

    def test_sequence_flags_the_bad_date_and_recovers(self) -> None:
        sequence = sector_capped_problem(0.6).sequence()

        first = sequence.solve_next()
        self.assertEqual(first.status, "solved")

        bad = sequence.solve_next(
            inequality_rhs=np.array([0.2, 0.2]), raise_on_failure=False
        )
        self.assertEqual(bad.status, "primal infeasible")
        self.assertEqual(bad.certificate.kind, "primal")

        recovered = sequence.solve_next(inequality_rhs=np.array([0.7, 0.7]))
        self.assertEqual(recovered.status, "solved")


if __name__ == "__main__":
    unittest.main()
