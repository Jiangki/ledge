"""Problem / result serialization for bug reproduction (roadmap 3.3).

``PortfolioProblem.to_json`` / ``from_json`` must round-trip the full
problem specification bit-exactly, and ``SolveResult.to_json`` must carry
the complete solver output (all dual blocks, residuals, certificates), so
one JSON pair attached to an issue reproduces a report.
"""

import json
import unittest

import numpy as np

from ledge import PortfolioProblem


def problem(**kwargs) -> PortfolioProblem:
    rng = np.random.default_rng(20260722)
    assets, factor_count = 24, 3
    factors = 0.3 * rng.standard_normal((assets, factor_count))
    omega = np.diag(np.linspace(0.05, 0.08, factor_count))
    specific = np.linspace(0.08, 0.12, assets)
    expected = 0.05 + 0.03 * np.cos(0.7 * np.arange(assets))
    return PortfolioProblem(factors, omega, specific, expected, **kwargs)


class ProblemRoundTripTest(unittest.TestCase):
    def test_round_trip_replays_the_solve_bit_exactly(self) -> None:
        assets = 24
        previous = np.full(assets, 1.0 / assets)
        original = problem(
            risk_aversion=4.0,
            previous_weights=previous,
            turnover_penalty=0.5,
            l1_turnover_costs=2.0e-3,
            benchmark_weights=previous,
            industry_ids=[asset % 3 for asset in range(assets)],
            max_weight=0.3,
        )
        restored = PortfolioProblem.from_json(original.to_json())

        self.assertEqual(restored.dimension, original.dimension)
        self.assertEqual(original.to_json(), restored.to_json())
        np.testing.assert_array_equal(
            original.solve().weights, restored.solve().weights
        )

    def test_unbounded_sides_survive_json(self) -> None:
        assets = 24
        lower = np.full(assets, -np.inf)
        upper = np.full(assets, 0.5)
        original = problem(lower_bounds=lower, upper_bounds=upper)
        dump = json.loads(original.to_json())
        self.assertIn(None, dump["lower_bounds"])

        restored = PortfolioProblem.from_json(original.to_json())
        self.assertEqual(original.to_json(), restored.to_json())

    def test_tampered_dumps_are_rejected_with_construction_errors(self) -> None:
        dump = json.loads(problem().to_json())
        dump["risk_aversion"] = -1.0
        with self.assertRaisesRegex(ValueError, "risk_aversion"):
            PortfolioProblem.from_json(json.dumps(dump))

        dump = json.loads(problem().to_json())
        dump["expected_returns"] = dump["expected_returns"][:-1]
        with self.assertRaises(ValueError):
            PortfolioProblem.from_json(json.dumps(dump))


class ResultSerializationTest(unittest.TestCase):
    def test_result_json_carries_the_full_solver_output(self) -> None:
        result = problem().solve()
        dump = json.loads(result.to_json())

        self.assertEqual(dump["status"], "Solved")
        np.testing.assert_array_equal(np.array(dump["x"]), result.weights)
        self.assertEqual(dump["iterations"], result.iterations)
        self.assertEqual(dump["polished"], result.polished)
        # The full dual blocks are present even though SolveResult does not
        # expose them as attributes — bug reports need the multipliers.
        for block in ("equalities", "inequalities", "bounds", "l1"):
            self.assertIn(block, dump["dual"])
        self.assertEqual(len(dump["dual"]["bounds"]), problem().dimension)

    def test_infeasible_result_json_carries_the_certificate(self) -> None:
        assets = 24
        # The budget forces sum(w) = 1 while this row demands sum(w) <= 0.5.
        infeasible = problem(
            inequality_matrix=np.ones((1, assets)),
            inequality_rhs=np.array([0.5]),
        )
        result = infeasible.solve(raise_on_failure=False)
        self.assertEqual(result.status, "primal infeasible")

        dump = json.loads(result.to_json())
        self.assertEqual(dump["status"], "PrimalInfeasible")
        certificate = dump["certificate"]["Primal"]
        np.testing.assert_array_equal(
            np.array(certificate["bound_dual"]), result.certificate.bound_dual
        )


if __name__ == "__main__":
    unittest.main()
