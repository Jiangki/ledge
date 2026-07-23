"""Multi-account solve_batch API tests (roadmap 3.2)."""

import unittest

import numpy as np

from ledge import PortfolioProblem, solve_batch

ASSETS = 30
FACTORS = 3
ACCOUNTS = 3
DATES = 4


def base_arrays():
    rng = np.random.default_rng(21)
    factors = rng.normal(0.0, 0.2, size=(ASSETS, FACTORS))
    omega = np.eye(FACTORS) * 0.05
    specific = rng.uniform(0.05, 0.10, size=ASSETS)
    return factors, omega, specific


def account_returns(account, date):
    rng = np.random.default_rng(100 * account + date)
    return 0.05 + rng.normal(0.0, 0.01, size=ASSETS)


def build_problem(account, **extra):
    factors, omega, specific = base_arrays()
    return PortfolioProblem(
        factors,
        omega,
        specific,
        account_returns(account, 0),
        risk_aversion=6.0,
        lower_bounds=np.zeros(ASSETS),
        upper_bounds=np.full(ASSETS, 0.2),
        **extra,
    )


def account_steps(account):
    return [
        {} if date == 0 else {"expected_returns": account_returns(account, date)}
        for date in range(DATES)
    ]


class SolveBatchTest(unittest.TestCase):
    def test_batch_matches_per_account_sequences(self) -> None:
        problems = [build_problem(account) for account in range(ACCOUNTS)]
        steps = [account_steps(account) for account in range(ACCOUNTS)]

        results = solve_batch(problems, steps)

        self.assertEqual(len(results), ACCOUNTS)
        for account in range(ACCOUNTS):
            self.assertEqual(len(results[account]), DATES)
            sequence = problems[account].sequence()
            for date in range(DATES):
                reference = sequence.solve_next(**steps[account][date])
                batch = results[account][date]
                self.assertEqual(batch.status, "solved")
                self.assertEqual(batch.iterations, reference.iterations)
                np.testing.assert_array_equal(batch.weights, reference.weights)

    def test_chained_anchors_follow_solved_weights(self) -> None:
        anchor = np.full(ASSETS, 1.0 / ASSETS)
        problems = [
            build_problem(0, previous_weights=anchor, turnover_penalty=0.5)
        ]
        steps = [account_steps(0)]

        results = solve_batch(problems, steps, chain_previous_weights=True)

        sequence = problems[0].sequence()
        held = None
        for date in range(DATES):
            kwargs = dict(steps[0][date])
            if held is not None:
                kwargs["previous_weights"] = held
            reference = sequence.solve_next(**kwargs)
            np.testing.assert_array_equal(results[0][date].weights, reference.weights)
            held = reference.weights

    def test_chaining_requires_a_turnover_term(self) -> None:
        with self.assertRaisesRegex(ValueError, "account 0.*turnover"):
            solve_batch(
                [build_problem(0)], [account_steps(0)], chain_previous_weights=True
            )

    def test_step_values_accept_lists_and_none(self) -> None:
        problems = [build_problem(0)]
        returns = account_returns(0, 1)
        steps = [[{"expected_returns": None}, {"expected_returns": list(returns)}]]

        results = solve_batch(problems, steps)

        reference = build_problem(0).sequence()
        reference.solve_next()
        rolled = reference.solve_next(expected_returns=returns)
        np.testing.assert_array_equal(results[0][1].weights, rolled.weights)

    def test_invalid_input_reports_the_account(self) -> None:
        problems = [build_problem(0), build_problem(1)]
        good = account_steps(1)
        with self.assertRaisesRegex(ValueError, "unknown step key 'budgett'"):
            solve_batch(problems, [[{"budgett": 0.9}], good])
        with self.assertRaisesRegex(ValueError, "account 1"):
            solve_batch(problems, [good, [{"expected_returns": np.zeros(3)}]])
        with self.assertRaisesRegex(ValueError, "same length"):
            solve_batch(problems, [good])

    def test_raise_on_failure_names_account_and_date(self) -> None:
        problems = [build_problem(account) for account in range(2)]
        steps = [account_steps(account)[:2] for account in range(2)]

        with self.assertRaisesRegex(RuntimeError, "account 0, step 0"):
            solve_batch(problems, steps, max_iterations=1)

        results = solve_batch(
            problems, steps, max_iterations=1, raise_on_failure=False
        )
        for account_results in results:
            for result in account_results:
                self.assertEqual(result.status, "maximum iterations reached")


if __name__ == "__main__":
    unittest.main()
