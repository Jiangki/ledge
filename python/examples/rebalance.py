"""Two-date factor-model rebalance with a warm start and L2 turnover control."""

import numpy as np

from ledge import PortfolioProblem


def factor_snapshot(
    rng: np.random.Generator, assets: int, factors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    exposures = rng.normal(0.0, 0.25, size=(assets, factors))
    factor_variance = np.diag(rng.uniform(0.04, 0.16, size=factors))
    specific_variance = rng.uniform(0.05, 0.12, size=assets)
    expected_returns = rng.normal(0.08, 0.025, size=assets)
    return exposures, factor_variance, specific_variance, expected_returns


def main() -> None:
    rng = np.random.default_rng(42)
    assets, factors = 60, 6
    lower = np.zeros(assets)
    upper = np.full(assets, 0.05)

    exposures, omega, specific, expected = factor_snapshot(rng, assets, factors)
    first_problem = PortfolioProblem(
        exposures,
        omega,
        specific,
        expected,
        risk_aversion=8.0,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    first = first_problem.solve()

    # A new alpha vector arrives. Penalize large moves around yesterday's
    # portfolio and initialize ADMM at the previous solution.
    next_expected = expected + rng.normal(0.0, 0.01, size=assets)
    next_problem = PortfolioProblem(
        exposures,
        omega,
        specific,
        next_expected,
        risk_aversion=8.0,
        lower_bounds=lower,
        upper_bounds=upper,
        previous_weights=first.weights,
        turnover_penalty=0.5,
    )
    second = next_problem.solve(warm_start=first.weights)

    print(first)
    print(second)
    print(f"budget: {second.weights.sum():.10f}")
    print(f"one-way turnover: {0.5 * np.abs(second.weights - first.weights).sum():.6f}")


if __name__ == "__main__":
    main()
