"""Rolling multi-date rebalance through the PortfolioSequence API.

One fixed factor structure, ten dates of new expected returns anchored on the
previous date's weights. The sequence keeps the equilibration and the reduced
factorizations cached across dates and chains warm starts internally, so each
date only pays iteration cost.
"""

import numpy as np

from ledge import PortfolioProblem


def main() -> None:
    rng = np.random.default_rng(42)
    assets, factors, dates = 60, 6, 10

    exposures = rng.normal(0.0, 0.25, size=(assets, factors))
    omega = np.diag(rng.uniform(0.04, 0.16, size=factors))
    specific = rng.uniform(0.05, 0.12, size=assets)
    expected = rng.normal(0.08, 0.025, size=assets)
    anchor = np.full(assets, 1.0 / assets)

    problem = PortfolioProblem(
        exposures,
        omega,
        specific,
        expected,
        risk_aversion=8.0,
        lower_bounds=np.zeros(assets),
        upper_bounds=np.full(assets, 0.05),
        previous_weights=anchor,
        turnover_penalty=0.5,
    )
    sequence = problem.sequence()

    print("date | status | iterations | factorizations | one-way turnover")
    previous_weights = None
    for date in range(dates):
        if date == 0:
            result = sequence.solve_next()
        else:
            result = sequence.solve_next(
                expected_returns=expected + rng.normal(0.0, 0.01, size=assets),
                previous_weights=previous_weights,
            )
        turnover = (
            0.0
            if previous_weights is None
            else 0.5 * float(np.abs(result.weights - previous_weights).sum())
        )
        print(
            f"{date} | {result.status} | {result.iterations} | "
            f"{sequence.factorizations} | {turnover:.6f}"
        )
        previous_weights = result.weights

    print(f"total reduced factorizations across {dates} dates: "
          f"{sequence.factorizations}")


if __name__ == "__main__":
    main()
