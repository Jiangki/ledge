"""Public-data-style rolling rebalance backtest (roadmap 2.7).

A monthly momentum strategy on a synthetic-but-realistic equity universe:
a market + sector + style factor model generates 36 months of returns, a
12-1 momentum signal produces each month's expected returns, and the
portfolio is rebalanced with exact proportional transaction costs (10 bps,
roadmap 2.1) against a tracking benchmark (roadmap 2.6).

The point of the example is the last table: the same 24 rebalance dates are
solved twice —

- cold: a fresh ``PortfolioProblem.solve()`` per date (equilibration,
  reduced factorization, and ADMM from scratch every time), and
- rolling: one ``PortfolioProblem.sequence()`` where equilibration and the
  SMW-reduced factorizations are built once, per-date data arrives as
  ``solve_next(...)`` updates, and warm starts chain automatically.

Everything is seeded, so the published numbers in ``README.md`` are
reproducible with ``python rolling_backtest.py``.
"""

import time

import numpy as np

from ledge import PortfolioProblem, solve_mean_variance_factor

SEED = 7
ASSETS = 300
SECTORS = 10
MONTHS = 36
LOOKBACK = 12
REBALANCE_DATES = MONTHS - LOOKBACK  # 24 monthly rebalances
RISK_AVERSION = 8.0
L1_COST_BPS = 10.0
MAX_WEIGHT = 0.02
INFORMATION_COEFFICIENT = 0.05


def build_universe(rng):
    """Market + sector + momentum factor structure with monthly returns."""
    market_beta = rng.normal(1.0, 0.3, size=ASSETS)
    sector_of = rng.integers(0, SECTORS, size=ASSETS)
    sector_loadings = np.zeros((ASSETS, SECTORS))
    sector_loadings[np.arange(ASSETS), sector_of] = 1.0
    momentum_loading = rng.normal(0.0, 1.0, size=ASSETS)

    exposures = np.column_stack([market_beta, sector_loadings, momentum_loading])
    factor_count = exposures.shape[1]

    # Monthly factor vols: ~4.5% market, ~2% sectors, ~1.5% style.
    factor_vol = np.concatenate([[0.045], np.full(SECTORS, 0.02), [0.015]])
    omega = np.diag(factor_vol**2)
    specific_vol = rng.uniform(0.03, 0.08, size=ASSETS)
    specific = specific_vol**2

    factor_returns = rng.normal(0.0, factor_vol, size=(MONTHS, factor_count))
    specific_returns = rng.normal(0.0, specific_vol, size=(MONTHS, ASSETS))
    asset_returns = factor_returns @ exposures.T + specific_returns

    return exposures, omega, specific, asset_returns


def momentum_signal(asset_returns, date):
    """12-1 momentum, cross-sectionally standardized, IC-scaled."""
    window = asset_returns[date : date + LOOKBACK - 1]
    cumulative = np.prod(1.0 + window, axis=0) - 1.0
    zscore = (cumulative - cumulative.mean()) / cumulative.std()
    return INFORMATION_COEFFICIENT * 0.05 * zscore


def cap_weight_benchmark(rng):
    """A fixed cap-weight-style benchmark inside the box constraints."""
    size = rng.lognormal(0.0, 1.0, size=ASSETS)
    benchmark = size / size.sum()
    return np.minimum(benchmark, MAX_WEIGHT * 0.9)


def main() -> None:
    rng = np.random.default_rng(SEED)
    exposures, omega, specific, asset_returns = build_universe(rng)
    benchmark = cap_weight_benchmark(rng)
    dates = range(REBALANCE_DATES)

    def problem_for(date, previous_weights):
        return {
            "risk_aversion": RISK_AVERSION,
            "lower_bounds": np.zeros(ASSETS),
            "upper_bounds": np.full(ASSETS, MAX_WEIGHT),
            "previous_weights": previous_weights,
            "l1_turnover_costs": L1_COST_BPS * 1.0e-4,
            "benchmark_weights": benchmark,
        }

    # ---- cold pass: a fresh problem (and factorization) every date --------
    cold_iterations, cold_seconds = [], []
    previous = benchmark.copy()
    for date in dates:
        expected = momentum_signal(asset_returns, date)
        start = time.perf_counter()
        result = solve_mean_variance_factor(
            exposures, omega, specific, expected, **problem_for(date, previous)
        )
        cold_seconds.append(time.perf_counter() - start)
        cold_iterations.append(result.iterations)
        assert result.status == "solved"
        previous = result.weights

    # ---- rolling pass: one sequence, per-date updates, chained warm starts
    problem = PortfolioProblem(
        exposures,
        omega,
        specific,
        momentum_signal(asset_returns, 0),
        **problem_for(0, benchmark.copy()),
    )
    sequence = problem.sequence()

    warm_iterations, warm_seconds, turnover = [], [], []
    previous = benchmark.copy()
    for date in dates:
        start = time.perf_counter()
        if date == 0:
            result = sequence.solve_next()
        else:
            result = sequence.solve_next(
                expected_returns=momentum_signal(asset_returns, date),
                previous_weights=previous,
            )
        warm_seconds.append(time.perf_counter() - start)
        warm_iterations.append(result.iterations)
        assert result.status == "solved"
        turnover.append(0.5 * float(np.abs(result.weights - previous).sum()))
        previous = result.weights

    # ---- report ------------------------------------------------------------
    print(
        f"universe: {ASSETS} assets, {1 + SECTORS + 1} factors, "
        f"{REBALANCE_DATES} monthly rebalances"
    )
    print(
        f"model: momentum signal, {L1_COST_BPS:.0f} bps proportional costs, "
        f"tracking benchmark, weights in [0, {MAX_WEIGHT}]"
    )
    print()
    print("date | cold iters | rolling iters | cold ms | rolling ms | turnover")
    for date in dates:
        print(
            f"{date:4d} | {cold_iterations[date]:10d} | "
            f"{warm_iterations[date]:13d} | {1e3 * cold_seconds[date]:7.2f} | "
            f"{1e3 * warm_seconds[date]:10.2f} | {turnover[date]:.4f}"
        )
    print()
    warm_only_iters = warm_iterations[1:]
    warm_only_secs = warm_seconds[1:]
    cold_after = cold_iterations[1:]
    cold_after_secs = cold_seconds[1:]
    print(
        f"warm dates (2..{REBALANCE_DATES}): iterations "
        f"{np.mean(cold_after):.0f} -> {np.mean(warm_only_iters):.0f} per date "
        f"({np.mean(cold_after) / np.mean(warm_only_iters):.1f}x), wall time "
        f"{1e3 * np.mean(cold_after_secs):.2f} -> "
        f"{1e3 * np.mean(warm_only_secs):.2f} ms per date "
        f"({np.mean(cold_after_secs) / np.mean(warm_only_secs):.1f}x)"
    )
    print(
        f"reduced factorizations: cold pass {REBALANCE_DATES}, "
        f"rolling pass {sequence.factorizations}"
    )


if __name__ == "__main__":
    main()
