"""Backtest = `StrategyRunner` + `HistoricalBarFeed` + `SimulatedBroker` (no backtest-only trading logic)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from quantlab.backtest.metrics import compute_metrics
from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.execution.router import Trade
from quantlab.execution.simulated import SimulatedBroker
from quantlab.runtime.bar_feed import HistoricalBarFeed
from quantlab.runtime.runner import StrategyRunner
from quantlab.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    equity: pd.Series
    #: closed trades, then the positions still open at the end marked at the last close (reason "open")
    trades: list[Trade]

    def trades_frame(self) -> pd.DataFrame:
        rows = [
            {**t.__dict__, "side": t.side.value, "pnl": t.pnl, "pnl_net": t.pnl_net, "return_pct": t.return_pct}
            for t in self.trades
        ]
        return pd.DataFrame(rows)


def backtest(
    strategies: Sequence[Strategy],
    training_data: pd.DataFrame,
    backtest_data: pd.DataFrame,
    *,
    capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    risk_limits: RiskLimits | None = None,
    precompute_features: bool = True,
) -> BacktestResult:
    """
    Fit the strategies on `training_data`, then trade `backtest_data` bar by bar.

    :param precompute_features: compute the features of the whole backtest at once (batch mode): same
        values as the bar-by-bar live path (enforced by the parity tests), much faster. Set False to
        exercise the live path.
    """
    if backtest_data.empty or not strategies:
        raise ValueError("Backtest data and at least one strategy are required.")
    for strategy in strategies:
        strategy.fit(training_data)
        if precompute_features:
            strategy.prepare(backtest_data)

    broker = SimulatedBroker(fee_bps=fee_bps, slippage_bps=slippage_bps)
    runner = StrategyRunner(strategies, broker, RiskManager(risk_limits), capital=capital)
    logger.info(
        "Backtest of %s on %d bars (%s to %s)",
        ", ".join(s.strategy_id for s in strategies),
        len(backtest_data),
        backtest_data.index[0],
        backtest_data.index[-1],
    )
    runner.run(HistoricalBarFeed(backtest_data))

    trades = runner.router.trades + runner.router.open_trades(runner.last_prices(), backtest_data.index[-1])
    metrics = compute_metrics(runner.equity, trades, capital)
    logger.info(
        "Backtest: trades=%d win_rate=%.1f%% return=%.3f%% fees=%.2f sharpe=%.2f max_dd=%.3f%%",
        metrics["total_trades"],
        metrics["win_rate"],
        metrics["total_return_pct"],
        metrics["total_fees"],
        metrics["sharpe_ratio"],
        metrics["max_drawdown_pct"],
    )
    return BacktestResult(metrics=metrics, equity=runner.equity, trades=trades)
