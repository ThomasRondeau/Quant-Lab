"""Performance metrics from an equity curve and a trade list (shared by backtest and replay reports)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantlab.execution.router import Trade

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    """Close-to-close daily returns of an intraday equity curve (the first day starts from the first point)."""
    if equity.empty:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex(equity.index)
    days = index.tz_convert("America/New_York").date if index.tz is not None else index.date
    closes = equity.groupby(days).last()
    previous = closes.shift(1)
    previous.iloc[0] = equity.iloc[0]
    return closes / previous - 1


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative percentage."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((equity - peak) / peak).min() * 100)


def compute_metrics(equity: pd.Series, trades: Sequence[Trade], initial_capital: float) -> dict[str, float]:
    returns = daily_returns(equity)
    std = returns.std() if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    downside_dev = math.sqrt((downside**2).sum() / len(returns)) if len(returns) > 1 else 0.0

    pnl_net = np.array([t.pnl_net for t in trades], dtype=float)
    gains, losses = pnl_net[pnl_net > 0].sum(), -pnl_net[pnl_net < 0].sum()
    final = float(equity.iloc[-1]) if not equity.empty else initial_capital
    return {
        "total_trades": float(len(trades)),
        "win_rate": float((pnl_net > 0).mean() * 100) if len(trades) else 0.0,
        "total_return_pct": (final / initial_capital - 1) * 100,
        "total_pnl_net": float(pnl_net.sum()),
        "total_fees": float(sum(t.fees for t in trades)),
        "avg_trade_return_pct": float(np.mean([t.return_pct for t in trades])) if len(trades) else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else (math.inf if gains > 0 else 0.0),
        "sharpe_ratio": float(returns.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0,
        "sortino_ratio": float(returns.mean() / downside_dev * math.sqrt(TRADING_DAYS)) if downside_dev > 0 else 0.0,
        "max_drawdown_pct": max_drawdown(equity),
    }
