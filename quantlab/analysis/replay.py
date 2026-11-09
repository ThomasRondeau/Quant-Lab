"""
Replay a journaled live session in the backtester and measure the live / backtest gap.

The strategies are reloaded from the session snapshot (as they were before the first live bar) and run
by the same `StrategyRunner` on the journaled bars, with a `SimulatedBroker`. Both journals are then
compared:

- signals: entry orders per (bar, strategy, symbol, side), found live only, in the replay only, or both.
  Bars caught up after a restart are flagged: the live session could not decide on them;
- entry fills of the matched signals: live price vs simulated price (adverse slippage in bp);
- features: largest live / replay difference per feature (should be ~0: same code, same bars);
- summary: agreement rate, P&L, trades, latency.

`bar_differences` compares the live (aggregated) bars to historical bars downloaded afterwards.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.execution.simulated import SimulatedBroker
from quantlab.runtime.bar_feed import HistoricalBarFeed
from quantlab.runtime.journal import Journal, JournalData, read_journal
from quantlab.runtime.runner import StrategyRunner
from quantlab.runtime.session import SessionFiles

logger = logging.getLogger(__name__)

SIGNAL_KEY = ["bar_ts", "strategy_id", "symbol", "side"]


@dataclass
class ReplayReport:
    summary: dict[str, float]
    signals: pd.DataFrame
    entry_fills: pd.DataFrame
    feature_diffs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def log(self) -> None:
        logger.info("Replay summary:\n%s", "\n".join(f"  {k:<28} {v:,.4f}" for k, v in self.summary.items()))
        mismatched = self.signals[self.signals["match"] != "both"]
        if not mismatched.empty:
            logger.info("Signals not matched:\n%s", mismatched.to_string())


def _risk_limits(saved: dict[str, Any]) -> RiskLimits:
    """The session's limits (fields of older versions, such as the former global stop / target, are dropped)."""
    known = {f.name for f in fields(RiskLimits)}
    return RiskLimits(**{k: v for k, v in saved.items() if k in known})


def replay_session(
    session_dir: str | Path, fee_bps: float | None = None, slippage_bps: float | None = None
) -> ReplayReport:
    files = SessionFiles(session_dir)
    meta = files.meta()
    live = read_journal(files.journal_path)
    bars = live.bars.drop(columns="catch_up")
    if bars.empty:
        raise ValueError(f"No bar in the journal of {session_dir}.")

    runner_kwargs: dict[str, Any] = {
        "risk": RiskManager(_risk_limits(meta["risk"])),
        "capital": float(meta["capital"]),
    }
    broker = SimulatedBroker(
        fee_bps=meta.get("fee_bps", 1.0) if fee_bps is None else fee_bps,
        slippage_bps=meta.get("slippage_bps", 1.0) if slippage_bps is None else slippage_bps,
    )
    with tempfile.TemporaryDirectory() as tmp:
        journal = Journal(Path(tmp) / "replay.sqlite")
        runner = StrategyRunner(files.load_strategies(), broker, journal=journal, **runner_kwargs)
        runner.run(HistoricalBarFeed(bars))
        journal.close()
        replayed = read_journal(Path(tmp) / "replay.sqlite")
    return compare_journals(live, replayed)


def _order_prices(journal: JournalData) -> pd.DataFrame:
    fills = journal.kind("fill")
    if fills.empty:
        return pd.DataFrame(columns=["price", "filled"])
    fills["notional"] = fills["price"] * fills["quantity"]
    grouped = fills.groupby("order_id").agg(notional=("notional", "sum"), filled=("quantity", "sum"))
    return pd.DataFrame({"price": grouped["notional"] / grouped["filled"], "filled": grouped["filled"]})


def _signals(journal: JournalData) -> pd.DataFrame:
    signals = journal.kind("signal")
    if signals.empty:
        return pd.DataFrame(columns=[*SIGNAL_KEY, "quantity", "sent_quantity", "order_id", "price", "filled"])
    return signals.join(_order_prices(journal), on="order_id")


def compare_journals(live: JournalData, replay: JournalData) -> ReplayReport:
    columns = ["quantity", "sent_quantity", "price", "filled"]
    merged = pd.merge(
        _signals(live)[[*SIGNAL_KEY, *columns]],
        _signals(replay)[[*SIGNAL_KEY, *columns]],
        on=SIGNAL_KEY,
        how="outer",
        suffixes=("_live", "_replay"),
        indicator="match",
    )
    merged["match"] = merged["match"].map({"both": "both", "left_only": "live only", "right_only": "replay only"})
    caught_up = live.bars.index[live.bars["catch_up"].astype(bool)]
    merged["catch_up_bar"] = merged["bar_ts"].isin(caught_up)
    merged = merged.sort_values("bar_ts").reset_index(drop=True)

    fills = merged[(merged["match"] == "both") & merged["price_live"].notna() & merged["price_replay"].notna()].copy()
    sign = np.where(fills["side"] == "BUY", 1.0, -1.0)
    fills["slippage_bps"] = sign * (fills["price_live"] / fills["price_replay"] - 1) * 1e4

    feature_diffs = _feature_diffs(live, replay)
    summary = _summary(live, replay, merged, fills, feature_diffs)
    return ReplayReport(summary=summary, signals=merged, entry_fills=fills, feature_diffs=feature_diffs)


def _feature_diffs(live: JournalData, replay: JournalData) -> pd.Series:
    a, b = live.kind("features"), replay.kind("features")
    if a.empty or b.empty:
        return pd.Series(dtype=float)
    key = ["bar_ts", "strategy_id"]
    names = [c for c in a.columns if c not in ("bar_ts", "wall_ts", "strategy_id") and c in b.columns]
    joined = a[key + names].merge(b[key + names], on=key, suffixes=("_live", "_replay"))
    diffs = {
        name: float(
            (joined[f"{name}_live"].astype(float) - joined[f"{name}_replay"].astype(float)).abs().max(skipna=True)
        )
        for name in names
    }
    return pd.Series(diffs, dtype=float).sort_values(ascending=False)


def _summary(
    live: JournalData, replay: JournalData, signals: pd.DataFrame, fills: pd.DataFrame, features: pd.Series
) -> dict[str, float]:
    def pnl(journal: JournalData) -> tuple[float, float]:
        trades = journal.kind("trade")
        return (float(len(trades)), float(trades["pnl_net"].sum()) if not trades.empty else 0.0)

    def final_equity(journal: JournalData) -> float:
        equity = journal.kind("equity")
        return float(equity["equity"].iloc[-1]) if not equity.empty else float("nan")

    live_trades, live_pnl = pnl(live)
    replay_trades, replay_pnl = pnl(replay)
    equity = live.kind("equity")
    latency = equity["latency_ms"].dropna().astype(float) if not equity.empty else pd.Series(dtype=float)
    decided = signals[~signals["catch_up_bar"]]
    return {
        "bars": float(len(live.bars)),
        "bars_caught_up": float(live.bars["catch_up"].sum()) if "catch_up" in live.bars else 0.0,
        "signals_live": float((signals["match"] != "replay only").sum()),
        "signals_replay": float((signals["match"] != "live only").sum()),
        "signal_agreement_pct": float((decided["match"] == "both").mean() * 100) if len(decided) else 100.0,
        "entry_slippage_bps_mean": float(fills["slippage_bps"].mean()) if not fills.empty else 0.0,
        "trades_live": live_trades,
        "trades_replay": replay_trades,
        "pnl_net_live": live_pnl,
        "pnl_net_replay": replay_pnl,
        "final_equity_live": final_equity(live),
        "final_equity_replay": final_equity(replay),
        "max_feature_diff": float(features.max()) if not features.empty else 0.0,
        "latency_ms_p50": float(np.percentile(latency, 50)) if len(latency) else float("nan"),
        "latency_ms_p95": float(np.percentile(latency, 95)) if len(latency) else float("nan"),
    }


def bar_differences(live_bars: pd.DataFrame, reference: pd.DataFrame, rel_tol: float = 1e-6) -> pd.DataFrame:
    """Bars whose OHLCV differ between the live feed and `reference` (e.g. history downloaded later)."""
    columns = [c for c in ("open", "high", "low", "close", "volume") if c in live_bars and c in reference]
    joined = live_bars[columns].join(reference[columns], how="outer", lsuffix="_live", rsuffix="_ref")
    differs = pd.Series(False, index=joined.index)
    for column in columns:
        a, b = joined[f"{column}_live"], joined[f"{column}_ref"]
        differs |= ~np.isclose(a, b, rtol=rel_tol, equal_nan=False)
    return joined[differs]
