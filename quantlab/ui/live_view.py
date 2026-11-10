"""
What the live page shows of a running (or ended) session, read from its files only: the journal (SQLite, safe
to read while the session writes it) and the state saved after every bar. No Streamlit here, so it is testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.schema import bar_value
from quantlab.runtime.journal import read_journal
from quantlab.runtime.session import SessionFiles

#: journal events shown in the activity feed
ACTIVITY_KINDS = ("signal", "order", "fill", "trade", "exit", "cancel", "status", "session")


@dataclass
class SessionView:
    bars: int = 0
    last_bar: pd.Timestamp | None = None
    equity: float | None = None
    cash: float | None = None
    realized_pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    positions: dict[str, int] = field(default_factory=dict)
    halted: bool = False
    latency_p50_ms: float | None = None
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    #: open lots: strategy, symbol, side, quantity, entry, last price, unrealized P&L, stop, target, bars held
    lots: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: newest first: bar time, kind, strategy, description
    activity: pd.DataFrame = field(default_factory=pd.DataFrame)
    closed_trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def read_session(path: str | Path, activity_rows: int = 40) -> SessionView:
    files = SessionFiles(path)
    view = SessionView()
    if not files.journal_path.exists():
        return view
    journal = read_journal(files.journal_path)
    view.bars = len(journal.bars)
    view.last_bar = journal.bars.index[-1] if view.bars else None
    equity = journal.kind("equity")
    if not equity.empty:
        last = equity.iloc[-1]
        view.equity, view.cash = float(last["equity"]), float(last["cash"])
        view.positions = {k: int(v) for k, v in (last["positions"] or {}).items() if v}
        view.halted = bool(last["halted"])
        latency = equity["latency_ms"].dropna().astype(float)
        view.latency_p50_ms = float(latency.median()) if len(latency) else None
        view.equity_curve = pd.Series(
            equity["equity"].astype(float).to_numpy(), index=pd.DatetimeIndex(equity["bar_ts"]), name="equity"
        )
    trades = journal.kind("trade")
    if not trades.empty:
        view.trades = len(trades)
        view.realized_pnl = float(trades["pnl_net"].sum())
        view.wins = int((trades["pnl_net"] > 0).sum())
        view.closed_trades = trades
    last_bar = journal.bars.iloc[-1] if view.bars else None
    view.lots = open_lots(files.state_path, last_bar)
    rows = journal.events[journal.events["kind"].isin(ACTIVITY_KINDS)].tail(activity_rows).iloc[::-1]
    view.activity = pd.DataFrame(
        {
            "bar": rows["bar_ts"],
            "event": rows["kind"],
            "strategy": rows["strategy_id"].fillna(""),
            "details": [describe_event(k, d) for k, d in zip(rows["kind"], rows["data"], strict=True)],
        }
    ).reset_index(drop=True)
    return view


def open_lots(state_path: Path, last_bar: pd.Series | None) -> pd.DataFrame:
    """Open lots from the state saved after the last bar, marked at that bar's close."""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # not written yet, or being replaced
        return pd.DataFrame()
    rows = []
    for lot in state.get("router", {}).get("lots", []):
        if not lot.get("quantity"):
            continue
        sign = 1 if lot["side"] == "BUY" else -1
        price = _price(last_bar, lot["symbol"])
        unrealized = sign * (price - lot["entry_price"]) * lot["quantity"] if price is not None else None
        rows.append(
            {
                "strategy": lot["strategy_id"],
                "symbol": lot["symbol"],
                "side": lot["side"],
                "quantity": lot["quantity"],
                "entry": round(lot["entry_price"], 2),
                "last": round(price, 2) if price is not None else None,
                "unrealized P&L": round(unrealized, 2) if unrealized is not None else None,
                "stop": lot.get("stop_price"),
                "target": lot.get("target_price"),
                "bars held": lot.get("bars_held", 0),
                "group": lot.get("group_id") or "",
            }
        )
    return pd.DataFrame(rows)


def _price(bar: pd.Series | None, symbol: str) -> float | None:
    if bar is None:
        return None
    try:
        return float(bar_value(bar, symbol, "close"))
    except KeyError:
        return None


def describe_event(kind: str, data: dict[str, Any]) -> str:
    """One readable line per journal event."""
    if kind == "signal":
        sent = data.get("sent_quantity", 0)
        size = f"{data.get('quantity')} → {sent}" if sent != data.get("quantity") else f"{sent}"
        plan = ", ".join(f"{k} {data[k]}" for k in ("stop", "target") if data.get(k))
        refused = " (refused by the risk checks)" if not sent else ""
        return f"{data.get('side')} {size} {data.get('symbol')}{refused}" + (f" · {plan}" if plan else "")
    if kind == "order":
        price = data.get("stop_price") or data.get("limit_price")
        return f"{data.get('side')} {data.get('quantity')} {data.get('symbol')} {data.get('order_type')}" + (
            f" @ {price:.2f}" if price else ""
        )
    if kind == "fill":
        return f"{data.get('side')} {data.get('quantity')} {data.get('symbol')} filled @ {data.get('price', 0):.2f}"
    if kind == "trade":
        return (
            f"{data.get('side')} {data.get('quantity')} {data.get('symbol')} closed ({data.get('reason')}): "
            f"{data.get('entry_price', 0):.2f} → {data.get('exit_price', 0):.2f}, net P&L {data.get('pnl_net', 0):+.2f}"
        )
    if kind == "exit":
        return f"exit requested on {data.get('symbol') or 'all symbols'} ({data.get('reason')})"
    if kind == "cancel":
        return "order cancelled"
    if kind == "status":
        return f"order {str(data.get('status', '')).lower()}" + (f": {data['message']}" if data.get("message") else "")
    if kind == "session":
        return f"session {data.get('action')}"
    return json.dumps(data, default=str)[:120]
