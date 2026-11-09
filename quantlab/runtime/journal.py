"""
Session journal: everything the runner sees and does, bar by bar, in a SQLite file (WAL, crash-safe,
append-only, readable while the session runs).

Tables:
- ``bars``: every bar fed to the strategies (live bars, and the bars caught up after a restart);
- ``events``: one row per event, with the bar it happened on, the wall-clock time and a JSON payload.
  Kinds: ``features`` (feature row of a strategy), ``decision`` (its diagnostics, e.g. predictions),
  ``signal`` (an entry order emitted by a strategy, and what the risk check let through), ``order``,
  ``cancel``, ``fill``, ``status``, ``trade`` (closed round trip), ``equity`` (per bar: equity, cash,
  positions, halted, latency), ``session`` (start / resume / stop).

`JournalingBroker` wraps the real broker so every order sent and every fill / status received is
recorded at the boundary, whichever component caused it.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.execution.broker import Broker, FillHandler, StatusHandler
from quantlab.execution.orders import Fill, Order, OrderStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (ts TEXT PRIMARY KEY, catch_up INTEGER NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_ts TEXT,
    wall_ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    strategy_id TEXT,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_kind ON events (kind);
"""


def _clean(value: Any) -> Any:
    """JSON-safe value: NaN / inf become None, numpy scalars become Python numbers."""
    if hasattr(value, "item") and not isinstance(value, (list, dict, str)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _dumps(data: Any) -> str:
    return json.dumps(_clean(data), separators=(",", ":"))


def fill_to_dict(fill: Fill) -> dict[str, Any]:
    return {
        "order_id": fill.order.order_id,
        "symbol": fill.order.symbol,
        "side": fill.order.side.value,
        "price": fill.price,
        "quantity": fill.quantity,
        "fees": fill.fees,
        "fill_ts": fill.timestamp.isoformat(),
        "exec_id": fill.exec_id,
    }


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self.bar_ts: pd.Timestamp | None = None  # bar the next events are attributed to

    def close(self) -> None:
        self._db.commit()
        self._db.close()

    def commit(self) -> None:
        self._db.commit()

    def bar(self, bar: pd.Series, catch_up: bool = False) -> None:
        self.bar_ts = pd.Timestamp(bar.name)  # type: ignore[arg-type]
        self._db.execute(
            "INSERT OR REPLACE INTO bars VALUES (?, ?, ?)",
            (self.bar_ts.isoformat(), int(catch_up), _dumps(bar.to_dict())),
        )

    def event(self, kind: str, data: dict[str, Any], strategy_id: str | None = None) -> None:
        self._db.execute(
            "INSERT INTO events (bar_ts, wall_ts, kind, strategy_id, data) VALUES (?, ?, ?, ?, ?)",
            (
                self.bar_ts.isoformat() if self.bar_ts is not None else None,
                pd.Timestamp.now(tz="UTC").isoformat(),
                kind,
                strategy_id,
                _dumps(data),
            ),
        )


@dataclass
class JournalData:
    bars: pd.DataFrame
    events: pd.DataFrame

    def kind(self, kind: str) -> pd.DataFrame:
        """Events of one kind with their payload expanded into columns."""
        rows = self.events[self.events["kind"] == kind]
        if rows.empty:
            return pd.DataFrame()
        payload = pd.DataFrame(list(rows["data"]), index=rows.index)
        payload = payload.drop(columns=[c for c in ("bar_ts", "wall_ts", "strategy_id") if c in payload])
        return pd.concat([rows[["bar_ts", "wall_ts", "strategy_id"]], payload], axis=1).reset_index(drop=True)


def read_journal(path: str | Path) -> JournalData:
    with sqlite3.connect(Path(path)) as db:
        bars = pd.read_sql_query("SELECT ts, catch_up, data FROM bars ORDER BY ts", db)
        events = pd.read_sql_query("SELECT * FROM events ORDER BY id", db)
    if bars.empty:
        frame = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="Datetime"))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True, format="ISO8601"), name="Datetime")
        frame = pd.DataFrame([json.loads(d) for d in bars["data"]], index=index)
        frame = frame.astype(float)
        frame["catch_up"] = bars["catch_up"].to_numpy().astype(bool)
    events["data"] = [json.loads(d) for d in events["data"]]
    for column in ("bar_ts", "wall_ts"):
        events[column] = pd.to_datetime(events[column], utc=True, format="ISO8601")
    return JournalData(bars=frame, events=events)


class JournalingBroker:
    """Broker proxy recording orders, cancels, fills and statuses in the journal."""

    def __init__(self, broker: Broker, journal: Journal) -> None:
        self.broker = broker
        self.journal = journal
        self._on_fill: FillHandler | None = None
        self._on_status: StatusHandler | None = None

    def set_handlers(self, on_fill: FillHandler, on_status: StatusHandler) -> None:
        self._on_fill, self._on_status = on_fill, on_status
        self.broker.set_handlers(self._fill, self._status)

    def submit(self, order: Order) -> None:
        self.journal.event("order", order.to_dict(), order.strategy_id)
        self.broker.submit(order)

    def cancel(self, order_id: str) -> None:
        self.journal.event("cancel", {"order_id": order_id})
        self.broker.cancel(order_id)

    def positions(self) -> dict[str, int]:
        return self.broker.positions()

    def process_events(self) -> None:
        self.broker.process_events()

    def on_bar(self, bar: pd.Series) -> None:
        self.broker.on_bar(bar)

    def get_state(self) -> dict[str, Any]:
        return self.broker.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        self.broker.set_state(state)

    def _fill(self, fill: Fill) -> None:
        self.journal.event("fill", fill_to_dict(fill), fill.order.strategy_id)
        if self._on_fill is not None:
            self._on_fill(fill)

    def _status(self, order_id: str, status: OrderStatus, message: str) -> None:
        self.journal.event("status", {"order_id": order_id, "status": status.value, "message": message})
        if self._on_status is not None:
            self._on_status(order_id, status, message)
