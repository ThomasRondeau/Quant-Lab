"""
The single trading loop, for backtest and live alike:

    BarFeed -> broker.on_bar (simulator fills) -> router time exits / kill switch
            -> strategy.on_bar -> router.exit_strategy (close requests) / submit_entries (risk) -> Broker

A backtest is this runner on a `HistoricalBarFeed` with a `SimulatedBroker`; live trading is the same runner
on a live bar feed with a live broker, so no trading logic exists for the backtest only. Everything runs on the
caller's thread: a live broker would queue its network-thread events and deliver them in `process_events`.

Live sessions add a `Journal` (every bar, feature row, decision, order, fill and the equity) and a
`StateStore` (portfolio, lots, pending orders, risk and broker state, saved after every bar) so a
session can resume after a restart (`set_state` + `catch_up`) and be replayed against the backtester.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

import pandas as pd

from quantlab.data.schema import bar_time, bar_value
from quantlab.execution.broker import Broker
from quantlab.execution.orders import Fill
from quantlab.execution.portfolio import Portfolio
from quantlab.execution.risk import RiskManager
from quantlab.execution.router import OrderRouter
from quantlab.runtime.bar_feed import BarFeed
from quantlab.runtime.journal import Journal, JournalingBroker
from quantlab.runtime.state import StateStore
from quantlab.strategies.base import Strategy

logger = logging.getLogger(__name__)

STATE_VERSION = 1


class StrategyRunner:
    def __init__(
        self,
        strategies: Sequence[Strategy],
        broker: Broker,
        risk: RiskManager | None = None,
        capital: float = 100_000.0,
        tick_size: float = 0.01,
        journal: Journal | None = None,
        state_store: StateStore | None = None,
        journal_features: bool = True,
    ) -> None:
        ids = [s.strategy_id for s in strategies]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Strategy ids must be unique (got {ids}); pass strategy_id=... to tell them apart.")
        self.strategies = list(strategies)
        self.journal = journal
        self.state_store = state_store
        self.journal_features = journal_features
        self.broker: Broker = JournalingBroker(broker, journal) if journal is not None else broker
        self.portfolio = Portfolio(capital)
        self.router = OrderRouter(self.broker, self.portfolio, risk, tick_size)
        self.router.fill_listeners.append(self._dispatch_fill)
        self._by_id = {s.strategy_id: s for s in self.strategies}
        self._equity_times: list[pd.Timestamp] = []
        self._equity_values: list[float] = []
        self.latencies_ms: deque[float] = deque(maxlen=10_000)  # bar received -> orders sent
        self.last_bar: pd.Series | None = None
        self.last_bar_time: pd.Timestamp | None = None
        self._stop_requested = False
        self._journaled_trades = 0

    @property
    def equity(self) -> pd.Series:
        return pd.Series(self._equity_values, index=pd.DatetimeIndex(self._equity_times), name="equity", dtype=float)

    def run(self, feed: BarFeed) -> None:
        self._stop_requested = False
        for strategy in self.strategies:
            strategy.start()
        try:
            for bar in feed:
                self.broker.process_events()
                if bar is not None:
                    self.step(bar)
                elif self.journal is not None:
                    self.journal.commit()  # fills / statuses delivered between bars
                if self._stop_requested:
                    break
            self.broker.process_events()
        finally:
            for strategy in self.strategies:
                strategy.stop()
            self._persist()
            if self.journal is not None:
                self.journal.commit()

    def stop(self) -> None:
        """Ask `run` to return after the current bar (thread-safe)."""
        self._stop_requested = True

    def step(self, bar: pd.Series) -> None:
        start = time.perf_counter()
        timestamp = bar_time(bar)
        if self.journal is not None:
            self.journal.bar(bar)
        self.broker.on_bar(bar)
        self.broker.process_events()
        self._mark(bar)
        self.router.on_bar_close(timestamp)
        for strategy in self.strategies:
            orders = strategy.on_bar(bar)
            for symbol, reason in strategy.pop_exit_requests():
                closed = self.router.exit_strategy(strategy.strategy_id, symbol, reason)
                if closed and self.journal is not None:
                    self.journal.event(
                        "exit", {"symbol": symbol, "reason": reason, "lots": closed}, strategy.strategy_id
                    )
            for order, sent in zip(orders, self.router.submit_entries(orders), strict=True):
                if self.journal is not None:
                    plan = order.exit_plan.describe() if order.exit_plan else {}
                    signal = {
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": order.quantity,
                        "sent_quantity": sent.quantity if sent else 0,
                        "group_id": order.group_id,
                        **plan,
                        **(order.sizing.describe() if order.sizing else {}),
                        "conviction": order.conviction,
                    }
                    self.journal.event("signal", signal, strategy.strategy_id)
        latency = (time.perf_counter() - start) * 1e3
        self.latencies_ms.append(latency)
        self._close_bar(bar, timestamp, latency)

    def catch_up(self, bars: pd.DataFrame) -> None:
        """
        After a restart: advance the book over the bars missed while down (the strategies' feeds already
        hold them from the warm-up history). No decision is taken on them: the simulator matches the
        pending orders, holding periods advance and time exits are sent.
        """
        count = 0
        for _, bar in bars.iterrows():
            timestamp = bar_time(bar)
            if self.last_bar_time is not None and timestamp <= self.last_bar_time:
                continue
            if self.journal is not None:
                self.journal.bar(bar, catch_up=True)
            self.broker.on_bar(bar)
            self.broker.process_events()
            self._mark(bar)
            self.router.on_bar_close(timestamp)
            self._close_bar(bar, timestamp, latency=None)
            count += 1
        if count:
            logger.info("Caught up %d bars missed while down", count)

    def last_prices(self) -> dict[str, float]:
        return dict(self.portfolio.last_prices)

    # --- persistence -----------------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "last_bar_time": self.last_bar_time.isoformat() if self.last_bar_time is not None else None,
            "portfolio": self.portfolio.get_state(),
            "risk": self.router.risk.get_state(),
            "router": self.router.get_state(),
            "broker": self.broker.get_state(),
            "strategies": {s.strategy_id: s.get_state() for s in self.strategies},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"Unsupported runner state version {state.get('version')!r}.")
        self.last_bar_time = pd.Timestamp(state["last_bar_time"]) if state["last_bar_time"] else None
        self.portfolio.set_state(state["portfolio"])
        self.router.risk.set_state(state["risk"])
        self.router.set_state(state["router"])
        self.broker.set_state(state["broker"])
        for strategy in self.strategies:
            if strategy.strategy_id in state["strategies"]:
                strategy.set_state(state["strategies"][strategy.strategy_id])

    # --- internals -------------------------------------------------------------------------------------

    def _close_bar(self, bar: pd.Series, timestamp: pd.Timestamp, latency: float | None) -> None:
        self._equity_times.append(timestamp)
        self._equity_values.append(self.portfolio.equity)
        self.last_bar, self.last_bar_time = bar, timestamp
        if self.journal is not None:
            self._journal_bar(latency)
        self._persist()

    def _journal_bar(self, latency: float | None) -> None:
        journal = self.journal
        assert journal is not None
        if latency is not None:  # a live bar: the strategies decided on it
            for strategy in self.strategies:
                if self.journal_features and strategy.feed.ready:
                    journal.event("features", {str(k): v for k, v in strategy.feed.row().items()}, strategy.strategy_id)
                if strategy.diagnostics:
                    journal.event("decision", dict(strategy.diagnostics), strategy.strategy_id)
        for trade in self.router.trades[self._journaled_trades :]:
            payload = {**trade.__dict__, "side": trade.side.value, "pnl_net": trade.pnl_net}
            journal.event("trade", payload, trade.strategy_id)
        self._journaled_trades = len(self.router.trades)
        equity = {
            "equity": self.portfolio.equity,
            "cash": self.portfolio.cash,
            "positions": self.portfolio.net_positions(),
            "halted": self.router.risk.halted,
            "latency_ms": latency,
        }
        journal.event("equity", equity)
        journal.commit()

    def _persist(self) -> None:
        if self.state_store is not None:
            self.state_store.save(self.get_state())

    def _mark(self, bar: pd.Series) -> None:
        # held symbols, and the strategies' ones: the router sizes new entries at these closes
        symbols = dict.fromkeys([*self.portfolio.positions, *(s for st in self.strategies for s in st.symbols)])
        prices = {}
        for symbol in symbols:
            try:
                price = bar_value(bar, symbol, "close")
            except KeyError:
                continue
            if math.isfinite(price):
                prices[symbol] = price
        self.portfolio.mark(prices)

    def _dispatch_fill(self, fill: Fill) -> None:
        strategy = self._by_id.get(fill.order.strategy_id)
        if strategy is not None:
            strategy.on_fill(fill)
