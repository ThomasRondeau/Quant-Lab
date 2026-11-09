"""
Simulated broker for backtests.

Orders submitted while processing bar `t` are matched from bar `t + 1` on, so a strategy that decides on
the close of `t` never trades at that close:

- market orders fill at the next bar's open, plus `slippage_bps` against the order;
- stop orders trigger when the bar trades through the stop and fill at the stop, or at the open if the
  bar gapped through it (plus slippage); limit orders fill at the limit or a better open;
- protective orders placed when an entry fills at the open are matched against the rest of that bar;
  when a bar touches both legs of an OCA pair the stop is assumed to fill first (conservative);
- fees: `fee_bps` of the traded notional.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from quantlab.data.schema import bar_time, bar_value
from quantlab.execution.broker import FillHandler, StatusHandler
from quantlab.execution.orders import Fill, Order, OrderStatus, OrderType, Side

logger = logging.getLogger(__name__)

BASIS_POINT = 1e-4


class SimulatedBroker:
    def __init__(self, fee_bps: float = 1.0, slippage_bps: float = 1.0) -> None:
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self._pending: dict[str, Order] = {}  # insertion-ordered
        self._positions: dict[str, int] = {}
        self._on_fill: FillHandler | None = None
        self._on_status: StatusHandler | None = None

    def set_handlers(self, on_fill: FillHandler, on_status: StatusHandler) -> None:
        self._on_fill, self._on_status = on_fill, on_status

    def submit(self, order: Order) -> None:
        self._pending[order.order_id] = order

    def cancel(self, order_id: str) -> None:
        if self._pending.pop(order_id, None) is not None:
            self._status(order_id, OrderStatus.CANCELLED, "")

    def positions(self) -> dict[str, int]:
        return {symbol: qty for symbol, qty in self._positions.items() if qty}

    def process_events(self) -> None:
        """Fills are delivered synchronously from `on_bar`."""

    def get_state(self) -> dict[str, Any]:
        return {"pending": [o.to_dict() for o in self._pending.values()], "positions": dict(self._positions)}

    def set_state(self, state: dict[str, Any]) -> None:
        self._pending = {o.order_id: o for o in map(Order.from_dict, state["pending"])}
        self._positions = {s: int(q) for s, q in state["positions"].items()}

    @property
    def pending_orders(self) -> list[Order]:
        return list(self._pending.values())

    def on_bar(self, bar: pd.Series) -> None:
        timestamp = bar_time(bar)
        for order in [o for o in self._pending.values() if o.order_type is OrderType.MARKET]:
            open_ = bar_value(bar, order.symbol, "open")
            if math.isfinite(open_):
                self._fill(order, self._slipped(open_, order.side), timestamp)
        # Stops first: the conservative choice when a bar touches both legs of an OCA pair. Includes the
        # protective orders just placed by the market fills above.
        resting = [o for o in self._pending.values() if o.order_type is not OrderType.MARKET]
        resting.sort(key=lambda o: o.order_type is not OrderType.STOP)
        for order in resting:
            if order.order_id in self._pending:  # not cancelled by an OCA sibling meanwhile
                price = self._trigger_price(order, bar)
                if price is not None:
                    self._fill(order, price, timestamp)

    # --- internals -------------------------------------------------------------------------------------

    def _slipped(self, price: float, side: Side) -> float:
        return price * (1 + side.sign * self.slippage_bps * BASIS_POINT)

    def _trigger_price(self, order: Order, bar: pd.Series) -> float | None:
        open_ = bar_value(bar, order.symbol, "open")
        high = bar_value(bar, order.symbol, "high")
        low = bar_value(bar, order.symbol, "low")
        if not all(map(math.isfinite, (open_, high, low))):
            return None
        buy = order.side is Side.BUY
        if order.order_type is OrderType.STOP:
            stop = order.stop_price
            assert stop is not None
            if buy and high >= stop:
                return self._slipped(max(open_, stop), order.side)
            if not buy and low <= stop:
                return self._slipped(min(open_, stop), order.side)
        elif order.order_type is OrderType.LIMIT:
            limit = order.limit_price
            assert limit is not None
            if buy and low <= limit:
                return min(open_, limit)
            if not buy and high >= limit:
                return max(open_, limit)
        return None

    def _fill(self, order: Order, price: float, timestamp: pd.Timestamp) -> None:
        del self._pending[order.order_id]
        if order.oca_group is not None:
            for other in [o for o in self._pending.values() if o.oca_group == order.oca_group]:
                self.cancel(other.order_id)
        self._positions[order.symbol] = self._positions.get(order.symbol, 0) + order.side.sign * order.quantity
        fees = order.quantity * price * self.fee_bps * BASIS_POINT
        if self._on_fill is not None:
            self._on_fill(Fill(order, price, order.quantity, timestamp, fees))

    def _status(self, order_id: str, status: OrderStatus, message: str) -> None:
        if self._on_status is not None:
            self._on_status(order_id, status, message)
