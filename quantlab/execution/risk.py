"""
Pre-trade risk checks and the kill switch, applied by the `OrderRouter` to every entry order.

Exits (time exits, protective stops / targets, flattening) are never blocked. The stop, target and
holding period of each position are not risk limits: they belong to the entry's `ExitPlan`, decided by
the strategy (`execution/exits.py`). The limits here protect the account.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from quantlab.execution.orders import Order

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskLimits:
    #: refuse every entry whose exit plan has no stop
    require_stop: bool = True
    #: entry orders are clipped to this quantity
    max_order_quantity: int | None = None
    #: absolute net position per symbol (including pending entries) that entries may not exceed
    max_position_quantity: int | None = None
    #: kill switch: halt when equity falls this fraction below the day's starting equity
    max_daily_loss_pct: float | None = None
    #: on halt, close every open position
    flatten_on_halt: bool = True


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.halted = False
        self.halt_reason = ""
        self._day: date | None = None
        self._day_start_equity = 0.0

    def halt(self, reason: str) -> None:
        if not self.halted:
            logger.error("Kill switch: trading halted (%s)", reason)
        self.halted, self.halt_reason = True, reason

    def resume(self) -> None:
        logger.warning("Trading resumed (was halted: %s)", self.halt_reason)
        self.halted, self.halt_reason = False, ""

    def check_entry(self, order: Order, exposure: int) -> Order | None:
        """
        The order to send (possibly with a reduced quantity), or None if it is refused.
        :param exposure: current net position plus pending entries of the order's symbol (signed).
        """
        if self.halted:
            logger.warning("Order refused (halted: %s): %s", self.halt_reason, order)
            return None
        quantity = order.quantity
        if self.limits.max_order_quantity is not None:
            quantity = min(quantity, self.limits.max_order_quantity)
        if self.limits.max_position_quantity is not None:
            room = self.limits.max_position_quantity - exposure * order.side.sign
            quantity = min(quantity, max(room, 0))
        if quantity <= 0:
            logger.info("Order refused by position limit (exposure %d): %s", exposure, order)
            return None
        if quantity != order.quantity:
            logger.info("Order quantity clipped %d -> %d: %s", order.quantity, quantity, order)
            return dataclasses.replace(order, quantity=quantity)
        return order

    def get_state(self) -> dict[str, Any]:
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "day": self._day.isoformat() if self._day else None,
            "day_start_equity": self._day_start_equity,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.halted, self.halt_reason = bool(state["halted"]), str(state["halt_reason"])
        self._day = date.fromisoformat(state["day"]) if state["day"] else None
        self._day_start_equity = float(state["day_start_equity"])

    def on_equity(self, timestamp: pd.Timestamp, equity: float) -> bool:
        """Track the daily loss; returns True when this update triggers the kill switch."""
        day = timestamp.date()
        if day != self._day:
            self._day, self._day_start_equity = day, equity
        limit = self.limits.max_daily_loss_pct
        if self.halted or limit is None or self._day_start_equity <= 0:
            return False
        loss = 1 - equity / self._day_start_equity
        if loss >= limit:
            self.halt(f"daily loss {loss:.2%} >= {limit:.2%}")
            return True
        return False
