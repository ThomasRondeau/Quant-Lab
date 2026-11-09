"""
Broker contract: the simulated broker implements it, and so would a connector to a real broker.

Brokers report executions and terminal statuses through the handlers set by the `OrderRouter`.
A live broker may call them from its network thread: the runner then queues them and processes
them on the strategy thread (see `StrategyRunner`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd

from quantlab.execution.orders import Fill, Order, OrderStatus

FillHandler = Callable[[Fill], None]
StatusHandler = Callable[[str, OrderStatus, str], None]  # order_id, status, message


class Broker(Protocol):
    def set_handlers(self, on_fill: FillHandler, on_status: StatusHandler) -> None: ...

    def submit(self, order: Order) -> None: ...

    def cancel(self, order_id: str) -> None: ...

    def positions(self) -> dict[str, int]:
        """Net position per symbol as known by the broker."""
        ...

    def process_events(self) -> None:
        """Deliver the queued fills / statuses on the calling (strategy) thread; no-op for the simulator."""
        ...

    def on_bar(self, bar: pd.Series) -> None:
        """Called with each new bar before the strategies see it (the simulator matches orders here)."""
        ...

    def get_state(self) -> dict[str, Any]:
        """JSON-serializable state needed to resume after a restart."""
        ...

    def set_state(self, state: dict[str, Any]) -> None: ...
