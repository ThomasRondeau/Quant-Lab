"""Cash, net positions and mark-to-market equity, updated from fills (same accounting in backtest and live)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Position:
    quantity: int = 0
    avg_price: float = 0.0


class Portfolio:
    def __init__(self, cash: float = 100_000.0) -> None:
        self.initial_cash = cash
        self.cash = cash
        self.positions: dict[str, Position] = {}
        self.last_prices: dict[str, float] = {}
        self.fees_paid = 0.0

    def apply_fill(self, symbol: str, signed_quantity: int, price: float, fees: float) -> None:
        self.cash -= signed_quantity * price + fees
        self.fees_paid += fees
        self.last_prices.setdefault(symbol, price)
        pos = self.positions.setdefault(symbol, Position())
        new_quantity = pos.quantity + signed_quantity
        if new_quantity == 0:
            pos.avg_price = 0.0
        elif pos.quantity == 0 or (pos.quantity > 0) != (new_quantity > 0):
            pos.avg_price = price  # opened, or flipped through zero
        elif abs(new_quantity) > abs(pos.quantity):
            pos.avg_price = (pos.avg_price * pos.quantity + price * signed_quantity) / new_quantity
        pos.quantity = new_quantity

    def mark(self, prices: dict[str, float]) -> None:
        self.last_prices.update(prices)

    def quantity(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0

    def net_positions(self) -> dict[str, int]:
        return {symbol: pos.quantity for symbol, pos in self.positions.items() if pos.quantity}

    @property
    def market_value(self) -> float:
        return sum(pos.quantity * self.last_prices.get(symbol, pos.avg_price) for symbol, pos in self.positions.items())

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    def get_state(self) -> dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "fees_paid": self.fees_paid,
            "positions": {s: [p.quantity, p.avg_price] for s, p in self.positions.items() if p.quantity},
            "last_prices": dict(self.last_prices),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.initial_cash, self.cash, self.fees_paid = state["initial_cash"], state["cash"], state["fees_paid"]
        self.positions = {s: Position(int(q), float(p)) for s, (q, p) in state["positions"].items()}
        self.last_prices = dict(state["last_prices"])
