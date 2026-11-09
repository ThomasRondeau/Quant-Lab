"""Orders emitted by strategies (and by the router), and the fills / status updates reported by a broker."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from quantlab.execution.exits import ExitPlan
from quantlab.execution.sizing import Sizing


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    def __str__(self) -> str:
        return self.value


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"


class OrderStatus(str, Enum):
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


def _new_order_id() -> str:
    # Unique across sessions: order ids end up in the journal and in the persisted live state.
    return uuid.uuid4().hex[:16]


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    quantity: int
    strategy_id: str
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    #: orders sharing an OCA ("one cancels all") group: the first fill cancels the others
    oca_group: str | None = None
    order_id: str = field(default_factory=_new_order_id)
    #: entries only: how the lot opened by this order is closed (None: the router's empty plan)
    exit_plan: ExitPlan | None = None
    #: entries only: orders sharing a group (the legs of a pair) are risk-checked and closed together
    group_id: str | None = None
    #: entries only: the rule the router sizes the order with (None: `quantity` as given)
    sizing: Sizing | None = None
    #: entries only: 0-1 multiplier of the size (e.g. a model's confidence)
    conviction: float = 1.0

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be > 0 (got {self.quantity}).")
        object.__setattr__(self, "side", Side(self.side))
        object.__setattr__(self, "order_type", OrderType(self.order_type))
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("A limit order needs a limit_price.")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("A stop order needs a stop_price.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "side": self.side.value,
            "order_type": self.order_type.value,
            "exit_plan": self.exit_plan.to_dict() if self.exit_plan else None,
            "sizing": self.sizing.to_dict() if self.sizing else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Order:
        return cls(
            **{
                **data,
                "exit_plan": ExitPlan.from_dict(data.get("exit_plan")),
                "sizing": Sizing.from_dict(data.get("sizing")),
            }
        )


@dataclass(frozen=True)
class Fill:
    """A (possibly partial) execution of `order`: `quantity` shares at `price`, `fees` in cash."""

    order: Order
    price: float
    quantity: int
    timestamp: pd.Timestamp
    fees: float = 0.0
    exec_id: str | None = None

    @property
    def signed_quantity(self) -> int:
        return self.order.side.sign * self.quantity
