"""
Position sizing: how many shares an entry buys or sells, decided by the router when the entry is sent.

A strategy says only "buy / sell, with this conviction (0-1)"; its sizing rule (`Sizing`, built from the
strategy's size parameters and frozen into the order like the exit plan) gives the quantity, from the
signal's close and the account equity:

- `fixed`: `quantity` shares (x conviction, at least 1);
- `notional`: `value` dollars;
- `equity_pct`: `value` percent of the equity;
- `risk`: lose at most `value` percent of the equity at the stop: equity x value% / distance to the stop,
  the distance being floored at `min_stop_pct` percent of the price (a tiny stop would mean a huge size).

A group (the legs of a pair) is sized as a whole: the rule gives the group's gross notional (`fixed`: the
quantity of the first leg on every leg; `risk`: the risk divided by the combined stop), shared equally
between the legs, so every leg holds the same dollar amount.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from quantlab.execution.exits import ExitPlan, LevelKind

logger = logging.getLogger(__name__)


class SizeMode(str, Enum):
    FIXED = "fixed"
    NOTIONAL = "notional"
    EQUITY_PCT = "equity_pct"
    RISK = "risk"


SIZE_MODES = tuple(mode.value for mode in SizeMode)


@dataclass(frozen=True)
class Sizing:
    mode: SizeMode = SizeMode.FIXED
    quantity: int = 10
    #: dollars (notional) or percent of the equity (equity_pct, risk); unused in fixed mode
    value: float | None = None
    #: risk mode: floor of the distance to the stop, in percent of the price
    min_stop_pct: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", SizeMode(self.mode))
        if self.quantity < 1:
            raise ValueError(f"Sizing quantity must be >= 1 (got {self.quantity}).")
        if self.mode is not SizeMode.FIXED and not (self.value and self.value > 0):
            raise ValueError(f"Sizing mode {self.mode.value} needs a value > 0 (size_value).")
        if self.min_stop_pct <= 0:
            raise ValueError(f"min_stop_pct must be > 0 (got {self.min_stop_pct}).")

    def describe(self) -> dict[str, Any]:
        """Flat fields for the journal."""
        return {"size_mode": self.mode.value, "size_value": self.value if self.mode is not SizeMode.FIXED else None}

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "mode": self.mode.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Sizing | None:
        return cls(**data) if data else None

    # --- quantities ------------------------------------------------------------------------------------

    def quantity_for(self, side_sign: int, price: float, equity: float, plan: ExitPlan, conviction: float) -> int:
        """Shares of a single entry at `price` (0: too small to trade)."""
        conviction = min(max(conviction, 0.0), 1.0)
        if self.mode is SizeMode.FIXED:
            return max(1, int(self.quantity * conviction))
        if price <= 0 or not math.isfinite(price):
            return 0
        if self.mode is SizeMode.RISK:
            if plan.stop is None:
                return 0  # no stop, no risk-based size (the entry is refused anyway when a stop is required)
            distance = abs(price - plan.stop.price(price, -side_sign))
            distance = max(distance, price * self.min_stop_pct / 100)
            return int(self._budget(equity) * conviction / distance)
        return int(self._budget(equity) * conviction / price)

    def group_quantities(self, prices: Sequence[float], equity: float, plan: ExitPlan, conviction: float) -> list[int]:
        """Shares of each leg: the same notional on every leg (0 for a leg: the group is too small)."""
        conviction = min(max(conviction, 0.0), 1.0)
        if any(p <= 0 or not math.isfinite(p) for p in prices):
            return [0] * len(prices)
        if self.mode is SizeMode.FIXED:
            leg_notional = max(1, int(self.quantity * conviction)) * prices[0]
        elif self.mode is SizeMode.RISK:
            if plan.stop is None or plan.stop.kind is not LevelKind.PCT:
                return [0] * len(prices)
            stop = max(plan.stop.value, self.min_stop_pct / 100)
            leg_notional = self._budget(equity) * conviction / stop / len(prices)
        else:
            leg_notional = self._budget(equity) * conviction / len(prices)
        return [round(leg_notional / p) for p in prices]

    def _budget(self, equity: float) -> float:
        """Dollars of the notional (notional mode), or of the equity share (equity_pct, risk)."""
        assert self.value is not None
        return self.value if self.mode is SizeMode.NOTIONAL else max(equity, 0.0) * self.value / 100
