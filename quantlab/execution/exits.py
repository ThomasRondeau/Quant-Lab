"""
Exit plans: how each lot is closed, decided by the strategy when it emits the entry.

A plan travels with the entry order (`Order.exit_plan`) and is frozen into the lot the router opens, so
the stop, target and holding period of a position never change after the signal (restarts included).

Stop and target levels are anchored at the lot's fill price, which the strategy cannot know:

* `Level.pct(0.02)`: 2 % away from the fill price;
* `Level.distance(0.42)`: 0.42 $ away from the fill price (e.g. 1.5 x the ATR measured at the signal);
* `Level.at(187.3)`: this absolute price (e.g. the Bollinger middle band at the signal). If the fill is
  already past it (gap), the lot is closed at market at once.

In a group (the legs of a pair, `Order.group_id`), `stop` and `target` must be `pct` levels and apply to
the group's combined P&L as a fraction of its gross entry notional; the router checks them at each bar
close and closes every leg together. `leg_stop` (a fraction) is a wide protective stop placed at the
broker on each leg, so a leg stays protected if the process dies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class LevelKind(str, Enum):
    PCT = "pct"
    DISTANCE = "distance"
    PRICE = "price"


@dataclass(frozen=True)
class Level:
    kind: LevelKind
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", LevelKind(self.kind))
        if not self.value > 0:
            raise ValueError(f"An exit level must be > 0 (got {self.kind.value} {self.value}).")

    @classmethod
    def pct(cls, fraction: float) -> Level:
        return cls(LevelKind.PCT, fraction)

    @classmethod
    def distance(cls, amount: float) -> Level:
        return cls(LevelKind.DISTANCE, amount)

    @classmethod
    def at(cls, price: float) -> Level:
        return cls(LevelKind.PRICE, price)

    def price(self, entry_price: float, direction: int) -> float:
        """
        The level's price for a lot filled at `entry_price`.
        :param direction: +1 for a level above the entry (long target, short stop), -1 below it.
        """
        if self.kind is LevelKind.PCT:
            return entry_price * (1 + direction * self.value)
        if self.kind is LevelKind.DISTANCE:
            return entry_price + direction * self.value
        return self.value

    def __str__(self) -> str:
        if self.kind is LevelKind.PCT:
            return f"{self.value:.4%}"
        return f"{self.kind.value} {self.value:g}"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Level | None:
        return None if data is None else cls(LevelKind(data["kind"]), float(data["value"]))


@dataclass(frozen=True)
class ExitPlan:
    #: protective stop (None: no stop; refused by the router while `RiskLimits.require_stop`)
    stop: Level | None = None
    #: profit target (None: none)
    target: Level | None = None
    #: bars after which the lot is closed at market (None: no time exit)
    holding_bars: int | None = None
    #: groups only: protective stop placed at the broker on each leg (fraction of the leg's fill price)
    leg_stop: float | None = None

    def __post_init__(self) -> None:
        if self.holding_bars is not None and self.holding_bars < 1:
            raise ValueError(f"holding_bars must be >= 1 or None (got {self.holding_bars}).")
        if self.leg_stop is not None and not self.leg_stop > 0:
            raise ValueError(f"leg_stop must be > 0 or None (got {self.leg_stop}).")

    def check_group(self) -> None:
        """Raise if the plan cannot apply to a group (combined P&L levels must be fractions)."""
        for name, level in (("stop", self.stop), ("target", self.target)):
            if level is not None and level.kind is not LevelKind.PCT:
                raise ValueError(f"A group's {name} must be a pct level of its gross notional (got {level}).")

    def describe(self) -> dict[str, Any]:
        """Flat summary for the journal."""
        return {
            "stop": str(self.stop) if self.stop else None,
            "target": str(self.target) if self.target else None,
            "holding_bars": self.holding_bars,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop": self.stop.to_dict() if self.stop else None,
            "target": self.target.to_dict() if self.target else None,
            "holding_bars": self.holding_bars,
            "leg_stop": self.leg_stop,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExitPlan | None:
        if data is None:
            return None
        return cls(
            stop=Level.from_dict(data.get("stop")),
            target=Level.from_dict(data.get("target")),
            holding_bars=data.get("holding_bars"),
            leg_stop=data.get("leg_stop"),
        )
