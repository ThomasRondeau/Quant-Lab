"""
Order routing shared by backtest and live: risk checks, trade lifecycle and fill attribution.

Every entry order a strategy emits opens a *lot* owned by that strategy, with the entry's `ExitPlan`
frozen into it (`execution/exits.py`). Once the entry is filled, the router places the lot's protective
orders at the broker (an OCA pair: the plan's stop + target, anchored at the fill price), so they keep
working if the process dies. After the plan's `holding_bars` it cancels them and closes the lot at
market. A strategy may also ask to close its own lots (`exit_strategy`). Several strategies may hold
lots on the same symbol: the broker position is their net sum, and each lot's exit moves it by exactly
that lot's quantity.

Entries sharing a `group_id` (the legs of a pair) form a group: risk-checked all or nothing, closed
together. The group's stop / target apply to its combined P&L and are checked at each bar close; each
leg only carries the plan's wide `leg_stop` at the broker, and when one leg is closed (leg stop, entry
not filled) the others are closed too.

Fills are attributed to lots through the order ids. A live race (a protective order filling while the
time exit is in flight) can over-close a lot: the excess becomes a new lot that is closed at once.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from quantlab.execution.broker import Broker
from quantlab.execution.exits import ExitPlan
from quantlab.execution.orders import Fill, Order, OrderStatus, OrderType, Side
from quantlab.execution.portfolio import Portfolio
from quantlab.execution.risk import RiskManager

logger = logging.getLogger(__name__)


class Role(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    PROTECT = "protect"


@dataclass(frozen=True)
class Trade:
    """A closed (or, at the end of a backtest, marked-to-market) round trip of one lot."""

    strategy_id: str
    symbol: str
    side: Side
    quantity: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    fees: float
    bars_held: int
    reason: str
    group_id: str | None = None
    #: protective levels actually placed (None: no such order)
    stop_price: float | None = None
    target_price: float | None = None

    @property
    def pnl(self) -> float:
        return self.side.sign * (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl_net(self) -> float:
        return self.pnl - self.fees

    @property
    def return_pct(self) -> float:
        """Net P&L as a percentage of the entry notional."""
        return 100 * self.pnl_net / (self.entry_price * self.quantity)


@dataclass
class Lot:
    lot_id: str
    strategy_id: str
    symbol: str
    side: Side
    plan: ExitPlan = field(default_factory=ExitPlan)
    entry_order_id: str = ""
    group_id: str | None = None
    quantity: int = 0  # still open
    entry_quantity: int = 0
    entry_price: float = 0.0  # average fill price
    entry_time: pd.Timestamp | None = None
    entry_done: bool = False
    fees: float = 0.0
    exit_quantity: int = 0
    exit_value: float = 0.0
    exit_time: pd.Timestamp | None = None
    exit_reason: str = ""
    bars_held: int = 0
    protective_ids: set[str] = field(default_factory=set)
    exit_order_id: str | None = None
    stop_price: float | None = None
    target_price: float | None = None
    #: reason of a requested close (strategy signal, group exit, flatten): the lot exits as soon as it can
    close_requested: str = ""

    _TIMES = ("entry_time", "exit_time")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["side"] = self.side.value
        data["plan"] = self.plan.to_dict()
        data["protective_ids"] = sorted(self.protective_ids)
        for key in self._TIMES:
            data[key] = data[key].isoformat() if data[key] is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lot:
        data = dict(data, side=Side(data["side"]), protective_ids=set(data["protective_ids"]))
        if "holding_bars" in data:  # state saved before exit plans: its stops are already at the broker
            data["plan"] = {"holding_bars": data.pop("holding_bars")}
        data["plan"] = ExitPlan.from_dict(data["plan"]) or ExitPlan()
        for key in cls._TIMES:
            data[key] = pd.Timestamp(data[key]) if data[key] is not None else None
        return cls(**data)


@dataclass
class _Tracked:
    order: Order
    role: Role
    lot_id: str
    reason: str = ""
    filled: int = 0

    @property
    def remaining(self) -> int:
        return self.order.quantity - self.filled


class OrderRouter:
    def __init__(
        self, broker: Broker, portfolio: Portfolio, risk: RiskManager | None = None, tick_size: float = 0.01
    ) -> None:
        self.broker = broker
        self.portfolio = portfolio
        self.risk = risk or RiskManager()
        self.tick_size = tick_size
        self.lots: dict[str, Lot] = {}
        self.trades: list[Trade] = []
        self.fill_listeners: list[Callable[[Fill], None]] = []
        self._orders: dict[str, _Tracked] = {}
        self._lot_seq = 0
        broker.set_handlers(self.on_fill, self.on_status)

    # --- entries ---------------------------------------------------------------------------------------

    def exposure(self, symbol: str) -> int:
        """Net position plus the unfilled part of pending entries (signed)."""
        pending = sum(
            t.order.side.sign * t.remaining
            for t in self._orders.values()
            if t.role is Role.ENTRY and t.order.symbol == symbol and t.remaining > 0
        )
        return self.portfolio.quantity(symbol) + pending

    def submit_entries(self, orders: Sequence[Order]) -> list[Order | None]:
        """`submit_entry` of each order, groups (same `group_id`) all or nothing; the orders sent, in order."""
        sent: dict[str, Order | None] = {}
        groups: dict[str, list[Order]] = {}
        for order in orders:
            if order.group_id is None:
                sent[order.order_id] = self.submit_entry(order)
            else:
                groups.setdefault(order.group_id, []).append(order)
        for legs in groups.values():
            for leg, result in zip(legs, self._submit_group(legs), strict=True):
                sent[leg.order_id] = result
        return [sent[order.order_id] for order in orders]

    def submit_entry(self, order: Order) -> Order | None:
        """Size, risk-check and send `order`; returns the order actually sent (None if refused)."""
        if order.group_id is not None:
            return self._submit_group([order])[0]
        plan = order.exit_plan or ExitPlan()
        if not self._stop_ok(order, plan):
            return None
        sized = self._size(order, plan)
        if sized is None:
            return None
        checked = self.risk.check_entry(sized, self.exposure(order.symbol))
        if checked is None:
            return None
        self._open(checked, plan)
        return checked

    def exit_strategy(self, strategy_id: str, symbol: str | None = None, reason: str = "signal") -> int:
        """Close the lots of a strategy (on `symbol` only, if given; groups as a whole); returns their count."""
        lots = [
            lot
            for lot in self.lots.values()
            if lot.strategy_id == strategy_id and (symbol is None or lot.symbol == symbol) and not lot.close_requested
        ]
        groups = {lot.group_id for lot in lots if lot.group_id is not None}
        lots += [lot for lot in self.lots.values() if lot.group_id in groups and lot not in lots]
        for lot in lots:
            self._request_close(lot, reason)
        return len(lots)

    # --- broker events ---------------------------------------------------------------------------------

    def on_fill(self, fill: Fill) -> None:
        self.portfolio.apply_fill(fill.order.symbol, fill.signed_quantity, fill.price, fill.fees)
        tracked = self._orders.get(fill.order.order_id)
        if tracked is None:
            logger.warning("Fill of an order not sent by this router (applied to the portfolio only): %s", fill)
            return
        tracked.filled += fill.quantity
        lot = self.lots.get(tracked.lot_id)
        if lot is None:
            logger.error("Fill for the already closed %s: %s", tracked.lot_id, fill)
            self._open_correction(fill.order.symbol, fill.order.side, fill.quantity, fill)
        elif tracked.role is Role.ENTRY:
            total = lot.entry_quantity + fill.quantity
            lot.entry_price = (lot.entry_price * lot.entry_quantity + fill.price * fill.quantity) / total
            lot.entry_quantity, lot.quantity = total, lot.quantity + fill.quantity
            lot.entry_time = lot.entry_time or fill.timestamp
            lot.fees += fill.fees
            if tracked.remaining <= 0:
                lot.entry_done = True
                self._protect(lot)
        else:
            group_id = lot.group_id
            self._reduce(lot, fill, tracked.reason)
            if tracked.role is Role.EXIT and tracked.remaining <= 0 and lot.lot_id in self.lots:
                lot.exit_order_id = None  # entry fills arrived after the exit: exited again at the next bar close
            if tracked.role is Role.PROTECT and group_id is not None:
                self._close_group(group_id, tracked.reason)  # a leg was stopped out: close the other legs
        for listener in self.fill_listeners:
            listener(fill)

    def on_status(self, order_id: str, status: OrderStatus, message: str = "") -> None:
        tracked = self._orders.get(order_id)
        if tracked is None:
            return
        logger.debug("Order %s %s %s", order_id, status.value, message)
        lot = self.lots.get(tracked.lot_id)
        if lot is None:
            return
        if tracked.role is Role.ENTRY and not lot.entry_done:
            tracked.filled = tracked.order.quantity  # nothing more will fill
            lot.entry_done = True
            if lot.quantity > 0:  # partially filled: protect what we have
                self._protect(lot)
            elif lot.exit_quantity > 0:  # already closed by an exit
                self._close(lot)
            else:
                del self.lots[lot.lot_id]
                if lot.group_id is not None:
                    self._close_group(lot.group_id, "leg_missing")  # a pair without one leg is not a pair
        elif tracked.role is Role.PROTECT:
            lot.protective_ids.discard(order_id)
        elif tracked.role is Role.EXIT and lot.exit_order_id == order_id:
            lot.exit_order_id = None  # retried at the next bar close

    # --- bar close: time exits and kill switch ---------------------------------------------------------

    def on_bar_close(self, timestamp: pd.Timestamp) -> None:
        for lot in list(self.lots.values()):
            if lot.quantity <= 0:
                continue
            lot.bars_held += 1
            if lot.exit_order_id is not None:
                continue
            holding = lot.plan.holding_bars
            if lot.close_requested:
                self._exit(lot, lot.close_requested)
            elif holding is not None and lot.bars_held >= holding:
                if lot.group_id is not None:
                    self._close_group(lot.group_id, "time")
                else:
                    self._exit(lot, "time")
        self._check_groups()
        if self.risk.on_equity(timestamp, self.portfolio.equity) and self.risk.limits.flatten_on_halt:
            self.flatten("kill switch")

    def flatten(self, reason: str) -> None:
        """Cancel pending entries and close every open lot at market."""
        for lot in list(self.lots.values()):
            self._request_close(lot, reason)

    def open_trades(self, prices: dict[str, float], timestamp: pd.Timestamp) -> list[Trade]:
        """Open lots as trades marked at `prices` (end of a backtest); nothing is sent."""
        trades = []
        for lot in self.lots.values():
            if lot.quantity > 0 and lot.entry_time is not None and lot.symbol in prices:
                trades.append(self._trade(lot, lot.quantity, prices[lot.symbol], timestamp, "open"))
        return trades

    # --- persistence -----------------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Open lots and the orders that can still affect them (closed trades live in the journal)."""
        orders = [
            {"order": t.order.to_dict(), "role": t.role.value, "lot_id": t.lot_id, "reason": t.reason,
             "filled": t.filled}
            for t in self._orders.values()
            if t.lot_id in self.lots or t.remaining > 0
        ]  # fmt: skip
        return {"lot_seq": self._lot_seq, "lots": [lot.to_dict() for lot in self.lots.values()], "orders": orders}

    def set_state(self, state: dict[str, Any]) -> None:
        self._lot_seq = int(state["lot_seq"])
        self.lots = {lot.lot_id: lot for lot in map(Lot.from_dict, state["lots"])}
        self._orders = {
            o["order"]["order_id"]: _Tracked(Order.from_dict(o["order"]), Role(o["role"]), o["lot_id"], o["reason"],
                                             int(o["filled"]))
            for o in state["orders"]
        }  # fmt: skip

    def orders(self) -> list[Order]:
        """Every order the router knows of (to re-map broker orders after a restart)."""
        return [t.order for t in self._orders.values()]

    # --- internals -------------------------------------------------------------------------------------

    def _next_lot_id(self) -> str:
        self._lot_seq += 1
        return f"lot{self._lot_seq}"

    def _send(self, order: Order, role: Role, lot_id: str, reason: str = "") -> None:
        self._orders[order.order_id] = _Tracked(order, role, lot_id, reason)
        self.broker.submit(order)

    def _round(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def _stop_ok(self, order: Order, plan: ExitPlan) -> bool:
        if not self.risk.limits.require_stop:
            return True
        if plan.stop is None:
            logger.warning("Entry refused, its exit plan has no stop (RiskLimits.require_stop): %s", order)
            return False
        # a group needs its combined stop (checked by the router) and, on top, a leg stop at the broker,
        # which still protects each leg if the application is down
        if order.group_id is not None and plan.leg_stop is None:
            logger.warning("Group entry refused, its exit plan has no leg stop (RiskLimits.require_stop): %s", order)
            return False
        return True

    def _size(self, order: Order, plan: ExitPlan) -> Order | None:
        """`order` with the quantity of its sizing rule, at the signal's close (None: too small to trade)."""
        if order.sizing is None:
            return order
        price = self.portfolio.last_prices.get(order.symbol, float("nan"))
        quantity = order.sizing.quantity_for(order.side.sign, price, self.portfolio.equity, plan, order.conviction)
        if quantity <= 0:
            logger.info("Entry refused, its size rounds to 0 shares (%s at %s): %s", order.sizing, price, order)
            return None
        return order if quantity == order.quantity else dataclasses.replace(order, quantity=quantity)

    def _size_group(self, legs: Sequence[Order], plan: ExitPlan) -> list[Order] | None:
        """The legs with the same notional each (None: a leg rounds to 0 shares)."""
        sizing = legs[0].sizing
        if sizing is None:
            return list(legs)
        prices = [self.portfolio.last_prices.get(leg.symbol, float("nan")) for leg in legs]
        quantities = sizing.group_quantities(prices, self.portfolio.equity, plan, legs[0].conviction)
        if any(q <= 0 for q in quantities):
            logger.info("Group %s refused, a leg rounds to 0 shares (prices %s)", legs[0].group_id, prices)
            return None
        return [dataclasses.replace(leg, quantity=q) for leg, q in zip(legs, quantities, strict=True)]

    def _open(self, order: Order, plan: ExitPlan) -> None:
        lot = Lot(
            self._next_lot_id(), order.strategy_id, order.symbol, order.side, plan, order.order_id, order.group_id
        )
        self.lots[lot.lot_id] = lot
        self._send(order, Role.ENTRY, lot.lot_id)

    def _submit_group(self, legs: Sequence[Order]) -> list[Order | None]:
        plan = legs[0].exit_plan or ExitPlan()
        plan.check_group()
        refused: list[Order | None] = [None] * len(legs)
        if not self._stop_ok(legs[0], plan):
            return refused
        sized = self._size_group(legs, plan)
        if sized is None:
            return refused
        checked = [self.risk.check_entry(leg, self.exposure(leg.symbol)) for leg in sized]
        if any(c is None or c.quantity != leg.quantity for c, leg in zip(checked, sized, strict=True)):
            logger.warning("Group %s refused: every leg must pass the risk checks unchanged", legs[0].group_id)
            return refused
        for leg in sized:
            self._open(leg, plan)
        return list(sized)

    def _group(self, group_id: str) -> list[Lot]:
        return [lot for lot in self.lots.values() if lot.group_id == group_id]

    def _close_group(self, group_id: str, reason: str) -> None:
        for lot in self._group(group_id):
            if not lot.close_requested:
                self._request_close(lot, reason)

    def _check_groups(self) -> None:
        """Combined stop / target of each group, on the bar close prices."""
        prices = self.portfolio.last_prices
        for group_id in sorted({lot.group_id for lot in self.lots.values() if lot.group_id is not None}):
            legs = self._group(group_id)
            plan = legs[0].plan
            if plan.stop is None and plan.target is None:
                continue
            if any(not lot.entry_done or lot.close_requested for lot in legs):
                continue
            if any(lot.quantity > 0 and lot.symbol not in prices for lot in legs):
                continue
            notional = sum(lot.entry_price * lot.entry_quantity for lot in legs)
            pnl = sum(lot.side.sign * (prices[lot.symbol] - lot.entry_price) * lot.quantity for lot in legs)
            if notional <= 0:
                continue
            if plan.stop is not None and pnl <= -plan.stop.value * notional:
                self._close_group(group_id, "stop_loss")
            elif plan.target is not None and pnl >= plan.target.value * notional:
                self._close_group(group_id, "take_profit")

    def _request_close(self, lot: Lot, reason: str) -> None:
        lot.close_requested = lot.close_requested or reason
        if not lot.entry_done:
            self.broker.cancel(lot.entry_order_id)
        if lot.quantity > 0 and lot.exit_order_id is None:
            self._exit(lot, lot.close_requested)

    def _protect(self, lot: Lot) -> None:
        if lot.quantity <= 0 or lot.exit_order_id is not None:
            return
        if lot.close_requested:  # closed while its entry was still filling
            self._exit(lot, lot.close_requested)
            return
        plan, sign, exit_side = lot.plan, lot.side.sign, lot.side.opposite
        oca = f"oca-{lot.entry_order_id}"
        protective: list[tuple[Order, str]] = []
        if lot.group_id is not None:  # the group's own levels are checked on its combined P&L
            if plan.leg_stop is not None:
                lot.stop_price = self._round(lot.entry_price * (1 - sign * plan.leg_stop))
                protective.append(
                    (Order(lot.symbol, exit_side, lot.quantity, lot.strategy_id, OrderType.STOP,
                           stop_price=lot.stop_price, oca_group=oca), "leg_stop")
                )  # fmt: skip
        else:
            if plan.stop is not None:
                lot.stop_price = self._round(plan.stop.price(lot.entry_price, -sign))
            if plan.target is not None:
                lot.target_price = self._round(plan.target.price(lot.entry_price, sign))
            # an absolute level the fill already went through (gap): close at market now
            if lot.stop_price is not None and sign * (lot.entry_price - lot.stop_price) <= 0:
                self._exit(lot, "stop_gap")
                return
            if lot.target_price is not None and sign * (lot.target_price - lot.entry_price) <= 0:
                self._exit(lot, "target_gap")
                return
            if lot.stop_price is not None:
                protective.append(
                    (Order(lot.symbol, exit_side, lot.quantity, lot.strategy_id, OrderType.STOP,
                           stop_price=lot.stop_price, oca_group=oca), "stop_loss")
                )  # fmt: skip
            if lot.target_price is not None:
                protective.append(
                    (Order(lot.symbol, exit_side, lot.quantity, lot.strategy_id, OrderType.LIMIT,
                           limit_price=lot.target_price, oca_group=oca), "take_profit")
                )  # fmt: skip
        for order, reason in protective:
            lot.protective_ids.add(order.order_id)
            self._send(order, Role.PROTECT, lot.lot_id, reason)

    def _exit(self, lot: Lot, reason: str) -> None:
        for order_id in list(lot.protective_ids):
            self.broker.cancel(order_id)
        lot.protective_ids.clear()
        order = Order(lot.symbol, lot.side.opposite, lot.quantity, lot.strategy_id)
        lot.exit_order_id = order.order_id
        self._send(order, Role.EXIT, lot.lot_id, reason)

    def _reduce(self, lot: Lot, fill: Fill, reason: str) -> None:
        used = min(fill.quantity, lot.quantity)
        if used:
            lot.quantity -= used
            lot.exit_quantity += used
            lot.exit_value += used * fill.price
            lot.fees += fill.fees
            lot.exit_time, lot.exit_reason = fill.timestamp, reason
        if fill.quantity > used:
            logger.error("Fill over-closes %s by %d shares: opening a correction", lot.lot_id, fill.quantity - used)
            self._open_correction(lot.symbol, fill.order.side, fill.quantity - used, fill)
        if lot.quantity == 0 and lot.entry_done:
            self._close(lot)

    def _close(self, lot: Lot) -> None:
        for order_id in lot.protective_ids:
            self.broker.cancel(order_id)  # the OCA sibling, if the broker has not cancelled it yet
        exit_order = self._orders.get(lot.exit_order_id or "")
        if exit_order is not None and exit_order.remaining > 0:
            self.broker.cancel(exit_order.order.order_id)
        del self.lots[lot.lot_id]
        assert lot.exit_time is not None
        self.trades.append(
            self._trade(lot, lot.exit_quantity, lot.exit_value / lot.exit_quantity, lot.exit_time, lot.exit_reason)
        )

    @staticmethod
    def _trade(lot: Lot, quantity: int, exit_price: float, exit_time: pd.Timestamp, reason: str) -> Trade:
        assert lot.entry_time is not None
        return Trade(
            strategy_id=lot.strategy_id,
            symbol=lot.symbol,
            side=lot.side,
            quantity=quantity,
            entry_time=lot.entry_time,
            exit_time=exit_time,
            entry_price=lot.entry_price,
            exit_price=exit_price,
            fees=lot.fees,
            bars_held=lot.bars_held,
            reason=reason,
            group_id=lot.group_id,
            stop_price=lot.stop_price,
            target_price=lot.target_price,
        )

    def _open_correction(self, symbol: str, side: Side, quantity: int, fill: Fill) -> None:
        """Unintended shares (over-fill): track them as a lot and close it immediately."""
        lot = Lot(self._next_lot_id(), fill.order.strategy_id, symbol, side, entry_done=True)
        lot.quantity = lot.entry_quantity = quantity
        lot.entry_price, lot.entry_time = fill.price, fill.timestamp
        self.lots[lot.lot_id] = lot
        self._exit(lot, "correction")
