"""Exit plans per lot, groups of lots (pairs) and exits requested by the strategies."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from test_execution import T0, ManualBroker, bar, entry

from quantlab.execution.exits import ExitPlan, Level, LevelKind
from quantlab.execution.orders import Order, OrderStatus, OrderType, Side
from quantlab.execution.portfolio import Portfolio
from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.execution.router import Lot, OrderRouter
from quantlab.execution.simulated import SimulatedBroker
from quantlab.features.technical import ATR
from quantlab.runtime.bar_feed import HistoricalBarFeed
from quantlab.runtime.runner import StrategyRunner
from quantlab.strategies.base import Strategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.pair_trading import PairTradingStrategy


def make_router(limits: RiskLimits | None = None) -> tuple[OrderRouter, SimulatedBroker]:
    broker = SimulatedBroker(fee_bps=0.0, slippage_bps=0.0)
    return OrderRouter(broker, Portfolio(10_000), RiskManager(limits)), broker


def pair_bar(i: int, a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> pd.Series:
    values = {}
    for symbol, prices in (("A", a), ("B", b)):
        for field, price in zip(("open", "high", "low", "close"), prices, strict=True):
            values[f"{field}_{symbol}"] = price
        values[f"volume_{symbol}"] = 1000.0
    return pd.Series(values, name=T0 + pd.Timedelta(minutes=i))


def flat(price: float) -> tuple[float, float, float, float]:
    return (price, price, price, price)


def pair_orders(plan: ExitPlan, quantity_a: int = 10, quantity_b: int = 10) -> list[Order]:
    return [
        Order("A", Side.BUY, quantity_a, "p", exit_plan=plan, group_id="g1"),
        Order("B", Side.SELL, quantity_b, "p", exit_plan=plan, group_id="g1"),
    ]


# --- exit plans --------------------------------------------------------------------------------------------


def test_levels_are_anchored_at_the_fill_price() -> None:
    assert Level.pct(0.02).price(100.0, -1) == pytest.approx(98.0)
    assert Level.distance(0.5).price(100.0, 1) == 100.5
    assert Level.at(97.0).price(100.0, -1) == 97.0
    with pytest.raises(ValueError):
        Level.pct(0)
    plan = ExitPlan(stop=Level.distance(0.4), target=Level.at(103.0), holding_bars=7, leg_stop=0.05)
    assert ExitPlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan


def test_each_lot_gets_its_own_plan() -> None:
    router, broker = make_router()
    router.submit_entry(Order("X", Side.BUY, 10, "s", exit_plan=ExitPlan(stop=Level.distance(1.5))))
    router.submit_entry(Order("X", Side.SELL, 5, "s", exit_plan=ExitPlan(stop=Level.pct(0.01), target=Level.at(95))))
    broker.on_bar(bar(1, 100, 100.2, 99.8, 100))
    orders = {(o.side, o.order_type): o for o in broker.pending_orders}
    assert orders[(Side.SELL, OrderType.STOP)].stop_price == 98.5  # long: 1.5 $ under its fill
    assert orders[(Side.BUY, OrderType.STOP)].stop_price == 101.0  # short: 1 % above its fill
    assert orders[(Side.BUY, OrderType.LIMIT)].limit_price == 95.0
    assert (Side.SELL, OrderType.LIMIT) not in orders  # the long lot has no target


def test_entry_without_a_stop_is_refused() -> None:
    router, broker = make_router()
    assert router.submit_entry(entry(Side.BUY, 10, stop=None)) is None
    assert router.lots == {} and broker.pending_orders == []
    permissive, _ = make_router(RiskLimits(require_stop=False))
    assert permissive.submit_entry(entry(Side.BUY, 10, stop=None)) is not None


def test_absolute_target_already_passed_at_the_fill_closes_at_market() -> None:
    router, broker = make_router()
    router.submit_entry(Order("X", Side.BUY, 10, "s", exit_plan=ExitPlan(stop=Level.pct(0.02), target=Level.at(99))))
    broker.on_bar(bar(1, 100, 100, 100, 100))  # filled at 100, above its target
    (exit_order,) = broker.pending_orders
    assert exit_order.order_type is OrderType.MARKET
    broker.on_bar(bar(2, 101, 101, 101, 101))
    (trade,) = router.trades
    assert (trade.reason, trade.exit_price) == ("target_gap", 101)


def test_plan_is_frozen_in_the_saved_lot() -> None:
    router, broker = make_router()
    router.submit_entry(entry(Side.BUY, 10, stop=0.02, target=0.05, holding=3))
    broker.on_bar(bar(1, 100, 100, 100, 100))
    state = json.loads(json.dumps(router.get_state()))
    restored, _ = make_router()
    restored.set_state(state)
    (lot,) = restored.lots.values()
    assert lot.plan == ExitPlan(stop=Level.pct(0.02), target=Level.pct(0.05), holding_bars=3)
    assert (lot.stop_price, lot.target_price) == (98.0, 105.0)


def test_lot_saved_before_exit_plans_is_migrated() -> None:
    lot = Lot("lot1", "s", "X", Side.BUY, ExitPlan(holding_bars=4)).to_dict()
    del lot["plan"], lot["group_id"], lot["stop_price"], lot["target_price"], lot["close_requested"]
    lot["holding_bars"] = 4
    assert Lot.from_dict(lot).plan == ExitPlan(holding_bars=4)


# --- exits requested by the strategy -----------------------------------------------------------------------


def test_strategy_exit_closes_only_its_lots_and_cancels_pending_entries() -> None:
    router, broker = make_router()
    router.submit_entry(entry(Side.BUY, 10, "s"))
    router.submit_entry(entry(Side.BUY, 5, "t"))
    broker.on_bar(bar(1, 100, 100, 100, 100))
    router.submit_entry(entry(Side.BUY, 3, "s"))  # still pending
    assert router.exit_strategy("s", reason="signal") == 2
    broker.on_bar(bar(2, 101, 101, 101, 101))
    assert [(t.strategy_id, t.reason, t.quantity) for t in router.trades] == [("s", "signal", 10)]
    assert [lot.strategy_id for lot in router.lots.values()] == ["t"]
    assert router.portfolio.quantity("X") == 5


class BuyThenExit(Strategy):
    """Buys on bar 1, asks to close on bar 3."""

    def __init__(self) -> None:
        super().__init__(symbol="X", holding_bars=None)
        self.count = 0

    def decide(self, bar: pd.Series) -> list[Order]:
        self.count += 1
        if self.count == 3:
            self.close_positions(reason="flip")
        return [self.order(Side.BUY, 10)] if self.count == 1 else []


def test_runner_executes_the_strategy_exit_requests() -> None:
    bars = pd.DataFrame([bar(i, 100 + i, 100.5 + i, 99.5 + i, 100 + i) for i in range(8)])
    strategy = BuyThenExit()
    strategy.fit(bars.iloc[:1])
    runner = StrategyRunner([strategy], SimulatedBroker(fee_bps=0, slippage_bps=0), RiskManager())
    runner.run(HistoricalBarFeed(bars.iloc[1:]))
    (trade,) = runner.router.trades
    # bought at bar 2's open, close requested at bar 3's close, sold at bar 4's open
    assert (trade.reason, trade.entry_price, trade.exit_price) == ("flip", 102, 104)


# --- strategy exit parameters ------------------------------------------------------------------------------


class Idle(Strategy):
    def decide(self, bar: pd.Series) -> list[Order]:
        return []


def history(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame([bar(i, 100 + i % 3, 101 + i % 3, 99 + i % 3, 100.5 + i % 3) for i in range(n)])


def test_default_plan_from_the_strategy_parameters() -> None:
    strategy = Idle(stop_loss_pct=1.5, take_profit_pct=0, holding_bars=0)
    strategy.fit(history())
    assert strategy.exit_plan() == ExitPlan(stop=Level.pct(0.015))


def test_atr_levels_are_measured_at_the_signal() -> None:
    strategy = Idle(stop_atr=2.0, take_profit_atr=3.0, atr_window=5)
    strategy.fit(history())
    atr = strategy.feed.row()[ATR(5).name]
    plan = strategy.exit_plan()
    assert plan.stop == Level.distance(2.0 * atr) and plan.target == Level.distance(3.0 * atr)
    assert plan.holding_bars == 5


def test_mean_reversion_can_target_the_middle_band() -> None:
    strategy = MeanReversionStrategy(target_middle_band=True)
    data = history(60)
    strategy.fit(data)
    oversold = data.iloc[-1].copy()
    oversold[["close", "low"]] = 50.0
    oversold["volume"] = 1e6
    oversold.name = data.index[-1] + pd.Timedelta(minutes=1)
    strategy.start()
    orders = strategy.on_bar(oversold)
    assert orders, "an extreme oversold bar should trigger a buy"
    (order,) = orders
    assert order.exit_plan is not None and order.exit_plan.target is not None
    assert order.exit_plan.target.kind is LevelKind.PRICE
    middle = (strategy.feed.row()[strategy.bollinger.upper] + strategy.feed.row()[strategy.bollinger.lower]) / 2
    assert order.exit_plan.target.value == pytest.approx(middle)


# --- groups (pairs) ----------------------------------------------------------------------------------------


#: router mechanics of groups without leg stops (a pair strategy always has one, see the tests below)
NO_LEG_STOP = RiskLimits(require_stop=False)


def test_group_needs_a_combined_stop_and_a_leg_stop() -> None:
    for plan in (ExitPlan(leg_stop=0.05), ExitPlan(stop=Level.pct(0.01))):
        router, broker = make_router()
        assert router.submit_entries(pair_orders(plan)) == [None, None] and broker.pending_orders == []
    router, _ = make_router()
    assert all(router.submit_entries(pair_orders(ExitPlan(stop=Level.pct(0.01), leg_stop=0.05))))


def test_group_is_refused_as_a_whole_when_one_leg_fails_the_risk_checks() -> None:
    router, broker = make_router(RiskLimits(max_order_quantity=8))
    sent = router.submit_entries(pair_orders(ExitPlan(stop=Level.pct(0.01)), quantity_a=5, quantity_b=10))
    assert sent == [None, None]
    assert router.lots == {} and broker.pending_orders == []


def test_group_combined_stop_closes_both_legs() -> None:
    router, broker = make_router(NO_LEG_STOP)  # combined stop only: nothing at the broker
    plan = ExitPlan(stop=Level.pct(0.005), target=Level.pct(0.01))
    assert all(router.submit_entries(pair_orders(plan)))
    broker.on_bar(pair_bar(1, flat(100), flat(100)))
    assert broker.pending_orders == []  # no per-leg stop / target at the broker without leg_stop
    router.portfolio.mark({"A": 99.6, "B": 100.3})  # -4 - 3 = -7 > -10 (0.5 % of 2000): kept
    router.on_bar_close(T0 + pd.Timedelta(minutes=1))
    assert len(router.lots) == 2 and broker.pending_orders == []
    router.portfolio.mark({"A": 99.5, "B": 100.6})  # -5 - 6 = -11: stop
    router.on_bar_close(T0 + pd.Timedelta(minutes=2))
    broker.on_bar(pair_bar(3, flat(99.4), flat(100.7)))
    assert sorted((t.symbol, t.reason, t.group_id) for t in router.trades) == [
        ("A", "stop_loss", "g1"),
        ("B", "stop_loss", "g1"),
    ]
    assert router.portfolio.net_positions() == {}


def test_group_target_and_time_exit_close_both_legs_together() -> None:
    router, broker = make_router(NO_LEG_STOP)
    router.submit_entries(pair_orders(ExitPlan(stop=Level.pct(0.5), target=Level.pct(0.01))))
    broker.on_bar(pair_bar(1, flat(100), flat(100)))
    router.portfolio.mark({"A": 101.0, "B": 99.0})  # +10 + 10 = 1 % of 2000
    router.on_bar_close(T0 + pd.Timedelta(minutes=1))
    broker.on_bar(pair_bar(2, flat(101), flat(99)))
    assert {t.reason for t in router.trades} == {"take_profit"} and len(router.trades) == 2

    router, broker = make_router(NO_LEG_STOP)
    router.submit_entries(pair_orders(ExitPlan(stop=Level.pct(0.5), holding_bars=2)))
    for i in (1, 2, 3):
        broker.on_bar(pair_bar(i, flat(100), flat(100)))
        router.portfolio.mark({"A": 100.0, "B": 100.0})
        router.on_bar_close(T0 + pd.Timedelta(minutes=i))
    assert [t.reason for t in router.trades] == ["time", "time"]


def test_leg_stop_at_the_broker_closes_the_other_leg() -> None:
    router, broker = make_router()
    router.submit_entries(pair_orders(ExitPlan(stop=Level.pct(0.5), leg_stop=0.05)))
    broker.on_bar(pair_bar(1, flat(100), flat(100)))
    stops = {o.symbol: o for o in broker.pending_orders}
    assert (stops["A"].order_type, stops["A"].stop_price) == (OrderType.STOP, 95.0)
    assert (stops["B"].side, stops["B"].stop_price) == (Side.BUY, 105.0)
    broker.on_bar(pair_bar(2, (99, 99, 94, 94), flat(100)))  # A falls through its stop
    broker.on_bar(pair_bar(3, flat(94), flat(100.5)))  # B closed at the next open
    assert sorted((t.symbol, t.reason, t.exit_price) for t in router.trades) == [
        ("A", "leg_stop", 95.0),
        ("B", "leg_stop", 100.5),
    ]
    assert router.lots == {} and broker.pending_orders == []


def test_leg_not_filled_closes_the_other_leg() -> None:
    broker = ManualBroker()
    router = OrderRouter(broker, Portfolio(), RiskManager())
    leg_a, leg_b = router.submit_entries(pair_orders(ExitPlan(stop=Level.pct(0.01), leg_stop=0.05)))
    assert leg_a is not None and leg_b is not None and broker.on_status is not None
    broker.fill(leg_a, 10, 100.0)
    broker.on_status(leg_b.order_id, OrderStatus.REJECTED, "not shortable")
    exit_order = broker.submitted[-1]
    assert (exit_order.symbol, exit_order.side, exit_order.quantity) == ("A", Side.SELL, 10)
    broker.fill(exit_order, 10, 100.5)
    (trade,) = router.trades
    assert trade.reason == "leg_missing" and router.lots == {}


def test_pair_strategy_emits_a_group_and_exits_on_the_zscore() -> None:
    strategy = PairTradingStrategy(symbol_a="A", symbol_b="B", leg_stop_pct=4)
    orders = strategy.pair(Side.SELL, Side.BUY)
    assert {o.group_id for o in orders} != {None} and len({o.group_id for o in orders}) == 1
    plan = orders[0].exit_plan
    assert plan == ExitPlan(stop=Level.pct(0.02), target=Level.pct(0.05), holding_bars=5, leg_stop=0.04)
    with pytest.raises(ValueError):
        PairTradingStrategy(stop_atr=1.0)
