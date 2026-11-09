"""Position sizing: every mode, the stop floor, groups at equal notional, and the router applying it."""

from __future__ import annotations

import pytest
from test_execution import ManualBroker

from quantlab.execution.exits import ExitPlan, Level
from quantlab.execution.orders import Order, Side
from quantlab.execution.portfolio import Portfolio
from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.execution.router import OrderRouter
from quantlab.execution.sizing import SizeMode, Sizing
from quantlab.strategies.pair_trading import PairTradingStrategy
from quantlab.strategies.trend_following import TrendFollowingStrategy

PLAN = ExitPlan(stop=Level.pct(0.02), target=Level.pct(0.05))


def test_fixed_mode_times_the_conviction() -> None:
    sizing = Sizing(SizeMode.FIXED, quantity=10)
    assert sizing.quantity_for(1, 100.0, 1e5, PLAN, 1.0) == 10
    assert sizing.quantity_for(1, 100.0, 1e5, PLAN, 0.37) == 3
    assert sizing.quantity_for(1, 100.0, 1e5, PLAN, 0.0) == 1  # at least one share


@pytest.mark.parametrize("amplitude", [0.06, 0.4, 1.3, 2.49, 3.0, 4.99, 5.0, 7.5, -0.8, -6.0])
def test_fixed_mode_reproduces_the_old_cascade_quantity(amplitude: float) -> None:
    confidence = min(abs(amplitude) / 5.0, 1.0)
    old = max(1, int(confidence * 10))
    assert Sizing(SizeMode.FIXED, quantity=10).quantity_for(1, 123.4, 1e5, PLAN, confidence) == old


def test_notional_and_equity_modes() -> None:
    assert Sizing(SizeMode.NOTIONAL, value=5_000).quantity_for(1, 123.0, 1e5, PLAN, 1.0) == 40
    assert Sizing(SizeMode.NOTIONAL, value=5_000).quantity_for(-1, 123.0, 1e5, PLAN, 0.5) == 20
    assert Sizing(SizeMode.EQUITY_PCT, value=10).quantity_for(1, 200.0, 50_000, PLAN, 1.0) == 25
    assert Sizing(SizeMode.NOTIONAL, value=50).quantity_for(1, 123.0, 1e5, PLAN, 1.0) == 0  # too small


def test_risk_mode_divides_the_risk_by_the_distance_to_the_stop() -> None:
    sizing = Sizing(SizeMode.RISK, value=0.5)  # lose at most 0.5 % of 100k = $500 at the stop
    assert sizing.quantity_for(1, 100.0, 1e5, PLAN, 1.0) == 250  # stop 2 % = $2 away
    atr_plan = ExitPlan(stop=Level.distance(1.25))
    assert sizing.quantity_for(-1, 100.0, 1e5, atr_plan, 1.0) == 400
    absolute = ExitPlan(stop=Level.at(95.0))
    assert sizing.quantity_for(1, 100.0, 1e5, absolute, 1.0) == 100
    # a tiny stop is floored at min_stop_pct of the price: no huge size
    tiny = ExitPlan(stop=Level.distance(0.001))
    assert sizing.quantity_for(1, 100.0, 1e5, tiny, 1.0) == 5_000  # floor 0.1 % = $0.10
    assert Sizing(SizeMode.RISK, value=0.5, min_stop_pct=1.0).quantity_for(1, 100.0, 1e5, tiny, 1.0) == 500
    assert sizing.quantity_for(1, 100.0, 1e5, ExitPlan(), 1.0) == 0  # no stop, no risk-based size


def test_invalid_sizing_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a value"):
        Sizing(SizeMode.RISK)
    with pytest.raises(ValueError):
        Sizing(SizeMode.FIXED, quantity=0)


def test_groups_hold_the_same_notional_per_leg() -> None:
    # fixed: `quantity` shares of the first leg, the same dollars of the second
    assert Sizing(SizeMode.FIXED, quantity=10).group_quantities([250.0, 400.0], 1e5, PLAN, 1.0) == [10, 6]
    assert Sizing(SizeMode.NOTIONAL, value=10_000).group_quantities([250.0, 400.0], 1e5, PLAN, 1.0) == [20, 12]
    # risk: $500 at a combined stop of 2 % of the gross notional -> $25,000 gross, $12,500 per leg
    assert Sizing(SizeMode.RISK, value=0.5).group_quantities([250.0, 400.0], 1e5, PLAN, 1.0) == [50, 31]
    assert Sizing(SizeMode.NOTIONAL, value=100).group_quantities([250.0, 400.0], 1e5, PLAN, 1.0) == [0, 0]


def test_sizing_round_trips_with_the_order() -> None:
    order = Order("X", Side.BUY, 1, "s", exit_plan=PLAN, sizing=Sizing(SizeMode.RISK, value=0.5), conviction=0.4)
    assert Order.from_dict(order.to_dict()) == order
    plain = Order("X", Side.BUY, 3, "s")
    assert Order.from_dict(plain.to_dict()) == plain


def make_router(limits: RiskLimits | None = None) -> tuple[OrderRouter, ManualBroker, Portfolio]:
    broker, portfolio = ManualBroker(), Portfolio(100_000)
    portfolio.mark({"A": 250.0, "B": 400.0, "X": 100.0})
    return OrderRouter(broker, portfolio, RiskManager(limits)), broker, portfolio  # type: ignore[arg-type]


def test_router_sizes_entries_at_the_signal_close() -> None:
    router, broker, _ = make_router()
    sized = Order("X", Side.BUY, 1, "s", exit_plan=PLAN, sizing=Sizing(SizeMode.RISK, value=0.5))
    assert router.submit_entry(sized) is not None
    assert broker.submitted[-1].quantity == 250
    assert router.submit_entry(Order("X", Side.SELL, 7, "s", exit_plan=PLAN)) is not None  # no rule: as given
    assert broker.submitted[-1].quantity == 7
    too_small = Order("X", Side.BUY, 1, "s", exit_plan=PLAN, sizing=Sizing(SizeMode.NOTIONAL, value=10))
    assert router.submit_entry(too_small) is None


def test_risk_limits_still_clip_the_sized_order() -> None:
    router, broker, _ = make_router(RiskLimits(max_order_quantity=100))
    order = Order("X", Side.BUY, 1, "s", exit_plan=PLAN, sizing=Sizing(SizeMode.RISK, value=0.5))
    assert router.submit_entry(order) is not None and broker.submitted[-1].quantity == 100


def test_router_sizes_a_pair_at_equal_notional() -> None:
    router, broker, _ = make_router()
    strategy = PairTradingStrategy(symbol_a="A", symbol_b="B")
    legs = strategy.pair(Side.SELL, Side.BUY)
    sent = router.submit_entries(legs)
    assert [o.quantity for o in sent if o] == [10, 6]
    assert [(o.symbol, o.side, o.quantity) for o in broker.submitted] == [("A", Side.SELL, 10), ("B", Side.BUY, 6)]
    # a leg rounding to 0 shares refuses the whole group
    router, broker, _ = make_router()
    small = PairTradingStrategy(symbol_a="A", symbol_b="B", size_mode="notional", size_value=300)
    assert router.submit_entries(small.pair(Side.BUY, Side.SELL)) == [None, None] and broker.submitted == []


def test_a_pair_needs_both_stops() -> None:
    router, _, _ = make_router()
    assert all(router.submit_entries(PairTradingStrategy(symbol_a="A", symbol_b="B").pair(Side.BUY, Side.SELL)))
    for stops in ({"stop_loss_pct": None}, {"leg_stop_pct": None}):
        router, broker, _ = make_router()
        strategy = PairTradingStrategy(symbol_a="A", symbol_b="B", **stops)
        assert router.submit_entries(strategy.pair(Side.BUY, Side.SELL)) == [None, None] and broker.submitted == []


def test_strategy_orders_carry_their_sizing_rule() -> None:
    strategy = TrendFollowingStrategy(size_mode="equity_pct", size_value=5)
    order = strategy.order(Side.BUY, conviction=0.5)
    assert order.sizing == Sizing(SizeMode.EQUITY_PCT, 10, 5.0) and order.conviction == 0.5
    assert strategy.order(Side.BUY, 3).sizing is None  # an explicit quantity is sent as is
    with pytest.raises(ValueError):
        TrendFollowingStrategy(size_mode="risk")
