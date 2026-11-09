"""Shared execution layer: simulated broker, order router (lots, risk, kill switch), portfolio, metrics, runner."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.metrics import compute_metrics, daily_returns, max_drawdown
from quantlab.backtest.run import backtest
from quantlab.execution.broker import FillHandler, StatusHandler
from quantlab.execution.exits import ExitPlan, Level
from quantlab.execution.orders import Fill, Order, OrderStatus, OrderType, Side
from quantlab.execution.portfolio import Portfolio
from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.execution.router import OrderRouter, Trade
from quantlab.execution.simulated import SimulatedBroker
from quantlab.runtime.bar_feed import HistoricalBarFeed
from quantlab.runtime.runner import StrategyRunner
from quantlab.strategies.base import Strategy

T0 = pd.Timestamp("2026-01-05 15:00", tz="UTC")


def bar(i: int, open_: float, high: float, low: float, close: float) -> pd.Series:
    return pd.Series(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0}, name=T0 + pd.Timedelta(minutes=i)
    )


class Recorder:
    def __init__(self) -> None:
        self.fills: list[Fill] = []
        self.statuses: list[tuple[str, OrderStatus]] = []

    def on_fill(self, fill: Fill) -> None:
        self.fills.append(fill)

    def on_status(self, order_id: str, status: OrderStatus, message: str) -> None:
        self.statuses.append((order_id, status))


def sim(fee_bps: float = 0.0, slippage_bps: float = 0.0) -> tuple[SimulatedBroker, Recorder]:
    broker, rec = SimulatedBroker(fee_bps=fee_bps, slippage_bps=slippage_bps), Recorder()
    broker.set_handlers(rec.on_fill, rec.on_status)
    return broker, rec


# --- SimulatedBroker ---------------------------------------------------------------------------------------


def test_market_order_fills_at_next_open_with_slippage_and_fees() -> None:
    broker, rec = sim(fee_bps=1.0, slippage_bps=10.0)
    broker.submit(Order("X", Side.BUY, 10, "s"))
    assert rec.fills == []  # nothing fills at submission (the decision bar's close)
    broker.on_bar(bar(1, 100, 101, 99, 100.5))
    (fill,) = rec.fills
    assert fill.price == pytest.approx(100 * 1.001)
    assert fill.fees == pytest.approx(10 * fill.price * 1e-4)
    assert fill.timestamp == T0 + pd.Timedelta(minutes=1)
    assert broker.positions() == {"X": 10}


def test_sell_stop_fills_at_stop_or_at_a_gapped_open() -> None:
    broker, rec = sim()
    broker.submit(Order("X", Side.SELL, 5, "s", OrderType.STOP, stop_price=98))
    broker.on_bar(bar(1, 100, 101, 98.5, 99))  # not touched
    assert rec.fills == []
    broker.on_bar(bar(2, 99, 99.5, 97, 97.5))  # trades through the stop
    assert rec.fills[-1].price == 98

    broker.submit(Order("X", Side.SELL, 5, "s", OrderType.STOP, stop_price=98))
    broker.on_bar(bar(3, 95, 96, 94, 95))  # gap below the stop: filled at the open
    assert rec.fills[-1].price == 95


def test_take_profit_limit_and_oca_sibling_cancelled() -> None:
    broker, rec = sim()
    stop = Order("X", Side.SELL, 5, "s", OrderType.STOP, stop_price=95, oca_group="g")
    target = Order("X", Side.SELL, 5, "s", OrderType.LIMIT, limit_price=105, oca_group="g")
    broker.submit(stop)
    broker.submit(target)
    broker.on_bar(bar(1, 104, 106, 103, 105.5))
    assert [(f.order.order_id, f.price) for f in rec.fills] == [(target.order_id, 105)]
    assert rec.statuses == [(stop.order_id, OrderStatus.CANCELLED)]
    assert broker.pending_orders == []


def test_stop_wins_when_a_bar_touches_both_legs() -> None:
    broker, rec = sim()
    stop = Order("X", Side.SELL, 5, "s", OrderType.STOP, stop_price=95, oca_group="g")
    broker.submit(Order("X", Side.SELL, 5, "s", OrderType.LIMIT, limit_price=105, oca_group="g"))
    broker.submit(stop)
    broker.on_bar(bar(1, 100, 106, 94, 100))
    assert [f.order.order_id for f in rec.fills] == [stop.order_id]


# --- OrderRouter --------------------------------------------------------------------------------------------


def entry(
    side: Side,
    quantity: int,
    strategy_id: str = "s",
    stop: float | None = 0.02,
    target: float | None = None,
    holding: int | None = None,
    symbol: str = "X",
) -> Order:
    """Entry order on X with an exit plan of pct levels."""
    plan = ExitPlan(
        stop=Level.pct(stop) if stop else None, target=Level.pct(target) if target else None, holding_bars=holding
    )
    return Order(symbol, side, quantity, strategy_id, exit_plan=plan)


NO_STOP = RiskLimits(require_stop=False)


def make_router(limits: RiskLimits | None = None) -> tuple[OrderRouter, SimulatedBroker]:
    broker = SimulatedBroker(fee_bps=0.0, slippage_bps=0.0)
    return OrderRouter(broker, Portfolio(10_000), RiskManager(limits)), broker


def test_entry_places_protective_oca_pair_after_fill() -> None:
    router, broker = make_router()
    router.submit_entry(entry(Side.BUY, 10, stop=0.02, target=0.05))
    broker.on_bar(bar(1, 100, 100.5, 99.5, 100))
    stop, target = sorted(broker.pending_orders, key=lambda o: o.order_type.value)[::-1]
    assert (stop.order_type, stop.stop_price, stop.side, stop.quantity) == (OrderType.STOP, 98.0, Side.SELL, 10)
    assert (target.order_type, target.limit_price) == (OrderType.LIMIT, 105.0)
    assert stop.oca_group == target.oca_group is not None

    broker.on_bar(bar(2, 99, 99, 97, 97.5))  # stop hit
    (trade,) = router.trades
    assert (trade.reason, trade.entry_price, trade.exit_price, trade.quantity) == ("stop_loss", 100, 98, 10)
    assert trade.pnl == pytest.approx(-20)
    assert router.lots == {}
    assert broker.pending_orders == []
    assert router.portfolio.cash == pytest.approx(10_000 - 20)


def test_protective_orders_matched_on_the_entry_bar() -> None:
    router, broker = make_router()
    router.submit_entry(entry(Side.BUY, 1, stop=0.01))
    broker.on_bar(bar(1, 100, 100.2, 98, 98.5))  # entry at the open, then the bar falls through the stop
    assert [t.reason for t in router.trades] == ["stop_loss"]


def test_time_exit_cancels_protective_orders() -> None:
    router, broker = make_router()
    router.submit_entry(entry(Side.SELL, 3, target=0.05, holding=2))
    for i in range(1, 4):
        broker.on_bar(bar(i, 100 + i, 100.5 + i, 99.5 + i, 100 + i))
        router.on_bar_close(T0 + pd.Timedelta(minutes=i))
    # entry at the open of bar 1, held bars 1-2, exit sent at the close of bar 2, filled at the open of bar 3
    (trade,) = router.trades
    assert (trade.reason, trade.entry_price, trade.exit_price, trade.bars_held) == ("time", 101, 103, 2)
    assert trade.pnl == pytest.approx(-6)
    assert broker.pending_orders == [] and broker.positions() == {}


def test_lots_of_several_strategies_are_attributed_separately() -> None:
    router, broker = make_router(NO_STOP)
    router.submit_entry(entry(Side.BUY, 5, "a", stop=None, holding=1))
    router.submit_entry(entry(Side.SELL, 3, "b", stop=None, holding=2))
    broker.on_bar(bar(1, 100, 100, 100, 100))
    assert router.portfolio.quantity("X") == 2  # the broker position is the net sum
    for i in (1, 2, 3):
        if i > 1:
            broker.on_bar(bar(i, 100 + i, 100 + i, 100 + i, 100 + i))
        router.on_bar_close(T0 + pd.Timedelta(minutes=i))
    by_strategy = {t.strategy_id: t for t in router.trades}
    assert by_strategy["a"].exit_price == 102 and by_strategy["a"].pnl == pytest.approx(10)
    assert by_strategy["b"].exit_price == 103 and by_strategy["b"].pnl == pytest.approx(-9)
    assert router.portfolio.quantity("X") == 0


class ManualBroker:
    """Broker driven by the test: records orders, the test injects fills and statuses."""

    def __init__(self) -> None:
        self.submitted: list[Order] = []
        self.cancelled: list[str] = []
        self.on_fill: FillHandler | None = None
        self.on_status: StatusHandler | None = None

    def set_handlers(self, on_fill: FillHandler, on_status: StatusHandler) -> None:
        self.on_fill, self.on_status = on_fill, on_status

    def submit(self, order: Order) -> None:
        self.submitted.append(order)

    def cancel(self, order_id: str) -> None:
        self.cancelled.append(order_id)

    def positions(self) -> dict[str, int]:
        return {}

    def process_events(self) -> None:
        pass

    def on_bar(self, bar: pd.Series) -> None:
        pass

    def fill(self, order: Order, quantity: int, price: float) -> None:
        assert self.on_fill is not None
        self.on_fill(Fill(order, price, quantity, T0))


def test_partial_fills_then_cancel_protect_the_filled_quantity() -> None:
    broker = ManualBroker()
    router = OrderRouter(broker, Portfolio(), RiskManager())
    sent = router.submit_entry(entry(Side.BUY, 10))
    assert sent is not None
    broker.fill(sent, 4, 100.0)
    broker.fill(sent, 2, 101.0)
    assert len(broker.submitted) == 1  # not protected until the entry is done
    assert broker.on_status is not None
    broker.on_status(sent.order_id, OrderStatus.CANCELLED, "")
    (lot,) = router.lots.values()
    assert (lot.quantity, lot.entry_price) == (6, pytest.approx((400 + 202) / 6))
    (stop,) = broker.submitted[1:]
    assert (stop.order_type, stop.quantity) == (OrderType.STOP, 6)


def test_rejected_entry_leaves_no_lot() -> None:
    broker = ManualBroker()
    router = OrderRouter(broker, Portfolio(), RiskManager())
    sent = router.submit_entry(entry(Side.BUY, 10, holding=5))
    assert sent is not None and broker.on_status is not None
    broker.on_status(sent.order_id, OrderStatus.REJECTED, "no permissions")
    assert router.lots == {} and router.exposure("X") == 0


def test_overfill_race_opens_a_correction() -> None:
    """Live race: the stop fills while the time exit is in flight -> the excess is closed at once."""
    broker = ManualBroker()
    router = OrderRouter(broker, Portfolio(), RiskManager())
    sent = router.submit_entry(entry(Side.BUY, 10, holding=1))
    assert sent is not None
    broker.fill(sent, 10, 100.0)
    stop = broker.submitted[-1]
    router.on_bar_close(T0)  # time exit sent, stop cancel requested
    exit_order = broker.submitted[-1]
    assert exit_order.order_type is OrderType.MARKET and stop.order_id in broker.cancelled
    broker.fill(stop, 10, 98.0)  # the stop filled before the cancel reached the exchange
    assert [t.reason for t in router.trades] == ["stop_loss"]
    broker.fill(exit_order, 10, 97.0)  # ...and the exit too: now short 10
    correction = broker.submitted[-1]
    assert (correction.side, correction.quantity) == (Side.BUY, 10)
    broker.fill(correction, 10, 97.5)
    assert router.portfolio.quantity("X") == 0
    assert router.lots == {}


def test_risk_clips_and_refuses_entries() -> None:
    broker = ManualBroker()
    limits = RiskLimits(max_order_quantity=8, max_position_quantity=10)
    router = OrderRouter(broker, Portfolio(), RiskManager(limits))
    first = router.submit_entry(entry(Side.BUY, 20))
    assert first is not None and first.quantity == 8
    second = router.submit_entry(entry(Side.BUY, 5))  # pending 8 -> room 2
    assert second is not None and second.quantity == 2
    assert router.submit_entry(entry(Side.BUY, 1)) is None
    short = router.submit_entry(entry(Side.SELL, 5))  # reduces exposure: allowed
    assert short is not None and short.quantity == 5


def test_kill_switch_halts_and_flattens_on_daily_loss() -> None:
    router, broker = make_router(RiskLimits(require_stop=False, max_daily_loss_pct=0.01))
    router.submit_entry(entry(Side.BUY, 50, stop=None))
    router.on_bar_close(T0)  # day start equity = 10 000
    broker.on_bar(bar(1, 100, 100, 100, 100))
    router.portfolio.mark({"X": 97.0})  # -150 = -1.5 %
    router.on_bar_close(T0 + pd.Timedelta(minutes=1))
    assert router.risk.halted
    assert router.submit_entry(entry(Side.BUY, 1, stop=None)) is None
    broker.on_bar(bar(2, 97, 97, 97, 97))
    assert [t.reason for t in router.trades] == ["kill switch"]
    assert router.portfolio.quantity("X") == 0


# --- Portfolio and metrics ---------------------------------------------------------------------------------


def test_portfolio_equity_and_average_price() -> None:
    p = Portfolio(1000)
    p.apply_fill("X", 10, 10.0, 1.0)
    p.apply_fill("X", 10, 12.0, 1.0)
    assert p.positions["X"].avg_price == pytest.approx(11.0)
    p.mark({"X": 13.0})
    assert p.equity == pytest.approx(1000 - 220 - 2 + 260)
    p.apply_fill("X", -30, 13.0, 0.0)  # flips short
    assert (p.quantity("X"), p.positions["X"].avg_price) == (-10, 13.0)


def test_metrics_on_a_known_equity_curve() -> None:
    index = pd.DatetimeIndex(
        ["2026-01-05 15:00", "2026-01-05 20:00", "2026-01-06 15:00", "2026-01-06 20:00", "2026-01-07 20:00"], tz="UTC"
    )
    equity = pd.Series([100.0, 110.0, 99.0, 99.0, 108.9], index=index)
    assert list(daily_returns(equity).round(10)) == [0.1, -0.1, 0.1]
    assert max_drawdown(equity) == pytest.approx(-10.0)

    trade = Trade("s", "X", Side.BUY, 1, index[0], index[1], 100, 110, 1.0, 3, "time")
    metrics = compute_metrics(equity, [trade], 100.0)
    returns = np.array([0.1, -0.1, 0.1])
    assert metrics["total_return_pct"] == pytest.approx(8.9)
    assert metrics["sharpe_ratio"] == pytest.approx(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
    assert metrics["sortino_ratio"] == pytest.approx(returns.mean() / math.sqrt(0.01 / 3) * math.sqrt(252))
    assert (metrics["win_rate"], metrics["total_pnl_net"], metrics["profit_factor"]) == (100.0, 9.0, math.inf)


# --- runner ------------------------------------------------------------------------------------------------


class BuyOnce(Strategy):
    def __init__(self) -> None:
        super().__init__(symbol="X", holding_bars=3)
        self.fills: list[Fill] = []

    def decide(self, bar: pd.Series) -> list[Order]:
        return [self.order(Side.BUY, 10)] if bar.name == T0 + pd.Timedelta(minutes=1) else []

    def on_fill(self, fill: Fill) -> None:
        self.fills.append(fill)


def test_runner_trades_next_open_and_reports_fills_to_the_strategy() -> None:
    bars = pd.DataFrame([bar(i, 100 + i, 100.5 + i, 99.5 + i, 100 + i) for i in range(10)])
    strategy = BuyOnce()
    strategy.fit(bars.iloc[:1])
    runner = StrategyRunner([strategy], SimulatedBroker(fee_bps=0, slippage_bps=0), RiskManager(), capital=1000)
    runner.run(HistoricalBarFeed(bars.iloc[1:]))
    (trade,) = runner.router.trades
    # signal on bar 1's close -> bought at bar 2's open (102), held bars 2-4, sold at bar 5's open (105)
    assert (trade.entry_price, trade.exit_price, trade.reason) == (102, 105, "time")
    assert [f.order.side for f in strategy.fills] == [Side.BUY, Side.SELL]
    assert runner.equity.iloc[-1] == pytest.approx(1000 + 30)
    assert len(runner.latencies_ms) == len(bars) - 1


def test_backtest_marks_open_positions_at_the_end() -> None:
    bars = pd.DataFrame([bar(i, 100 + i, 100.5 + i, 99.5 + i, 100 + i) for i in range(4)])
    result = backtest([BuyOnce()], bars.iloc[:1], bars.iloc[1:], fee_bps=0, slippage_bps=0)
    (trade,) = result.trades
    assert (trade.reason, trade.entry_price, trade.exit_price) == ("open", 102, 103)


def test_runner_rejects_duplicate_strategy_ids() -> None:
    with pytest.raises(ValueError):
        StrategyRunner([BuyOnce(), BuyOnce()], SimulatedBroker())
