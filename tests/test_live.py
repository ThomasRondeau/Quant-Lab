"""Simulated live trading: paced feeds, synthetic market, sessions (replayed exactly), background registry, page."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from quantlab.analysis.replay import replay_session
from quantlab.data.loading import load_bars
from quantlab.live.feeds import PacedBarFeed, session_times, synthetic_bars
from quantlab.live.session import SessionLimitError, SimulatedConfig, SimulatedSession, SimulatedSessions
from quantlab.runtime.journal import read_journal
from quantlab.ui.live_view import describe_event, read_session


def wait_until(condition, timeout: float = 60.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met")
        time.sleep(0.05)


def bars(n: int = 20) -> pd.DataFrame:
    return load_bars(["AAPL"], "sample", 60).iloc[:n]


# --- feeds -------------------------------------------------------------------------------------------------


def test_paced_feed_delivers_every_bar_in_order() -> None:
    data = bars(20)
    out = [b for b in PacedBarFeed(data, bars_per_second=0) if b is not None]
    assert [b.name for b in out] == list(data.index)


def test_paced_feed_paces_with_heartbeats_and_stops() -> None:
    feed = PacedBarFeed(bars(20), bars_per_second=20, heartbeat=0.01)
    start = time.monotonic()
    items = list(feed)
    assert time.monotonic() - start >= 0.9  # 19 intervals of 50 ms
    assert sum(i is None for i in items) > 0 and feed.delivered == 20
    feed = PacedBarFeed(bars(20), bars_per_second=5, heartbeat=0.01)
    received = 0
    for item in feed:
        if item is not None:
            received += 1
            if received == 2:
                feed.stop()
    assert received == 2 and feed.stopped
    limited = PacedBarFeed(bars(20), bars_per_second=2, heartbeat=0.01, max_seconds=0.3)
    assert sum(i is not None for i in limited) < 20 and limited.stopped


def test_session_times_follow_new_york_sessions() -> None:
    times = session_times(pd.Timestamp("2026-11-06 20:58", tz="UTC"), 400)  # a Friday
    local = times.tz_convert("America/New_York")
    assert len(times) == 400 and local[0] == pd.Timestamp("2026-11-09 09:30", tz="America/New_York")  # Monday
    assert (local.weekday < 5).all() and local.time.min() >= pd.Timestamp("09:30").time()
    assert local.time.max() < pd.Timestamp("16:00").time()


def test_synthetic_market_continues_the_history() -> None:
    history = load_bars(["SPY", "QQQ"], "sample", 60)
    future = synthetic_bars(history, ["SPY", "QQQ"], 300, seed=3)
    assert len(future) == 300 and future.index[0] > history.index[-1]
    assert set(history.columns) >= set(future.columns) >= {"close_SPY", "close_QQQ", "close"}
    for symbol in ("SPY", "QQQ"):
        o, h, low, c = (future[f"{f}_{symbol}"] for f in ("open", "high", "low", "close"))
        assert (h >= o.combine(c, max)).all() and (low <= o.combine(c, min)).all() and (low > 0).all()
        assert abs(c.iloc[0] / history[f"close_{symbol}"].iloc[-1] - 1) < 0.02
    assert future["close_SPY"].pct_change().corr(future["close_QQQ"].pct_change()) > 0.5
    pd.testing.assert_frame_equal(future, synthetic_bars(history, ["SPY", "QQQ"], 300, seed=3))  # reproducible


# --- sessions ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "symbols", "source"),
    [("mean_reversion", ("AAPL",), "replay"), ("pair", ("SPY", "QQQ"), "synthetic")],
)
def test_session_streams_journals_and_replays_exactly(
    tmp_path: Path, strategy: str, symbols: tuple[str, ...], source: str
) -> None:
    config = SimulatedConfig(strategy, symbols, source=source, live_bars=300, bars_per_second=0, seed=5)  # type: ignore[arg-type]
    session = SimulatedSession(tmp_path / "s", config)
    session.prepare()
    runner = session.run()
    assert session.status == "finished" and session.progress == (300, 300)
    journal = read_journal(session.files.journal_path)
    assert len(journal.bars) == 300 and {"signal", "order", "fill", "equity"} <= set(journal.events["kind"])
    assert session.files.meta()["mode"] == "simulated"
    report = replay_session(session.files.root)
    assert report.summary["signal_agreement_pct"] == 100.0  # the same code on the same bars
    assert report.summary["pnl_net_live"] == pytest.approx(report.summary["pnl_net_replay"])
    assert runner.router.trades


def test_stopping_a_session_keeps_what_happened(tmp_path: Path) -> None:
    session = SimulatedSession(tmp_path / "s", SimulatedConfig("trend", ("AAPL",), live_bars=500, bars_per_second=50))
    session.prepare()
    stopper = __import__("threading").Timer(0.3, session.stop)
    stopper.start()
    session.run()
    delivered, total = session.progress
    assert session.status == "stopped" and 0 < delivered < total
    assert len(read_journal(session.files.journal_path).bars) == delivered


def test_registry_runs_in_the_background_with_limits(tmp_path: Path) -> None:
    registry = SimulatedSessions(tmp_path, max_running=2, per_owner=1)
    slow = SimulatedConfig("trend", ("AAPL",), live_bars=400, bars_per_second=20)
    first = registry.start("a", slow, owner="alice")
    with pytest.raises(SessionLimitError, match="Stop your running session"):
        registry.start("b", slow, owner="alice")
    second = registry.start("c", slow, owner="bob")
    with pytest.raises(SessionLimitError, match="sessions are running"):
        registry.start("d", slow, owner="carol")
    assert [e.name for e in registry.entries("alice")] == ["a"]
    for entry in (first, second):
        entry.stop()
    wait_until(lambda: not first.alive and not second.alive)
    assert first.status == "stopped" and first.path == tmp_path / "alice" / "a" and first.path.exists()
    assert registry.cleanup(max_age_seconds=0) == 2 and not first.path.exists() and registry.entries() == []


def test_registry_reports_errors(tmp_path: Path) -> None:
    registry = SimulatedSessions(tmp_path)
    entry = registry.start("x", SimulatedConfig("trend", ("NOPE",), live_bars=10, bars_per_second=0))
    wait_until(lambda: not entry.alive)
    assert entry.status == "error" and "No bundled sample" in (entry.error or "")


# --- live view ---------------------------------------------------------------------------------------------


def test_live_view_of_a_session(tmp_path: Path) -> None:
    config = SimulatedConfig("pair", ("SPY", "QQQ"), source="synthetic", live_bars=250, bars_per_second=0, seed=1,
                             params={"holding_bars": 200, "exit_zscore": None})  # fmt: skip
    session = SimulatedSession(tmp_path / "s", config)
    session.prepare()
    session.run()
    view = read_session(session.files.root)
    assert view.bars == 250 and view.equity is not None and len(view.equity_curve) == 250
    assert not view.activity.empty and view.activity["bar"].is_monotonic_decreasing
    assert {"signal", "fill"} <= set(view.activity["event"])
    if not view.lots.empty:  # lots still open at the end are marked at the last close
        assert {"symbol", "unrealized P&L", "stop"} <= set(view.lots.columns)
    assert read_session(tmp_path / "missing").bars == 0


def test_event_descriptions() -> None:
    signal = {"side": "BUY", "quantity": 10, "sent_quantity": 0, "symbol": "AAPL", "stop": "2%"}
    assert describe_event("signal", signal) == "BUY 10 → 0 AAPL (refused by the risk checks) · stop 2%"
    fill = {"side": "SELL", "quantity": 5, "symbol": "MSFT", "price": 401.237}
    assert describe_event("fill", fill) == "SELL 5 MSFT filled @ 401.24"
    trade = {"side": "BUY", "quantity": 10, "symbol": "AAPL", "reason": "take_profit", "entry_price": 100.0,
             "exit_price": 105.0, "pnl_net": 49.8}  # fmt: skip
    assert describe_event("trade", trade) == "BUY 10 AAPL closed (take_profit): 100.00 → 105.00, net P&L +49.80"
    order = {"side": "SELL", "quantity": 10, "symbol": "AAPL", "order_type": "STP", "stop_price": 98.0}
    assert describe_event("order", order) == "SELL 10 AAPL STP @ 98.00"


# --- page --------------------------------------------------------------------------------------------------


def test_live_page_runs_a_session_and_replays_it(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import quantlab.ui.app as app

    def script(sessions: str) -> None:
        from pathlib import Path

        import quantlab.ui.app as app

        if Path(sessions) != app.SESSIONS_DIR:
            app.SESSIONS_DIR = Path(sessions)
            app.simulated_sessions.clear()
        app.page_live()

    original = app.SESSIONS_DIR
    try:
        at = AppTest.from_function(script, args=(str(tmp_path),), default_timeout=120).run()
        assert not at.exception and any("No session yet" in i.value for i in at.info)
        at.number_input(key="live_bars").set_value(65).run()
        at.select_slider(key="live_speed").set_value(20.0).run()
        next(b for b in at.button if b.label == "Start the session").click().run()
        assert not at.exception
        registry = app.simulated_sessions(False)
        (entry,) = registry.entries("local")
        wait_until(lambda: not entry.alive)
        at.run()
        assert entry.status == "finished" and {"Equity", "Realized P&L"} <= {m.label for m in at.metric}
        next(b for b in at.button if b.label == "Replay in the backtester").click().run()
        assert any(m.label == "Signal agreement" and m.value == "100.0 %" for m in at.metric)
    finally:
        app.SESSIONS_DIR = original
        app.simulated_sessions.clear()
