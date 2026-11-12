"""Simulated live trading: paced feeds, synthetic market, sessions (replayed exactly), background registry, page."""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import pandas as pd
import pytest

from quantlab.analysis.replay import replay_session
from quantlab.data.loading import load_bars
from quantlab.data.market import MarketDataError
from quantlab.live.feeds import LiveBarFeed, PacedBarFeed, session_times, synthetic_bars
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


# --- real market data (paper trading) ----------------------------------------------------------------------


class Clock:
    """A clock that moves `step` seconds each time it is read: the feed never waits for real minutes."""

    def __init__(self, start: pd.Timestamp, step: float = 7.0) -> None:
        self.time, self.step = start, pd.Timedelta(seconds=step)

    def __call__(self) -> pd.Timestamp:
        self.time += self.step
        return self.time


class Exchange:
    """
    A fake exchange: one candle per minute up to the current minute of `clock` (that one is still moving), with the
    prices of the bundled AAPL sample. `quiet` minutes have no candle, and `down` makes the next requests fail.
    """

    def __init__(self, clock=None) -> None:  # type: ignore[no-untyped-def]
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.sample = load_bars(["AAPL"], "sample", 60)[["open", "high", "low", "close", "volume"]]
        self.quiet: set[pd.Timestamp] = set()
        self.down = 0
        self.asked: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []  # (clock, start, end)

    def fetch(self, product: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        now = self.clock()
        self.asked.append((now, start, end))
        if self.down:
            self.down -= 1
            raise MarketDataError("exchange unreachable")
        times = [t for t in pd.date_range(start, min(end, now.floor("min")), freq="min") if t not in self.quiet]
        rows = [self.sample.iloc[int(t.timestamp() // 60) % len(self.sample)].to_numpy() for t in times]
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name="Datetime"), columns=list(self.sample.columns))


T0 = pd.Timestamp("2026-11-12 12:00", tz="UTC")
MINUTE = pd.Timedelta(minutes=1)


def live_feed(exchange: Exchange, backfill: int = 5, max_bars: int = 4, **options: float) -> LiveBarFeed:
    history = exchange.fetch("SOL-USD", T0 - backfill * MINUTE, T0 - MINUTE)
    exchange.asked.clear()
    return LiveBarFeed(
        lambda start, end: exchange.fetch("SOL-USD", start, end),
        history,
        max_bars,
        heartbeat=0,
        clock=exchange.clock,
        **options,
    )


def test_live_feed_catches_up_then_delivers_each_candle_once_it_is_closed() -> None:
    exchange = Exchange(Clock(T0 + pd.Timedelta(seconds=20)))
    feed = live_feed(exchange)
    assert feed.total == 9 and feed.status == "catching up"

    delivered: list[tuple[pd.Timestamp, pd.Timestamp]] = []  # (bar, clock when it was delivered)
    for bar in feed:
        if bar is not None:
            delivered.append((pd.Timestamp(bar.name), exchange.clock.time))

    times = [t for t, _ in delivered]
    assert times == list(pd.date_range(T0 - 5 * MINUTE, periods=9, freq="min"))  # no hole, no duplicate
    assert feed.delivered == 9 and feed.status == "ended" and feed.last_bar == times[-1] and feed.last_error is None
    # a live candle is delivered once its minute has been over for `delay` (10 s): never the moving one
    assert all(clock >= bar + MINUTE + pd.Timedelta(seconds=10) for bar, clock in delivered[5:])
    assert all(end + MINUTE + pd.Timedelta(seconds=10) <= now for now, _, end in exchange.asked)
    # one request per candle here: the candle is there when it is asked for
    assert len(exchange.asked) == 4


def test_live_feed_skips_quiet_minutes_and_never_asks_the_same_range_twice() -> None:
    exchange = Exchange(Clock(T0 + pd.Timedelta(seconds=20)))
    exchange.quiet = {T0 + MINUTE}
    feed = live_feed(exchange, max_bars=3)
    times = [pd.Timestamp(bar.name) for bar in feed if bar is not None][5:]
    assert times == [T0, T0 + 2 * MINUTE, T0 + 3 * MINUTE]
    retries = [(start, end) for _, start, end in exchange.asked if end == T0 + MINUTE]
    # the exchange caches every URL for minutes: each retry of the quiet minute has its own range
    assert len(retries) > 1 and len(set(retries)) == len(retries)


def test_live_feed_survives_an_outage_and_delivers_what_it_missed() -> None:
    exchange = Exchange(Clock(T0 + pd.Timedelta(seconds=20)))
    feed = live_feed(exchange, max_bars=6, retry=30, max_retry=60)
    seen: list[pd.Timestamp] = []
    errors: list[str | None] = []
    heartbeats = 0
    for bar in feed:
        if bar is None:
            heartbeats += 1
            errors.append(feed.last_error)
        else:
            seen.append(pd.Timestamp(bar.name))
            if len(seen) == 6:  # the first live candle went through: the exchange goes down
                exchange.down = 4
    assert seen[5:] == list(pd.date_range(T0, periods=6, freq="min"))  # nothing lost
    assert any(e and "exchange unreachable" in e for e in errors) and feed.last_error is None and heartbeats > 10
    failed = [now for now, _, _ in exchange.asked][1:5]
    gaps = [(b - a).total_seconds() for a, b in itertools.pairwise(failed)]
    assert gaps == sorted(gaps) and gaps[0] >= 30 and gaps[-1] >= 60  # retries further and further apart


def test_live_feed_stops() -> None:
    exchange = Exchange(Clock(T0 + pd.Timedelta(seconds=20)))
    feed = live_feed(exchange, max_bars=1000)
    count = 0
    for bar in feed:
        count += bar is not None
        if count == 7:
            feed.stop()
    assert feed.stopped and feed.delivered == 7 < feed.total


def test_market_session_trades_real_candles_on_paper_and_replays_exactly(tmp_path: Path) -> None:
    exchange = Exchange(Clock(T0 + pd.Timedelta(seconds=20), step=20))
    candles = exchange.fetch("SOL-USD", T0 - 299 * MINUTE, T0 - MINUTE)
    size = {"size_mode": "notional", "size_value": 20_000.0}
    config = SimulatedConfig(
        "trend",
        ("SOL-USD",),
        params=size,
        extra_strategies={"mean_reversion": size},
        source="market",
        live_bars=60,
        history_bars=179,
        backfill_bars=120,
    )
    assert list(config.strategies) == ["trend", "mean_reversion"]
    session = SimulatedSession(tmp_path / "market", config)
    session.prepare(bars=candles, fetch=exchange.fetch)
    assert isinstance(session.feed, LiveBarFeed) and session.progress == (0, 180)
    session.feed.clock, session.feed.heartbeat = exchange.clock, 0

    runner = session.run()

    assert session.status == "finished" and session.progress == (180, 180)
    journal = read_journal(session.files.journal_path)
    assert len(journal.bars) == 180 and journal.bars.index[-1] == T0 + 59 * MINUTE
    assert session.files.meta()["source"] == "market" and len(session.files.load_warmup()) == 179
    assert {"TrendFollowingStrategy", "MeanReversionStrategy"} == set(session.files.meta()["strategies"])
    assert runner.router.trades and all(t.symbol == "SOL-USD" for t in runner.router.trades)
    assert any(t.quantity > 50 for t in runner.router.trades)  # sized by notional, not 10 shares
    report = replay_session(session.files.root)
    assert report.summary["signal_agreement_pct"] == 100.0 and report.summary["signals_live"] > 0
    assert report.summary["pnl_net_live"] == pytest.approx(report.summary["pnl_net_replay"])


def test_market_session_needs_one_product_and_a_history(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one product"):
        SimulatedConfig("pair", ("SOL-USD", "ETH-USD"), source="market")
    exchange = Exchange(Clock(T0))
    session = SimulatedSession(tmp_path / "m", SimulatedConfig("trend", ("SOL-USD",), source="market"))
    with pytest.raises(ValueError, match="not enough for a history"):
        session.prepare(bars=exchange.fetch("SOL-USD", T0 - 99 * MINUTE, T0 - MINUTE), fetch=exchange.fetch)
    with pytest.raises(MarketDataError, match="product name"):
        SimulatedSession(tmp_path / "n", SimulatedConfig("trend", ("AAPL",), source="market")).prepare()


def test_registry_keeps_a_session_running(tmp_path: Path) -> None:
    registry = SimulatedSessions(tmp_path, max_running=1, per_owner=1)
    quick = SimulatedConfig("trend", ("AAPL",), live_bars=30, bars_per_second=0)
    assert registry.keep_running("shared", quick, owner="market", keep=2)
    assert not registry.keep_running("shared", quick, owner="market") and registry.kept("market")
    # a session that ended is replaced by a new one, and only the last folders are kept
    names: set[str] = set()

    def several_sessions() -> bool:
        names.update(e.name for e in registry.entries("market"))
        return len(names) >= 4

    wait_until(several_sessions)
    assert len(registry.entries("market")) <= 2 and len(list((tmp_path / "market").iterdir())) <= 2
    # the kept sessions are outside the limits of the visitors
    visitor = registry.start("mine", quick, "alice")
    registry.stop_kept("market")
    wait_until(lambda: not registry.kept("market") and not visitor.alive)
    assert all(not e.alive for e in registry.entries("market")) and visitor.status == "finished"


def test_registry_retries_a_kept_session_that_fails(tmp_path: Path) -> None:
    registry = SimulatedSessions(tmp_path)
    broken = SimulatedConfig("trend", ("NOPE",), live_bars=10, bars_per_second=0)
    registry.keep_running("shared", broken, owner="market", retry_seconds=0.05, keep=1)
    seen: set[str] = set()

    def several_attempts() -> bool:
        seen.update(e.name for e in registry.entries("market") if e.status == "error")
        return len(seen) >= 3

    wait_until(several_attempts)
    registry.stop_kept("market")
    wait_until(lambda: not registry.kept("market"))
    assert len(registry.entries("market")) == 1  # the failed attempts do not pile up


def market_page(monkeypatch: pytest.MonkeyPatch, sessions: Path, fresh: bool = True) -> tuple[Exchange, object]:
    """The live page on a fake exchange that follows the real clock: (exchange, AppTest)."""
    from streamlit.testing.v1 import AppTest

    import quantlab.ui.app as app
    from quantlab.data import market

    def script() -> None:
        import quantlab.ui.app as app

        app.page_live()

    exchange = Exchange()
    if fresh:
        monkeypatch.setattr(market, "fetch_candles", exchange.fetch)
        monkeypatch.setattr(app, "SESSIONS_DIR", sessions)
        monkeypatch.setattr(app, "MARKET_BARS", {"history_bars": 200, "backfill_bars": 90, "live_bars": 5})
        app.simulated_sessions.clear()
    return exchange, AppTest.from_function(script, default_timeout=120)


def test_market_tab_shows_the_session_shared_by_every_visitor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    import quantlab.ui.app as app

    monkeypatch.setenv("QUANTLAB_MARKET", "1")
    _, page = market_page(monkeypatch, tmp_path)
    at = page.run()  # type: ignore[attr-defined]
    registry = app.simulated_sessions(False)
    try:
        assert not at.exception and registry.kept("market")
        wait_until(lambda: any(read_session(e.path).bars >= 90 for e in registry.entries("market")))
        (entry,) = registry.entries("market")
        at.run()
        assert not at.exception and {"Equity", "Realized P&L", "Last bar"} <= {m.label for m in at.metric}
        assert any("SOL-USD" in m.value and "Trend following + Mean reversion" in m.value for m in at.markdown)
        assert any("90 candles of the hours before" in c.value for c in at.caption)
        # caught up, it waits for the next real candle; the visitors cannot stop it
        delivered, total = entry.session.progress
        assert entry.alive and entry.session.feed.status == "live" and 90 <= delivered < total == 95
        assert not [b for b in at.button if "market session" in b.label]
        # the replay runs on the session as it is so far
        next(b for b in at.button if b.label == "Replay in the backtester").click().run()
        assert any(m.label == "Signal agreement" and m.value == "100.0 %" for m in at.metric)
        # another visitor watches the same session: no second one is started
        _, other_page = market_page(monkeypatch, tmp_path, fresh=False)
        other = other_page.run()  # type: ignore[attr-defined]
        assert not other.exception and "Equity" in {m.label for m in other.metric}
        assert registry.entries("market") == [entry]
    finally:
        registry.stop_all()
        wait_until(lambda: not registry.kept("market"))
        app.simulated_sessions.clear()


def test_market_tab_reads_the_exchange_only_once_started_locally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("streamlit")
    import quantlab.ui.app as app

    monkeypatch.delenv("QUANTLAB_MARKET", raising=False)
    exchange, page = market_page(monkeypatch, tmp_path)
    at = page.run()  # type: ignore[attr-defined]
    registry = app.simulated_sessions(False)
    try:
        assert not at.exception and not exchange.asked and not registry.kept("market")
        at.text_input(key="market_product").set_value("not a product").run()
        next(b for b in at.button if b.label == "Start the market session").click().run()
        assert any("product name" in w.value for w in at.warning) and not registry.kept("market")
        at.text_input(key="market_product").set_value("ltc-usd").run()
        next(b for b in at.button if b.label == "Start the market session").click().run()
        assert not at.exception and registry.kept("market")
        wait_until(lambda: any(read_session(e.path).bars >= 90 for e in registry.entries("market")))
        at.run()
        assert any("LTC-USD" in m.value for m in at.markdown) and "Equity" in {m.label for m in at.metric}
        next(b for b in at.button if b.label == "Stop the market session").click().run()
        wait_until(lambda: not registry.kept("market"))
        assert all(not e.alive for e in registry.entries("market"))
    finally:
        registry.stop_all()
        app.simulated_sessions.clear()
