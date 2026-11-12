"""
Simulated live trading: a strategy trades a real-time stream of bars, with orders, fills, positions, a journal
and a persisted state, exactly as it would live, but on replayed or synthetic data and a simulated broker.

A session is the live stack of the platform, wired to a `PacedBarFeed`:

    load bars -> history / live split -> strategy.fit(history) -> snapshot (strategies.pkl, session.json)
    -> StrategyRunner(journal, state store).run(PacedBarFeed(live bars))

With the `market` source, the bars are the real 1-minute candles of a crypto exchange (`data/market.py`): the
strategies are fitted on the oldest ones, and a `LiveBarFeed` delivers the last hours at once, then every candle as
it closes. The broker stays simulated: this is paper trading on real market data, and no order is sent anywhere.

Its folder has the same files as any session (`runtime/session.py`), so `analysis.replay.replay_session`
compares it with the backtester. `SimulatedSessions` runs sessions in background threads for the web interface.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from quantlab.data import loading, market
from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.execution.simulated import SimulatedBroker
from quantlab.live.feeds import LiveBarFeed, PacedBarFeed, synthetic_bars
from quantlab.runtime.journal import Journal
from quantlab.runtime.runner import StrategyRunner
from quantlab.runtime.session import SessionFiles
from quantlab.runtime.state import StateStore
from quantlab.strategies.base import Strategy
from quantlab.strategies.registry import build_strategy

logger = logging.getLogger(__name__)

Source = Literal["replay", "synthetic", "market"]
SOURCES: tuple[Source, ...] = ("replay", "synthetic", "market")


@dataclass(frozen=True)
class SimulatedConfig:
    strategy: str
    symbols: tuple[str, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    #: other strategies trading in the same account (name -> parameters)
    extra_strategies: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: replay: the last `live_bars` bars of the data are streamed; synthetic: new bars continue the data;
    #: market: the real candles of the product `symbols[0]`, at most `live_bars` of them after the backfill
    source: Source = "replay"
    live_bars: int = 975  # five sessions of 2-minute bars
    #: market: candles the strategies are fitted on, then candles of the last hours delivered at once
    history_bars: int = 720
    backfill_bars: int = 360
    bars_per_second: float = 2.0
    #: wall-clock limit of the streaming (None: until the bars run out or stop())
    max_seconds: float | None = None
    seed: int | None = None
    capital: float = 100_000.0
    risk: RiskLimits = field(default_factory=RiskLimits)
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    #: where the bars come from (`data.loading.load_bars`)
    data_source: str = "sample"
    days: int = 60

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"Unknown source {self.source!r}; available: {', '.join(SOURCES)}.")
        if self.live_bars < 1:
            raise ValueError(f"live_bars must be >= 1 (got {self.live_bars}).")
        if self.source == "market":
            if len(self.symbols) != 1:
                raise ValueError(f"The market source trades one product (got {list(self.symbols)}).")
            if self.history_bars < 1 or self.backfill_bars < 0:
                raise ValueError("history_bars must be >= 1 and backfill_bars >= 0.")

    @property
    def strategies(self) -> dict[str, Mapping[str, Any]]:
        """Every strategy of the session, with its parameters."""
        return {self.strategy: self.params, **self.extra_strategies}


def split_live(config: SimulatedConfig, bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(history to fit on, bars to stream)."""
    if config.source == "synthetic":
        live = synthetic_bars(bars, config.symbols, config.live_bars, seed=config.seed)
        return bars[list(live.columns)], live  # same columns on both sides
    if len(bars) <= config.live_bars:
        raise ValueError(f"{len(bars)} bars: not enough for {config.live_bars} live bars plus a training history.")
    return bars.iloc[: -config.live_bars], bars.iloc[-config.live_bars :]


class SimulatedSession:
    """One simulated live session in `root`. `prepare()` then `run()` (blocking); `stop()` from any thread."""

    def __init__(self, root: str | Path, config: SimulatedConfig) -> None:
        self.files = SessionFiles(root)
        self.config = config
        self.status = "created"
        self.runner: StrategyRunner | None = None
        self.feed: PacedBarFeed | LiveBarFeed | None = None
        self.journal: Journal | None = None
        self._stop = threading.Event()

    @property
    def progress(self) -> tuple[int, int]:
        """(bars streamed, bars to stream)."""
        return (self.feed.delivered, self.feed.total) if self.feed is not None else (0, self.config.live_bars)

    def prepare(
        self,
        bars: pd.DataFrame | None = None,
        strategies: Sequence[Strategy] | None = None,
        fetch: market.Fetch | None = None,
    ) -> None:
        """
        Load the bars, fit the strategies on the history and write the session snapshot.
        :param fetch: market source: how the candles are read (default: from the exchange).
        """
        config = self.config
        if self.files.exists():
            raise FileExistsError(f"Session {self.files.root} already exists.")
        self.status = "loading data"
        feed: PacedBarFeed | LiveBarFeed
        if config.source == "market":
            read = fetch or market.fetch_candles
            product = market.check_product(config.symbols[0])
            if bars is None:
                bars = market.history(product, config.history_bars + config.backfill_bars, fetch=read)
            cut = len(bars) - config.backfill_bars
            if cut < 1:
                raise ValueError(f"{len(bars)} candles: not enough for a history before the backfill.")
            history, live = bars.iloc[:cut], bars.iloc[cut:]
            feed = LiveBarFeed(lambda start, end: read(product, start, end), live, max_bars=config.live_bars)
        else:
            if bars is None:
                bars = loading.load_bars(list(config.symbols), config.data_source, config.days)
            history, live = split_live(config, bars)
            feed = PacedBarFeed(live, config.bars_per_second, max_seconds=config.max_seconds)
        if strategies is None:
            strategies = [build_strategy(name, config.symbols, params) for name, params in config.strategies.items()]
        self.status = "training"
        for strategy in strategies:
            strategy.fit(history)
        meta = {
            "mode": "simulated",
            "source": config.source,
            "symbols": list(config.symbols),
            "bars_per_second": config.bars_per_second,
            "live_bars": feed.total,
            "capital": config.capital,
            "risk": dataclasses.asdict(config.risk),
            "fee_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
            "strategies": [s.strategy_id for s in strategies],
            "params": {s.strategy_id: _jsonable(getattr(s, "params", {})) for s in strategies},
        }
        self.files.save_snapshot(strategies, history, meta)
        self.journal = Journal(self.files.journal_path)
        broker = SimulatedBroker(fee_bps=config.fee_bps, slippage_bps=config.slippage_bps)
        self.runner = StrategyRunner(
            strategies,
            broker,
            RiskManager(config.risk),
            config.capital,
            journal=self.journal,
            state_store=StateStore(self.files.state_path),
        )
        self.feed = feed
        if self._stop.is_set():
            self.feed.stop()
        self.journal.event("session", {"action": "start", "mode": "simulated", "warmup_bars": len(history)})

    def run(self) -> StrategyRunner:
        if self.runner is None or self.feed is None or self.journal is None:
            raise RuntimeError("prepare() the session before running it.")
        self.status = "running"
        try:
            self.runner.run(self.feed)
        finally:
            self.journal.event("session", {"action": "stop", "equity": self.runner.portfolio.equity})
            self.journal.close()
        delivered, total = self.progress
        self.status = "finished" if delivered >= total else "stopped"
        logger.info("Session %s %s after %d / %d bars", self.files.root.name, self.status, delivered, total)
        return self.runner

    def stop(self) -> None:
        self._stop.set()
        if self.feed is not None:
            self.feed.stop()


def _jsonable(params: Mapping[str, Any]) -> dict[str, Any]:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}


# --- background sessions (web interface) -----------------------------------------------------------------


@dataclass
class SessionEntry:
    name: str
    owner: str
    session: SimulatedSession
    started: float = field(default_factory=time.time)
    error: str | None = None
    thread: threading.Thread | None = None

    @property
    def path(self) -> Path:
        return self.session.files.root

    @property
    def status(self) -> str:
        return "error" if self.error else self.session.status

    @property
    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def stop(self) -> None:
        self.session.stop()


class SessionLimitError(RuntimeError):
    """Too many sessions running."""


class SimulatedSessions:
    """
    Sessions running in background threads, for as long as the process lives (the web interface reruns its
    script on every interaction). `max_running` bounds them across all owners, `per_owner` per visitor.
    `prepare_guard()` wraps the data loading and training (a shared compute slot in the hosted demo).
    """

    def __init__(
        self,
        root: str | Path,
        max_running: int | None = None,
        per_owner: int | None = None,
        prepare_guard: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> None:
        self.root = Path(root)
        self.max_running = max_running
        self.per_owner = per_owner
        self.prepare_guard = prepare_guard
        self._lock = threading.Lock()
        self._entries: dict[str, SessionEntry] = {}
        #: owners whose session is kept running (`keep_running`), with the flag that ends it
        self._kept: dict[str, threading.Event] = {}

    def entries(self, owner: str | None = None) -> list[SessionEntry]:
        with self._lock:
            return [e for e in self._entries.values() if owner is None or e.owner == owner]

    def kept(self, owner: str) -> bool:
        """True while `keep_running` runs sessions for `owner`."""
        with self._lock:
            return owner in self._kept

    def keep_running(
        self, prefix: str, config: SimulatedConfig, owner: str, retry_seconds: float = 60.0, keep: int = 2
    ) -> bool:
        """
        Run sessions of `config` for `owner` one after the other, for as long as the process lives: a new one
        starts when the previous one ends (its `live_bars` are reached), or `retry_seconds` after a failure. Only
        the folders of the last `keep` sessions are kept. These sessions are outside the limits of `start`.
        Returns False when `owner` is kept running already.
        """
        with self._lock:
            if owner in self._kept:
                return False
            stop = self._kept[owner] = threading.Event()
        args = (prefix, config, owner, stop, retry_seconds, keep)
        threading.Thread(target=self._keep, args=args, name=f"sessions-{owner}", daemon=True).start()
        return True

    def stop_all(self) -> None:
        """Stop every session, kept or not (the code is about to be reloaded, or the process to end)."""
        with self._lock:
            owners = list(self._kept)
            current = list(self._entries.values())
        for owner in owners:
            self.stop_kept(owner)
        for entry in current:
            entry.stop()

    def stop_kept(self, owner: str) -> None:
        """End `keep_running` for `owner` and stop its current session."""
        with self._lock:
            stop = self._kept.get(owner)
            current = [e for e in self._entries.values() if e.owner == owner]
        if stop is not None:
            stop.set()
        for entry in current:
            entry.stop()

    def _keep(
        self, prefix: str, config: SimulatedConfig, owner: str, stop: threading.Event, retry_seconds: float, keep: int
    ) -> None:
        try:
            while not stop.is_set():
                name = f"{prefix}_{pd.Timestamp.now(tz='UTC'):%Y%m%d_%H%M%S_%f}"
                entry = SessionEntry(name, owner, SimulatedSession(self.root / owner / name, config))
                with self._lock:
                    self._entries[f"{owner}/{name}"] = entry
                    old = sorted((e for e in self._entries.values() if e.owner == owner), key=lambda e: e.started)
                    old = old[:-keep] if keep else old
                    for previous in old:
                        del self._entries[f"{owner}/{previous.name}"]
                for previous in old:
                    shutil.rmtree(previous.path, ignore_errors=True)
                entry.thread = threading.Thread(target=self._run, args=(entry,), name=f"session-{name}", daemon=True)
                entry.thread.start()
                if stop.is_set():  # stop_kept ran while the session was being created
                    entry.stop()
                entry.thread.join()
                if entry.error or entry.status != "finished":
                    stop.wait(retry_seconds)  # a failure, or a session cut short: not in a tight loop
        finally:
            with self._lock:
                self._kept.pop(owner, None)

    def start(self, name: str, config: SimulatedConfig, owner: str = "local") -> SessionEntry:
        with self._lock:
            running = [e for e in self._entries.values() if e.alive and e.owner not in self._kept]
            if self.max_running is not None and len(running) >= self.max_running:
                raise SessionLimitError(f"{len(running)} sessions are running already; try again in a few minutes.")
            if self.per_owner is not None and sum(e.owner == owner for e in running) >= self.per_owner:
                raise SessionLimitError("Stop your running session before starting another one.")
            key = f"{owner}/{name}"
            if key in self._entries and self._entries[key].alive:
                raise SessionLimitError(f"Session {name!r} is already running.")
            path = self.root / owner / name if owner != "local" else self.root / name
            entry = SessionEntry(name, owner, SimulatedSession(path, config))
            self._entries[key] = entry
        entry.thread = threading.Thread(target=self._run, args=(entry,), name=f"session-{name}", daemon=True)
        entry.thread.start()
        return entry

    def _run(self, entry: SessionEntry) -> None:
        try:
            with self.prepare_guard():
                entry.session.prepare()
            entry.session.run()
        except Exception as exc:
            logger.exception("Simulated session %s failed", entry.name)
            entry.error = f"{type(exc).__name__}: {exc}"

    def cleanup(self, max_age_seconds: float) -> int:
        """Forget the sessions that ended more than `max_age_seconds` ago and delete their folders."""
        now = time.time()
        with self._lock:
            old = [k for k, e in self._entries.items() if not e.alive and now - e.started > max_age_seconds]
            removed = [self._entries.pop(k) for k in old]
        for entry in removed:
            shutil.rmtree(entry.path, ignore_errors=True)
        return len(removed)
