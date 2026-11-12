"""
Real-time bar feeds for simulated live trading.

`PacedBarFeed` delivers bars one by one at a chosen pace (e.g. 2 bars per second: a 2-minute bar every half
second, 240x real time), with heartbeats in between, exactly like a live feed would. The bars come from a
replay of recorded data or from `synthetic_bars`, a random market generated on the fly.

`LiveBarFeed` delivers the candles of a real market as they close (`data/market.py`), after the candles of the
last hours.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from quantlab.data.market import INTERVAL, MAX_CANDLES, MarketDataError, last_closed
from quantlab.data.pairs import combine_symbols

logger = logging.getLogger(__name__)

#: New York regular session
SESSION_OPEN, SESSION_CLOSE = pd.Timedelta(hours=9, minutes=30), pd.Timedelta(hours=16)


class PacedBarFeed:
    """
    A `BarFeed` that yields `bars` at `bars_per_second` (0: as fast as possible), with a heartbeat (None) at
    least every `heartbeat` seconds while waiting. `stop()` is thread-safe; `max_seconds` bounds the run.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        bars_per_second: float = 2.0,
        heartbeat: float = 0.25,
        max_seconds: float | None = None,
    ) -> None:
        if bars_per_second < 0:
            raise ValueError(f"bars_per_second must be >= 0 (got {bars_per_second}).")
        self.bars = bars
        self.bars_per_second = bars_per_second
        self.heartbeat = heartbeat
        self.max_seconds = max_seconds
        self.delivered = 0
        self._stop = threading.Event()

    @property
    def total(self) -> int:
        return len(self.bars)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def __iter__(self) -> Iterator[pd.Series | None]:
        interval = 1.0 / self.bars_per_second if self.bars_per_second else 0.0
        start = time.monotonic()
        next_at = start
        for _, bar in self.bars.iterrows():
            while not self._stop.is_set():
                now = time.monotonic()
                if self.max_seconds is not None and now - start >= self.max_seconds:
                    self._stop.set()
                    break
                if now >= next_at:
                    break
                self._stop.wait(min(self.heartbeat, next_at - now))
                yield None
            if self._stop.is_set():
                return
            self.delivered += 1
            next_at = max(next_at + interval, time.monotonic()) if interval else 0.0
            yield bar


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class LiveBarFeed:
    """
    A `BarFeed` on real market data. It first yields the `backfill` bars at once (the candles of the last hours,
    so the session starts with a history), then every 1-minute candle as it closes, read with `fetch(start, end)`.

    - A candle is asked for `delay` seconds after its minute ended, then every `poll` seconds until it is there.
      In the first seconds after the close the exchange may still answer with an unfinished candle, hence the
      delay. A minute without a trade has no candle: the next one is delivered when it comes.
    - A failed request is retried after `retry` seconds, doubled at each failure up to `max_retry`; `last_error`
      holds its message until a request succeeds.
    - After an interruption, the candles missed are delivered at once (the last 290 at most).
    - Heartbeats (None) are yielded at least every `heartbeat` seconds. The feed ends after `max_bars` live bars
      or on `stop()`, which is thread-safe.
    """

    #: candles asked for at once after an interruption; the rest of the 300 allowed keeps the retries apart
    CATCH_UP = MAX_CANDLES - 10

    def __init__(
        self,
        fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
        backfill: pd.DataFrame,
        max_bars: int,
        delay: float = 10.0,
        poll: float = 5.0,
        heartbeat: float = 0.25,
        retry: float = 5.0,
        max_retry: float = 60.0,
        clock: Callable[[], pd.Timestamp] = _now,
    ) -> None:
        if max_bars < 1:
            raise ValueError(f"max_bars must be >= 1 (got {max_bars}).")
        self.fetch = fetch
        self.backfill = backfill
        self.max_bars = max_bars
        self.delay = pd.Timedelta(seconds=delay)
        self.poll = pd.Timedelta(seconds=poll)
        self.heartbeat = heartbeat
        self.retry, self.max_retry = retry, max_retry
        self.clock = clock
        self.delivered = 0
        self.status = "catching up"
        self.last_bar: pd.Timestamp | None = None
        self.last_error: str | None = None
        self._stop = threading.Event()

    @property
    def total(self) -> int:
        return len(self.backfill) + self.max_bars

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def _deliver(self, bar: pd.Series) -> pd.Series:
        self.delivered += 1
        self.last_bar = pd.Timestamp(bar.name)  # type: ignore[arg-type]
        return bar

    def __iter__(self) -> Iterator[pd.Series | None]:
        for _, bar in self.backfill.iterrows():
            if self._stop.is_set():
                return
            yield self._deliver(bar)
        self.status = "live"
        live, attempts, retry = 0, 0, self.retry
        asked: pd.Timestamp | None = None
        next_request: pd.Timestamp | None = None
        while live < self.max_bars and not self._stop.is_set():
            now = self.clock()
            due = last_closed(now - self.delay)  # the newest candle that has been closed for `delay`
            waiting = self.last_bar is None or due > self.last_bar
            if waiting and (next_request is None or now >= next_request):
                if due != asked:
                    asked, attempts = due, 0
                first = due - INTERVAL * (self.CATCH_UP - 1)
                if self.last_bar is not None:
                    first = max(first, self.last_bar + INTERVAL)
                try:
                    # every attempt has its own range, hence its own URL: the exchange caches each URL for minutes
                    candles = self.fetch(first - INTERVAL * (attempts % 10), due)
                except MarketDataError as exc:
                    self.last_error = str(exc)
                    logger.warning("Market data unavailable, next try in %.0f s: %s", retry, exc)
                    next_request = now + pd.Timedelta(seconds=retry)
                    retry = min(retry * 2, self.max_retry)
                else:
                    self.last_error, retry = None, self.retry
                    attempts += 1
                    next_request = now + self.poll
                    if len(candles):
                        candles = candles[(candles.index >= first) & (candles.index <= due)]
                    for _, bar in candles.iterrows():
                        if live >= self.max_bars or self._stop.is_set():
                            break
                        live += 1
                        yield self._deliver(bar)
                    continue
            self._stop.wait(self.heartbeat)
            yield None
        self.status = "ended"


def session_times(start: pd.Timestamp, n_bars: int, bar_minutes: int = 2) -> pd.DatetimeIndex:
    """`n_bars` bar timestamps (UTC) in New York regular sessions, weekdays only, from the session after `start`."""
    step = pd.Timedelta(minutes=bar_minutes)
    per_session = int((SESSION_CLOSE - SESSION_OPEN) / step)
    day = start.tz_convert("America/New_York").normalize() + pd.Timedelta(days=1)
    times: list[pd.Timestamp] = []
    while len(times) < n_bars:
        if day.weekday() < 5:
            opening = day + SESSION_OPEN
            times.extend(opening + step * i for i in range(min(per_session, n_bars - len(times))))
        day += pd.Timedelta(days=1)
    return pd.DatetimeIndex(times).tz_convert("UTC").rename("Datetime")


def synthetic_bars(
    history: pd.DataFrame,
    symbols: Sequence[str],
    n_bars: int,
    seed: int | None = None,
    correlation: float = 0.8,
    bar_minutes: int = 2,
) -> pd.DataFrame:
    """
    A random market continuing `history` (same layout, one or several symbols): correlated log-normal returns
    with each symbol's realized volatility, an intraday volatility smile, and volumes around the history's
    median. Prices start from the last close of `history`. Not a forecast: a plausible stream to trade on.
    """
    rng = np.random.default_rng(seed)
    times = session_times(history.index[-1], n_bars, bar_minutes)
    n = len(symbols)
    cov = np.full((n, n), correlation) + np.eye(n) * (1 - correlation)
    shocks = rng.multivariate_normal(np.zeros(n), cov, size=n_bars)
    local = times.tz_convert("America/New_York")
    hours = (local - local.normalize()).total_seconds().to_numpy() / 3600
    phase = (hours - 9.5) / 6.5  # 0 at the open, 1 at the close
    smile = 0.7 + 2.4 * (phase - 0.5) ** 2  # higher volatility near the open (1.3x) and the close
    frames: dict[str, pd.DataFrame] = {}
    for k, symbol in enumerate(symbols):
        closes = _series(history, symbol, "close", n > 1)
        volumes = _series(history, symbol, "volume", n > 1)
        sigma = float(np.log(closes).diff().std()) or 1e-3
        returns = shocks[:, k] * sigma * smile
        close = float(closes.iloc[-1]) * np.exp(np.cumsum(returns))
        open_ = np.concatenate([[float(closes.iloc[-1])], close[:-1]])
        wick = np.abs(rng.normal(0, sigma / 2, size=(2, n_bars))) * close
        high = np.maximum(open_, close) + wick[0]
        low = np.minimum(open_, close) - wick[1]
        volume = float(volumes.median()) * rng.lognormal(0, 0.4, n_bars) * smile
        frames[symbol] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=times
        )
    return combine_symbols(frames)


def _series(history: pd.DataFrame, symbol: str, field: str, multi: bool) -> pd.Series:
    column = f"{field}_{symbol}" if multi else field
    if column not in history.columns:
        raise ValueError(f"The history has no {column!r} column.")
    series: pd.Series = history[column].astype(float)
    return series
