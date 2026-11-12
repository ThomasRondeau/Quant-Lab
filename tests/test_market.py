"""Public market data client (Coinbase candles): parsing, request shape, errors, pagination. No network."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from quantlab.data import market
from quantlab.data.market import MarketDataError, fetch_candles, history, last_closed

T0 = pd.Timestamp("2026-11-12 12:00", tz="UTC")


def candle(start: pd.Timestamp, close: float = 82.0) -> list[float]:
    """A candle as the exchange sends it: time, low, high, open, close, volume."""
    return [int(start.timestamp()), close - 0.2, close + 0.3, close - 0.1, close, 150.0]


class Exchange:
    """A fake `urlopen`: answers with the candles of the asked range, newest first, and records the URLs."""

    def __init__(self, missing: frozenset[pd.Timestamp] = frozenset(), payload: Any = None) -> None:
        self.urls: list[str] = []
        self.missing = missing
        self.payload = payload

    def __call__(self, request: Any, timeout: float) -> io.BytesIO:
        self.urls.append(request.full_url)
        assert request.get_header("User-agent") == market.USER_AGENT and timeout == market.TIMEOUT_SECONDS
        if self.payload is not None:
            return io.BytesIO(json.dumps(self.payload).encode())
        query = parse_qs(urlparse(request.full_url).query)
        start, end = pd.Timestamp(query["start"][0]), pd.Timestamp(query["end"][0])
        times = [t for t in pd.date_range(start.ceil("min"), end, freq="min") if t not in self.missing]
        return io.BytesIO(json.dumps([candle(t, 80 + t.minute / 100) for t in reversed(times)]).encode())


def test_candles_are_read_oldest_first_in_the_ohlcv_contract() -> None:
    exchange = Exchange()
    bars = fetch_candles("sol-usd", T0, T0 + pd.Timedelta(minutes=4), opener=exchange)

    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert list(bars.index) == list(pd.date_range(T0, periods=5, freq="min")) and str(bars.index.tz) == "UTC"
    first = bars.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == pytest.approx(
        (79.9, 80.3, 79.8, 80.0, 150.0)
    )
    url = urlparse(exchange.urls[0])
    assert url.netloc == "api.exchange.coinbase.com" and url.path == "/products/SOL-USD/candles"
    # its own start / end: the CDN caches each URL for minutes, a poll must not read a stale copy
    assert parse_qs(url.query) == {
        "granularity": ["60"],
        "start": ["2026-11-12T12:00:00Z"],
        "end": ["2026-11-12T12:04:00Z"],
    }


def test_candles_outside_the_range_are_dropped_and_no_candle_is_an_empty_frame() -> None:
    early = [candle(T0 + pd.Timedelta(minutes=1)), candle(T0), candle(T0 - pd.Timedelta(minutes=1))]
    bars = fetch_candles("SOL-USD", T0, T0 + pd.Timedelta(minutes=1), opener=Exchange(payload=early))
    assert list(bars.index) == [T0, T0 + pd.Timedelta(minutes=1)]
    empty = fetch_candles("SOL-USD", T0, T0 + pd.Timedelta(minutes=1), opener=Exchange(payload=[]))
    assert empty.empty and list(empty.columns) == ["open", "high", "low", "close", "volume"]


def test_errors_are_market_data_errors() -> None:
    def offline(request: Any, timeout: float) -> Any:
        raise urllib.error.URLError("no route to host")

    end = T0 + pd.Timedelta(minutes=1)
    with pytest.raises(MarketDataError, match="no route to host"):
        fetch_candles("SOL-USD", T0, end, opener=offline)
    with pytest.raises(MarketDataError, match="NotFound"):
        fetch_candles("NOPE-USD", T0, end, opener=Exchange(payload={"message": "NotFound"}))
    with pytest.raises(MarketDataError, match="Unreadable"):
        fetch_candles("SOL-USD", T0, end, opener=Exchange(payload=[[1, 2, 3]]))
    with pytest.raises(MarketDataError, match="product name"):
        fetch_candles("SOL-USD/../accounts", T0, end, opener=Exchange())
    with pytest.raises(ValueError, match="At most 300"):
        fetch_candles("SOL-USD", T0, T0 + pd.Timedelta(minutes=300), opener=Exchange())


def test_history_reads_windows_of_300_closed_candles() -> None:
    exchange = Exchange()
    now = T0 + pd.Timedelta(seconds=42)  # the 12:00 candle is still moving
    assert last_closed(now) == T0 - pd.Timedelta(minutes=1)

    bars = history("SOL-USD", 700, end=now, fetch=lambda p, s, e: fetch_candles(p, s, e, opener=exchange))

    assert len(bars) == 700 and bars.index[-1] == T0 - pd.Timedelta(minutes=1)
    assert bars.index.is_monotonic_increasing and bars.index.is_unique
    assert (bars.index[1:] - bars.index[:-1] == pd.Timedelta(minutes=1)).all()
    assert len(exchange.urls) == 3  # three windows of 300 candles, the oldest 200 dropped
    spans = [parse_qs(urlparse(u).query) for u in exchange.urls]
    assert all(pd.Timestamp(q["end"][0]) - pd.Timestamp(q["start"][0]) == pd.Timedelta(minutes=299) for q in spans)


def test_history_reads_further_back_over_minutes_without_trades() -> None:
    quiet = frozenset(pd.date_range(T0 - pd.Timedelta(minutes=50), periods=20, freq="min"))
    exchange = Exchange(missing=quiet)
    bars = history("SOL-USD", 300, end=T0, fetch=lambda p, s, e: fetch_candles(p, s, e, opener=exchange))
    assert len(bars) == 300 and not set(bars.index) & quiet and len(exchange.urls) == 2
    with pytest.raises(MarketDataError, match="No candle"):
        history("SOL-USD", 10, end=T0, fetch=lambda p, s, e: fetch_candles(p, s, e, opener=Exchange(payload=[])))
