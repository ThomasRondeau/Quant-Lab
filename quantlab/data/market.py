"""
Public market data of a crypto exchange: the 1-minute candles of Coinbase Exchange, read without key or account.

Only market data is read here: no order is ever sent anywhere. What the endpoint does, and what this module
does about it:

- at most 300 candles per request, both ends of the range included. A wider range silently loses its most recent
  candles, so `history` asks for windows of exactly 300 candles;
- the responses are cached for several minutes per URL by the exchange's CDN. Every request therefore carries its
  own `start` / `end`: a poll never reads a stale copy of the "latest candles";
- a minute without a trade has no candle, and the current minute's candle is still moving. `history` only returns
  closed candles.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from quantlab.data.schema import INDEX_NAME, OHLCV_COLUMNS, validate_ohlcv

API_URL = "https://api.exchange.coinbase.com"
INTERVAL = pd.Timedelta(minutes=1)
MAX_CANDLES = 300
TIMEOUT_SECONDS = 10.0
USER_AGENT = "market-data-client/1.0"
#: order of the fields of a candle in the exchange's response
CANDLE_FIELDS = ("time", "low", "high", "open", "close", "volume")
PRODUCT = re.compile(r"[A-Z0-9]{2,10}-[A-Z0-9]{2,10}")

#: (product, start, end) -> candles whose start lies in [start, end], oldest first
Fetch = Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame]
Opener = Callable[..., Any]


class MarketDataError(RuntimeError):
    """The market data could not be read (network, exchange error, unknown product)."""


def check_product(product: str) -> str:
    """`product` as the exchange names it (`SOL-USD`); raises on anything else."""
    name = product.strip().upper()
    if not PRODUCT.fullmatch(name):
        raise MarketDataError(f"{product!r} is not a product name such as SOL-USD.")
    return name


def _utc(time: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(time)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OHLCV_COLUMNS), index=pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME), dtype=float)


def fetch_candles(
    product: str, start: pd.Timestamp, end: pd.Timestamp, opener: Opener = urllib.request.urlopen
) -> pd.DataFrame:
    """
    The 1-minute candles of `product` whose start lies in [start, end], oldest first, in the OHLCV contract.
    The range must hold at most 300 candles; the candle of the current minute, if asked for, is not final.
    """
    start, end = _utc(start), _utc(end)
    if (end - start) / INTERVAL >= MAX_CANDLES:
        raise ValueError(f"At most {MAX_CANDLES} candles per request ({start} to {end} asked).")
    query = urlencode(
        {
            "granularity": int(INTERVAL.total_seconds()),
            "start": f"{start:%Y-%m-%dT%H:%M:%SZ}",
            "end": f"{end:%Y-%m-%dT%H:%M:%SZ}",
        }
    )
    url = f"{API_URL}/products/{check_product(product)}/candles?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with opener(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:  # network errors, HTTP errors, timeouts, invalid JSON
        raise MarketDataError(f"Market data request failed for {product}: {exc}") from exc
    if not isinstance(payload, list):
        raise MarketDataError(f"Unexpected market data response for {product}: {str(payload)[:200]}")
    if not payload:
        return _empty()
    try:
        frame = pd.DataFrame(payload, columns=list(CANDLE_FIELDS))
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("time"), unit="s", utc=True))
        bars = validate_ohlcv(frame)[list(OHLCV_COLUMNS)]
    except (ValueError, TypeError) as exc:
        raise MarketDataError(f"Unreadable candles for {product}: {exc}") from exc
    return bars[(bars.index >= start) & (bars.index <= end)]


def last_closed(now: pd.Timestamp) -> pd.Timestamp:
    """Start of the most recent candle that is closed at `now`."""
    return _utc(now).floor("min") - INTERVAL


def history(product: str, n_bars: int, end: pd.Timestamp | None = None, fetch: Fetch = fetch_candles) -> pd.DataFrame:
    """
    The last `n_bars` closed candles of `product` at `end` (default: now), oldest first. Minutes without a
    trade have no candle, so a few more windows than `n_bars / 300` may be read.
    """
    if n_bars < 1:
        raise ValueError(f"n_bars must be >= 1 (got {n_bars}).")
    cursor = last_closed(pd.Timestamp.now(tz="UTC") if end is None else end)
    window = INTERVAL * (MAX_CANDLES - 1)
    frames: list[pd.DataFrame] = []
    count = 0
    for _ in range(-(-n_bars // MAX_CANDLES) + 3):
        candles = fetch(product, cursor - window, cursor)
        frames.append(candles)
        count += len(candles)
        if count >= n_bars:
            break
        cursor -= window + INTERVAL
    bars = pd.concat(frames[::-1])
    bars = bars[~bars.index.duplicated(keep="first")].sort_index()
    if bars.empty:
        raise MarketDataError(f"No candle for {product}.")
    return bars.iloc[-n_bars:]
