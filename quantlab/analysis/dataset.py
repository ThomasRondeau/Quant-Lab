"""
Descriptive analysis of a set of bars: one frame per symbol, summary, sessions, bar returns, intraday
profile and quality checks. Everything is computed on New York wall-clock times; a session is a New York date.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from quantlab.data.schema import OHLCV_COLUMNS

EXCHANGE_TZ = "America/New_York"
SESSIONS_PER_YEAR = 252
#: widths (minutes) tried for the time-of-day buckets of the intraday profile
PROFILE_WIDTHS = (1, 2, 5, 10, 15, 30, 60)


def exchange_index(index: pd.Index) -> pd.DatetimeIndex:
    """New York wall-clock times (naive); naive timestamps are read as UTC."""
    times = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return times.tz_convert(EXCHANGE_TZ).tz_localize(None)


def symbol_bars(bars: pd.DataFrame, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """One OHLCV frame per symbol, indexed in New York time, from single- or multi-symbol bars."""
    index = exchange_index(bars.index)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        suffixed = {f"{field}_{symbol}": field for field in OHLCV_COLUMNS}
        if all(column in bars.columns for column in suffixed):
            frame = bars[list(suffixed)].rename(columns=suffixed)
        elif len(symbols) == 1 and all(field in bars.columns for field in OHLCV_COLUMNS):
            frame = bars[list(OHLCV_COLUMNS)]
        else:
            raise ValueError(f"No bars for {symbol} in this dataset.")
        frame = frame.astype(float).set_axis(index)
        frames[symbol] = frame[~frame.index.duplicated(keep="first")].sort_index()
    return frames


def bar_interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Most frequent spacing between consecutive bars (zero for fewer than two bars)."""
    steps = pd.Series(index).diff().dropna()
    return pd.Timedelta(steps.mode().iloc[0]) if not steps.empty else pd.Timedelta(0)


def _session(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(frame.index).normalize()


def intraday_returns(frame: pd.DataFrame) -> pd.Series:
    """Close-to-close returns of the bars inside each session (the overnight gap is left out)."""
    returns: pd.Series = frame["close"].groupby(_session(frame)).pct_change().dropna()
    return returns


def sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per session: OHLC, return on the previous close (first session: on its open), range, volume, bars."""
    table = frame.groupby(_session(frame)).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        bars=("close", "size"),
    )
    previous = table["close"].shift().fillna(table["open"])
    table.insert(4, "return_pct", (table["close"] / previous - 1) * 100)
    table.insert(5, "range_pct", (table["high"] / table["low"] - 1) * 100)
    table.index = pd.Index(pd.DatetimeIndex(table.index).date, name="session")
    return table


def summary(frame: pd.DataFrame) -> dict[str, Any]:
    """
    Headline figures of the bars. The volatility is the standard deviation of the intraday bar returns,
    annualized on the typical number of bars per session and 252 sessions (overnight gaps left out).
    """
    per_session = frame.groupby(_session(frame)).size()
    returns = intraday_returns(frame)
    bars_per_year = float(per_session.median()) * SESSIONS_PER_YEAR
    return {
        "bars": len(frame),
        "sessions": len(per_session),
        "start": frame.index[0],
        "end": frame.index[-1],
        "interval": bar_interval(pd.DatetimeIndex(frame.index)),
        "return_pct": float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1) * 100,
        "low": float(frame["low"].min()),
        "high": float(frame["high"].max()),
        "volatility_pct": float(returns.std() * np.sqrt(bars_per_year)) * 100 if len(returns) > 1 else float("nan"),
        "session_volume": float(frame["volume"].groupby(_session(frame)).sum().mean()),
    }


def return_stats(returns: pd.Series) -> dict[str, float]:
    """Shape of a return distribution; means and extremes in basis points."""
    if returns.empty:
        return {}
    return {
        "mean_bp": float(returns.mean()) * 1e4,
        "std_bp": float(returns.std()) * 1e4,
        "min_bp": float(returns.min()) * 1e4,
        "max_bp": float(returns.max()) * 1e4,
        "skew": float(cast(float, returns.skew())),
        "excess_kurtosis": float(cast(float, returns.kurt())),
        "positive_pct": float((returns > 0).mean()) * 100,
    }


def intraday_profile(frame: pd.DataFrame, max_buckets: int = 100) -> pd.DataFrame:
    """
    Average volume per bar and average absolute bar return (bp) by time of day (`HH:MM` index). The buckets
    are the narrowest of PROFILE_WIDTHS that hold a full bar and keep the profile under `max_buckets` rows.
    """
    index = pd.DatetimeIndex(frame.index)
    minutes = np.asarray(index.hour * 60 + index.minute)
    span = int(minutes.max() - minutes.min()) + 1
    bar_minutes = bar_interval(index) / pd.Timedelta(minutes=1)
    width = next((w for w in PROFILE_WIDTHS if w >= bar_minutes and span / w <= max_buckets), PROFILE_WIDTHS[-1])
    bucket = minutes // width * width
    moves = frame["close"].groupby(_session(frame)).pct_change().abs() * 1e4
    profile = pd.DataFrame(
        {"volume": frame["volume"].groupby(bucket).mean(), "abs_return_bp": moves.groupby(bucket).mean()}
    )
    profile.index = pd.Index([f"{m // 60:02d}:{m % 60:02d}" for m in profile.index], name="time")
    return profile


def quality(frame: pd.DataFrame) -> dict[str, Any]:
    """
    Checks of the bars. Missing bars are the holes of each session between its first and last bar, on the
    grid of the bar interval; a session is incomplete when it holds fewer bars than the typical one.
    """
    index = pd.DatetimeIndex(frame.index)
    interval = bar_interval(index)
    times = pd.Series(index, index=index).groupby(_session(frame))
    sizes = times.size()
    expected = int(((times.last() - times.first()) / interval).round().sum()) + len(sizes) if interval else len(frame)
    gaps = pd.to_timedelta(times.diff().dropna())
    body_high, body_low = frame[["open", "close"]].max(axis=1), frame[["open", "close"]].min(axis=1)
    return {
        "missing_bars": max(expected - len(frame), 0),
        "incomplete_sessions": int((sizes.to_numpy() < float(sizes.median())).sum()),
        "largest_gap": gaps.max() if not gaps.empty else pd.Timedelta(0),
        "zero_volume_bars": int((frame["volume"] <= 0).sum()),
        "inconsistent_bars": int(((frame["high"] < body_high) | (frame["low"] > body_low) | (frame["low"] <= 0)).sum()),
        "missing_values": int(frame.isna().any(axis=1).sum()),
    }


def return_correlation(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Correlation of the intraday bar returns of several symbols."""
    return pd.DataFrame({symbol: intraday_returns(frame) for symbol, frame in frames.items()}).corr()
