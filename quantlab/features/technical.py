"""Base technical indicators as stateless, causal features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantlab.features.base import (
    Computed,
    Feature,
    Source,
    ewm_warmup,
    source_columns,
    source_inputs,
    source_name,
    source_series,
)


def _suffix(source: Source) -> str:
    """Name suffix for a source; the default `close` column is implicit."""
    name = source_name(source)
    return "" if name == "close" else f"_{name}"


class SMA(Feature):
    """Simple moving average."""

    def __init__(self, window: int = 14, source: Source = "close") -> None:
        self.window = window
        self.source = source
        self.required_columns = source_columns(source)

    @property
    def name(self) -> str:
        return f"sma_{self.window}{_suffix(self.source)}"

    def inputs(self) -> Sequence[Feature]:
        return source_inputs(self.source)

    @property
    def warmup(self) -> int:
        return self.window - 1

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series:
        return source_series(self.source, data, computed).rolling(window=self.window).mean()


class RSI(Feature):
    """Relative Strength Index with simple-moving-average smoothing of gains and losses."""

    required_columns = ("close",)

    def __init__(self, window: int = 14) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"rsi_{self.window}"

    @property
    def warmup(self) -> int:
        return self.window  # diff (1) + rolling (window - 1)

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series:
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class ATR(Feature):
    """Average True Range (simple moving average of the true range)."""

    required_columns = ("high", "low", "close")

    def __init__(self, window: int = 14) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"atr_{self.window}"

    @property
    def warmup(self) -> int:
        return self.window  # previous close (1) + rolling (window - 1)

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series:
        prev_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"], (data["high"] - prev_close).abs(), (data["low"] - prev_close).abs()], axis=1
        ).max(axis=1)
        return true_range.rolling(window=self.window).mean()


class MACD(Feature):
    """MACD line, signal line and histogram (EMA-based)."""

    required_columns = ("close",)

    def __init__(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> None:
        if short_window >= long_window:
            raise ValueError(f"MACD: short_window ({short_window}) must be < long_window ({long_window}).")
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    @property
    def name(self) -> str:
        return f"macd_{self.short_window}_{self.long_window}_{self.signal_window}"

    @property
    def line(self) -> str:
        return self.name

    @property
    def signal(self) -> str:
        return f"macd_signal_{self.short_window}_{self.long_window}_{self.signal_window}"

    @property
    def histogram(self) -> str:
        return f"macd_hist_{self.short_window}_{self.long_window}_{self.signal_window}"

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.line, self.signal, self.histogram)

    @property
    def warmup(self) -> int:
        return ewm_warmup(self.long_window) + ewm_warmup(self.signal_window)

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.DataFrame:
        close = data["close"]
        macd = (
            close.ewm(span=self.short_window, adjust=False).mean()
            - close.ewm(span=self.long_window, adjust=False).mean()
        )
        signal = macd.ewm(span=self.signal_window, adjust=False).mean()
        return pd.DataFrame({self.line: macd, self.signal: signal, self.histogram: macd - signal})


class BollingerBands(Feature):
    required_columns = ("close",)

    def __init__(self, window: int = 20, num_std_dev: float = 2) -> None:
        self.window = window
        self.num_std_dev = num_std_dev

    @property
    def name(self) -> str:
        return f"bb_{self.window}_{self.num_std_dev:g}"

    @property
    def upper(self) -> str:
        return f"bb_upper_{self.window}_{self.num_std_dev:g}"

    @property
    def lower(self) -> str:
        return f"bb_lower_{self.window}_{self.num_std_dev:g}"

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.upper, self.lower)

    @property
    def warmup(self) -> int:
        return self.window - 1

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.DataFrame:
        close = data["close"]
        mid = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()
        return pd.DataFrame({self.upper: mid + self.num_std_dev * std, self.lower: mid - self.num_std_dev * std})


class VolumeRatio(Feature):
    """Volume divided by its moving average."""

    required_columns = ("volume",)

    def __init__(self, window: int = 14) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"volume_ratio_{self.window}"

    @property
    def warmup(self) -> int:
        return self.window - 1

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series:
        return data["volume"] / data["volume"].rolling(window=self.window).mean()


class Ratio(Feature):
    """Ratio of two sources (e.g. the prices of a pair)."""

    def __init__(self, numerator: Source, denominator: Source) -> None:
        self.numerator = numerator
        self.denominator = denominator
        self.required_columns = source_columns(numerator, denominator)

    @property
    def name(self) -> str:
        return f"ratio_{source_name(self.numerator)}_{source_name(self.denominator)}"

    def inputs(self) -> Sequence[Feature]:
        return source_inputs(self.numerator, self.denominator)

    @property
    def warmup(self) -> int:
        return 0

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series:
        return source_series(self.numerator, data, computed) / source_series(self.denominator, data, computed)


class ZScore(Feature):
    """Rolling z-score of a source."""

    def __init__(self, source: Source, window: int = 20) -> None:
        self.source = source
        self.window = window
        self.required_columns = source_columns(source)

    @property
    def name(self) -> str:
        return f"zscore_{self.window}_{source_name(self.source)}"

    def inputs(self) -> Sequence[Feature]:
        return source_inputs(self.source)

    @property
    def warmup(self) -> int:
        return self.window - 1

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series:
        series = source_series(self.source, data, computed)
        return (series - series.rolling(window=self.window).mean()) / series.rolling(window=self.window).std()


class ADX(Feature):
    """Average Directional Index (rolling-sum smoothing)."""

    required_columns = ("high", "low", "close")

    def __init__(self, window: int = 14) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"adx_{self.window}"

    @property
    def warmup(self) -> int:
        return 2 * self.window - 1  # shift (1) + rolling sums (window - 1) + rolling mean (window - 1)

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series:
        high, low, close = data["high"], data["low"], data["close"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr_smooth = tr.rolling(window=self.window).sum()
        plus_di = 100 * (plus_dm.rolling(window=self.window).sum() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(window=self.window).sum() / tr_smooth)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx.rolling(window=self.window).mean()


class VWAP(Feature):
    """
    Session VWAP with volume-weighted standard-deviation bands. It restarts at each session
    (calendar date in `session_tz`), so the history must reach back to the session start:
    `warmup` is `max_session_bars`.
    """

    required_columns = ("close", "high", "low", "volume")

    def __init__(self, std_multiplier: float = 3, session_tz: str = "America/New_York", max_session_bars: int = 1440):
        self.std_multiplier = std_multiplier
        self.session_tz = session_tz
        self.max_session_bars = max_session_bars

    @property
    def name(self) -> str:
        return "vwap"

    @property
    def outputs(self) -> tuple[str, ...]:
        m = f"{self.std_multiplier:g}"
        return ("vwap", f"vwap_upper_{m}", f"vwap_lower_{m}", f"vwap_far_upper_{m}", f"vwap_far_lower_{m}")

    @property
    def warmup(self) -> int:
        return self.max_session_bars

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.DataFrame:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("VWAP: a DatetimeIndex is required to reset the VWAP at each session.")
        index = data.index if data.index.tz is not None else data.index.tz_localize("UTC")
        session = pd.Series(index.tz_convert(self.session_tz).date, index=data.index)

        typical_price = (data["close"] + data["high"] + data["low"]) / 3
        volume = data["volume"]
        cum_volume = volume.groupby(session).cumsum()
        vwap = (typical_price * volume).groupby(session).cumsum() / cum_volume
        # Volume-weighted standard deviation of the typical price around the VWAP
        mean_sq = (typical_price**2 * volume).groupby(session).cumsum() / cum_volume
        band = np.sqrt((mean_sq - vwap**2).clip(lower=0)) * self.std_multiplier

        _, upper, lower, far_upper, far_lower = self.outputs
        return pd.DataFrame(
            {
                "vwap": vwap,
                upper: vwap + band,
                lower: vwap - band,
                far_upper: vwap + 2 * band,
                far_lower: vwap - 2 * band,
            }
        )
