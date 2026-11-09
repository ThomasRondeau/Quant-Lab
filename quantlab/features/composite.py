"""
Composite feature sets, one per ML model (amplitude, direction, opportunity).

Each set receives the base indicators it is built on, so a strategy controls their parameters
(e.g. the volume window) explicitly, and its own windows (momentum, slope, ...) as arguments. Output order
is the model's input column order. The set's name carries its windows when they differ from the defaults,
so two sets with different windows never collide in a pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from quantlab.features.base import Computed, Feature
from quantlab.features.technical import ATR, MACD, RSI, SMA, BollingerBands, VolumeRatio


class _FeatureSet(Feature):
    prefix: ClassVar[str]
    #: the set's own windows and their defaults (the name shows the ones that differ)
    defaults: ClassVar[dict[str, Any]]
    required_columns: tuple[str, ...] = ("close",)
    columns: tuple[str, ...]

    def _set_windows(self, **windows: Any) -> None:
        for key, value in windows.items():
            if any(int(v) < 1 for v in (value if isinstance(value, tuple) else (value,))):
                raise ValueError(f"{type(self).__name__}: {key} must be >= 1 (got {value}).")
        self.windows = windows

    @property
    def name(self) -> str:
        changed = [
            "-".join(map(str, v)) if isinstance(v, tuple) else str(v)
            for k, v in self.windows.items()
            if v != self.defaults[k]
        ]
        return "_".join((self.prefix, *changed))

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}_{c}" for c in self.columns)

    def _frame(self, values: dict[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame({f"{self.prefix}_{c}": values[c] for c in self.columns})


class AmplitudeFeatures(_FeatureSet):
    """Magnitude of moves, volatility and activity."""

    prefix = "amplitude"
    defaults: ClassVar[dict[str, Any]] = {"momentum": 5}

    def __init__(
        self,
        atr: ATR | None = None,
        sma: SMA | None = None,
        bollinger: BollingerBands | None = None,
        rsi: RSI | None = None,
        macd: MACD | None = None,
        volume: VolumeRatio | None = None,
        momentum: int = 5,
    ) -> None:
        """:param momentum: bars of the absolute momentum."""
        self.atr = atr or ATR(14)
        self.sma = sma or SMA(20)
        self.bollinger = bollinger or BollingerBands(20, 2)
        self.rsi = rsi or RSI(14)
        self.macd = macd or MACD()
        self.volume = volume or VolumeRatio(14)
        self._set_windows(momentum=int(momentum))
        self.momentum = int(momentum)
        self.columns = (
            "volatility",
            "dist_sma",
            "bb_width",
            "rsi_extreme",
            "macd_hist_abs",
            f"momentum_{self.momentum}",
            "volume_ratio",
        )

    def inputs(self) -> Sequence[Feature]:
        return (self.atr, self.sma, self.bollinger, self.rsi, self.macd, self.volume)

    @property
    def warmup(self) -> int:
        return self.momentum  # pct_change(momentum) on the close

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.DataFrame:
        close = data["close"]
        return self._frame(
            {
                "volatility": computed[self.atr.name] / close,
                "dist_sma": abs(close - computed[self.sma.name]) / close,
                "bb_width": (computed[self.bollinger.upper] - computed[self.bollinger.lower]) / close,
                "rsi_extreme": abs(computed[self.rsi.name] - 50),
                "macd_hist_abs": abs(computed[self.macd.histogram]) / close,
                f"momentum_{self.momentum}": abs(close.pct_change(self.momentum)),
                "volume_ratio": computed[self.volume.name],
            }
        )


class DirectionFeatures(_FeatureSet):
    """Stationarized trend and momentum measures (ATR-normalized)."""

    prefix = "direction"
    defaults: ClassVar[dict[str, Any]] = {"momentum": (5, 10, 20), "slope": 3}

    def __init__(
        self,
        atr: ATR | None = None,
        sma: SMA | None = None,
        bollinger: BollingerBands | None = None,
        rsi: RSI | None = None,
        macd: MACD | None = None,
        volume: VolumeRatio | None = None,
        momentum: Sequence[int] = (5, 10, 20),
        slope: int = 3,
    ) -> None:
        """
        :param momentum: bars of each momentum column.
        :param slope: bars of the SMA slope.
        """
        self.atr = atr or ATR(14)
        self.sma = sma or SMA(20)
        self.bollinger = bollinger or BollingerBands(20, 2)
        self.rsi = rsi or RSI(14)
        self.macd = macd or MACD()
        self.volume = volume or VolumeRatio(14)
        self.momentum = tuple(int(w) for w in momentum)
        self.slope = int(slope)
        if not self.momentum or len(set(self.momentum)) != len(self.momentum):
            raise ValueError(f"DirectionFeatures: momentum windows must be distinct (got {self.momentum}).")
        self._set_windows(momentum=self.momentum, slope=self.slope)
        self.columns = (
            "rsi",
            "macd_hist",
            "dist_sma",
            "sma_slope",
            "bb_pos",
            "vol_trend",
            "volatility",
            *(f"mom_{w}" for w in self.momentum),
        )

    def inputs(self) -> Sequence[Feature]:
        return (self.atr, self.sma, self.bollinger, self.rsi, self.macd, self.volume)

    @property
    def warmup(self) -> int:
        # pct_change(max momentum) on the close; the SMA slope adds `slope` bars to the SMA's own warmup
        return max(*self.momentum, self.slope)

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.DataFrame:
        close = data["close"]
        atr = computed[self.atr.name]
        sma = computed[self.sma.name]
        upper, lower = computed[self.bollinger.upper], computed[self.bollinger.lower]
        return self._frame(
            {
                "rsi": computed[self.rsi.name],
                "macd_hist": computed[self.macd.histogram] / atr,
                "dist_sma": (close - sma) / atr,
                "sma_slope": sma.diff(self.slope) / atr,
                "bb_pos": (close - lower) / (upper - lower),
                "vol_trend": computed[self.volume.name] * np.sign(close.diff()),
                "volatility": atr / close,
                **{f"mom_{w}": close.pct_change(w) for w in self.momentum},
            }
        )


class OpportunityFeatures(_FeatureSet):
    """Volatility regime, activity and distance to the bands."""

    prefix = "opportunity"
    defaults: ClassVar[dict[str, Any]] = {
        "atr_mean": 20,
        "atr_change": 3,
        "volume_fast": 5,
        "volume_slow": 20,
        "accel": 3,
    }
    columns = (
        "bb_width",
        "vol_ratio",
        "vol_change",
        "volume_ratio",
        "volume_trend",
        "momentum_align",
        "dist_lower",
        "dist_upper",
        "dist_sma",
        "price_accel",
    )
    required_columns = ("close", "volume")

    def __init__(
        self,
        atr: ATR | None = None,
        sma_fast: SMA | None = None,
        sma: SMA | None = None,
        bollinger: BollingerBands | None = None,
        rsi: RSI | None = None,
        macd: MACD | None = None,
        volume: VolumeRatio | None = None,
        atr_mean: int = 20,
        atr_change: int = 3,
        volume_fast: int = 5,
        volume_slow: int = 20,
        accel: int = 3,
    ) -> None:
        """
        :param atr_mean: bars of the ATR mean (volatility regime: ATR / its mean).
        :param atr_change: bars of the ATR change.
        :param volume_fast: bars of the fast raw-volume mean (volume trend: fast / slow).
        :param volume_slow: bars of the slow raw-volume mean.
        :param accel: bars of the price change ("acceleration").
        """
        self.atr = atr or ATR(14)
        self.sma_fast = sma_fast or SMA(5)
        self.sma = sma or SMA(20)
        self.bollinger = bollinger or BollingerBands(20, 2)
        self.rsi = rsi or RSI(14)
        self.macd = macd or MACD()
        self.volume = volume or VolumeRatio(14)
        self.atr_mean, self.atr_change, self.accel = int(atr_mean), int(atr_change), int(accel)
        self.volume_fast, self.volume_slow = int(volume_fast), int(volume_slow)
        self._set_windows(
            atr_mean=self.atr_mean,
            atr_change=self.atr_change,
            volume_fast=self.volume_fast,
            volume_slow=self.volume_slow,
            accel=self.accel,
        )

    def inputs(self) -> Sequence[Feature]:
        return (self.atr, self.sma_fast, self.sma, self.bollinger, self.rsi, self.macd, self.volume)

    @property
    def warmup(self) -> int:
        # rolling / pct_change on the ATR and on the raw volume and close
        return max(self.atr_mean, self.atr_change, self.volume_fast, self.volume_slow, self.accel)

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.DataFrame:
        close = data["close"]
        atr = computed[self.atr.name]
        upper, lower = computed[self.bollinger.upper], computed[self.bollinger.lower]
        raw_volume = data["volume"]
        momentum_alignment = (
            (computed[self.macd.line] > computed[self.macd.signal]).astype(int)
            + (computed[self.rsi.name] > 50).astype(int)
            + (close > computed[self.sma_fast.name]).astype(int)
        )
        return self._frame(
            {
                "bb_width": (upper - lower) / close,
                "vol_ratio": atr / atr.rolling(window=self.atr_mean).mean(),
                "vol_change": atr.pct_change(self.atr_change),
                "volume_ratio": computed[self.volume.name],
                "volume_trend": raw_volume.rolling(window=self.volume_fast).mean()
                / raw_volume.rolling(window=self.volume_slow).mean(),
                "momentum_align": momentum_alignment,
                "dist_lower": (close - lower) / atr,
                "dist_upper": (upper - close) / atr,
                "dist_sma": (close - computed[self.sma.name]) / atr,
                "price_accel": close.pct_change(self.accel),
            }
        )
