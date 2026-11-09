from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from quantlab.execution.exits import Level
from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.technical import ATR, RSI, BollingerBands, VolumeRatio
from quantlab.strategies.base import Strategy
from quantlab.strategies.params import Param


class MeanReversionStrategy(Strategy):
    """
    Fade moves well outside the Bollinger Bands when the RSI is extreme and volume spikes.

    With `target_middle_band`, the target of each position is the Bollinger middle band at the signal
    (the mean the price should revert to) instead of `take_profit_pct`.
    """

    PARAMS = (
        Param("rsi_threshold", "float", 30, "Survente si RSI < seuil, surachat si RSI > 100 - seuil.", "signal",
              min=0, max=50),
        Param("volume_threshold", "float", 1.2, "Ratio de volume minimal (pic de volume).", "signal", min=0, max=100),
        Param("band_atr_multiple", "float", 0.2, "Le prix doit sortir des bandes d'au moins k x ATR.", "signal",
              min=0, max=20),
        Param("target_middle_band", "bool", False, "Objectif = bande de Bollinger médiane au signal.", "sorties"),
        Param("bb_window", "int", 20, "Fenêtre des bandes de Bollinger.", "indicateurs", min=2, max=1000),
        Param("bb_std", "float", 2.0, "Écart des bandes, en écarts-types.", "indicateurs", min=0.1, max=10),
        Param("rsi_window", "int", 14, "Fenêtre du RSI.", "indicateurs", min=2, max=1000),
        Param("volume_window", "int", 20, "Fenêtre du ratio de volume.", "indicateurs", min=1, max=1000),
        Param("band_atr_window", "int", 14, "Fenêtre de l'ATR du filtre de bandes.", "indicateurs", min=1, max=1000),
        *Strategy.PARAMS,
    )  # fmt: skip
    # Class values: the defaults of strategies pickled before these parameters existed
    volume_threshold: float = 1.2
    band_atr_multiple: float = 0.2

    def __init__(
        self,
        rsi_threshold: float = 30,
        symbol: str = "AAPL",
        target_middle_band: bool = False,
        volume_threshold: float = 1.2,
        band_atr_multiple: float = 0.2,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_window: int = 14,
        volume_window: int = 20,
        band_atr_window: int = 14,
        **kwargs: Any,
    ) -> None:
        super().__init__(symbol=symbol, **kwargs)
        self.rsi_threshold = rsi_threshold
        self.target_middle_band = target_middle_band
        self.volume_threshold, self.band_atr_multiple = volume_threshold, band_atr_multiple
        self.bollinger, self.rsi = BollingerBands(bb_window, bb_std), RSI(rsi_window)
        self.volume, self.atr = VolumeRatio(volume_window), ATR(band_atr_window)

    def features(self) -> Sequence[Feature]:
        return (self.bollinger, self.rsi, self.volume, self.atr)

    def decide(self, bar: pd.Series) -> list[Order]:
        row = self.feed.row()
        price, rsi, atr = bar["close"], row[self.rsi.name], row[self.atr.name]
        upper, lower = row[self.bollinger.upper], row[self.bollinger.lower]

        volume_spike = row[self.volume.name] > self.volume_threshold  # volume confirms the potential reversal
        # The price must be significantly outside the bands
        oversold = price < (lower - self.band_atr_multiple * atr)
        overbought = price > (upper + self.band_atr_multiple * atr)

        plan = self.exit_plan()
        if self.target_middle_band:
            plan = replace(plan, target=Level.at((upper + lower) / 2))
        if oversold and rsi < self.rsi_threshold and volume_spike:
            return [self.order(Side.BUY, exit_plan=plan)]
        if overbought and rsi > (100 - self.rsi_threshold) and volume_spike:
            return [self.order(Side.SELL, exit_plan=plan)]
        return []
