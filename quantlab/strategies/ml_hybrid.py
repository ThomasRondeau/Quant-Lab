from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.composite import DirectionFeatures
from quantlab.features.technical import ATR, MACD, RSI, SMA, BollingerBands, VolumeRatio
from quantlab.models.sklearn_model import SklearnModel
from quantlab.strategies.base import Strategy
from quantlab.strategies.ml_params import check_ml, indicator_params, target_params, warn_ml, xgb_params
from quantlab.strategies.params import Param
from quantlab.targets import ForwardTarget

logger = logging.getLogger(__name__)


class MLHybridStrategy(Strategy):
    """Mean reversion entries (Bollinger + RSI + volume) confirmed by an XGBoost direction model."""

    PARAMS = (
        Param("rsi_buy", "float", 35, "Achat si le prix est sous la bande basse et RSI < ce seuil.", "signal",
              min=0, max=100),
        Param("rsi_sell", "float", 65, "Vente si le prix est sur la bande haute et RSI > ce seuil.", "signal",
              min=0, max=100),
        Param("volume_threshold", "float", 1.1, "Ratio de volume minimal.", "signal", min=0, max=100),
        *xgb_params("", "direction"),
        *target_params(cv_folds=False),
        *indicator_params(volume_window=20),
        *Strategy.PARAMS,
    )  # fmt: skip
    # Class values: the defaults of strategies pickled before these parameters existed
    rsi_buy: float = 35
    rsi_sell: float = 65
    volume_threshold: float = 1.1

    @classmethod
    def check(cls, values: Mapping[str, Any]) -> list[str]:
        errors = check_ml(values)
        if values["rsi_buy"] >= values["rsi_sell"]:
            errors.append("rsi_buy must be smaller than rsi_sell.")
        return errors

    @classmethod
    def warnings(cls, values: Mapping[str, Any]) -> list[str]:
        return warn_ml(values)

    def __init__(
        self,
        max_depth: int = 4,
        symbol: str = "AAPL",
        rsi_buy: float = 35,
        rsi_sell: float = 65,
        volume_threshold: float = 1.1,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        horizon: int = 5,
        training_margin: int = 30,
        random_state: int = 42,
        rsi_window: int = 14,
        bb_window: int = 20,
        bb_std: float = 2.0,
        volume_window: int = 20,
        feature_atr_window: int = 14,
        sma_window: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        momentum_windows: Sequence[int] = (5, 10, 20),
        slope_window: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(symbol=symbol, **kwargs)
        self.rsi_buy, self.rsi_sell, self.volume_threshold = rsi_buy, rsi_sell, volume_threshold
        self.training_data_shift = training_margin  # look-ahead safety margin of the training set (>= horizon)
        self.bollinger = BollingerBands(bb_window, bb_std)
        self.rsi, self.volume = RSI(rsi_window), VolumeRatio(volume_window)
        self.direction_features = DirectionFeatures(
            ATR(feature_atr_window),
            SMA(sma_window),
            self.bollinger,
            self.rsi,
            MACD(macd_fast, macd_slow, macd_signal),
            self.volume,
            momentum=momentum_windows,
            slope=slope_window,
        )
        self.direction_model = SklearnModel(
            xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            self.direction_features.outputs,
            ForwardTarget(horizon, "direction"),
            accuracy_score,
        )
        self._dataset: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None

    def features(self) -> Sequence[Feature]:
        return (self.direction_features,)

    def train(self, history: pd.DataFrame) -> None:
        features = self.feed.pipeline.compute(history)
        self._dataset = self.direction_model.dataset(history, features, drop_last=self.training_data_shift)
        self.direction_model.fit(*self._dataset)

    def decide(self, bar: pd.Series) -> list[Order]:
        row = self.feed.row()
        price = bar["close"]
        rsi, volume_ratio = row[self.rsi.name], row[self.volume.name]
        upper, lower = row[self.bollinger.upper], row[self.bollinger.lower]

        tech_buy = price <= lower and rsi < self.rsi_buy and volume_ratio > self.volume_threshold
        tech_sell = price >= upper and rsi > self.rsi_sell and volume_ratio > self.volume_threshold
        if not (tech_buy or tech_sell):
            return []

        up = self.direction_model.predict_row(row) == 1
        self.diagnostics = {"direction_up": int(up)}
        if tech_buy and up:
            return [self.order(Side.BUY)]
        if tech_sell and not up:
            return [self.order(Side.SELL)]
        return []
