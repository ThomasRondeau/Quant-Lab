from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score

from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.composite import AmplitudeFeatures, DirectionFeatures, OpportunityFeatures
from quantlab.features.technical import ATR, MACD, RSI, SMA, BollingerBands, VolumeRatio
from quantlab.models.sklearn_model import SklearnModel
from quantlab.strategies.base import Strategy
from quantlab.strategies.ml_params import (
    check_ml,
    forest_params,
    indicator_params,
    target_params,
    warn_ml,
    xgb_params,
)
from quantlab.strategies.params import Param
from quantlab.targets import ForwardTarget

logger = logging.getLogger(__name__)

HORIZON = 5  # default bars ahead predicted by the three models


class CascadeStrategy(Strategy):
    """
    Three models in cascade: is there an opportunity? -> which direction? -> how large?
    A trade is taken only when the three agree and the predicted amplitude is large enough.
    """

    PARAMS = (
        Param("min_amplitude_threshold", "float", 0.05, "Amplitude prévue minimale pour trader, en %.", "signal",
              min=0, max=100),
        Param("conviction_cap_pct", "float", 5.0, "Amplitude prévue (en %) qui donne la conviction maximale.",
              "signal", min=0.001, max=100),
        *forest_params("amplitude_", "amplitude", min_samples_leaf=10),
        *xgb_params("direction_", "direction"),
        *forest_params("opportunity_", "opportunity", min_samples_leaf=15, class_weight=True),
        Param("opportunity_threshold_pct", "float", 0.05, "Mouvement (en %) qui compte comme une opportunité.",
              "cible", min=0, max=100),
        *target_params(),
        *indicator_params(volume_window=14),
        Param("sma_fast_window", "int", 5, "Moyenne mobile rapide du modèle d'opportunité.", "indicateurs", min=1,
              max=1000),
        Param("amplitude_momentum", "int", 5, "Momentum du modèle d'amplitude (barres).", "indicateurs", min=1,
              max=1000),
        Param("atr_mean_window", "int", 20, "Moyenne de l'ATR (régime de volatilité).", "indicateurs", min=1,
              max=1000),
        Param("atr_change_window", "int", 3, "Variation de l'ATR (barres).", "indicateurs", min=1, max=1000),
        Param("volume_fast_window", "int", 5, "Moyenne rapide du volume brut.", "indicateurs", min=1, max=1000),
        Param("volume_slow_window", "int", 20, "Moyenne lente du volume brut.", "indicateurs", min=1, max=1000),
        Param("accel_window", "int", 3, "Variation du prix (« accélération », barres).", "indicateurs", min=1,
              max=1000),
        *Strategy.PARAMS,
    )  # fmt: skip
    # Class values: the defaults of strategies pickled before these parameters existed
    conviction_cap_pct: float = 5.0
    cv_folds: int = 5

    @classmethod
    def check(cls, values: Mapping[str, Any]) -> list[str]:
        errors = check_ml(values)
        if values["volume_fast_window"] >= values["volume_slow_window"]:
            errors.append("volume_fast_window must be smaller than volume_slow_window.")
        return errors

    @classmethod
    def warnings(cls, values: Mapping[str, Any]) -> list[str]:
        return warn_ml(values)

    def __init__(
        self,
        min_amplitude_threshold: float = 0.05,
        symbol: str = "AAPL",
        conviction_cap_pct: float = 5.0,
        amplitude_n_estimators: int = 100,
        amplitude_max_depth: int = 5,
        amplitude_min_samples_leaf: int = 10,
        direction_n_estimators: int = 100,
        direction_max_depth: int = 4,
        direction_learning_rate: float = 0.05,
        direction_subsample: float = 0.8,
        direction_colsample_bytree: float = 0.8,
        opportunity_n_estimators: int = 100,
        opportunity_max_depth: int = 5,
        opportunity_min_samples_leaf: int = 15,
        opportunity_class_weight: str | None = "balanced",
        opportunity_threshold_pct: float = 0.05,
        horizon: int = HORIZON,
        training_margin: int = 30,
        random_state: int = 42,
        cv_folds: int = 5,
        rsi_window: int = 14,
        bb_window: int = 20,
        bb_std: float = 2.0,
        volume_window: int = 14,
        feature_atr_window: int = 14,
        sma_window: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        momentum_windows: Sequence[int] = (5, 10, 20),
        slope_window: int = 3,
        sma_fast_window: int = 5,
        amplitude_momentum: int = 5,
        atr_mean_window: int = 20,
        atr_change_window: int = 3,
        volume_fast_window: int = 5,
        volume_slow_window: int = 20,
        accel_window: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        :param min_amplitude_threshold: minimum predicted amplitude to trade, in percent (same unit as the
                                        `return_pct` and `move` targets).
        :param conviction_cap_pct: predicted amplitude giving the full conviction (the order's size is
                                   multiplied by min(|amplitude| / cap, 1)).
        """
        super().__init__(symbol=symbol, **kwargs)
        self.min_amplitude_threshold = min_amplitude_threshold
        self.conviction_cap_pct = conviction_cap_pct
        self.cv_folds = cv_folds
        self.training_data_shift = training_margin  # look-ahead safety margin of the training sets (>= horizon)

        atr, rsi = ATR(feature_atr_window), RSI(rsi_window)
        macd, bollinger = MACD(macd_fast, macd_slow, macd_signal), BollingerBands(bb_window, bb_std)
        volume, sma_fast, sma = VolumeRatio(volume_window), SMA(sma_fast_window), SMA(sma_window)
        self.amplitude_features = AmplitudeFeatures(atr, sma, bollinger, rsi, macd, volume, momentum=amplitude_momentum)
        self.direction_features = DirectionFeatures(
            atr, sma, bollinger, rsi, macd, volume, momentum=momentum_windows, slope=slope_window
        )
        self.opportunity_features = OpportunityFeatures(
            atr,
            sma_fast,
            sma,
            bollinger,
            rsi,
            macd,
            volume,
            atr_mean=atr_mean_window,
            atr_change=atr_change_window,
            volume_fast=volume_fast_window,
            volume_slow=volume_slow_window,
            accel=accel_window,
        )

        self.amplitude_model = SklearnModel(
            RandomForestRegressor(
                n_estimators=amplitude_n_estimators,
                max_depth=amplitude_max_depth,
                min_samples_leaf=amplitude_min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
            ),
            self.amplitude_features.outputs,
            ForwardTarget(horizon, "return_pct"),
            r2_score,
        )
        self.direction_model = SklearnModel(
            xgb.XGBClassifier(
                n_estimators=direction_n_estimators,
                max_depth=direction_max_depth,
                learning_rate=direction_learning_rate,
                subsample=direction_subsample,
                colsample_bytree=direction_colsample_bytree,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            self.direction_features.outputs,
            ForwardTarget(horizon, "direction"),
            accuracy_score,
        )
        self.opportunity_model = SklearnModel(
            RandomForestClassifier(
                n_estimators=opportunity_n_estimators,
                max_depth=opportunity_max_depth,
                min_samples_leaf=opportunity_min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
                class_weight=opportunity_class_weight,
            ),
            self.opportunity_features.outputs,
            ForwardTarget(horizon, "move", threshold_pct=opportunity_threshold_pct),
            balanced_accuracy_score,
        )
        self._datasets: dict[str, tuple[npt.NDArray[Any], npt.NDArray[Any]]] = {}

    @property
    def models(self) -> dict[str, SklearnModel]:
        return {
            "amplitude": self.amplitude_model,
            "direction": self.direction_model,
            "opportunity": self.opportunity_model,
        }

    def features(self) -> Sequence[Feature]:
        return (self.amplitude_features, self.direction_features, self.opportunity_features)

    def train(self, history: pd.DataFrame) -> None:
        features = self.feed.pipeline.compute(history)
        for name, model in self.models.items():
            X, y = model.dataset(history, features, drop_last=self.training_data_shift)
            model.fit(X, y)
            self._datasets[name] = (X, y)

    def evaluate(self, n_splits: int | None = None) -> dict[str, dict[str, float]]:
        """In-sample vs time-ordered cross-validated scores of the three models (overfitting check)."""
        report: dict[str, dict[str, float]] = {}
        for name, model in self.models.items():
            X, y = self._datasets[name]
            cv_mean, _ = model.cross_validate(X, y, n_splits=n_splits or self.cv_folds)
            report[name] = {"in_sample": model.in_sample_score(X, y), "cv": cv_mean}
            logger.info("%-11s model | in-sample %.2f | CV %.2f", name, report[name]["in_sample"], cv_mean)
        return report

    def decide(self, bar: pd.Series) -> list[Order]:
        row = self.feed.row()
        opportunity = int(self.opportunity_model.predict_row(row))
        self.diagnostics = {"opportunity": opportunity}
        if opportunity != 1:
            return []

        up = int(self.direction_model.predict_row(row)) == 1
        direction = "UP" if up else "DOWN"
        amplitude = float(self.amplitude_model.predict_row(row))  # percent
        self.diagnostics.update(direction_up=int(up), amplitude_pct=amplitude)
        if abs(amplitude) < self.min_amplitude_threshold:
            return []
        # The direction and the sign of the amplitude must agree
        if not ((direction == "UP" and amplitude > 0) or (direction == "DOWN" and amplitude < 0)):
            return []

        confidence = min(abs(amplitude) / self.conviction_cap_pct, 1.0)
        return [self.order(Side.BUY if direction == "UP" else Side.SELL, conviction=confidence)]
