"""Parameter blocks shared by the ML strategies: indicators, target, and estimator hyperparameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quantlab.strategies.params import Param


def indicator_params(volume_window: int) -> tuple[Param, ...]:
    """Windows of the base indicators the composite feature sets are built on."""
    g = "indicators"
    return (
        Param("rsi_window", "int", 14, "RSI window.", g, min=2, max=1000),
        Param("bb_window", "int", 20, "Bollinger Bands window.", g, min=2, max=1000),
        Param("bb_std", "float", 2.0, "Band width, in standard deviations.", g, min=0.1, max=10),
        Param("volume_window", "int", volume_window, "Volume-ratio window.", g, min=1, max=1000),
        Param("feature_atr_window", "int", 14, "Window of the feature ATR.", g, min=1, max=1000),
        Param("sma_window", "int", 20, "Window of the feature moving average.", g, min=1, max=1000),
        Param("macd_fast", "int", 12, "MACD fast window.", g, min=1, max=500),
        Param("macd_slow", "int", 26, "MACD slow window.", g, min=2, max=1000),
        Param("macd_signal", "int", 9, "MACD signal window.", g, min=1, max=500),
        Param("momentum_windows", "tuple", (5, 10, 20), "Momentum windows of the direction model (bars).", g, min=1,
              max=1000),
        Param("slope_window", "int", 3, "Slope of the moving average (bars).", g, min=1, max=500),
    )  # fmt: skip


def target_params(cv_folds: bool = True) -> tuple[Param, ...]:
    g = "target"
    params: tuple[Param, ...] = (
        Param("horizon", "int", 5, "Bars ahead predicted by the models.", g, min=1, max=1000),
        Param("training_margin", "int", 30, "Bars dropped at the end of the training set (no look-ahead, >= horizon).",
              g, min=1, max=10_000),
        Param("random_state", "int", 42, "Random seed of the models.", g, min=0, max=2**31 - 1),
    )  # fmt: skip
    if cv_folds:
        params += (Param("cv_folds", "int", 5, "Folds of the time-ordered cross-validation.", g, min=2, max=50),)
    return params


def xgb_params(prefix: str, model: str, max_depth: int = 4) -> tuple[Param, ...]:
    g = f"model:{model}"
    return (
        Param(f"{prefix}n_estimators", "int", 100, "Number of trees.", g, min=1, max=10_000),
        Param(f"{prefix}max_depth", "int", max_depth, "Maximum tree depth.", g, min=1, max=20),
        Param(f"{prefix}learning_rate", "float", 0.05, "Learning rate.", g, min=1e-4, max=1),
        Param(f"{prefix}subsample", "float", 0.8, "Fraction of the bars sampled per tree.", g, min=0.05, max=1),
        Param(
            f"{prefix}colsample_bytree", "float", 0.8, "Fraction of the features sampled per tree.", g, min=0.05, max=1
        ),
    )


def forest_params(prefix: str, model: str, min_samples_leaf: int, class_weight: bool = False) -> tuple[Param, ...]:
    g = f"model:{model}"
    params: tuple[Param, ...] = (
        Param(f"{prefix}n_estimators", "int", 100, "Number of trees.", g, min=1, max=10_000),
        Param(f"{prefix}max_depth", "int", 5, "Maximum tree depth.", g, min=1, max=50),
        Param(f"{prefix}min_samples_leaf", "int", min_samples_leaf, "Minimum bars per leaf.", g, min=1,
              max=10_000),
    )  # fmt: skip
    if class_weight:
        params += (
            Param(f"{prefix}class_weight", "choice", "balanced", "Class weighting (empty = none).", g,
                  choices=("balanced", "balanced_subsample"), optional=True),
        )  # fmt: skip
    return params


def check_ml(values: Mapping[str, Any]) -> list[str]:
    errors = []
    if values["macd_fast"] >= values["macd_slow"]:
        errors.append("macd_fast must be smaller than macd_slow.")
    if values["training_margin"] < values["horizon"]:
        errors.append("training_margin must be >= horizon (the last labels would see the future).")
    if len(set(values["momentum_windows"])) != len(values["momentum_windows"]):
        errors.append("momentum_windows must be distinct.")
    return errors


def warn_ml(values: Mapping[str, Any]) -> list[str]:
    holding, horizon = values.get("holding_bars"), values["horizon"]
    if holding and holding != horizon:
        return [f"holding_bars ({holding}) differs from the models' horizon ({horizon}): positions are not held "
                "for the period the models predict."]  # fmt: skip
    return []
