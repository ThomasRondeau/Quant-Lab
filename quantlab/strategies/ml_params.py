"""Parameter blocks shared by the ML strategies: indicators, target, and estimator hyperparameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quantlab.strategies.params import Param


def indicator_params(volume_window: int) -> tuple[Param, ...]:
    """Windows of the base indicators the composite feature sets are built on."""
    g = "indicateurs"
    return (
        Param("rsi_window", "int", 14, "Fenêtre du RSI.", g, min=2, max=1000),
        Param("bb_window", "int", 20, "Fenêtre des bandes de Bollinger.", g, min=2, max=1000),
        Param("bb_std", "float", 2.0, "Écart des bandes, en écarts-types.", g, min=0.1, max=10),
        Param("volume_window", "int", volume_window, "Fenêtre du ratio de volume.", g, min=1, max=1000),
        Param("feature_atr_window", "int", 14, "Fenêtre de l'ATR des features.", g, min=1, max=1000),
        Param("sma_window", "int", 20, "Fenêtre de la moyenne mobile des features.", g, min=1, max=1000),
        Param("macd_fast", "int", 12, "MACD : fenêtre rapide.", g, min=1, max=500),
        Param("macd_slow", "int", 26, "MACD : fenêtre lente.", g, min=2, max=1000),
        Param("macd_signal", "int", 9, "MACD : fenêtre du signal.", g, min=1, max=500),
        Param("momentum_windows", "tuple", (5, 10, 20), "Momentums du modèle de direction (barres).", g, min=1,
              max=1000),
        Param("slope_window", "int", 3, "Pente de la moyenne mobile (barres).", g, min=1, max=500),
    )  # fmt: skip


def target_params(cv_folds: bool = True) -> tuple[Param, ...]:
    g = "cible"
    params: tuple[Param, ...] = (
        Param("horizon", "int", 5, "Barres à l'avance prédites par les modèles.", g, min=1, max=1000),
        Param("training_margin", "int", 30, "Barres retirées à la fin de l'entraînement (anti look-ahead, ≥ horizon).",
              g, min=1, max=10_000),
        Param("random_state", "int", 42, "Graine aléatoire des modèles.", g, min=0, max=2**31 - 1),
    )  # fmt: skip
    if cv_folds:
        params += (Param("cv_folds", "int", 5, "Plis de la validation croisée temporelle.", g, min=2, max=50),)
    return params


def xgb_params(prefix: str, model: str, max_depth: int = 4) -> tuple[Param, ...]:
    g = f"modèle:{model}"
    return (
        Param(f"{prefix}n_estimators", "int", 100, "Nombre d'arbres.", g, min=1, max=10_000),
        Param(f"{prefix}max_depth", "int", max_depth, "Profondeur maximale des arbres.", g, min=1, max=20),
        Param(f"{prefix}learning_rate", "float", 0.05, "Taux d'apprentissage.", g, min=1e-4, max=1),
        Param(f"{prefix}subsample", "float", 0.8, "Fraction des barres tirée par arbre.", g, min=0.05, max=1),
        Param(f"{prefix}colsample_bytree", "float", 0.8, "Fraction des features tirée par arbre.", g, min=0.05, max=1),
    )


def forest_params(prefix: str, model: str, min_samples_leaf: int, class_weight: bool = False) -> tuple[Param, ...]:
    g = f"modèle:{model}"
    params: tuple[Param, ...] = (
        Param(f"{prefix}n_estimators", "int", 100, "Nombre d'arbres.", g, min=1, max=10_000),
        Param(f"{prefix}max_depth", "int", 5, "Profondeur maximale des arbres.", g, min=1, max=50),
        Param(f"{prefix}min_samples_leaf", "int", min_samples_leaf, "Barres minimales par feuille.", g, min=1,
              max=10_000),
    )  # fmt: skip
    if class_weight:
        params += (
            Param(f"{prefix}class_weight", "choice", "balanced", "Pondération des classes (vide = aucune).", g,
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
