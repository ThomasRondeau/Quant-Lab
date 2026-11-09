from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from quantlab.targets import ForwardTarget

Array = npt.NDArray[Any]
Score = Callable[[Array, Array], float]


def build_dataset(
    data: pd.DataFrame,
    features: pd.DataFrame,
    columns: Sequence[str],
    target: ForwardTarget,
    drop_last: int = 0,
) -> tuple[Array, Array]:
    """
    Supervised dataset: rows of `features[columns]` paired with `target` computed on `data`.

    :param drop_last: trailing bars to discard (look-ahead safety margin); must be >= the target horizon
                      so that no label relies on bars after the end of `data`.
    Rows with NaN or inf anywhere are dropped.
    """
    if drop_last < target.horizon:
        raise ValueError(f"drop_last ({drop_last}) must be >= the target horizon ({target.horizon}).")
    frame = features[list(columns)].copy()
    frame["__target__"] = target.compute(data)
    frame = frame.iloc[:-drop_last]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    X = frame[list(columns)].to_numpy()
    y = frame["__target__"].to_numpy()
    if target.is_classification:
        y = y.astype(int)
    return X, y


def _n_jobs(estimator: Any) -> Any:
    return estimator.get_params().get("n_jobs") if hasattr(estimator, "get_params") else None


class SklearnModel:
    """
    A scikit-learn compatible estimator bound to its input columns and its target.

    Fitting uses the estimator's configured `n_jobs`; single-row predictions run on one thread,
    because spawning a thread pool per bar costs far more than the prediction itself (live latency).
    """

    def __init__(self, estimator: Any, features: Sequence[str], target: ForwardTarget, score: Score) -> None:
        self.estimator = estimator
        self._fit_n_jobs = _n_jobs(estimator)
        self.features: tuple[str, ...] = tuple(features)
        self.target = target
        self.score = score

    def dataset(self, data: pd.DataFrame, features: pd.DataFrame, drop_last: int) -> tuple[Array, Array]:
        return build_dataset(data, features, self.features, self.target, drop_last=drop_last)

    def _set_n_jobs(self, estimator: Any, n_jobs: Any) -> Any:
        if self._fit_n_jobs is not None:
            estimator.set_params(n_jobs=n_jobs)
        return estimator

    def fit(self, X: Array, y: Array) -> None:
        self._set_n_jobs(self.estimator, self._fit_n_jobs).fit(X, y)
        self._set_n_jobs(self.estimator, 1)

    def predict_row(self, row: pd.Series) -> Any:
        """Prediction for one bar's feature values."""
        x = np.array([[row[c] for c in self.features]], dtype=float)
        return self.estimator.predict(x)[0]

    def in_sample_score(self, X: Array, y: Array) -> float:
        """Score on the training data itself: optimistic, for overfitting diagnosis only."""
        return float(self.score(y, self.estimator.predict(X)))

    def cross_validate(self, X: Array, y: Array, n_splits: int = 5) -> tuple[float, list[float]]:
        """Time-ordered CV (TimeSeriesSplit) of clones of the configured estimator."""
        scores: list[float] = []
        for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
            model = self._set_n_jobs(clone(self.estimator), self._fit_n_jobs)
            model.fit(X[train_idx], y[train_idx])
            scores.append(float(self.score(y[test_idx], model.predict(X[test_idx]))))
        return float(np.mean(scores)), scores
