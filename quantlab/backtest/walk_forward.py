"""
Generic walk-forward hyperparameter optimization for a strategy.

Each fold (DataFrame positions, chronological order):

    [ train ][ embargo ][ validation ][ embargo ][ test ]

For each fold:
  1. every grid combination is trained on `train` and scored on `validation`;
  2. the best combination (by `objective`) is retrained on `train + validation`;
  3. it is scored once on `test`, which was never used to choose the parameters.

The concatenated `test` scores are the out-of-sample estimate. The embargo drops bars between
blocks so that targets built from future prices (forward targets, the training sets' drop_last) cannot
leak into the next block.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.backtest.run import backtest
from quantlab.strategies.base import Strategy

logger = logging.getLogger(__name__)

Params = dict[str, Any]
StrategyFactory = Callable[..., Strategy]
Metrics = dict[str, Any]


@dataclass(frozen=True)
class WalkForwardFold:
    """Block bounds of a fold (iloc positions, end excluded)."""

    index: int
    train: slice
    validation: slice
    test: slice


@dataclass(frozen=True)
class FoldResult:
    fold: WalkForwardFold
    best_params: Params
    validation_score: float
    test_score: float
    test_metrics: Metrics
    grid_scores: list[tuple[Params, float]]


@dataclass
class WalkForwardResult:
    objective: str
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """One row per fold: chosen parameters, validation score and test metrics."""
        rows: list[dict[str, Any]] = []
        for res in self.folds:
            row: dict[str, Any] = {"fold": res.fold.index, **res.best_params}
            row["validation_score"] = res.validation_score
            row["test_score"] = res.test_score
            row.update({f"test_{k}": v for k, v in res.test_metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows).set_index("fold") if rows else pd.DataFrame()

    @property
    def test_scores(self) -> list[float]:
        return [res.test_score for res in self.folds]

    @property
    def mean_test_score(self) -> float:
        valid = [s for s in self.test_scores if math.isfinite(s)]
        return sum(valid) / len(valid) if valid else float("nan")

    def param_stability(self) -> pd.Series:
        """How often each combination was chosen: a combination that changes every fold signals noise."""
        labels = [", ".join(f"{k}={v}" for k, v in res.best_params.items()) for res in self.folds]
        return pd.Series(labels, dtype="object").value_counts()


def expand_grid(param_grid: Mapping[str, Sequence[Any]]) -> list[Params]:
    """Cartesian product of a {name: [values]} grid -> list of parameter dicts."""
    if not param_grid:
        return [{}]
    names = list(param_grid)
    for name in names:
        if len(param_grid[name]) == 0:
            raise ValueError(f"The grid of parameter '{name}' is empty.")
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(param_grid[n] for n in names))]


def make_folds(
    n_bars: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
    anchored: bool = False,
) -> list[WalkForwardFold]:
    """
    Compute the walk-forward folds.

    :param anchored: if True the train block always starts at 0 and grows each fold;
                     otherwise the train window slides with a constant size.
    :param step: shift between two folds (defaults to `test_size`: contiguous test blocks).
    """
    for name, value in (("train_size", train_size), ("validation_size", validation_size), ("test_size", test_size)):
        if value <= 0:
            raise ValueError(f"{name} must be > 0 (got {value}).")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0 (got {embargo}).")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError(f"step must be > 0 (got {step}).")

    folds: list[WalkForwardFold] = []
    k = 0
    while True:
        offset = k * step
        train_start = 0 if anchored else offset
        train_end = train_size + offset
        val_start = train_end + embargo
        val_end = val_start + validation_size
        test_start = val_end + embargo
        test_end = test_start + test_size
        if test_end > n_bars:
            break
        folds.append(
            WalkForwardFold(
                index=k,
                train=slice(train_start, train_end),
                validation=slice(val_start, val_end),
                test=slice(test_start, test_end),
            )
        )
        k += 1
    return folds


@contextlib.contextmanager
def _quiet(enabled: bool) -> Iterator[None]:
    """Silence the per-backtest INFO logs of the engine and strategies."""
    if not enabled:
        yield
        return
    names = ("quantlab.backtest.run", "quantlab.strategies", "quantlab.execution", "quantlab.runtime")
    noisy = [logging.getLogger(name) for name in names]
    previous = [lg.level for lg in noisy]
    for lg in noisy:
        lg.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for lg, level in zip(noisy, previous, strict=True):
            lg.setLevel(level)


class WalkForwardOptimizer:
    """
    Walk-forward search of a strategy's hyperparameters.

    Example:
        optimizer = WalkForwardOptimizer(
            strategy_factory=TrendFollowingStrategy,
            param_grid={"sma_fast": [5, 10], "sma_slow": [20, 50]},
            train_size=4000, validation_size=1000, test_size=1000, embargo=30,
        )
        result = optimizer.run(data)
        print(result.summary())

    `strategy_factory(**params)` must return a *fresh* instance on every call: strategies register
    their models and feature buffer in `fit()` and are not meant to be refitted.
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        param_grid: Mapping[str, Sequence[Any]],
        train_size: int,
        validation_size: int,
        test_size: int,
        objective: str = "total_return_pct",
        step: int | None = None,
        embargo: int = 0,
        anchored: bool = False,
        min_trades: int = 1,
        verbose: bool = False,
    ) -> None:
        self.strategy_factory = strategy_factory
        self.candidates = expand_grid(param_grid)
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.objective = objective
        self.step = step
        self.embargo = embargo
        self.anchored = anchored
        self.min_trades = min_trades
        self.verbose = verbose

    def run(self, data: pd.DataFrame) -> WalkForwardResult:
        folds = make_folds(
            len(data),
            self.train_size,
            self.validation_size,
            self.test_size,
            step=self.step,
            embargo=self.embargo,
            anchored=self.anchored,
        )
        if not folds:
            needed = self.train_size + self.validation_size + self.test_size + 2 * self.embargo
            raise ValueError(f"Not enough data for a single fold: {len(data)} bars, {needed} needed.")

        logger.info("Walk-forward: %d folds x %d combinations", len(folds), len(self.candidates))
        result = WalkForwardResult(objective=self.objective)
        for fold in folds:
            result.folds.append(self._run_fold(data, fold))
        return result

    def _run_fold(self, data: pd.DataFrame, fold: WalkForwardFold) -> FoldResult:
        train = data.iloc[fold.train]
        validation = data.iloc[fold.validation]

        grid_scores: list[tuple[Params, float]] = []
        for params in self.candidates:
            score = self._score(self._backtest(params, train, validation))
            grid_scores.append((params, score))
            logger.debug("Fold %d %s -> %s=%.4f", fold.index, params, self.objective, score)

        best_params, best_score = max(grid_scores, key=lambda item: item[1])
        if not math.isfinite(best_score):
            logger.warning("Fold %d: no valid combination (errors or < %d trades).", fold.index, self.min_trades)

        # Refit on train + validation (embargo between them included), then a single test run
        refit = data.iloc[fold.train.start : fold.validation.stop]
        test = data.iloc[fold.test]
        test_metrics = self._backtest(best_params, refit, test)
        test_score = self._score(test_metrics)
        logger.info(
            "Fold %d: best=%s | validation %s=%.4f | test %s=%.4f",
            fold.index,
            best_params,
            self.objective,
            best_score,
            self.objective,
            test_score,
        )
        return FoldResult(
            fold=fold,
            best_params=best_params,
            validation_score=best_score,
            test_score=test_score,
            test_metrics=test_metrics,
            grid_scores=grid_scores,
        )

    def _backtest(self, params: Params, training_data: pd.DataFrame, backtest_data: pd.DataFrame) -> Metrics:
        """Train a fresh strategy on `training_data` and return the backtest metrics ({} on error)."""
        try:
            with _quiet(not self.verbose):
                metrics = backtest([self.strategy_factory(**params)], training_data, backtest_data).metrics
        except Exception:
            logger.exception("Backtest failed for %s", params)
            return {}
        return dict(metrics) if metrics else {}

    def _score(self, metrics: Metrics) -> float:
        if not metrics or metrics.get("total_trades", 0) < self.min_trades:
            return float("-inf")
        value = metrics.get(self.objective)
        if value is None:
            raise KeyError(f"Metric '{self.objective}' missing from the results; available: {sorted(metrics)}")
        value = float(value)
        return value if math.isfinite(value) else float("-inf")
