import math

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.walk_forward import WalkForwardOptimizer, expand_grid, make_folds
from quantlab.execution.orders import Order, Side
from quantlab.strategies.base import Strategy


class EveryNBarsStrategy(Strategy):
    """Dummy strategy: buys every `period` bars."""

    instances = 0

    def __init__(self, period: int = 10) -> None:
        super().__init__(symbol="TEST")
        self.period = period
        self.trained = False
        self._bar = 0
        EveryNBarsStrategy.instances += 1

    def train(self, history: pd.DataFrame) -> None:
        assert not self.trained, "an instance must be trained only once"
        self.trained = True

    def decide(self, bar: pd.Series) -> list[Order]:
        self._bar += 1
        return [self.order(Side.BUY, 1)] if self._bar % self.period == 0 else []


def make_data(n: int) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.full(n, 0.05))  # uptrend: buying often wins
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=index)


def test_expand_grid() -> None:
    assert expand_grid({}) == [{}]
    assert expand_grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
    with pytest.raises(ValueError):
        expand_grid({"a": []})


def test_rolling_folds_are_ordered_and_embargoed() -> None:
    folds = make_folds(100, train_size=40, validation_size=10, test_size=10, embargo=2)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train.stop - fold.train.start == 40
        assert fold.validation.start == fold.train.stop + 2
        assert fold.test.start == fold.validation.stop + 2
        assert fold.test.stop <= 100
    assert [f.test.start for f in folds] == [54, 64, 74, 84]


def test_anchored_folds_grow_train() -> None:
    folds = make_folds(100, train_size=40, validation_size=10, test_size=10, anchored=True)
    assert all(f.train.start == 0 for f in folds)
    assert [f.train.stop for f in folds] == [40, 50, 60, 70, 80][: len(folds)]


def test_not_enough_data() -> None:
    assert make_folds(50, 40, 10, 10) == []
    optimizer = WalkForwardOptimizer(EveryNBarsStrategy, {}, train_size=40, validation_size=10, test_size=10)
    with pytest.raises(ValueError):
        optimizer.run(make_data(50))


def test_optimizer_selects_best_params_and_uses_fresh_instances() -> None:
    EveryNBarsStrategy.instances = 0
    optimizer = WalkForwardOptimizer(
        EveryNBarsStrategy,
        {"period": [3, 50]},
        train_size=100,
        validation_size=60,
        test_size=60,
        embargo=5,
        objective="total_return_pct",
    )
    result = optimizer.run(make_data(400))

    n_folds = len(result.folds)
    assert n_folds > 0
    # 2 combinations evaluated + 1 refit per fold, each on a fresh instance.
    assert EveryNBarsStrategy.instances == n_folds * 3
    for fold_result in result.folds:
        assert fold_result.best_params == {"period": 3}
        assert math.isfinite(fold_result.test_score)
    summary = result.summary()
    assert list(summary["period"]) == [3] * n_folds
    assert result.param_stability().iloc[0] == n_folds
