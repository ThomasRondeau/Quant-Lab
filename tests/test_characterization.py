"""
Characterization tests: freeze the current backtest metrics of every strategy on fixed data.

They guard the refactoring steps: a pure restructuring must reproduce these numbers exactly.
Every case runs twice, with the features precomputed (backtest batch mode) and bar by bar (the live
path): both must match the same baseline, which is the strategy-level parity test.
When a change is *meant* to alter results (bug fix, engine change), regenerate the baseline
deliberately and review the diff:

    UPDATE_BASELINE=1 python -m pytest tests/test_characterization.py
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quantlab.backtest.run import backtest
from quantlab.strategies.arima import ArimaStrategy
from quantlab.strategies.base import Strategy
from quantlab.strategies.cascade import CascadeStrategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.ml_hybrid import MLHybridStrategy
from quantlab.strategies.pair_trading import PairTradingStrategy
from quantlab.strategies.trend_following import TrendFollowingStrategy

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE_PATH = FIXTURES / "baseline_metrics.json"
UPDATE = os.environ.get("UPDATE_BASELINE") == "1"


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / name)


# name -> (fixture, factory, train_bars, backtest_bars)
CASES: dict[str, tuple[str, Callable[[], Strategy], int, int]] = {
    "trend_following": ("aapl_1m.parquet", TrendFollowingStrategy, 4000, 1000),
    "mean_reversion": ("aapl_1m.parquet", MeanReversionStrategy, 4000, 1000),
    "ml_hybrid": ("aapl_1m.parquet", MLHybridStrategy, 4000, 1000),
    "cascade": ("aapl_1m.parquet", CascadeStrategy, 4000, 1000),
    "arima": ("aapl_1m.parquet", ArimaStrategy, 4000, 200),  # ARIMA refits every bar: keep it short
    "pair_trading": ("aapl_msft_2m.parquet", PairTradingStrategy, 1500, 450),
}


def _run(case: str, precompute: bool) -> dict[str, Any]:
    fixture, factory, n_train, n_test = CASES[case]
    data = _load(fixture)
    result = backtest(
        [factory()], data.iloc[:n_train], data.iloc[n_train : n_train + n_test], precompute_features=precompute
    )
    return {k: float(v) for k, v in result.metrics.items()}


def _read_baseline() -> dict[str, dict[str, float]]:
    if BASELINE_PATH.exists():
        result: dict[str, dict[str, float]] = json.loads(BASELINE_PATH.read_text())
        return result
    return {}


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["batch", "bar_by_bar"])
@pytest.mark.parametrize("case", list(CASES))
def test_metrics_match_baseline(case: str, mode: str) -> None:
    metrics = _run(case, precompute=mode == "batch")
    baseline = _read_baseline()

    if UPDATE:
        if mode != "batch":
            pytest.skip("the baseline is written from the batch run; the bar-by-bar run must then match it")
        baseline[case] = metrics
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        return

    assert case in baseline, f"No baseline for '{case}'; run with UPDATE_BASELINE=1 first."
    expected = baseline[case]
    assert set(metrics) == set(expected)
    for key, value in expected.items():
        actual = metrics[key]
        if math.isnan(value):
            assert math.isnan(actual), key
        else:
            assert actual == pytest.approx(value, rel=1e-9, abs=1e-12), (key, mode)
