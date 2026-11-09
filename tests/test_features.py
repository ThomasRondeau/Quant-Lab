"""
Live/backtest equivalence guarantees of the feature layer.

- causality: a feature value at bar t never depends on bars after t;
- warmup: `warmup` bars of history are enough for an exact value (this sizes the live buffer);
- parity: the bar-by-bar feed (live path) and the precomputed batch give the same values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.features.base import Feature
from quantlab.features.pipeline import FeaturePipeline
from quantlab.features.technical import ADX, SMA, VWAP, ZScore
from quantlab.runtime.buffer import BarBuffer
from quantlab.runtime.feature_feed import FeatureFeed
from quantlab.strategies.cascade import CascadeStrategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.ml_hybrid import MLHybridStrategy
from quantlab.strategies.pair_trading import PairTradingStrategy
from quantlab.strategies.trend_following import TrendFollowingStrategy

FIXTURES = Path(__file__).parent / "fixtures"

# Exponential averages forget their start only asymptotically (see features.base.EWM_TOLERANCE)
RTOL, ATOL = 1e-7, 1e-9


@pytest.fixture(scope="module")
def aapl() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "aapl_1m.parquet")


@pytest.fixture(scope="module")
def pair() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "aapl_msft_2m.parquet")


PIPELINES: dict[str, tuple[str, list[Feature]]] = {
    "cascade": ("aapl", list(CascadeStrategy().features())),
    # non-default windows of every composite set and of the base indicators
    "cascade_custom": (
        "aapl",
        list(
            CascadeStrategy(
                amplitude_momentum=8,
                momentum_windows=(3, 7, 30),
                slope_window=5,
                atr_mean_window=30,
                atr_change_window=6,
                volume_fast_window=3,
                volume_slow_window=40,
                accel_window=7,
                rsi_window=9,
                bb_window=30,
                macd_fast=8,
                macd_slow=30,
                sma_window=40,
            ).features()
        ),
    ),
    "ml_hybrid": ("aapl", list(MLHybridStrategy().features())),
    "mean_reversion": ("aapl", list(MeanReversionStrategy().features())),
    "trend_following": ("aapl", list(TrendFollowingStrategy(10, 50).features())),
    "pair_trading": ("pair", list(PairTradingStrategy().features())),
    "adx_vwap": ("aapl", [ADX(14), VWAP()]),
}


def _data(name: str, aapl: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    return aapl if PIPELINES[name][0] == "aapl" else pair


def _assert_rows_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    np.testing.assert_allclose(actual.to_numpy(float), expected.to_numpy(float), rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("name", list(PIPELINES))
def test_features_are_causal(name: str, aapl: pd.DataFrame, pair: pd.DataFrame) -> None:
    data = _data(name, aapl, pair).iloc[:1700]
    pipeline = FeaturePipeline(PIPELINES[name][1])
    full = pipeline.compute(data)
    for t in (100, 450, 1000, 1699):
        truncated = pipeline.compute(data.iloc[: t + 1])
        _assert_rows_equal(truncated.iloc[[-1]], full.iloc[[t]])


@pytest.mark.parametrize("name", list(PIPELINES))
def test_declared_warmup_is_enough(name: str, aapl: pd.DataFrame, pair: pd.DataFrame) -> None:
    data = _data(name, aapl, pair)
    pipeline = FeaturePipeline(PIPELINES[name][1])
    full = pipeline.compute(data)
    t = len(data) - 1
    window = data.iloc[t - pipeline.warmup : t + 1]  # exactly `warmup` bars of history + the current bar
    _assert_rows_equal(pipeline.compute(window).iloc[[-1]], full.iloc[[t]])


@pytest.mark.parametrize("name", list(PIPELINES))
def test_bar_by_bar_feed_matches_batch(name: str, aapl: pd.DataFrame, pair: pd.DataFrame) -> None:
    data = _data(name, aapl, pair)
    n_history = min(len(data) // 2, 2500)
    history, upcoming = data.iloc[:n_history], data.iloc[n_history : n_history + 150]
    pipeline = FeaturePipeline(PIPELINES[name][1])

    live = FeatureFeed(pipeline)
    live.warm_start(history)
    batch = FeatureFeed(pipeline)
    batch.warm_start(history)
    batch.precompute(upcoming)

    live_rows, batch_rows = [], []
    for _, bar in upcoming.iterrows():
        live.update(bar)
        batch.update(bar)
        live_rows.append(live.row())
        batch_rows.append(batch.row())
    assert batch.batch_mode and not live.batch_mode

    reference = pipeline.compute(data).iloc[n_history : n_history + 150]
    _assert_rows_equal(pd.DataFrame(live_rows), reference)
    _assert_rows_equal(pd.DataFrame(batch_rows), reference)


def test_feed_leaves_batch_mode_on_unexpected_bar(aapl: pd.DataFrame) -> None:
    feed = FeatureFeed(FeaturePipeline([SMA(5)]))
    feed.warm_start(aapl.iloc[:100])
    feed.precompute(aapl.iloc[100:110])
    feed.update(aapl.iloc[100])
    assert feed.batch_mode
    feed.update(aapl.iloc[105])  # skips bars: not the precomputed path any more
    assert not feed.batch_mode
    expected = aapl["close"].iloc[[*range(97, 101), 105]].mean()
    assert feed.row()["sma_5"] == pytest.approx(expected)


def test_pipeline_resolves_shared_inputs_once_and_sums_warmups() -> None:
    sma = SMA(10)
    pipeline = FeaturePipeline([ZScore(sma, 20), SMA(5, source=sma), sma])
    assert pipeline.columns.count("sma_10") == 1
    assert pipeline.columns[0] == "sma_10"  # inputs first
    assert pipeline.warmup == 9 + 19


def test_pipeline_rejects_name_collisions() -> None:
    class Fake(SMA):
        @property
        def outputs(self) -> tuple[str, ...]:
            return ("a", "b")

    with pytest.raises(ValueError, match="named"):
        FeaturePipeline([SMA(5), Fake(5)])


def test_bar_buffer_keeps_last_bars_in_order() -> None:
    buffer = BarBuffer(["close"], capacity=3)
    times = pd.date_range("2026-01-05 14:30", periods=5, freq="1min", tz="UTC")
    for i, ts in enumerate(times):
        buffer.append(ts, [float(i)])
    frame = buffer.frame()
    assert list(frame["close"]) == [2.0, 3.0, 4.0]
    assert list(frame.index) == list(times[2:])
    assert buffer.is_full and len(buffer) == 3
    with pytest.raises(ValueError, match="increasing"):
        buffer.append(times[0], [9.0])
