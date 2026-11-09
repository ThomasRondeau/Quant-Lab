"""Regression tests for the audit bugs A1-A12 (see the refactor plan). One block per bug."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import quantlab.data.data_manager as data_manager_module
import quantlab.models.arima as arima_module
import quantlab.models.sklearn_model as sklearn_model_module
from quantlab.data.data_manager import DataManager
from quantlab.data.schema import validate_ohlcv
from quantlab.execution.orders import Side
from quantlab.features.technical import ADX, MACD, VWAP
from quantlab.strategies.arima import ArimaStrategy
from quantlab.strategies.cascade import CascadeStrategy
from quantlab.strategies.ml_hybrid import MLHybridStrategy
from quantlab.strategies.pair_trading import PairTradingStrategy
from quantlab.strategies.trend_following import TrendFollowingStrategy
from quantlab.targets import ForwardTarget

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def aapl() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "aapl_1m.parquet")


def ohlcv(n: int, start: str = "2026-01-05 14:30", freq: str = "1min", tz: str | None = "UTC") -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq=freq, tz=tz)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": rng.integers(100, 1000, n),
        },
        index=index,
    )


# --- A4: one OHLCV contract -------------------------------------------------------------------------------


def test_validate_ohlcv_normalizes_yfinance_shape() -> None:
    raw = ohlcv(5, tz="America/New_York").rename(columns=str.title)
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])
    raw = raw.iloc[[0, 2, 1, 3, 4, 4]]  # unsorted + duplicate

    out = validate_ohlcv(raw)

    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "Datetime"
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert (out.dtypes == np.float64).all()


def test_validate_ohlcv_naive_is_utc_and_reads_datetime_column() -> None:
    raw = ohlcv(3, tz=None).reset_index(names="Datetime")
    out = validate_ohlcv(raw, datetime_column="Datetime")
    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2026-01-05 14:30", tz="UTC")


def test_validate_ohlcv_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="volume"):
        validate_ohlcv(ohlcv(3).drop(columns="volume"))


def test_yfinance_keeps_datetime_index_and_date_convention(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_download(symbol: str, **kwargs: Any) -> pd.DataFrame:
        calls.update(kwargs)
        return ohlcv(10, tz="America/New_York").rename(columns=str.title)

    monkeypatch.setattr(data_manager_module.yf, "download", fake_download)
    df = DataManager().collect_data_yfinance("AAPL", days_back=50, until=5)

    assert isinstance(df.index, pd.DatetimeIndex)  # used to be a RangeIndex -> Sharpe on 1970 dates
    today = datetime.now()
    assert calls["start"] == (today - timedelta(days=50)).strftime("%Y-%m-%d")
    assert calls["end"] == (today - timedelta(days=5)).strftime("%Y-%m-%d")


# --- A1: MACD windows --------------------------------------------------------------------------------------


def test_macd_rejects_inverted_windows() -> None:
    with pytest.raises(ValueError):
        MACD(short_window=26, long_window=12)


def test_strategies_use_macd_12_26() -> None:
    for macd in (CascadeStrategy().amplitude_features.macd, MLHybridStrategy().direction_features.macd):
        assert (macd.short_window, macd.long_window) == (12, 26)


# --- A2: ADX alignment -------------------------------------------------------------------------------------


def test_adx_keeps_data_index() -> None:
    data = ohlcv(100)
    adx = ADX(window=14).compute(data, {})
    assert adx.index.equals(data.index)
    assert adx.notna().sum() > 0
    assert adx.dropna().between(0, 100).all()


# --- A9: VWAP resets per session ---------------------------------------------------------------------------


def test_vwap_resets_each_session() -> None:
    day1 = ohlcv(30, start="2026-01-05 14:30")
    day2 = ohlcv(30, start="2026-01-06 14:30")
    data = pd.concat([day1, day2])
    vwap = VWAP().compute(data, {})

    typical = (data["close"] + data["high"] + data["low"]) / 3
    first_bar_day2 = day2.index[0]
    assert vwap.loc[first_bar_day2, "vwap"] == pytest.approx(typical[first_bar_day2])
    assert vwap.loc[first_bar_day2, "vwap_upper_3"] == pytest.approx(typical[first_bar_day2])  # zero dispersion


# --- A3: ARIMA ---------------------------------------------------------------------------------------------


class FakeArima:
    orders: ClassVar[list[tuple[int, int, int]]] = []
    forecast_delta = 0.0

    def __init__(self, series: np.ndarray, order: tuple[int, int, int]) -> None:
        FakeArima.orders.append(order)
        self.last = float(series[-1])

    def fit(self) -> FakeArima:
        return self

    def forecast(self) -> np.ndarray:
        return np.array([self.last + FakeArima.forecast_delta])


def _arima_orders(monkeypatch: pytest.MonkeyPatch, delta: float, aapl: pd.DataFrame) -> list[Any]:
    FakeArima.orders = []
    FakeArima.forecast_delta = delta
    monkeypatch.setattr(arima_module, "ARIMA", FakeArima)
    strategy = ArimaStrategy(order=(3, 1, 1))
    strategy.fit(aapl.iloc[:200])
    strategy.start()
    orders = [o for _, bar in aapl.iloc[200:210].iterrows() for o in strategy.on_bar(bar)]
    assert set(FakeArima.orders) == {(3, 1, 1)}  # order used to be ignored
    return orders


def test_arima_does_not_trade_without_edge(monkeypatch: pytest.MonkeyPatch, aapl: pd.DataFrame) -> None:
    assert _arima_orders(monkeypatch, delta=1e-6, aapl=aapl) == []  # used to trade every bar


def test_arima_trades_when_edge_exceeds_costs(monkeypatch: pytest.MonkeyPatch, aapl: pd.DataFrame) -> None:
    orders = _arima_orders(monkeypatch, delta=1.0, aapl=aapl)
    assert len(orders) == 10
    assert {o.side for o in orders} == {Side.BUY}


# --- A7: cross-validation uses the configured estimator ----------------------------------------------------


def test_cross_validate_clones_configured_model(monkeypatch: pytest.MonkeyPatch, aapl: pd.DataFrame) -> None:
    depths: list[int] = []
    real_clone = sklearn_model_module.clone

    def spy_clone(estimator: Any) -> Any:
        depths.append(estimator.get_params()["max_depth"])
        return real_clone(estimator)

    monkeypatch.setattr(sklearn_model_module, "clone", spy_clone)
    strategy = MLHybridStrategy(max_depth=7)
    strategy.fit(aapl.iloc[:600])
    assert strategy._dataset is not None
    strategy.direction_model.cross_validate(*strategy._dataset, n_splits=2)
    assert depths == [7, 7]  # used to be a hard-coded max_depth=4


# --- A8: opportunity threshold -----------------------------------------------------------------------------


def test_opportunity_threshold_is_used() -> None:
    data = pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.1, 100.0, 100.0, 100.0]})
    assert ForwardTarget(3, "move", threshold_pct=0.05).compute(data).iloc[0] == 1  # +0.1 % > 0.05 %
    assert ForwardTarget(3, "move", threshold_pct=0.5).compute(data).iloc[0] == 0  # +0.1 % < 0.5 %


# --- A10: trend-following feature names --------------------------------------------------------------------


def test_trend_following_features_match_windows() -> None:
    strategy = TrendFollowingStrategy(sma_fast=10, sma_slow=50)
    assert [f.name for f in strategy.features()] == ["sma_10", "sma_50"]


# --- A11: bounded history ----------------------------------------------------------------------------------


def test_pair_trading_history_is_bounded() -> None:
    pair = pd.read_parquet(FIXTURES / "aapl_msft_2m.parquet")
    strategy = PairTradingStrategy()
    strategy.fit(pair.iloc[:200])
    strategy.start()
    for _, bar in pair.iloc[200:600].iterrows():
        strategy.on_bar(bar)
    assert strategy.feed.buffer is not None
    assert len(strategy.feed.buffer) == strategy.feed.capacity < 200
