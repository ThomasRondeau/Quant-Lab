"""Streamlit UI: every page renders, and the offline actions work end to end (AppTest, no browser)."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from quantlab.ui import charts

FIXTURES = Path(__file__).parent / "fixtures"


def page_app(page: str) -> AppTest:
    def script(page: str) -> None:
        import quantlab.ui.app as app

        getattr(app, page)()

    return AppTest.from_function(script, args=(page,), default_timeout=120)


def no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("page", ["page_data", "page_backtest", "page_optimize", "page_live"])
def test_every_page_renders(page: str) -> None:
    at = page_app(page).run()
    no_errors(at)


def test_backtest_page_runs_on_a_file(tmp_path: Path) -> None:
    at = page_app("page_backtest").run()
    at.multiselect[0].set_value(["trend", "mean_reversion"]).run()
    at.selectbox(key="bt_source").set_value("file").run()
    at.text_input(key="bt_path").set_value(str(FIXTURES / "aapl_1m.parquet")).run()
    at.radio(key="bt_split").set_value("Last fraction").run()
    at.number_input(key="bt_trend_sma_fast").set_value(8).run()  # generated from the parameter sheet
    next(b for b in at.button if b.label == "Run the backtest").click().run()
    no_errors(at)
    labels = [m.label for m in at.metric]
    assert {"Return", "Trades", "Sharpe"} <= set(labels)
    assert len(at.dataframe) >= 2  # comparison of the runs + trades


def test_forms_are_generated_from_the_parameter_sheets() -> None:
    from quantlab.ui.app import from_widget, grouped, parse_extra, sheet, to_widget

    at = page_app("page_backtest").run()
    at.multiselect[0].set_value(["cascade", "pair"]).run()
    no_errors(at)
    for name in ("cascade", "pair"):
        keys = {w.key for w in [*at.number_input, *at.checkbox, *at.selectbox, *at.text_input] if w.key}
        assert {f"bt_{name}_{p.name}" for p in sheet(name)} <= keys
    assert {"Signal", "Indicators", "Model: direction", "Target", "Sizing", "Exits"} <= {t.label for t in at.tabs}
    assert "bt_pair_stop_atr" not in keys  # no ATR exit for a group

    # an inconsistent value is reported under the form, before running
    at.number_input(key="bt_cascade_training_margin").set_value(3).run()
    assert any("training_margin must be >= horizon" in e.value for e in at.error)
    # a warning, not an error
    at.number_input(key="bt_cascade_training_margin").set_value(30).run()
    at.number_input(key="bt_cascade_horizon").set_value(10).run()
    assert not at.error and any("holding_bars (5) differs" in w.value for w in at.warning)

    stop_atr = next(p for p in sheet("trend") if p.name == "stop_atr")
    assert to_widget(stop_atr, None) == 0.0 and from_widget(stop_atr, 0.0) is None and from_widget(stop_atr, 1.5) == 1.5
    assert list(grouped(sheet("trend")))[:2] == ["indicators", "sizing"]
    assert parse_extra("sma_fast=10" + chr(10) * 2 + "k=1.5") == {"sma_fast": 10, "k": 1.5}
    with pytest.raises(ValueError):
        parse_extra("oops")


def test_presets_load_reset_and_save(tmp_path: Path) -> None:
    import quantlab.ui.app as app
    from quantlab.strategies.params import load_preset, save_preset

    original = app.PRESETS_DIR  # the script below runs in this process and replaces it
    assert original.name == "presets" and original.is_dir()  # the module default: the versioned folder
    save_preset(tmp_path / "trend-court.json", "trend", {"sma_fast": 3, "sma_slow": 12, "stop_atr": 1.5})

    def script(presets: str) -> None:
        from pathlib import Path

        import quantlab.ui.app as app

        app.PRESETS_DIR = Path(presets)
        with app.st.container():
            app.param_inputs("trend", "t")

    at = AppTest.from_function(script, args=(str(tmp_path),), default_timeout=60).run()
    at.selectbox(key="t_trend_preset").set_value("trend-court").run()
    next(b for b in at.button if b.label == "Load").click().run()
    no_errors(at)
    assert at.number_input(key="t_trend_sma_fast").value == 3 and at.number_input(key="t_trend_stop_atr").value == 1.5
    at.text_input(key="t_trend_save_as").set_value("Copie Courte").run()
    next(b for b in at.button if b.label == "Save").click().run()
    assert load_preset(tmp_path / "trend-copie-courte.json") == ("trend", {"sma_fast": 3, "sma_slow": 12,
                                                                           "stop_atr": 1.5})  # fmt: skip
    next(b for b in at.button if b.label == "Defaults").click().run()
    assert at.number_input(key="t_trend_sma_fast").value == 5 and at.number_input(key="t_trend_stop_atr").value == 0
    app.PRESETS_DIR = original


def test_optimization_ranges_build_the_grid() -> None:
    at = page_app("page_optimize").run()
    at.multiselect(key="wf_grid_trend").set_value(["sma_fast", "stop_atr", "size_mode"]).run()
    at.number_input(key="wf_range_trend_sma_fast_min").set_value(3).run()
    at.number_input(key="wf_range_trend_sma_fast_max").set_value(9).run()
    at.number_input(key="wf_range_trend_sma_fast_step").set_value(3).run()
    at.number_input(key="wf_range_trend_stop_atr_min").set_value(1.0).run()
    at.number_input(key="wf_range_trend_stop_atr_max").set_value(2.0).run()
    at.number_input(key="wf_range_trend_stop_atr_step").set_value(0.5).run()
    no_errors(at)
    caption = " ".join(c.value for c in at.caption)
    assert "sma_fast: 3, 6, 9" in caption and "stop_atr: 1.0, 1.5, 2.0" in caption
    assert "size_mode: fixed, notional, equity_pct, risk" in caption


def test_data_page_analyzes_the_chosen_period() -> None:
    def tile(at: AppTest, label: str) -> str:
        return str(next(m.value for m in at.metric if m.label == label))

    at = page_app("page_data").run()  # the landing page: the bundled sample, without a click
    no_errors(at)
    assert tile(at, "Bars") == "5,800" and tile(at, "Sessions") == "30" and tile(at, "Bar interval") == "2 min"
    assert {"Returns", "Intraday profile", "Sessions", "Quality", "Bars"} <= {t.label for t in at.tabs}
    # the slider narrows the period: every figure follows
    last = at.slider[0].value[1]
    at.slider[0].set_range(last - pd.Timedelta(hours=12), last).run()  # the last session
    no_errors(at)
    assert tile(at, "Bars") == "195" and tile(at, "Sessions") == "1"
    # several symbols: one line each, and the correlation of their returns
    at.text_input(key="data_symbols").set_value("SPY, QQQ").run()
    no_errors(at)
    assert at.radio(key="data_symbol").options == ["SPY", "QQQ"]
    assert any(list(d.value.columns) == ["SPY", "QQQ"] for d in at.dataframe)
    # a file, with bars around the clock
    at.text_input(key="data_symbols").set_value("AAPL").run()
    at.selectbox(key="data_source").set_value("file").run()
    assert not at.metric and any("Upload a bars file" in i.value for i in at.info)
    at.text_input(key="data_path").set_value(str(FIXTURES / "aapl_1m.parquet")).run()
    no_errors(at)
    assert tile(at, "Bars") == "5,000" and tile(at, "Bar interval") == "1 min"


def test_dataset_analysis() -> None:
    from quantlab.analysis import dataset

    # two New York sessions of 5-minute bars (13:30 UTC = 09:30 in January), the second one with a hole
    index = pd.DatetimeIndex(
        [*pd.date_range("2026-01-05 14:30", periods=12, freq="5min", tz="UTC"),
         *pd.date_range("2026-01-06 14:30", periods=12, freq="5min", tz="UTC").delete([4, 5])]
    )  # fmt: skip
    close = pd.Series(100.0 + pd.RangeIndex(len(index)), index=index)
    bars = pd.DataFrame({"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close, "volume": 10.0})
    frame = dataset.symbol_bars(bars, ["X"])["X"]
    assert frame.index[0] == pd.Timestamp("2026-01-05 09:30")  # New York wall-clock time

    figures = dataset.summary(frame)
    assert figures["bars"] == 22 and figures["sessions"] == 2 and figures["interval"] == pd.Timedelta(minutes=5)
    assert figures["return_pct"] == pytest.approx(21.0) and figures["session_volume"] == 110.0
    assert len(dataset.intraday_returns(frame)) == 20  # the overnight return is left out

    table = dataset.sessions(frame)
    assert list(table["bars"]) == [12, 10] and table["open"].iloc[0] == 99.5 and table["close"].iloc[0] == 111.0
    assert table["return_pct"].iloc[1] == pytest.approx((121 / 111 - 1) * 100)

    checks = dataset.quality(frame)
    assert checks["missing_bars"] == 2 and checks["incomplete_sessions"] == 1
    assert checks["largest_gap"] == pd.Timedelta(minutes=15) and checks["inconsistent_bars"] == 0
    assert dataset.quality(frame.assign(high=frame["close"] - 2))["inconsistent_bars"] == 22

    profile = dataset.intraday_profile(frame)
    assert list(profile.index[:2]) == ["09:30", "09:35"] and profile["volume"].iloc[0] == 10.0
    assert dataset.return_stats(dataset.intraday_returns(frame))["positive_pct"] == 100.0

    # the multi-symbol layout gives one frame per symbol; a symbol absent from the data is refused
    pair = pd.read_parquet(FIXTURES / "aapl_msft_2m.parquet")
    frames = dataset.symbol_bars(pair, ["AAPL", "MSFT"])
    assert list(frames["MSFT"].columns) == ["open", "high", "low", "close", "volume"]
    assert dataset.return_correlation(frames).loc["AAPL", "MSFT"] == pytest.approx(
        dataset.intraday_returns(frames["AAPL"]).corr(dataset.intraday_returns(frames["MSFT"]))
    )
    with pytest.raises(ValueError, match="No bars for NVDA"):
        dataset.symbol_bars(pair, ["AAPL", "NVDA"])


def test_charts_build_valid_specs() -> None:
    index = pd.date_range("2026-01-05 14:30", periods=10, freq="min", tz="UTC")
    equity = pd.Series(range(100, 110), index=index, dtype=float)
    trades = pd.DataFrame({"pnl_net": [1.0, -2.0], "exit_time": index[:2], "strategy_id": "s", "symbol": "X",
                           "side": "BUY", "reason": "time"})  # fmt: skip
    for chart in (
        charts.equity_chart({"a": equity, "b": equity * 1.01}),
        charts.drawdown_chart(equity),
        charts.trade_pnl_chart(trades),
        charts.value_bars(pd.Series([1.0, 2.0], index=["x", "y"]), "v", "k"),
        charts.time_line(pd.Series(index), equity.reset_index(drop=True), "v"),
        charts.price_chart({"A": equity, "B.C": equity * 2}, "v"),
        charts.volume_chart(equity),
        charts.histogram(equity, "v"),
        charts.signed_bars(equity.diff().fillna(-1.0), "v", "k"),
        charts.value_bars(equity, "v", "k", ticks=[str(index[0])]),
    ):
        assert isinstance(chart, alt.TopLevelMixin)
        chart.to_dict()
    # New York wall-clock times, naive
    assert charts.exchange_time(index)[0] == pd.Timestamp("2026-01-05 09:30")
    # trading-time axis: one tick per session when there are many, per hour otherwise
    days = pd.DatetimeIndex([d + pd.Timedelta(minutes=m) for d in pd.date_range("2026-01-05 09:30", periods=3)
                             for m in (0, 30, 60)])  # fmt: skip
    assert charts._bar_ticks(days) == {0: "Jan 05", 3: "Jan 06", 6: "Jan 07"}
    assert charts._bar_ticks(days[:3]) == {0: "Jan 05 09:30", 1: "10:00"}
