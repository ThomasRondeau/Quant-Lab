"""Hosted demo mode: bundled data, limits, safe presets, no server paths, shared compute slots."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from quantlab.cli import build_parser
from quantlab.data.loading import load_bars, sample_symbols
from quantlab.ui.hosting import ComputeSlots, LimitError, Limits, hosted, preset_slug


def test_bundled_samples_load_offline() -> None:
    assert {"AAPL", "MSFT", "NVDA", "SPY", "QQQ"} <= set(sample_symbols())
    bars = load_bars(["AAPL"], "sample", 55)
    assert len(bars) > 4000 and str(bars.index.tz) == "UTC"
    pair = load_bars(["SPY", "QQQ"], "sample", 55)
    assert {"close_SPY", "close_QQQ", "close"} <= set(pair.columns)
    assert len(load_bars(["AAPL"], "sample", 3)) < len(bars)  # the window is honored
    with pytest.raises(ValueError, match="No bundled sample"):
        load_bars(["TSLA"], "sample", 55)
    assert build_parser().parse_args(["backtest", "--strategy", "trend", "--symbol", "AAPL"]).source == "sample"


def test_hosted_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUANTLAB_HOSTED", raising=False)
    assert not hosted()
    monkeypatch.setenv("QUANTLAB_HOSTED", "1")
    assert hosted()


def test_limits() -> None:
    limits = Limits()
    limits.check_backtest(["trend", "cascade"], 5800, 900)
    with pytest.raises(LimitError, match="at most 3 strategies"):
        limits.check_backtest(["trend", "mean_reversion", "pair", "cascade"], 5800, 900)
    with pytest.raises(LimitError, match="12,000 bars"):
        limits.check_backtest(["trend"], 20_000, 900)
    with pytest.raises(LimitError, match="ARIMA"):
        limits.check_backtest(["arima"], 5800, 900)
    limits.check_backtest(["arima"], 5800, 150)
    limits.check_optimize("trend", 12, 4, 5800)
    with pytest.raises(LimitError, match="allows 4 for cascade"):
        limits.check_optimize("cascade", 3, 2, 5800)
    with pytest.raises(LimitError, match="too slow to optimize"):
        limits.check_optimize("arima", 1, 1, 5800)


def test_preset_slugs_cannot_escape_the_folder() -> None:
    assert preset_slug("Wide Stops") == "wide-stops"
    assert preset_slug("../../etc/passwd") == "etc-passwd"
    assert preset_slug("..\\x") == "x"
    assert preset_slug("///") == ""


def test_compute_slots_limit_concurrency() -> None:
    slots = ComputeSlots(size=1, timeout=0.05)
    with slots.acquire():
        outcome: list[str] = []

        def other() -> None:
            try:
                with slots.acquire():
                    outcome.append("ran")
            except LimitError:
                outcome.append("busy")

        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
    assert outcome == ["busy"]
    with slots.acquire():  # released
        pass


def hosted_app(page: str, presets: str) -> AppTest:
    def script(page: str, presets: str) -> None:
        from pathlib import Path

        import quantlab.ui.app as app

        app.PRESETS_DIR = Path(presets)
        getattr(app, page)()

    return AppTest.from_function(script, args=(page, presets), default_timeout=120)


def test_hosted_backtest_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import quantlab.ui.app as app

    original = app.PRESETS_DIR
    monkeypatch.setenv("QUANTLAB_HOSTED", "1")
    at = hosted_app("page_backtest", str(tmp_path)).run()
    assert at.selectbox(key="bt_source").value == "sample"
    # the default run works on the bundled data
    next(b for b in at.button if b.label == "Run the backtest").click().run()
    assert not at.exception and {"Return", "Trades"} <= {m.label for m in at.metric}
    # no path on the server can be typed
    at.selectbox(key="bt_source").set_value("file").run()
    assert "bt_path" not in {w.key for w in at.text_input}
    # presets stay in the visitor's session
    at.selectbox(key="bt_source").set_value("sample").run()
    at.text_input(key="bt_trend_save_as").set_value("../../escape").run()
    next(b for b in at.button if b.label == "Save").click().run()
    assert list(tmp_path.iterdir()) == [] and not list(original.parent.glob("escape*"))
    at.run()  # the preset list is drawn above the Save button: it shows the new preset on the next run
    assert "trend-escape" in at.selectbox(key="bt_trend_preset").options
    # a request over the limits is refused with a message, not an error
    at.multiselect[0].set_value(["arima"]).run()
    next(b for b in at.button if b.label == "Run the backtest").click().run()
    assert not at.exception and any("ARIMA refits a model" in w.value for w in at.warning)
    app.PRESETS_DIR = original
