"""Hosted demo mode: bundled data, limits, safe presets, no server paths, shared compute slots."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from quantlab.cli import build_parser
from quantlab.data.loading import load_bars, sample_symbols
from quantlab.ui.hosting import ComputeSlots, LimitError, Limits, hosted, market_product, preset_slug


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


def test_hosted_live_page_identifies_the_visitor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import quantlab.ui.app as app

    monkeypatch.setattr(app, "PRESETS_DIR", app.PRESETS_DIR)  # the script below replaces it: restored after
    monkeypatch.setenv("QUANTLAB_HOSTED", "1")
    at = hosted_app("page_live", str(tmp_path)).run()
    assert not at.exception
    visitor = at.query_params["visitor"]
    visitor = visitor if isinstance(visitor, str) else visitor[0]
    assert len(visitor) == 12 and int(visitor, 16) >= 0
    assert at.number_input(key="live_bars").max == 975  # bounded sessions online


def test_market_session_is_off_unless_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUANTLAB_MARKET", raising=False)
    assert market_product() is None
    for value, product in (("1", "SOL-USD"), ("ltc-usd", "LTC-USD"), ("off", None), ("", None)):
        monkeypatch.setenv("QUANTLAB_MARKET", value)
        assert market_product() == product


def test_hosted_market_tab_reads_the_exchange_only_when_asked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pandas as pd

    import quantlab.ui.app as app
    from quantlab.data import market

    calls: list[str] = []

    def candles(product: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        calls.append(product)
        if product != "SOL-USD":
            raise market.MarketDataError("blocked by the exchange")
        row = {"open": 82.4, "high": 82.5, "low": 82.3, "close": 82.45, "volume": 120.0}
        return pd.DataFrame([row], index=pd.DatetimeIndex([end], name="Datetime"))

    monkeypatch.setattr(app, "PRESETS_DIR", app.PRESETS_DIR)  # the script below replaces it: restored after
    monkeypatch.setattr(market, "fetch_candles", candles)
    monkeypatch.setenv("QUANTLAB_HOSTED", "1")
    monkeypatch.delenv("QUANTLAB_MARKET", raising=False)
    app.market_probe.clear()
    at = hosted_app("page_live", str(tmp_path)).run()
    assert not at.exception and not calls and any("off on this server" in i.value for i in at.info)

    monkeypatch.setenv("QUANTLAB_MARKET", "1")
    at = hosted_app("page_live", str(tmp_path)).run()
    assert calls == ["SOL-USD"] and any("SOL-USD closed at 82.45 $" in s.value for s in at.success)

    monkeypatch.setenv("QUANTLAB_MARKET", "DOGE-USD")
    at = hosted_app("page_live", str(tmp_path)).run()
    assert not at.exception and any("blocked by the exchange" in e.value for e in at.error)
    app.market_probe.clear()


def test_hosted_data_page_is_the_landing_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import quantlab.ui.app as app

    monkeypatch.setattr(app, "PRESETS_DIR", app.PRESETS_DIR)  # the script below replaces it: restored after
    monkeypatch.setenv("QUANTLAB_HOSTED", "1")
    at = hosted_app("page_data", str(tmp_path)).run()
    assert not at.exception and not at.error
    assert at.selectbox(key="data_source").value == "sample" and {"Bars", "Sessions"} <= {m.label for m in at.metric}
    at.selectbox(key="data_source").set_value("file").run()
    assert "data_path" not in {w.key for w in at.text_input}  # no path on the server can be typed


def test_requirements_pin_the_declared_dependencies() -> None:
    """`requirements.txt` (what the online demo installs) covers pyproject.toml, at versions it allows."""
    tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")
    requirement = pytest.importorskip("packaging.requirements")

    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = [requirement.Requirement(d) for d in [*project["dependencies"],
                                                     *project["optional-dependencies"]["ui"]]]  # fmt: skip
    lines = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pins = dict(line.split("==") for line in lines if line.strip() and not line.startswith("#"))
    for dependency in declared:
        assert dependency.name in pins, f"{dependency.name} is missing from requirements.txt"
        assert dependency.specifier.contains(pins[dependency.name]), f"{dependency.name}=={pins[dependency.name]}"
