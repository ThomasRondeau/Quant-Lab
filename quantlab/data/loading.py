"""Bar loading and train / test split shared by the CLI and the UI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from quantlab.data.pairs import combine_symbols
from quantlab.data.schema import validate_ohlcv

SOURCES = ("sample", "yfinance", "file")
#: bundled 2-minute bars (offline demo data): one parquet file per symbol
SAMPLES_DIR = Path(__file__).parent / "samples"


def sample_symbols() -> list[str]:
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.parquet"))


def read_sample(symbol: str, days: int) -> pd.DataFrame:
    """Bundled bars of `symbol`, limited to the last `days` calendar days of the sample."""
    path = SAMPLES_DIR / f"{symbol.upper()}.parquet"
    if not path.exists():
        raise ValueError(f"No bundled sample for {symbol!r}; available: {', '.join(sample_symbols())}.")
    bars = validate_ohlcv(pd.read_parquet(path))
    recent: pd.DataFrame = bars[bars.index >= bars.index[-1] - pd.Timedelta(days=days)]
    return recent


def read_bars_file(path: str | Path, symbols: Sequence[str] = ()) -> pd.DataFrame:
    """Bars from a parquet / csv file; multi-symbol files (`close_<SYMBOL>` columns) are returned as is."""
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if symbols and any(str(c).endswith(f"_{symbols[-1]}") for c in raw.columns):
        return raw
    return validate_ohlcv(raw)


def load_bars(symbols: Sequence[str], source: str, days: int, file: str | Path | None = None) -> pd.DataFrame:
    """Bars of `symbols` over the last `days` calendar days (several symbols: the multi-symbol layout)."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; available: {', '.join(SOURCES)}.")
    if source == "file":
        if not file:
            raise ValueError("The file source needs a file path.")
        return read_bars_file(file, symbols)
    if source == "sample":
        return combine_symbols({symbol: read_sample(symbol, days) for symbol in symbols})

    from quantlab.data.data_manager import DataManager

    manager = DataManager()
    frames = {symbol: manager.collect_data_yfinance(symbol, days, 0) for symbol in symbols}
    for symbol, frame in frames.items():
        if frame.empty:
            raise ValueError(f"No {source} data for {symbol}.")
    return combine_symbols(frames)


def split(data: pd.DataFrame, test_days: float, test_fraction: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training / test blocks: the last `test_fraction` of the bars, or the last `test_days` calendar days."""
    if test_fraction is not None:
        cut = int(len(data) * (1 - test_fraction))
        return data.iloc[:cut], data.iloc[cut:]
    boundary = data.index[-1] - pd.Timedelta(days=test_days)
    return data[data.index <= boundary], data[data.index > boundary]
