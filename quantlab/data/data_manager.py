"""Historical intraday bars from yfinance (free, no API key)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from quantlab.data.schema import validate_ohlcv

logger = logging.getLogger(__name__)


class DataManager:
    """Window convention `(symbol, days_back, until)`: from `days_back` days ago to `until` days ago."""

    def collect_data_yfinance(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        """2-minute bars (yfinance serves ~60 days of intraday history)."""
        logger.info("Collecting yfinance data for %s from D-%d to D-%d...", symbol, days_back, until)
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() - timedelta(days=until)
        df = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="2m",
            auto_adjust=False,
            progress=False,
        )
        df = validate_ohlcv(df)
        logger.info("yfinance data received: %d bars", len(df))
        return df
