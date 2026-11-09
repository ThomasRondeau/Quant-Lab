"""Bar sources for the `StrategyRunner`: a replay of historical bars (a live feed implements the same protocol)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import pandas as pd


class BarFeed(Protocol):
    def __iter__(self) -> Iterator[pd.Series | None]:
        """
        Completed bars (Series named by their timestamp), in time order. A live feed also yields None
        as a heartbeat while waiting, so the runner can process broker events between bars.
        """
        ...


class HistoricalBarFeed:
    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars

    def __iter__(self) -> Iterator[pd.Series | None]:
        for _, bar in self.bars.iterrows():
            yield bar
