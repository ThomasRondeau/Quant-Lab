"""Live-path latency budget: bar received -> orders decided, for the heaviest strategy."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.strategies.cascade import CascadeStrategy

BUDGET_MS = 100.0


@pytest.mark.slow
def test_cascade_on_bar_latency_within_budget() -> None:
    data = pd.read_parquet(Path(__file__).parent / "fixtures" / "aapl_1m.parquet")
    strategy = CascadeStrategy()
    strategy.fit(data.iloc[:3000])
    strategy.start()
    assert not strategy.feed.batch_mode  # the live (bar-by-bar) path

    timings = []
    for _, bar in data.iloc[3000:3100].iterrows():
        start = time.perf_counter()
        strategy.on_bar(bar)
        # Force the full decision path (features + 3 model predictions) even when no opportunity is seen
        row = strategy.feed.row()
        for model in strategy.models.values():
            model.predict_row(row)
        timings.append((time.perf_counter() - start) * 1e3)

    median = float(np.median(timings))
    assert median < BUDGET_MS, f"median on_bar latency {median:.1f} ms exceeds {BUDGET_MS} ms"
