"""
Serves feature values to a strategy, bar by bar, identically in backtest and live.

- Incremental mode (live, and the reference): every bar is appended to a `BarBuffer` sized from the
  pipeline warmup, and the features are computed on that buffer.
- Batch mode (backtest acceleration only): `precompute(upcoming_bars)` computes the features of all
  upcoming bars at once; `update` then just advances a cursor. Because every feature is causal and the
  buffer covers the warmup, both modes give the same values (enforced by the parity tests). A bar that
  does not match the precomputed data switches the feed back to incremental mode.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quantlab.features.pipeline import FeaturePipeline
from quantlab.runtime.buffer import BarBuffer

# Extra bars kept beyond the warmup (also covers strategies reading a few past feature rows).
DEFAULT_MARGIN = 32


class FeatureFeed:
    def __init__(self, pipeline: FeaturePipeline, min_history: int = 0, capacity: int | None = None) -> None:
        """
        :param min_history: raw bars the strategy itself reads through `bars(n)` (e.g. an ARIMA window).
        :param capacity: buffer size; defaults to what the features and `min_history` need, plus a margin.
        """
        if not pipeline.causal:
            raise ValueError("Only causal features can be served bar by bar.")
        self.pipeline = pipeline
        needed = max(pipeline.warmup + 1, min_history)
        self.capacity = capacity if capacity is not None else needed + DEFAULT_MARGIN
        if self.capacity < needed:
            raise ValueError(f"capacity {self.capacity} is below the {needed} bars the features need.")
        self.buffer: BarBuffer | None = None
        self._latest: pd.DataFrame | None = None  # cache of the incremental computation
        self._batch: pd.DataFrame | None = None
        self._cursor = -1

    # --- feeding ---------------------------------------------------------------------------------------

    def warm_start(self, history: pd.DataFrame, columns: Sequence[str] | None = None) -> None:
        """Initialize the buffer with the tail of `history` (numeric columns by default)."""
        if columns is None:
            columns = [c for c in history.columns if pd.api.types.is_numeric_dtype(history[c])]
        tz = str(history.index.tz) if isinstance(history.index, pd.DatetimeIndex) and history.index.tz else None
        self.buffer = BarBuffer(columns, self.capacity, tz=tz)
        self.buffer.extend(history)
        self._latest, self._batch, self._cursor = None, None, -1

    def update(self, bar: pd.Series) -> None:
        buffer = self._require_buffer()
        buffer.append_bar(bar)
        self._latest = None
        if self._batch is not None:
            nxt = self._cursor + 1
            if nxt < len(self._batch) and self._batch.index[nxt] == bar.name:
                self._cursor = nxt
            else:
                self._batch = None  # off the precomputed path: back to incremental

    def precompute(self, upcoming: pd.DataFrame) -> None:
        """Batch mode: compute the features of the buffered history plus all `upcoming` bars at once."""
        buffer = self._require_buffer()
        history = buffer.frame()
        frame = pd.concat([history, upcoming[list(buffer.columns)].astype(float)])
        self._batch = self.pipeline.compute(frame)
        self._cursor = len(history) - 1

    # --- reading ---------------------------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once the buffer holds enough history for exact feature values."""
        return self.buffer is not None and len(self.buffer) > self.pipeline.warmup

    @property
    def batch_mode(self) -> bool:
        return self._batch is not None

    def latest(self, n: int = 1) -> pd.DataFrame:
        """Feature rows of the last `n` bars (oldest first)."""
        if self._batch is not None:
            return self._batch.iloc[self._cursor - n + 1 : self._cursor + 1]
        if self._latest is None or len(self._latest) < n:
            self._latest = self.pipeline.compute(self._require_buffer().frame())
        return self._latest.iloc[-n:]

    def row(self) -> pd.Series:
        """Feature values of the current bar."""
        return self.latest(1).iloc[0]

    def bars(self, n: int) -> pd.DataFrame:
        """The last `n` raw bars."""
        return self._require_buffer().frame(last=n)

    def _require_buffer(self) -> BarBuffer:
        if self.buffer is None:
            raise RuntimeError("FeatureFeed: call warm_start(history) first.")
        return self.buffer
