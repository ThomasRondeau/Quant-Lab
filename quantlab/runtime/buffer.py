from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


class BarBuffer:
    """
    Fixed-capacity ring buffer of numeric bars (preallocated, no per-bar DataFrame concat).
    Keeps the most recent `capacity` bars in chronological order.
    """

    def __init__(self, columns: Sequence[str], capacity: int, tz: str | None = "UTC") -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0 (got {capacity}).")
        self.columns: tuple[str, ...] = tuple(columns)
        self.capacity = capacity
        self.tz = tz
        self._values = np.full((capacity, len(self.columns)), np.nan)
        self._times = np.zeros(capacity, dtype="int64")  # epoch nanoseconds
        self._next = 0  # slot written by the next append
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def append(self, timestamp: pd.Timestamp, values: Sequence[float] | npt.NDArray[np.float64]) -> None:
        ts = pd.Timestamp(timestamp)
        if self._size and ts.value <= self._times[(self._next - 1) % self.capacity]:
            raise ValueError(f"Bars must arrive in strictly increasing time order (got {ts}).")
        self._values[self._next] = values
        self._times[self._next] = ts.value
        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def append_bar(self, bar: pd.Series) -> None:
        """Append a bar Series named by its timestamp (as yielded by `DataFrame.iterrows`)."""
        self.append(pd.Timestamp(bar.name), bar[list(self.columns)].to_numpy(dtype=float))  # type: ignore[arg-type]

    def extend(self, bars: pd.DataFrame) -> None:
        """Append many bars; only the last `capacity` are kept."""
        for timestamp, row in zip(
            bars.index[-self.capacity :], bars[list(self.columns)].to_numpy(dtype=float)[-self.capacity :], strict=True
        ):
            self.append(timestamp, row)

    def _order(self) -> npt.NDArray[np.int64]:
        start = (self._next - self._size) % self.capacity
        return (start + np.arange(self._size)) % self.capacity

    def frame(self, last: int | None = None) -> pd.DataFrame:
        """Buffered bars (optionally only the `last` ones) as a DataFrame indexed by timestamp."""
        order = self._order()
        if last is not None:
            order = order[-last:]
        index = pd.DatetimeIndex(pd.to_datetime(self._times[order], utc=True), name="Datetime")
        if self.tz is None:
            index = index.tz_localize(None)
        elif self.tz != "UTC":
            index = index.tz_convert(self.tz)
        return pd.DataFrame(self._values[order], index=index, columns=list(self.columns))

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        if not self._size:
            return None
        return pd.Timestamp(self.frame(last=1).index[-1])
