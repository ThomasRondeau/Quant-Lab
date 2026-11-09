"""
Hosted mode of the web interface (`QUANTLAB_HOSTED=1`, e.g. on Streamlit Community Cloud).

A public demo runs untrusted input on a small shared machine, so this mode:

- reads data from the bundled samples, uploads or yfinance only, never from a path on the server;
- keeps presets saved by a visitor in that visitor's session (no file written on the server);
- caps the work of one request (bars, strategies, ARIMA test bars, walk-forward runs);
- lets only a few heavy computations run at once (`ComputeSlots`).

Locally, everything stays available and uncapped.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

#: environment variable that turns the hosted mode on (Streamlit Community Cloud: a root-level secret)
HOSTED_ENV = "QUANTLAB_HOSTED"


def hosted() -> bool:
    return os.environ.get(HOSTED_ENV, "").strip().lower() in ("1", "true", "yes", "on")


class LimitError(ValueError):
    """A request exceeds the limits of the hosted demo."""


@dataclass(frozen=True)
class Limits:
    max_bars: int = 12_000
    max_strategies: int = 3
    #: ARIMA refits a model at every bar: its test block is capped
    max_arima_test_bars: int = 200
    #: walk-forward runs (combinations x folds) per strategy; strategies missing here cannot be optimized
    max_optimize_runs: dict[str, int] = field(
        default_factory=lambda: {"trend": 60, "mean_reversion": 60, "pair": 60, "ml_hybrid": 24, "cascade": 4}
    )

    def check_backtest(self, names: Sequence[str], n_bars: int, n_test_bars: int) -> None:
        if len(names) > self.max_strategies:
            raise LimitError(f"The online demo runs at most {self.max_strategies} strategies at once.")
        if n_bars > self.max_bars:
            raise LimitError(f"The online demo is limited to {self.max_bars:,} bars ({n_bars:,} requested).")
        if "arima" in names and n_test_bars > self.max_arima_test_bars:
            raise LimitError(
                f"ARIMA refits a model at every bar: the online demo tests it on at most "
                f"{self.max_arima_test_bars} bars ({n_test_bars} requested). Use a smaller test block."
            )

    def check_optimize(self, name: str, n_candidates: int, n_folds: int, n_bars: int) -> None:
        if n_bars > self.max_bars:
            raise LimitError(f"The online demo is limited to {self.max_bars:,} bars ({n_bars:,} requested).")
        cap = self.max_optimize_runs.get(name)
        if cap is None:
            raise LimitError(f"{name} is too slow to optimize in the online demo; run it locally.")
        runs = n_candidates * n_folds
        if runs > cap:
            raise LimitError(
                f"{n_candidates} combinations x {n_folds} folds = {runs} runs; the online demo allows {cap} for "
                f"{name}. Narrow the grid, or use larger blocks (fewer folds)."
            )


class ComputeSlots:
    """At most `size` heavy computations at once across all visitors; others wait up to `timeout` seconds."""

    def __init__(self, size: int = 2, timeout: float = 90.0) -> None:
        self._semaphore = threading.BoundedSemaphore(size)
        self.timeout = timeout

    @contextmanager
    def acquire(self) -> Iterator[None]:
        if not self._semaphore.acquire(timeout=self.timeout):
            raise LimitError("The demo server is busy with other computations; please try again in a minute.")
        try:
            yield
        finally:
            self._semaphore.release()


def preset_slug(label: str) -> str:
    """A safe preset file name part: lowercase letters, digits, `-` and `_` only (no path)."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", label.strip().lower()).strip("-_")
    return slug[:40]
