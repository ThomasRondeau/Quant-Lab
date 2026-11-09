"""
Files of a live trading session, in one directory:

    session.json      settings of the session (mode, symbols, capital, risk limits, simulator costs)
    strategies.pkl    the strategies right after `fit`, before the first live bar (models + feature buffer)
    warmup.parquet    the history they were fitted on
    journal.sqlite    everything that happened (`Journal`)
    state.json        the runner state after the last bar (`StateStore`), to resume after a restart

`strategies.pkl` is what makes a session reproducible: a restart reloads the same models, and
`analysis.replay` re-runs the same strategies on the journaled bars. Only load pickles you wrote.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.strategies.base import Strategy


class SessionFiles:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta_path = self.root / "session.json"
        self.strategies_path = self.root / "strategies.pkl"
        self.warmup_path = self.root / "warmup.parquet"
        self.journal_path = self.root / "journal.sqlite"
        self.state_path = self.root / "state.json"

    def exists(self) -> bool:
        return self.strategies_path.exists()

    def save_snapshot(self, strategies: Sequence[Strategy], warmup: pd.DataFrame, meta: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.strategies_path.open("wb") as f:
            pickle.dump(list(strategies), f, protocol=pickle.HIGHEST_PROTOCOL)
        warmup.to_parquet(self.warmup_path)
        self.meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")

    def load_strategies(self) -> list[Strategy]:
        with self.strategies_path.open("rb") as f:
            strategies: list[Strategy] = pickle.load(f)  # our own session file
        return strategies

    def load_warmup(self) -> pd.DataFrame:
        return pd.read_parquet(self.warmup_path)

    def meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return meta
