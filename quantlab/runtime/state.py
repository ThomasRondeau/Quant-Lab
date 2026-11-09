"""Atomic JSON persistence of the runner state (written after every bar of a live session)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, state: dict[str, Any]) -> None:
        """Write to a temporary file then rename it: a crash never leaves a half-written state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        state: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return state
