"""Atomic JSON persistence of the runner state (written after every bar of a live session)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

#: tries of the final rename before giving up (10 ms apart)
REPLACE_ATTEMPTS = 50


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
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                # Windows refuses to replace a file that a reader (the interface) has open at that instant
                if attempt == REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.01)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        state: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return state
