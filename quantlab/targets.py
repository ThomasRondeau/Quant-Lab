"""
Forward-looking labels for supervised learning.

A label at bar t uses closes after t by construction, so it must only be paired with features
of bars whose label window is fully *in the past* of the data the model is trained on. The
training-set builder (`quantlab.models.sklearn_model.build_dataset`) drops the trailing rows for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

TargetKind = Literal["price", "return_pct", "direction", "move"]


@dataclass(frozen=True)
class ForwardTarget:
    """
    Label built from the close `horizon` bars ahead:

    - ``price``: the future close;
    - ``return_pct``: signed move in percent, (close[t+h] - close[t]) / close[t] * 100;
    - ``direction``: 1 if close[t+h] > close[t], else 0;
    - ``move``: 1 if |return_pct| > `threshold_pct` (percent: 0.05 = 0.05 %), else 0.
    """

    horizon: int
    kind: TargetKind
    threshold_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0 (got {self.horizon}).")

    @property
    def name(self) -> str:
        suffix = f"_{self.threshold_pct:g}" if self.kind == "move" else ""
        return f"target_{self.kind}_{self.horizon}{suffix}"

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        future = close.shift(-self.horizon)
        if self.kind == "price":
            label = future
        elif self.kind == "return_pct":
            label = ((future - close) / close) * 100
        elif self.kind == "direction":
            label = (future > close).astype(int)
        elif self.kind == "move":
            label = ((((future - close) / close) * 100).abs() > self.threshold_pct).astype(int)
        else:
            raise ValueError(f"Unknown target kind {self.kind!r}")
        # The last `horizon` bars have no future close: their label is unknown, never 0
        return label.where(future.notna()).rename(self.name)

    @property
    def is_classification(self) -> bool:
        return self.kind in ("direction", "move")
