"""
Support / resistance trend lines drawn between local extrema.

WARNING: NOT CAUSAL. Extrema are detected with a centered window (they use `window` future bars)
and the scan is O(n^3). Use for charting only, never as a model feature or a live trading signal;
this is why these are plain functions and not pipeline `Feature`s.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


def _local_extrema(values: npt.NDArray[np.float64], window: int, kind: str) -> list[tuple[int, float]]:
    pick = min if kind == "min" else max
    return [
        (i, values[i])
        for i in range(window, len(values) - window)
        if values[i] == pick(values[i - window : i + window + 1])
    ]


def _score_lines(extrema: list[tuple[int, float]], min_length: int) -> pd.DataFrame:
    """Every line between two extrema at least `min_length` bars apart, scored by how close inner extrema are."""
    lines: list[dict[str, Any]] = []
    for idx_start, price_start in extrema:
        for idx_end, price_end in extrema:
            if idx_start < (idx_end - min_length):
                slope = (price_end - price_start) / (idx_end - idx_start)
                intercept = price_start - slope * idx_start
                points = {i: slope * i + intercept for i in range(idx_start, idx_end + 1)}
                score = 0.0
                for idx, price in extrema:
                    if idx_start < idx < idx_end:
                        distance = abs(points[idx] - price)
                        score += 1 / (distance * distance)
                lines.append(
                    {
                        "Score": score,
                        "idxStart": idx_start,
                        "idxEnd": idx_end,
                        "PriceStart": price_start,
                        "PriceEnd": price_end,
                        "points": points,
                    }
                )
    if not lines:
        return pd.DataFrame(columns=["Score", "idxStart", "idxEnd", "PriceStart", "PriceEnd", "points"])
    return pd.DataFrame(lines).sort_values(by="Score", ascending=False)


def support_lines(data: pd.DataFrame, window: int = 10, min_length: int = 50) -> pd.DataFrame:
    """Support lines through local lows, best score first (positional bar indices)."""
    return _score_lines(_local_extrema(data["low"].to_numpy(), window, "min"), min_length)


def resistance_lines(data: pd.DataFrame, window: int = 10, min_length: int = 50) -> pd.DataFrame:
    """Resistance lines through local highs, best score first (positional bar indices)."""
    return _score_lines(_local_extrema(data["high"].to_numpy(), window, "max"), min_length)
