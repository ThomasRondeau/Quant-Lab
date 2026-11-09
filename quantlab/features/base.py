"""
Stateless features.

A feature is a pure function of past bars: `compute(data, computed)` returns its output column(s)
for every row of `data`, using only rows up to each row (causal). It declares:

- `name` / `outputs`: unique column names, derived from the parameters (`sma_20`, `macd_hist_12_26_9`);
  two instances with the same name must compute the same thing;
- `inputs()`: the features it reads from `computed` (resolved and computed first by the pipeline);
- `warmup`: bars of its *own* history needed before its values are exact (the pipeline adds the
  inputs' warmup). It sizes the live bar buffer: with at least `total warmup` bars of history,
  the bar-by-bar (live) values equal the values computed on the full history.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Union

import pandas as pd

Computed = Mapping[str, pd.Series]
Source = Union[str, "Feature"]  # a raw data column or a single-output feature

# Relative weight below which the forgotten start of an exponential average is considered negligible.
EWM_TOLERANCE = 1e-10


class Feature(ABC):
    required_columns: tuple[str, ...] = ()
    causal: bool = True

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the feature (and of its output when it has a single one)."""

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.name,)

    def inputs(self) -> Sequence[Feature]:
        return ()

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Own history (in bars) needed for exact values, excluding the inputs' warmup."""

    @abstractmethod
    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series | pd.DataFrame:
        """Values for every row of `data`; a DataFrame must have exactly `outputs` as columns."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


def source_name(source: Source) -> str:
    return source if isinstance(source, str) else source.name


def source_inputs(*sources: Source) -> tuple[Feature, ...]:
    return tuple(s for s in sources if isinstance(s, Feature))


def source_columns(*sources: Source) -> tuple[str, ...]:
    return tuple(s for s in sources if isinstance(s, str))


def source_series(source: Source, data: pd.DataFrame, computed: Computed) -> pd.Series:
    if isinstance(source, str):
        return data[source]
    if len(source.outputs) != 1:
        raise ValueError(f"{source} has several outputs and cannot be used as a single source.")
    return computed[source.name]


def ewm_warmup(span: int, tolerance: float = EWM_TOLERANCE) -> int:
    """Bars after which the weight of the forgotten history of an EWM(span) falls below `tolerance`."""
    alpha = 2 / (span + 1)
    return math.ceil(math.log(tolerance) / math.log(1 - alpha))
