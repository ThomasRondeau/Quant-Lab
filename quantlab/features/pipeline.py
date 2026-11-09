from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quantlab.features.base import Feature


class FeaturePipeline:
    """
    Computes a set of features (and, transitively, their inputs) in dependency order.

    The same `compute` serves both modes: on the full history (backtest batch mode) or on the
    live bar buffer, which only needs `warmup` bars for the latest row to be exact.
    """

    def __init__(self, features: Iterable[Feature]) -> None:
        self.features: tuple[Feature, ...] = tuple(features)
        self._order: list[Feature] = []
        self._total_warmup: dict[str, int] = {}
        seen: dict[str, Feature] = {}
        for feature in self.features:
            self._visit(feature, seen, path=())
        self.required_columns: tuple[str, ...] = tuple(
            dict.fromkeys(col for f in self._order for col in f.required_columns)
        )
        self.columns: tuple[str, ...] = tuple(col for f in self._order for col in f.outputs)

    def _visit(self, feature: Feature, seen: dict[str, Feature], path: tuple[str, ...]) -> None:
        if feature.name in path:
            raise ValueError(f"Dependency cycle: {' -> '.join((*path, feature.name))}")
        if feature.name in seen:
            other = seen[feature.name]
            if other.outputs != feature.outputs or [d.name for d in other.inputs()] != [
                d.name for d in feature.inputs()
            ]:
                raise ValueError(f"Two different features are named {feature.name!r}.")
            return
        for dep in feature.inputs():
            self._visit(dep, seen, (*path, feature.name))
        seen[feature.name] = feature
        self._order.append(feature)
        deps_warmup = max((self._total_warmup[d.name] for d in feature.inputs()), default=0)
        self._total_warmup[feature.name] = feature.warmup + deps_warmup

    @property
    def warmup(self) -> int:
        """History (in bars) needed for the last row of `compute` to be exact."""
        return max(self._total_warmup.values(), default=0)

    @property
    def causal(self) -> bool:
        return all(f.causal for f in self._order)

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.required_columns if c not in data.columns]
        if missing:
            raise ValueError(f"Missing data columns for the feature pipeline: {missing}")

        computed: dict[str, pd.Series] = {}
        for feature in self._order:
            result = feature.compute(data, computed)
            if isinstance(result, pd.Series):
                computed[feature.name] = result
            else:
                if tuple(result.columns) != feature.outputs:
                    raise ValueError(f"{feature} returned {list(result.columns)}, expected {list(feature.outputs)}")
                computed.update({col: result[col] for col in result.columns})
        return pd.DataFrame(computed, index=data.index, columns=list(self.columns))
