from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from quantlab.execution.exits import ExitPlan, Level
from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.technical import SMA, Ratio, ZScore
from quantlab.strategies.base import Strategy, exit_sheet, size_sheet
from quantlab.strategies.params import Param


class PairTradingStrategy(Strategy):
    """
    Sell the outperformer and buy the underperformer when the smoothed z-score of the price ratio
    moves beyond `zscore_threshold`. Bars must carry `close_<SYMBOL>` columns for both symbols.

    Both legs form one group: entered all or nothing and closed together, when the z-score comes back
    within `exit_zscore`, on the time exit, or when the pair's combined P&L reaches `stop_loss_pct` /
    `take_profit_pct` of its gross notional (checked at each bar close). Each leg also carries a wide
    protective stop at the broker (`leg_stop_pct`); when it fires, the other leg is closed at once.

    The legs always hold the same notional (the router sizes the group): in fixed mode, `quantity` shares of
    the first symbol and the same dollar amount of the second, so the pair trades the ratio, not the prices.
    """

    is_group = True
    PARAMS = (
        Param("zscore_threshold", "float", 2.2, "Entrée quand |z-score| du ratio des prix dépasse ce seuil.",
              "signal", min=0.01, max=20),
        Param("exit_zscore", "float", 0.5, "Sortie quand |z-score| revient sous ce seuil. 0 = aucune.",
              "sorties", min=0, max=20, optional=True),
        Param("leg_stop_pct", "float", 5.0, "Stop de chaque jambe chez le broker, en % (filet si l'application "
              "tombe ; plus large que le stop combiné). Obligatoire.", "sorties",
              min=0.01, max=100),
        Param("zscore_window", "int", 30, "Fenêtre du z-score du ratio.", "indicateurs", min=2, max=5000),
        Param("smoothing_window", "int", 3, "Lissage du z-score (moyenne mobile).", "indicateurs", min=1, max=500),
        *size_sheet(group=True),
        *exit_sheet(group=True),
    )  # fmt: skip

    @classmethod
    def check(cls, values: Mapping[str, Any]) -> list[str]:
        errors = []
        if values["exit_zscore"] is not None and values["exit_zscore"] >= values["zscore_threshold"]:
            errors.append("exit_zscore must be smaller than zscore_threshold (the pair would exit at once).")
        if values.get("stop_atr") or values.get("take_profit_atr"):
            errors.append("Pair trading: the stop / target apply to the combined P&L, in percent (no ATR).")
        if not values["stop_loss_pct"] or not values["leg_stop_pct"]:
            errors.append("Pair trading needs a combined stop (stop_loss_pct) and a leg stop (leg_stop_pct).")
        return errors

    @classmethod
    def warnings(cls, values: Mapping[str, Any]) -> list[str]:
        if values["leg_stop_pct"] and values["stop_loss_pct"] and values["leg_stop_pct"] <= values["stop_loss_pct"]:
            return [f"leg_stop_pct ({values['leg_stop_pct']}) <= stop_loss_pct ({values['stop_loss_pct']}): a leg "
                    "stop can fire on a common market move while the pair itself is flat."]  # fmt: skip
        return []

    def __init__(
        self,
        zscore_threshold: float = 2.2,
        symbol_a: str = "AAPL",
        symbol_b: str = "MSFT",
        zscore_window: int = 30,
        smoothing_window: int = 3,
        exit_zscore: float | None = 0.5,
        leg_stop_pct: float | None = 5.0,
        **kwargs: Any,
    ) -> None:
        """
        :param exit_zscore: close the pair when |z-score| <= this (None: no z-score exit).
        :param leg_stop_pct: protective stop of each leg at the broker, in percent (0 / None: none).
        """
        super().__init__(symbol=symbol_a, **kwargs)
        if self.stop_atr or self.take_profit_atr:
            raise ValueError("Pair trading: the stop / target apply to the combined P&L, in percent (no ATR).")
        self.zscore_threshold = zscore_threshold
        self.exit_zscore = exit_zscore
        self.leg_stop_pct = leg_stop_pct if leg_stop_pct and leg_stop_pct > 0 else None
        self.symbol_a, self.symbol_b = symbol_a, symbol_b
        ratio = Ratio(f"close_{symbol_a}", f"close_{symbol_b}")
        # Smoothed over a few bars to avoid reacting to spikes
        self.zscore = SMA(smoothing_window, source=ZScore(ratio, zscore_window))

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.symbol_a, self.symbol_b)

    def features(self) -> Sequence[Feature]:
        return (self.zscore,)

    def decide(self, bar: pd.Series) -> list[Order]:
        z = self.feed.row()[self.zscore.name]
        if self.exit_zscore is not None and abs(z) <= self.exit_zscore:
            self.close_positions(reason="zscore")
        if z >= self.zscore_threshold:
            return self.pair(Side.SELL, Side.BUY)
        if z <= -self.zscore_threshold:
            return self.pair(Side.BUY, Side.SELL)
        return []

    def pair(self, side_a: Side, side_b: Side) -> list[Order]:
        plan = ExitPlan(
            stop=Level.pct(self.stop_loss_pct / 100) if self.stop_loss_pct else None,
            target=Level.pct(self.take_profit_pct / 100) if self.take_profit_pct else None,
            holding_bars=self.holding_bars,
            leg_stop=self.leg_stop_pct / 100 if self.leg_stop_pct else None,
        )
        legs = [
            self.order(side_a, symbol=self.symbol_a, exit_plan=plan),
            self.order(side_b, symbol=self.symbol_b, exit_plan=plan),
        ]
        return self.group(legs, plan)
