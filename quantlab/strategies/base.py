"""
Strategy contract shared by backtest and live.

A strategy declares its features (`features()`), optionally fits models in `fit`, and decides on
each bar in `decide`. The base class owns the running state and the `FeatureFeed`, so a concrete
strategy never manages buffers or caches and runs unchanged on historical or live bars.

Exits: every entry carries an `ExitPlan` (stop, target, holding period), built by `exit_plan()` from the
strategy's exit parameters below, which `decide` may adjust per signal (`self.order(..., exit_plan=...)`).
A strategy can also close its own positions with `close_positions()`; the router executes the exits.

Size: `self.order(side, conviction=...)` leaves the quantity to the router, which applies the strategy's
sizing rule (`sizing()`, from the size parameters: fixed quantity, notional, % of equity, or fixed risk at the
stop; `execution/sizing.py`).

Parameters: `PARAMS` is the strategy's parameter sheet (`strategies/params.py`), the base size and exit blocks
plus the strategy's own entries; `check` holds the rules linking several values. Every constructor argument is declared
there with the same default (tested).
"""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar

import pandas as pd

from quantlab.execution.exits import ExitPlan, Level
from quantlab.execution.orders import Fill, Order, Side
from quantlab.execution.sizing import SIZE_MODES, SizeMode, Sizing
from quantlab.features.base import Feature
from quantlab.features.pipeline import FeaturePipeline
from quantlab.features.technical import ATR
from quantlab.runtime.feature_feed import FeatureFeed
from quantlab.strategies.params import Param


def exit_sheet(group: bool = False) -> tuple[Param, ...]:
    """The base exit parameters; for a group (pair), stop / target apply to the combined P&L and no ATR."""
    if group:  # the combined stop is required for a pair
        stop = Param(
            "stop_loss_pct",
            "float",
            2.0,
            "Stop loss, en % du notionnel de la paire (P&L combiné), vérifié à chaque clôture de barre. Obligatoire.",
            "sorties",
            min=0.01,
            max=100,
        )
    else:
        stop = Param(
            "stop_loss_pct",
            "float",
            2.0,
            "Stop loss, en % du prix d'entrée. 0 = aucun (l'entrée est alors refusée).",
            "sorties",
            min=0,
            max=100,
            optional=True,
        )
    of = "du notionnel de la paire (P&L combiné)" if group else "du prix d'entrée"
    sheet = (
        stop,
        Param("take_profit_pct", "float", 5.0, f"Objectif, en % {of}. 0 = aucun.", "sorties", min=0, max=1000,
              optional=True),
        Param("holding_bars", "int", 5, "Sortie au marché après ce nombre de barres. 0 = aucune.", "sorties",
              min=0, max=100_000, optional=True),
        Param("stop_atr", "float", None, "Stop en multiples d'ATR mesuré au signal (remplace le %). 0 = non.",
              "sorties", min=0, max=100, optional=True),
        Param("take_profit_atr", "float", None, "Objectif en multiples d'ATR (remplace le %). 0 = non.",
              "sorties", min=0, max=100, optional=True),
        Param("atr_window", "int", 14, "Fenêtre de l'ATR des sorties.", "sorties", min=1, max=1000),
    )  # fmt: skip
    return tuple(p for p in sheet if not (group and p.name in ("stop_atr", "take_profit_atr", "atr_window")))


#: constructor parameters of the base class (exit parameters), shown with every strategy's own ones
EXIT_PARAMS = tuple(p.name for p in exit_sheet())


def size_sheet(group: bool = False) -> tuple[Param, ...]:
    """The base size parameters (executed by the router, `execution/sizing.py`)."""
    quantity = "Actions de la première jambe (même notionnel sur l'autre)" if group else "Actions par entrée"
    return (
        Param("size_mode", "choice", "fixed", "Taille : fixed (quantité), notional ($), equity_pct (% du capital) "
              "ou risk (% du capital perdu au stop).", "taille", choices=SIZE_MODES),
        Param("quantity", "int", 10, f"{quantity} en mode fixed, fois la conviction.", "taille", min=1,
              max=1_000_000),
        Param("size_value", "float", None, "$ (notional) ou % du capital (equity_pct, risk) ; vide en mode fixed.",
              "taille", min=0, max=1e9, optional=True),
        Param("size_min_stop_pct", "float", 0.1, "Mode risk : distance minimale au stop, en % du prix.", "taille",
              min=0.001, max=100),
    )  # fmt: skip


SIZE_PARAMS = tuple(p.name for p in size_sheet())


def _positive(value: float | None) -> float | None:
    """0 / None / negative mean "none"."""
    return float(value) if value is not None and value > 0 else None


def check_size(values: Mapping[str, Any]) -> list[str]:
    if values.get("size_mode", "fixed") != "fixed" and not values.get("size_value"):
        return [f"size_mode {values['size_mode']} needs size_value (dollars or percent of the equity)."]
    return []


class Strategy(ABC):
    #: parameter sheet: the base exit block, extended by each strategy (`PARAMS = Strategy.PARAMS + (...)`)
    PARAMS: ClassVar[tuple[Param, ...]] = (*size_sheet(), *exit_sheet())
    #: True when the entries are groups (pair legs): exits apply to the combined P&L
    is_group: ClassVar[bool] = False
    #: raw bars the strategy reads itself through `self.feed.bars(n)` (beyond the features' warmup)
    min_history: int = 0
    # Exit parameters (class values: the defaults of strategies pickled before they existed)
    holding_bars: int | None = 5
    stop_loss_pct: float | None = 2.0
    take_profit_pct: float | None = 5.0
    stop_atr: float | None = None
    take_profit_atr: float | None = None
    atr_window: int = 14
    # Size parameters (same: fixed quantity 10 for strategies pickled before they existed)
    size_mode: str = "fixed"
    quantity: int = 10
    size_value: float | None = None
    size_min_stop_pct: float = 0.1

    def __init__(
        self,
        symbol: str = "AAPL",
        strategy_id: str | None = None,
        stop_loss_pct: float | None = 2.0,
        take_profit_pct: float | None = 5.0,
        stop_atr: float | None = None,
        take_profit_atr: float | None = None,
        atr_window: int = 14,
        holding_bars: int | None = 5,
        size_mode: str = "fixed",
        quantity: int = 10,
        size_value: float | None = None,
        size_min_stop_pct: float = 0.1,
    ) -> None:
        """
        :param stop_loss_pct: stop of each position, in percent of its fill price (0 / None: none; a stop is
            required by default, see `RiskLimits.require_stop`).
        :param take_profit_pct: target, in percent of the fill price (0 / None: none).
        :param stop_atr: if set, the stop is this many ATRs (measured at the signal) from the fill price,
            instead of `stop_loss_pct`.
        :param take_profit_atr: same for the target, instead of `take_profit_pct`.
        :param atr_window: window of that ATR.
        :param holding_bars: bars after which a position is closed at market (0 / None: no time exit).
        :param size_mode: fixed, notional, equity_pct or risk (see `execution/sizing.py`).
        :param quantity: shares per entry in fixed mode (times the conviction).
        :param size_value: dollars (notional) or percent of the equity (equity_pct, risk).
        :param size_min_stop_pct: risk mode: floor of the distance to the stop, in percent of the price.
        """
        self.symbol = symbol
        self.strategy_id = strategy_id or type(self).__name__
        self.stop_loss_pct, self.take_profit_pct = _positive(stop_loss_pct), _positive(take_profit_pct)
        self.stop_atr, self.take_profit_atr = _positive(stop_atr), _positive(take_profit_atr)
        if atr_window < 1:
            raise ValueError(f"atr_window must be >= 1 (got {atr_window}).")
        self.atr_window = int(atr_window)
        self.holding_bars = int(holding_bars) if holding_bars else None
        self.size_mode, self.quantity = SizeMode(size_mode).value, int(quantity)
        self.size_value, self.size_min_stop_pct = _positive(size_value), float(size_min_stop_pct)
        self.sizing()  # validates the size parameters
        self.is_running = False
        #: full parameter values when built by the registry (saved with live sessions); {} otherwise
        self.params: dict[str, Any] = {}
        self._feed: FeatureFeed | None = None
        #: values explaining the last decision (e.g. model predictions); journaled with each bar
        self.diagnostics: dict[str, float] = {}
        self._exit_requests: list[tuple[str | None, str]] = []

    # --- parameters ------------------------------------------------------------------------------------

    @classmethod
    def check(cls, values: Mapping[str, Any]) -> list[str]:
        """Errors of a full set of values (defaults merged with the chosen ones); [] when consistent."""
        return []

    @classmethod
    def warnings(cls, values: Mapping[str, Any]) -> list[str]:
        """Questionable but allowed combinations (logged, shown in the UI)."""
        return []

    @property
    def symbols(self) -> tuple[str, ...]:
        """Symbols the strategy trades (their closes are marked before it decides)."""
        return (self.symbol,)

    # --- to implement ----------------------------------------------------------------------------------

    def features(self) -> Sequence[Feature]:
        """Features read by `decide` (and by the models)."""
        return ()

    @abstractmethod
    def decide(self, bar: pd.Series) -> list[Order]:
        """Orders for the current bar; `self.feed` already includes it."""

    def train(self, history: pd.DataFrame) -> None:  # noqa: B027 - optional hook
        """Fit models on `history` (no-op for rule-based strategies)."""

    # --- lifecycle -------------------------------------------------------------------------------------

    @property
    def feed(self) -> FeatureFeed:
        if self._feed is None:
            raise RuntimeError(f"{self.strategy_id}: call fit(history) before using the feature feed.")
        return self._feed

    def fit(self, history: pd.DataFrame) -> None:
        """Warm the feature buffer with `history`, then train the models."""
        self.warm(history)
        self.train(history)

    def warm(self, history: pd.DataFrame) -> None:
        """(Re)build the feature buffer from `history` without retraining (restart of a live session)."""
        features = [*self.features(), *self._exit_features()]
        self._feed = FeatureFeed(FeaturePipeline(features), min_history=self.min_history)
        self._feed.warm_start(history)

    def prepare(self, upcoming: pd.DataFrame) -> None:
        """Backtest only: precompute the features of the upcoming bars (same values as bar by bar)."""
        self.feed.precompute(upcoming)

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def on_bar(self, bar: pd.Series) -> list[Order]:
        """Feed the bar and return the orders to send (none while stopped or not warmed up)."""
        if not self.is_running:
            return []
        self.diagnostics = {}
        self._exit_requests = []
        self.feed.update(bar)
        if not self.feed.ready:
            return []
        return self.decide(bar)

    def on_fill(self, fill: Fill) -> None:  # noqa: B027 - optional hook
        """Called by the broker when one of this strategy's orders is (partially) filled."""

    # --- persistence (live restarts) -------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id, "symbol": self.symbol, "is_running": self.is_running}

    def set_state(self, state: dict[str, Any]) -> None:
        self.is_running = bool(state.get("is_running", False))

    # --- helpers ---------------------------------------------------------------------------------------

    def order(
        self,
        side: Side,
        quantity: int | None = None,
        symbol: str | None = None,
        exit_plan: ExitPlan | None = None,
        conviction: float = 1.0,
    ) -> Order:
        """
        An entry order. Without `quantity`, the router sizes it with `sizing()` (times `conviction`, 0-1);
        its exit plan defaults to `exit_plan()`.
        """
        sizing = self.sizing() if quantity is None else None
        return Order(
            symbol=symbol or self.symbol,
            side=side,
            quantity=quantity if quantity is not None else sizing.quantity if sizing else 1,
            strategy_id=self.strategy_id,
            exit_plan=exit_plan or self.exit_plan(),
            sizing=sizing,
            conviction=conviction,
        )

    def sizing(self) -> Sizing:
        """Sizing rule of a new entry, from the size parameters."""
        return Sizing(SizeMode(self.size_mode), self.quantity, self.size_value, self.size_min_stop_pct)

    def group(self, orders: Sequence[Order], exit_plan: ExitPlan) -> list[Order]:
        """
        Make `orders` one group (the legs of a pair): risk-checked all or nothing and closed together. The
        plan's stop / target (pct levels) apply to the combined P&L; `leg_stop` is placed on each leg.
        """
        exit_plan.check_group()
        group_id = f"grp-{uuid.uuid4().hex[:12]}"
        return [replace(order, group_id=group_id, exit_plan=exit_plan) for order in orders]

    def exit_plan(self) -> ExitPlan:
        """Exit plan of a new entry, from the exit parameters (ATR levels measured on the current bar)."""
        atr: float | None = None
        if self.stop_atr or self.take_profit_atr:
            value = float(self.feed.row()[ATR(self.atr_window).name])
            atr = value if math.isfinite(value) and value > 0 else None

        def level(pct: float | None, multiple: float | None) -> Level | None:
            if multiple:
                return Level.distance(multiple * atr) if atr else None
            return Level.pct(pct / 100) if pct else None

        return ExitPlan(
            stop=level(self.stop_loss_pct, self.stop_atr),
            target=level(self.take_profit_pct, self.take_profit_atr),
            holding_bars=self.holding_bars,
        )

    def close_positions(self, symbol: str | None = None, reason: str = "signal") -> None:
        """Ask the router to close this strategy's open positions (on `symbol` only, if given)."""
        self._exit_requests.append((symbol, reason))

    def pop_exit_requests(self) -> list[tuple[str | None, str]]:
        """Close requests of the current bar, as (symbol, reason) (read by the runner)."""
        requests, self._exit_requests = getattr(self, "_exit_requests", []), []
        return requests

    def _exit_features(self) -> list[Feature]:
        return [ATR(self.atr_window)] if self.stop_atr or self.take_profit_atr else []
