# Quant Lab code guide

This document describes **how the code is organized**:
- the command-line reference;
- the modules and the main classes;
- the conventions and the tests;
- how to add a feature, a strategy, a model, a data source or a broker;
- the dependencies and the deployment of the online demo.

For how the system works as a whole (execution, risk, sessions), see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Quick start

```bash
python -m pip install -e ".[dev]"       # Python 3.10+
python -m quantlab --help               # command line (or `quantlab --help` once installed)
python -m quantlab ui                   # web interface (Streamlit)
python -m pytest                        # all tests (~3 min, offline)
python -m pytest -m "not slow"          # without the end-to-end backtests
python -m ruff check . && python -m ruff format .
python -m mypy quantlab                 # strict; must stay at zero errors
python -m pre_commit install            # ruff + mypy hooks before each commit
```

Example commands:

```bash
python -m quantlab backtest --strategy trend --symbol AAPL --test-days 5          # bundled sample, offline
python -m quantlab backtest --strategy pair --symbol AAPL --symbol MSFT --days 55 --param zscore_threshold=2.0
python -m quantlab optimize --strategy trend --symbol AAPL --range sma_fast=5:20:5 --grid sma_slow=30,50 \
    --train-size 3000 --validation-size 1000 --test-size 1000
python -m quantlab params --strategy mean_reversion
python -m quantlab live --strategy pair --symbol SPY --symbol QQQ --source synthetic --speed 10 --session demo
python -m quantlab replay --session demo
```

The extras of `pyproject.toml`: `.[dev]` installs everything (interface, tests, linters), `.[ui]` the interface without the development tools, and a plain `-e .` the library and the command line only.

### 1.1 Command-line reference

`python -m quantlab <command> --help` prints the options of a command. `-v`, before the command, turns the debug logs on.

| Command | Role |
|---|---|
| `backtest` | train the strategies on the start of the data and trade them on the end; prints the metrics |
| `optimize` | walk-forward search of the parameters of one strategy |
| `live` | simulated live trading of one strategy, saved as a session: on recorded bars, on a synthetic market, or on the real candles of a crypto product |
| `replay` | rerun a session in the backtester and compare both |
| `params` | list every parameter of the given strategies: group, type, default, bounds, help |
| `plot` | chart one session: candles, volume, VWAP bands, support and resistance lines |
| `ui` | open the web interface (`--port`, 8501 by default) |

Options shared by several commands:

| Option | Commands | Default | Role |
|---|---|---|---|
| `--symbol` | `backtest`, `optimize`, `live`, `plot` | required | repeat it for a two-symbol strategy |
| `--source` | `backtest`, `optimize`, `plot` | `sample` | `sample`, `yfinance` or `file` |
| `--file` | `backtest`, `optimize`, `plot` | | parquet or csv file of `--source file` |
| `--days` | `backtest`, `optimize`, `plot`, `live` | 55 (`live`: 60) | history window, in calendar days; ignored for a file |
| `--strategy` | `backtest`, `optimize`, `live`, `params` | required | `trend`, `mean_reversion`, `arima`, `ml_hybrid`, `cascade` or `pair`. `backtest` and `params` take several, `optimize` and `live` exactly one |
| `--param` | `backtest`, `optimize`, `live` | | `key=value` for every selected strategy that declares the key, `strategy.key=value` for one |
| `--params-file` | `backtest`, `optimize`, `live` | | preset JSON; a `--param` overrides it |
| `--capital` | `backtest`, `live` | 100,000 | starting cash, in dollars |
| `--fee-bps` | `backtest`, `live` | 1 | fees per side, in basis points of the notional |
| `--slippage-bps` | `backtest`, `live`, `replay` | 1 (`replay`: the session's) | simulated slippage, in basis points |
| `--max-order` | `backtest`, `live` | none | maximum shares per entry order |
| `--max-position` | `backtest`, `live` | none | maximum net shares per symbol |
| `--max-daily-loss` | `backtest`, `live` | none | kill switch, in percent of the day's starting equity |
| `--sessions-dir` | `live`, `replay` | `sessions` | parent folder of the sessions |

Options of one command:

| Command | Option | Default | Role |
|---|---|---|---|
| `backtest` | `--test-days` | 5 | the last N calendar days are the test block |
| | `--test-fraction` | | instead: the last fraction of the bars |
| | `--each` | off | with several strategies, also run each one alone |
| | `--bar-by-bar` | off | compute the features on the live path instead of the batch mode |
| | `--trades`, `--equity` | | write the trades, or the equity curve, to this CSV file |
| `optimize` | `--grid` | | `key=v1,v2,...`: the values to try |
| | `--range` | | `key=min:max:step`: a range of a numeric parameter, checked by its bounds |
| | `--train-size`, `--validation-size`, `--test-size` | required | block sizes, in bars |
| | `--step` | the test size | shift between two folds |
| | `--embargo` | 30 | bars dropped between two blocks |
| | `--anchored` | off | growing training block |
| | `--objective` | `total_return_pct` | metric to maximize |
| | `--min-trades` | 5 | fewer trades discard a combination |
| `live` | `--source` | `replay` | `replay`, `synthetic`, or `market`: the real 1-minute candles of the crypto product given as `--symbol` (`SOL-USD`), read from Coinbase without key. Needs the network; Ctrl+C stops the session |
| | `--data` | `sample` | `replay` and `synthetic`: where the bars come from, `sample` or `yfinance` |
| | `--bars` | 975 | bars to stream; `market`: live candles before the session ends |
| | `--speed` | 2 | `replay` and `synthetic`: bars per second; 0 for as fast as possible |
| | `--history` | 720 | `market`: candles the strategy is trained on |
| | `--backfill` | 360 | `market`: candles of the last hours, traded at once when the session starts |
| | `--seed` | | seed of the synthetic market |
| | `--session` | `<strategy>_<symbols>_<time>` | name of the session; it must be new |
| `replay` | `--session` | required | name of the session |
| `plot` | `--date` | the last session | New York date of the session |
| | `--output` | | save the chart to this image file instead of showing it |

`optimize` needs at least one `--grid` or `--range`. It has no capital or cost option: its backtests use the defaults.

---

## 2. Conventions

- **Everything in English**: names, docstrings, logs, interface labels and parameter help texts.
- **`logging` everywhere, never `print`**. Only the CLI prints its result tables.
- **The `__init__.py` files are empty**: always import from the module that defines the object (`from quantlab.features.technical import SMA`). This avoids circular imports and module shadowing.
- **Strict typing**: `mypy --strict` passes with zero errors, and the pre-commit hook enforces it. Untyped libraries (sklearn, xgboost) are tolerated through `ignore_missing_imports`.
- **ruff**: 120-character lines, rules `E F W I B UP SIM RUF`.
- **No backtest-only shortcut**: everything that decides or executes must work bar by bar, live.
- **Timestamps**: UTC internally. A bar is a `pd.Series` whose `name` is its timestamp, as `DataFrame.iterrows()` returns it.

---

## 3. Layout

```
quantlab/
├── cli.py                  entry point: backtest | optimize | live | replay | params | plot | ui
├── __main__.py             `python -m quantlab`
├── targets.py              ForwardTarget (model targets)
├── data/
│   ├── schema.py           validate_ohlcv, bar_value, bar_time
│   ├── pairs.py            combine_symbols (multi-symbol layout)
│   ├── loading.py          load_bars, split, read_sample (shared by the CLI and the interface)
│   ├── data_manager.py     DataManager: yfinance
│   ├── market.py           public 1-minute candles of a crypto exchange (Coinbase): fetch_candles, history
│   └── samples/            bundled 2-minute bars, one parquet file per symbol (the default source)
├── features/
│   ├── base.py             Feature (abstract), Source, ewm_warmup
│   ├── pipeline.py         FeaturePipeline (dependencies, total warmup)
│   ├── technical.py        SMA, RSI, ATR, MACD, BollingerBands, VolumeRatio, Ratio, ZScore, ADX, VWAP
│   ├── composite.py        AmplitudeFeatures, DirectionFeatures, OpportunityFeatures
│   └── levels.py           support_lines / resistance_lines (NOT causal, charts only)
├── models/
│   ├── sklearn_model.py    SklearnModel, build_dataset
│   └── arima.py            ArimaForecaster
├── strategies/
│   ├── base.py             Strategy (the contract shared by backtest and live), size and exit blocks
│   ├── params.py           Param, validate, presets, param_range
│   ├── ml_params.py        shared parameter blocks of the ML strategies
│   ├── registry.py         STRATEGIES, build_strategy (names used by the CLI)
│   └── trend_following.py, mean_reversion.py, ml_hybrid.py, cascade.py, arima.py, pair_trading.py
├── runtime/                the core shared by every mode
│   ├── buffer.py           BarBuffer (preallocated ring buffer)
│   ├── feature_feed.py     FeatureFeed (bar by bar or precomputed)
│   ├── bar_feed.py         BarFeed (protocol), HistoricalBarFeed
│   ├── runner.py           StrategyRunner (THE trading loop)
│   ├── journal.py          SQLite Journal, JournalingBroker, read_journal
│   ├── state.py            StateStore (atomic JSON)
│   └── session.py          SessionFiles (the folder of a session)
├── execution/
│   ├── orders.py           Side, OrderType, OrderStatus, Order, Fill
│   ├── exits.py            Level, ExitPlan
│   ├── sizing.py           SizeMode, Sizing
│   ├── broker.py           Broker (protocol)
│   ├── portfolio.py        Portfolio (cash, positions, equity)
│   ├── risk.py             RiskLimits, RiskManager (checks, kill switch)
│   ├── router.py           OrderRouter, Lot, Trade
│   └── simulated.py        SimulatedBroker
├── live/                   simulated live trading
│   ├── feeds.py            PacedBarFeed (real-time pacing), LiveBarFeed (real candles as they close),
│   │                       synthetic_bars, session_times
│   └── session.py          SimulatedConfig, SimulatedSession, SimulatedSessions (background registry, and the
│                           session it keeps running)
├── backtest/
│   ├── run.py              backtest() -> BacktestResult
│   ├── metrics.py          compute_metrics, daily_returns, max_drawdown
│   └── walk_forward.py     WalkForwardOptimizer, make_folds, expand_grid
├── analysis/
│   ├── replay.py           replay_session, compare_journals, bar_differences
│   ├── dataset.py          analysis of a set of bars: symbol_bars, summary, sessions, intraday_returns,
│   │                       intraday_profile, quality, return_correlation (New York time)
│   └── plot.py             session_bars, levels_figure, plot_levels (VWAP + support and resistance; CLI `plot`)
└── ui/                     Streamlit interface
    ├── app.py              pages: Data (landing page), Backtest, Optimization, Live trading
    ├── live_view.py        what the live page shows of a session, read from its journal and state
    ├── charts.py           Altair charts (equity, drawdown, per-trade P&L, bars, lines, price and volume on a
    │                       trading-time axis, histogram)
    └── hosting.py          hosted mode (QUANTLAB_HOSTED=1): limits, compute slots, safe preset names
presets/                    one parameter preset per strategy (JSON)
tests/                      offline test suite; fixtures/ holds the frozen bars and the baseline metrics
docs/                       ARCHITECTURE.md, this guide, strategies/ (one write-up per strategy)
pyproject.toml              dependencies, extras (ui, dev), ruff, mypy and pytest settings
requirements.txt            pinned environment of the online demo (§8)
.streamlit/config.toml      interface settings: uploads capped at 10 MB, no usage statistics
.pre-commit-config.yaml     ruff and mypy hooks
```

---

## 4. Main interfaces

### 4.1 `Feature` (`features/base.py`)

```python
class Feature(ABC):
    required_columns: tuple[str, ...] = ()   # raw columns read (close, volume…)
    causal: bool = True
    name: str                                # abstract property; unique name derived from the parameters
    outputs: tuple[str, ...]                 # (name,) by default; may hold several columns
    def inputs(self) -> Sequence[Feature]    # other features read from `computed`
    warmup: int                              # abstract property; history the feature itself needs
    def compute(self, data, computed) -> pd.Series | pd.DataFrame
```

A `Source` is either a raw column name (`"close"`) or a single-output feature. The helpers `source_name`, `source_inputs`, `source_columns` and `source_series` handle both cases.

### 4.2 `FeaturePipeline` and `FeatureFeed`

- `FeaturePipeline(features)`:
  - `.compute(data) -> DataFrame` returns every output;
  - also `.warmup`, `.columns`, `.required_columns`, `.causal`.
- `FeatureFeed(pipeline, min_history=0)`:
  - to feed it: `warm_start(history)`, `update(bar)` (live path), `precompute(bars)` (batch mode);
  - to read it: `ready` (warmup reached), `row()` (features of the current bar), `latest(n)` (last n rows), `bars(n)` (last n raw bars), `batch_mode`.

### 4.3 `Strategy` (`strategies/base.py`)

| To implement | Role |
|---|---|
| `features()` | features read by `decide` and by the models |
| `decide(bar) -> list[Order]` | **required**: entry orders for the current bar (`self.feed` already contains it) |
| `train(history)` | optional: fit the models |
| `on_fill(fill)` | optional: react to a fill |

| Attribute | Role |
|---|---|
| `stop_loss_pct`, `take_profit_pct` | stop and target in % of the entry price (2 and 5; 0 / `None` = none), constructor parameters |
| `stop_atr`, `take_profit_atr`, `atr_window` | levels in ATR multiples measured at the signal, instead of the %. The ATR is added to the features automatically |
| `holding_bars` | holding period before the exit at market (5; 0 / `None` = stop / target only), constructor parameter |
| `size_mode`, `quantity`, `size_value`, `size_min_stop_pct` | sizing rule applied by the router (`execution/sizing.py`): `fixed` 10 shares by default, `notional`, `equity_pct` or `risk`. Constructor parameters |
| `PARAMS`, `is_group` | parameter sheet (§4.4); `is_group` is `True` for a strategy whose entries are groups (pair) |
| `min_history` | raw bars read through `feed.bars(n)` on top of the features' warmup |
| `diagnostics` | dict reset at every bar and journaled (e.g. predictions) |
| `params` | full parameter values when built by the registry (saved with sessions) |

Provided by the base class:

- `fit(history)` runs `warm(history)`, then `train(history)`. `warm` rebuilds the buffer without retraining, which a session restart uses;
- `prepare(bars)`: precomputation for the backtest;
- `start()` and `stop()`;
- `on_bar(bar)`: filters a stopped strategy and the warmup, then calls `decide`;
- `get_state()` and `set_state()`;
- `order(side, quantity=None, symbol=None, exit_plan=None, conviction=1.0)` builds an `Order` owned by the strategy:
  - without `quantity`, the order carries the sizing rule `sizing()`, and the router applies it, times `conviction` (0 to 1);
  - with a quantity, the order is sent as is;
- `exit_plan()`: the plan of a new entry, built from the exit parameters;
- `close_positions(symbol=None, reason)`: ask the router to close this strategy's lots;
- `group(orders, plan)`: make orders one group (the legs of a pair);
- `symbols`: the traded symbols, whose close is marked before each decision. The router sizes the entries at these prices.

A `strategy_id` defaults to the class name. It must be unique within a runner: pass `strategy_id=…` to run two instances.

### 4.4 Parameter sheets (`strategies/params.py`)

- **`PARAMS`**: a class attribute, the tuple of the strategy's `Param(name, kind, default, help, group, min=, max=, choices=, optional=)`.
  - The base class provides the shared blocks `size_sheet()` and `exit_sheet()` (`Strategy.PARAMS`), and each strategy extends them: `PARAMS = (Param(...), ..., *Strategy.PARAMS)`.
  - To change a shared default, use `with_defaults(Strategy.PARAMS, quantity=5)` and pass `quantity=5` to the constructor as well.
  - For a pair, use `size_sheet(group=True)` and `exit_sheet(group=True)`. That exit block has no ATR parameters.
- **`check(values)`** (class method) returns the errors linking several values; **`warnings(values)`** returns the warnings.
- **`StrategySpec.build`** (`registry.py`) validates the values (types, bounds, rules) before calling the constructor, then stores the full values in `strategy.params`. The CLI and the interface both go through it.
- **Reusable ML blocks**: `strategies/ml_params.py` (`indicator_params`, `target_params`, `xgb_params(prefix, model)`, `forest_params`, `check_ml`, `warn_ml`).
- **Presets** and **ranges**: `save_preset` / `load_preset` for presets, `param_range(param, start, stop, step)` for ranges.

### 4.5 Orders, exits, sizing (`execution/`)

- `Order(symbol, side, quantity, strategy_id, order_type=MARKET, limit_price=None, stop_price=None, oca_group=None, order_id=<uuid>, exit_plan=None, group_id=None, sizing=None, conviction=1.0)`: frozen, and serializable with `to_dict` / `from_dict`.
- `Fill(order, price, quantity, timestamp, fees=0.0, exec_id=None)`, with `signed_quantity`.
- `Side`: `BUY` or `SELL`, with `sign` and `opposite`. `OrderType`: `MARKET`, `LIMIT` or `STOP`.
- `OrderStatus`: `CANCELLED` or `REJECTED`, the terminal statuses that brokers report.
- `Level.pct(0.02)`, `Level.distance(1.5)`, `Level.at(101.2)`, and `ExitPlan(stop, target, holding_bars, leg_stop)`.
- `Sizing(mode, quantity, value, min_stop_pct)`, with `quantity_for(...)` and `group_quantities(...)`.

### 4.6 `Broker` (`execution/broker.py`)

```python
class Broker(Protocol):
    def set_handlers(self, on_fill, on_status) -> None
    def submit(self, order) -> None
    def cancel(self, order_id) -> None
    def positions(self) -> dict[str, int]
    def process_events(self) -> None      # deliver queued events on the caller's thread
    def on_bar(self, bar) -> None         # the simulator fills here; a no-op for a live broker
    def get_state(self) -> dict ; def set_state(self, state) -> None
```

Implementations: `SimulatedBroker`, and `JournalingBroker`, a wrapper that records everything in the journal.

### 4.7 `OrderRouter` (`execution/router.py`)

- **Sending**: `submit_entries(orders)` or `submit_entry(order) -> Order | None`.
  - Each entry is sized, then risk-checked.
  - It opens a `Lot` that freezes `order.exit_plan`.
  - A plan without a stop is refused (`RiskLimits.require_stop`).
  - Orders sharing a `group_id` pass all together or not at all.
- **Strategy-requested exit**: `exit_strategy(strategy_id, symbol=None, reason)`. A group exits as a whole.
- **Broker callbacks**: `on_fill`, `on_status`.
- **At each bar**: `on_bar_close(ts)` handles the time exits, the pending close requests, the combined stop / target of groups, and the kill switch. `flatten(reason)` closes everything.
- **Reading**: `lots` (open), `trades` (closed `Trade`s: `pnl`, `pnl_net`, `return_pct`), `open_trades(prices, ts)`, `exposure(symbol)`.
- **Persistence**: `get_state()`, `set_state()`, `orders()`.

### 4.8 `StrategyRunner` (`runtime/runner.py`)

```python
runner = StrategyRunner(strategies, broker, RiskManager(limits), capital=100_000, journal=None, state_store=None)
# also: tick_size=0.01 (rounding of the protective prices), journal_features=True (journal every feature row)
runner.run(feed)  # blocking; runner.stop() from another thread
runner.step(bar)  # one bar
runner.catch_up(bars)  # catch up after a restart (no decision)
runner.equity, runner.latencies_ms, runner.portfolio, runner.router
runner.get_state() / runner.set_state(state)
```

### 4.9 High-level entry points

- `load_bars(symbols, source, days, file=None)` and `split(data, test_days, test_fraction)` (`data/loading.py`): the bars of a source, and their training / test blocks.
- `build_strategy(name, symbols, params=None)` (`strategies/registry.py`): a strategy from its command-line name, with validated parameters.
- `backtest(strategies, training, data, capital=…, fee_bps=1, slippage_bps=1, risk_limits=None, precompute_features=True) -> BacktestResult` (`metrics`, `equity`, `trades`, `trades_frame()`).
- `WalkForwardOptimizer(strategy_factory, param_grid, train_size, validation_size, test_size, objective="total_return_pct", step=None, embargo=0, anchored=False, min_trades=1)`. `.run(data)` returns a `WalkForwardResult` (`summary()`, `param_stability()`, `mean_test_score`, `folds`). The factory must return a **new** instance at each call. The command line and the interface pass an embargo of 30 bars and a minimum of 5 trades.
- `SimulatedSession(folder, SimulatedConfig(strategy, symbols, …))` (`live/session.py`): `prepare()` loads the bars, trains the strategy and writes the snapshot; `run()` streams the bars and blocks; `stop()` can be called from another thread. `source="market"` trades real candles, and `extra_strategies` adds strategies to the same account.
- `SimulatedSessions(root)`: `start(name, config, owner)` runs a session in a background thread, within the limits of the registry; `keep_running(prefix, config, owner)` runs sessions one after the other for as long as the process lives; `stop_kept(owner)` and `stop_all()` end them.
- `market.fetch_candles(product, start, end)` and `market.history(product, n_bars)` (`data/market.py`): the public 1-minute candles of a crypto product, in the OHLCV contract.
- `replay_session(folder, fee_bps=None, slippage_bps=None)` returns a `ReplayReport` (`summary`, `signals`, `entry_fills`, `feature_diffs`).
- `read_journal(path)` (`runtime/journal.py`) returns the bars and the events of a session; `.kind(name)` gives the events of one kind as a table.
- `analysis/dataset.py`, on the bars of one symbol in New York time (`symbol_bars`): `summary`, `sessions`, `intraday_returns`, `return_stats`, `intraday_profile`, `quality`, and `return_correlation` across symbols.

---

## 5. Extending the platform

### 5.1 Add a feature

In `features/technical.py`:

```python
class EMA(Feature):
    """Exponential moving average."""

    def __init__(self, span: int = 20, source: Source = "close") -> None:
        self.span = span
        self.source = source
        self.required_columns = source_columns(source)

    @property
    def name(self) -> str:
        return f"ema_{self.span}{_suffix(self.source)}"  # name = parameters

    def inputs(self) -> Sequence[Feature]:
        return source_inputs(self.source)

    @property
    def warmup(self) -> int:
        return ewm_warmup(self.span)  # EWM: converged to 1e-10

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series:
        return source_series(self.source, data, computed).ewm(span=self.span, adjust=False).mean()
```

Rules:

- **causal**: `rolling`, `shift(positive)`, `ewm` and `cumsum` are fine. `shift(-n)`, a centered window, or a normalization over the whole series are forbidden;
- **exact or upper-bound warmup**: `w − 1` for a `w`-bar moving average. A `diff` or a `shift(1)` adds 1. The pipeline sums the warmups of the inputs, so do not count them again;
- **unique name** derived from every parameter that changes the result;
- **add the feature to `PIPELINES` in `tests/test_features.py`**: the causality, warmup and parity tests then apply automatically.

### 5.2 Add a strategy

In `strategies/breakout.py`:

```python
class BreakoutStrategy(Strategy):
    """Buy when the close breaks above its moving average by k ATRs (and vice versa)."""

    PARAMS = (
        Param("window", "int", 20, "Moving-average window.", "indicators", min=2, max=1000),
        Param("k", "float", 2.0, "Breakout distance, in ATRs.", "signal", min=0, max=20),
        *with_defaults(Strategy.PARAMS, holding_bars=10),  # base size and exit parameters
    )

    def __init__(
        self, window: int = 20, k: float = 2.0, symbol: str = "AAPL", holding_bars: int | None = 10, **kwargs: Any
    ) -> None:
        super().__init__(symbol=symbol, holding_bars=holding_bars, **kwargs)
        self.k = k
        self.sma, self.atr = SMA(window), ATR(14)

    def features(self) -> Sequence[Feature]:
        return (self.sma, self.atr)

    def decide(self, bar: pd.Series) -> list[Order]:
        row = self.feed.row()
        mid, band = row[self.sma.name], self.k * row[self.atr.name]
        if bar["close"] > mid + band:
            return [self.order(Side.BUY)]  # quantity: the size parameters, applied by the router
        if bar["close"] < mid - band:
            return [self.order(Side.SELL)]
        return []
```

Then:

1. Register it in `strategies/registry.py`: `"breakout": StrategySpec(BreakoutStrategy)`. Add `n_symbols=2` for a two-asset strategy.
2. Give it a display name in `STRATEGY_LABELS` (`ui/app.py`). The pages look the name up there, and fail on a strategy without one.
3. It is now available everywhere:
   - on the command line: `quantlab backtest --strategy breakout --param k=1.5`, `quantlab params --strategy breakout`, `optimize --range k=1:3:0.5`, `live --strategy breakout`;
   - in the interface, with a form generated from `PARAMS`.

   `tests/test_params.py` checks that the sheet and the constructor agree.
4. For the online demo, give it a budget of walk-forward runs in `Limits.max_optimize_runs` (`ui/hosting.py`). Without one, the hosted mode refuses to optimize it.
5. Add a case to `tests/test_characterization.py` (then `UPDATE_BASELINE=1`, see §7), and its features to `tests/test_features.py`.
6. Document it: a write-up in `docs/strategies/`, a row in the strategy tables of the README and of ARCHITECTURE.md (§6), and optionally a preset in `presets/`.

Points to watch:

- **Data**: only use `self.feed`, the data up to the current bar; never a global DataFrame.
- **Exits** come from each entry's exit plan, built by `exit_plan()` from the exit parameters. The router executes every exit.
  - A level specific to one signal: `self.order(side, exit_plan=replace(self.exit_plan(), target=Level.at(price)))`.
  - Closing your positions on a signal: `self.close_positions(reason="...")`.
  - A pair: `self.group([leg_a, leg_b], plan)`.
- **ML models**: implement `train(history)` as in `MLHybridStrategy`, with `drop_last` at least equal to the target's horizon. Expose the hyperparameters with the blocks of `strategies/ml_params.py`, and the margin as `training_margin`, which `check_ml` validates.
- **Saved sessions**: a new value read in `decide` also needs a class-level default (e.g. `volume_threshold: float = 1.2`). Otherwise a strategy pickled before that change cannot resume a session.
- **Live latency**: keep predictions on a single row (`SklearnModel.predict_row`). The budget is 100 ms per bar.

### 5.3 Add a model

Any scikit-learn-compatible estimator (with `fit` and `predict`) can be wrapped:

```python
model = SklearnModel(
    GradientBoostingClassifier(max_depth=3), features.outputs, ForwardTarget(5, "direction"), accuracy_score
)
X, y = model.dataset(history, self.feed.pipeline.compute(history), drop_last=30)
model.fit(X, y)
model.predict_row(self.feed.row())
model.cross_validate(X, y)  # time-ordered cross-validation on clones
```

The estimator must be picklable: sessions save the fitted strategies.

### 5.4 Add a data source

1. Add a method `collect_data_<source>(symbol, days_back, until)` to `DataManager`. It **returns `validate_ohlcv(...)`**.
2. Add the source's name to `SOURCES` and to `load_bars` in `data/loading.py`. The command line picks it up.
3. Give it a display name in `SOURCE_LABELS` (`ui/app.py`): the source selector of the interface fails on a source without one.
4. Write a test with a faked response, without network, as `test_bugfixes.py` does for yfinance.

To add a bundled sample, save the 2-minute bars of the symbol as `quantlab/data/samples/<SYMBOL>.parquet`. The file is found by its name, and the interface lists it.

### 5.5 Add a broker

Implement the `Broker` protocol (§4.6):

- call `on_fill` and `on_status` **on the strategy thread**. If the API notifies from another thread, queue the events and deliver them in `process_events()`;
- ignore duplicate executions;
- provide `get_state` and `set_state` for restarts;
- keep the orders' ids in a field the broker echoes back, so that working orders can be found again after a restart.

A live feed implements `BarFeed`. It yields completed bars in time order, and `None` as a heartbeat while waiting, so that the runner can process broker events between bars. `LiveBarFeed` (`live/feeds.py`) is the example to follow: it takes the function that reads the candles, so another exchange only needs its own `fetch(product, start, end)` beside the one of `data/market.py`. Three things to check on a new source:

- when a closed candle is final. Coinbase can still serve an unfinished one a few seconds after the close, hence the delay of the feed;
- whether its responses are cached. Coinbase caches each URL for minutes, hence a distinct range for every request;
- what it sends for a period without a trade, and how many candles one request returns.

### 5.6 The web interface

The interface (`quantlab/ui/app.py`) holds no logic of its own: each page calls the same functions as the CLI. To add a feature, write it in the library, then wire both the CLI command **and** the page.

- **Generated forms**:
  - a strategy's form is generated from its `PARAMS` sheet (`param_inputs`), with one tab per group;
  - widgets are bounded according to the type, and the `check` errors are shown;
  - presets come from `presets/`;
  - the optimization grid is built from ranges (`grid_inputs`);
  - a new strategy registered in `registry.py` therefore needs no form code, only a display name in `STRATEGY_LABELS`.
- **Pages**: `page_data`, `page_backtest`, `page_optimize` and `page_live`, declared in `main()` with `st.navigation`. The first one is the landing page; the others have the addresses `/backtest`, `/optimization` and `/live`. What each page shows is described in [ARCHITECTURE.md §11](ARCHITECTURE.md).
- **Widget values** are kept in `st.session_state["<page>_<strategy>_<parameter>"]`. Presets write them there, before the widgets are drawn.
- **Reruns**: Streamlit reruns the script at every interaction. Results to keep go to `st.session_state`, and downloaded data to `st.cache_data`.
- **Live page**: sessions run in the `SimulatedSessions` registry (`st.cache_resource`), and the monitor is an `st.fragment(run_every=2)` that only runs while a session is alive. It reads the session's files (`read_session`), never the running objects, so the UI thread does not race the trading thread.
- **Market tab**: the shared session is started by `keep_running`, once per server process (`MARKET_STRATEGIES`, `MARKET_SIZE` and `MARKET_BARS` set what it trades). Its monitor is an `st.fragment(run_every=5)`, and the view is cached for all visitors by the modification time of the session's files (`shared_session_view`).
- **Code updates**: a deployment replaces the files under the running server, which reruns `app.py` but keeps the modules it imported. `refresh_code()`, at the top of `app.py`, detects a changed file of the package, stops the background sessions, forgets the modules and empties the caches. It needs one run to know the files, so it protects from the second deployment on; it also picks up local edits without a restart.
- **Hosted mode**: anything that reads a path, writes a file or can run for long must respect `hosted()` (`ui/hosting.py`): no server path, presets in the session, `Limits` checks before computing, `compute_slots()` around heavy work.
- **Data page**: it loads the bars without a button (the bundled sample by default) and recomputes everything from the period of its slider. The figures come from `analysis/dataset.py`, which has no Streamlit code.
- **Charts**: colors by role (a fixed categorical order, blue for a single series, blue / red for gains / losses), hover tooltips, New York time. The price and volume charts use the position of the bar as x scale, labeled with its time, so nights and weekends take no room. Bars drawn on an x range (`x` and `x2`) need `stack=None`, or Vega-Lite drops their width.

---

## 6. Pitfalls

- **Batch / live parity**: if a strategy behaves differently with `--bar-by-bar`, a feature is not causal or its warmup is too short. The tests of `test_features.py` pinpoint it.
- **Fixtures without a time zone**: `aapl_1m.parquet` has a naive index. The runner accepts it, but `validate_ohlcv` would convert it to UTC. Its 5,000 one-minute bars include the extended hours, about 700 bars per day, so 1,000 bars cover about a day and a half.
- **Streamlit keeps imported modules**: `refresh_code()` reloads the package when one of its files changes (§5.6). Running sessions are stopped by that reload.
- **Windows and the session state**: Windows refuses to replace `state.json` while a reader has it open. `StateStore.save` retries the rename for half a second.
- **No network in the tests**: `data/market.py` is the only module that contacts the exchange. The tests replace its `fetch_candles`, or pass their own `fetch` and clock to the feed.
- **Unique `strategy_id`** within a runner: otherwise `StrategyRunner` raises an error.
- **Pickle**: a strategy must be picklable for sessions, so no lambda and no connection in its attributes. `test_every_strategy_survives_the_session_snapshot` checks it.

---

## 7. Tests

| File | What it checks |
|---|---|
| `test_characterization.py` (slow) | every strategy, backtested on frozen fixtures in batch **and** bar-by-bar mode, reproduces `fixtures/baseline_metrics.json` exactly |
| `test_features.py` | causality, warmup sufficiency, batch / bar-by-bar parity (default and custom windows), pipeline, buffer |
| `test_execution.py` | simulated broker (next open, gaps, OCA, stop first), router (lots, partial fills, rejects, over-fills, risk, kill switch), portfolio, metrics, runner |
| `test_exits.py` | per-lot exit plans, refusal without a stop, gaps, groups (all or nothing, combined stop, leg stop, missing leg), strategy-requested exits |
| `test_sizing.py` | sizing modes, stop floor, equal notional per leg, Cascade's historical formula, application by the router |
| `test_params.py` | parameter sheets: match with the constructors, each parameter has an effect, conversion and bounds, consistency rules, versioned presets, `quantlab params` |
| `test_session.py` | journal, restart identical to an uninterrupted run, catch-up, replay (signal agreement, slippage), snapshot of every strategy |
| `test_cli.py` | argument parsing (`--param`, `--params-file`, `--range`), registry, `backtest` and `optimize` commands on the fixtures |
| `test_ui.py` | Streamlit interface without a browser (`AppTest`): every page renders, generated forms, presets, optimization ranges, backtest and data page end to end, the dataset analysis |
| `test_bugfixes.py` | regressions of the bugs fixed during the refactoring of the original prototype |
| `test_live.py` | paced feeds, synthetic market, simulated sessions (replayed with 100 % agreement), stop, background registry and its limits, live view, the live page end to end; on a fake exchange with its own clock: the real-market feed (catch-up, closed candles only, quiet minutes, outage), the market session, the session kept running, the Market tab |
| `test_market.py` | the public candles client: parsing, request ranges, errors, windows of 300 candles |
| `test_hosting.py` | hosted mode: bundled samples, limits, safe preset names, compute slots, the hosted backtest, live and data pages; `requirements.txt` pins every declared dependency |
| `test_walk_forward.py` | fold layout, embargo, parameter selection, fresh instances |
| `test_latency.py` (slow) | median `on_bar` latency of Cascade < 100 ms |

The tests **never** use the network.

**Changing a backtest result on purpose** (a bug fix, an engine change):

```bash
UPDATE_BASELINE=1 python -m pytest tests/test_characterization.py
git diff tests/fixtures/baseline_metrics.json     # review and justify every difference
```

A pure refactoring must **never** modify this file.

---

## 8. Dependencies and deployment

- **`pyproject.toml`** declares the dependencies with minimum versions, and two extras: `ui` (Streamlit) and `dev` (Streamlit, pytest, ruff, mypy, pandas-stubs, pre-commit). The charts use Altair, which Streamlit installs.
- **`requirements.txt`** is the environment of the online demo: Streamlit Community Cloud installs this file, not `pyproject.toml`. It pins every dependency to the version the tests run on. `test_requirements_pin_the_declared_dependencies` fails when a dependency of `pyproject.toml` is missing from it, or pinned outside its allowed range. When a dependency changes, update both files.
- **Python**: 3.10 or later. The online demo runs on Python 3.12, chosen in the settings of the app, because the pinned versions have no prebuilt package for newer versions.
- **Online demo**: the app runs `quantlab/ui/app.py` from the `main` branch, and every push to `main` redeploys it. The secret `QUANTLAB_HOSTED = "1"`, set in the settings of the app, reaches the code as an environment variable and turns the hosted mode on ([ARCHITECTURE.md §1.1](ARCHITECTURE.md)). Without it, the public app runs unrestricted.
- **Market session**: the optional secret `QUANTLAB_MARKET = "1"` starts the shared paper-trading session on real market data ([ARCHITECTURE.md §10.5](ARCHITECTURE.md)); another value names the product. Removing the secret is the switch that stops the app from contacting the exchange.
- **After a push**: the cloud updates the files without restarting the server, and `refresh_code()` reloads the package (§5.6). If a page still shows an import error after a deployment, reboot the app from its menu on share.streamlit.io.
- **`.streamlit/config.toml`** applies to local and hosted runs: uploads capped at 10 MB, usage statistics off.
