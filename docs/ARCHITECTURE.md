# How Quant Lab works

This document explains **how the system works**:
- the path of a market bar through the engine;
- how features stay causal;
- how models are trained without look-ahead;
- how orders are sized, simulated and protected;
- risk controls, metrics, and the session journal, restart and replay.

For the code layout and how to extend it, see [CODE_GUIDE.md](CODE_GUIDE.md).

---

## 1. Goal and guiding principle

Quant Lab is a research platform for intraday strategies built on technical indicators and machine learning. It is used to **backtest** them, **optimize** them with walk-forward validation, and run them **bar by bar exactly as they would run live**.

The guiding principle: **one code path for backtest and live**. Features, models, strategies, sizing, exits, risk and order routing are identical in every mode. Only two components change:

| Component | Backtest | Live |
|---|---|---|
| Bar source (`BarFeed`) | `HistoricalBarFeed`, a replay of a DataFrame | a real-time feed (same protocol) |
| Execution (`Broker`) | `SimulatedBroker` | a broker connector (same protocol) |

Any backtest-only optimization, such as precomputing the features, must be **proven identical** to the live path by a parity test.

The system is driven from the command line (`python -m quantlab …`) or from the **web interface** (`python -m quantlab ui`). Both call the same library functions. The interface adds charts: equity curve, drawdown, per-trade P&L, scores per fold, session charts.

### 1.1 Hosted mode

The interface also runs as a public online demo. With `QUANTLAB_HOSTED=1` (`ui/hosting.py`), the interface:

- reads data from the bundled samples, from uploads (10 MB at most, `.streamlit/config.toml`) or from yfinance, and never from a path typed on the server;
- keeps the presets a visitor saves in that visitor's session; nothing is written on the server;
- caps one request: 12,000 bars, 3 strategies at once, 200 test bars for ARIMA, and a number of walk-forward runs (combinations × folds) per strategy. ARIMA cannot be optimized online. A request over a limit gets a message instead of running;
- runs at most 2 heavy computations at once across all visitors, and the others wait;
- caches every backtest by its full specification for all visitors, so the default runs are instant after the first one.

---

## 2. Overview

```
                 ┌───────────────────────────────────────────┐
  Historical     │ yfinance · parquet / csv files            │
  bars           └───────────────┬───────────────────────────┘
                                 │ validate_ohlcv (UTC, float, sorted, unique)
                                 ▼
                       strategy.fit(history)
                       ├─ warm: fill the bar buffer (FeatureFeed)
                       └─ train: fit the ML models (if any)
                                 │
                                 ▼
 ┌──────────── StrategyRunner: one loop, one thread, the same everywhere ─────────────┐
 │  for each bar of the BarFeed:                                                       │
 │   1. broker.on_bar(bar)        simulator: fills pending orders                     │
 │   2. broker.process_events()   live broker: delivers fills received from the wire  │
 │   3. portfolio.mark(prices)    mark to the close                                   │
 │   4. router.on_bar_close()     time exits, combined stops of groups, kill switch   │
 │   5. strategy.on_bar(bar)      features → decision → entry orders                  │
 │   6. router.submit_entries()   sizing → risk check → broker.submit                 │
 │   7. equity curve, latency, journal, persisted state                               │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

A **backtest** is this runner fed by a `HistoricalBarFeed` and wired to a `SimulatedBroker`. There is no separate backtesting engine.

---

## 3. Data

### 3.1 Sources

| Source | Resolution | Depth | Requirement |
|---|---|---|---|
| `sample` (default) | 2 min | about six weeks: AAPL, MSFT, NVDA, SPY, QQQ | none: bundled in `quantlab/data/samples/`, offline |
| `yfinance` | 2 min | ~60 days | network (free, no API key) |
| `file` | any | any | parquet or csv |

The window follows one convention, `(symbol, days_back, until)`: from `days_back` days ago to `until` days ago. For the samples, the days count back from the end of the sample.

### 3.2 One format (`validate_ohlcv`)

Every source returns a normalized DataFrame:

- a **UTC** `DatetimeIndex` named `Datetime`, sorted and unique. A timestamp without a time zone is taken as UTC;
- lowercase `float` columns `open, high, low, close, volume`;
- no row without a close.

Everything is UTC internally. Only the split into sessions, used by the VWAP and the daily returns, uses New York time.

### 3.3 Several symbols

Multi-asset strategies (pair trading) receive bars aligned on their common timestamps (inner join). Each symbol has its own `<field>_<SYMBOL>` columns (`close_AAPL`, `open_MSFT`, …). The plain `open … volume` columns are those of the first symbol. `combine_symbols` builds this layout.

---

## 4. Features

### 4.1 Stateless, causal features

A feature is a **pure function of past bars**: for each row, `compute(data, computed)` uses only the rows up to that one. Each feature declares:

- a **name derived from its parameters** (`sma_20`, `macd_hist_12_26_9`, `bb_upper_20_2`). Two features with the same name compute the same thing, so a shared indicator is computed once;
- its **inputs** (`inputs()`), the other features it reads. For example, `SMA(3)` can smooth a `ZScore` of the `Ratio` of two prices;
- its **warmup**: the bars of history it needs for an exact value.
  - For a `w`-bar moving average, it is `w − 1`.
  - For an exponential average, it is the number of bars after which the weight of the forgotten history drops below 1e-10 (`ewm_warmup`).

Available indicators: SMA, RSI, ATR, MACD, Bollinger Bands, volume ratio, ratio of two series, z-score, ADX, and the session VWAP with its bands. Three **composite sets** feed the models (`amplitude_*`, `direction_*`, `opportunity_*`). They take their base indicators and their own windows as constructor arguments.

Support and resistance lines (`levels.py`) **look ahead**: they find extremes on a centered window. They are plain functions, used for charts only and never as features.

### 4.2 The pipeline

`FeaturePipeline` resolves the dependency graph. It sorts the features in topological order and detects cycles and name collisions. It computes each feature once and sums the warmups along the dependencies.

### 4.3 Two computation paths, the same values

`FeatureFeed` serves the features to a strategy in one of two ways:

- **Bar-by-bar path (live, and the reference)**:
  - each new bar enters a `BarBuffer`, a preallocated ring buffer without `pd.concat`;
  - its size comes from the declared warmup: total warmup + 1, more if the strategy reads raw bars itself, plus a margin of 32;
  - the features are computed on this buffer.
- **Batch mode (backtest speed-up)**:
  - `precompute(upcoming_bars)` computes everything at once, and each bar then only moves a cursor. It is 3 to 9 times faster;
  - if a bar does not match the precomputed data, the feed falls back to the bar-by-bar path.

Three families of tests guarantee that both paths are equivalent:

1. **Causality**: the value at `t` computed on the whole history equals the value computed on the history truncated at `t`.
2. **Warmup sufficiency**: with exactly the declared warmup, the value is exact. This is what guarantees that the live buffer is large enough.
3. **Parity**: batch and bar-by-bar give the same features, and every strategy produces the same backtest metrics in both modes.

---

## 5. Models and training

### 5.1 Targets (`ForwardTarget`)

A target looks `horizon` bars ahead:

| `kind` | Value |
|---|---|
| `price` | future close |
| `return_pct` | future return, in % |
| `direction` | 1 if the future close is higher, else 0 |
| `move` | 1 if \|future return\| exceeds `threshold_pct` (in %) |

The labels of the last `horizon` bars are NaN: they are unknown, and they never enter a training set.

### 5.2 No look-ahead

`build_dataset(..., drop_last=n)` removes the last `n` bars from the training history. `n` must be at least the horizon: the ML strategies use `training_margin = 30` for a horizon of 5, and the parameter sheet enforces `training_margin ≥ horizon`. No label depends on a bar after the end of the training history.

### 5.3 `SklearnModel`

A scikit-learn-compatible estimator (RandomForest, XGBoost, …) bundled with its input columns, its target and its score metric:

- training uses the configured `n_jobs`;
- **a single-bar prediction runs on one thread**. Spinning up a thread pool on every bar cost more than the prediction itself: 71 ms against 19 ms for the three models of Cascade;
- cross-validation is time-ordered (`TimeSeriesSplit`) and fits clones of the configured estimator;
- the in-sample score only serves to diagnose overfitting, by comparison with the cross-validated score (`CascadeStrategy.evaluate()`).

`ArimaForecaster` refits an ARIMA(p, d, q) on a rolling window of closes at each bar.

### 5.4 Latency budget

A test checks that the live `on_bar` of the heaviest strategy stays **under 100 ms at the median**. That strategy is Cascade: features, then three predictions. It runs around 50 ms today.

---

## 6. Strategies

A strategy declares its features, optionally trains models, and **emits entry orders**: a side, and optionally a conviction between 0 and 1. The router decides the quantity (§7.1) and executes every exit (§7.2).

| CLI name | Class | Logic |
|---|---|---|
| `trend` | `TrendFollowingStrategy` | crossover of two SMAs (5 / 20 by default) |
| `mean_reversion` | `MeanReversionStrategy` | price well outside the Bollinger Bands, extreme RSI and a volume spike: take the opposite side |
| `ml_hybrid` | `MLHybridStrategy` | mean-reversion trigger confirmed by an XGBoost model predicting the 5-bar direction |
| `cascade` | `CascadeStrategy` | three models in a chain (opportunity? → direction? → amplitude?); trades only when they agree and the predicted amplitude is large enough |
| `arima` | `ArimaStrategy` | trades when the ARIMA forecast beats the round-trip costs |
| `pair` | `PairTradingStrategy` | smoothed z-score of the ratio of two prices: sell the outperformer and buy the other beyond a threshold |

### 6.1 Parameter sheets

Each strategy declares a **parameter sheet** (`PARAMS`, `strategies/params.py`). For each parameter it gives:
- the name and the type (`int`, `float`, `bool`, `choice`, `tuple`);
- the default and the bounds;
- the group: signal, indicators, model, target, sizing or exits;
- a help text.

The sheet is the single reference:

- **CLI**:
  - `quantlab params --strategy NAME` lists every parameter;
  - `--param key=value` applies to each selected strategy that declares the key, and `--param name.key=value` to one;
  - `--params-file presets/X.json` loads a preset, and `--param` overrides it;
  - an unknown key, a wrong type or an out-of-bounds value is an error that lists the valid keys.
- **Web interface**:
  - each strategy's form is generated from its sheet, with one tab per group;
  - errors and warnings are shown before running;
  - presets can be loaded, saved, or reset to the defaults.
- **Optimization**: a min / max / step range per numeric parameter (`--range key=min:max:step`), or a list of values (`--grid`).
- **Sessions**: the full values are saved in `session.json`.

For an optional number, 0 (or `none`) means "none".

**Consistency rules** link several parameters (`Strategy.check`). A violation is an error. Examples:
- MACD fast < MACD slow;
- `training_margin` ≥ `horizon`;
- `exit_zscore` < `zscore_threshold`;
- no ATR level for a pair;
- both a combined stop and a leg stop for a pair.

A **warning** flags a combination that is allowed but questionable. For example, a `holding_bars` different from the horizon of the ML models means positions are not held for the period the models predict.

**Presets** are versioned JSON files in `presets/`, named `<strategy>-<name>.json`: `{"strategy": "trend", "params": {...}}`.

Tests check that each sheet matches its constructor (names and defaults), and that changing each declared parameter really changes the strategy.

Number of tunable parameters, including the shared sizing block (4) and the exit block (6, or 3 for the pair):

| Strategy | Parameters |
|---|---|
| Trend | 12 |
| Pair | 12 |
| ARIMA | 13 |
| Mean Reversion | 19 |
| ML Hybrid | 32 |
| Cascade | 47 |

### 6.2 Exit parameters

Every strategy has **exit parameters** (base class `Strategy`), set with `--param` (`trend.stop_atr=1.5`) or in the "Exits" tab of the interface:

| Parameter | Default | Role |
|---|---|---|
| `stop_loss_pct` | 2 | stop, in % of the entry price (0 = none: the entry is then refused) |
| `take_profit_pct` | 5 | target, in % of the entry price (0 = none) |
| `stop_atr`, `take_profit_atr` | none | distance in ATR multiples, measured at the signal; replaces the matching % |
| `atr_window` | 14 | window of that ATR |
| `holding_bars` | 5 | exit at market after this many bars (0 = none) |

For a pair, the stop and the target apply to the combined P&L, and the ATR parameters do not exist.

A strategy can also:
- adjust the plan of one signal: `mean_reversion` with `target_middle_band=true` targets the Bollinger middle band of the moment;
- ask to close its own positions (`close_positions()`): `pair` does so when the z-score comes back within `exit_zscore`;
- fill `diagnostics` (for example its predictions), which is journaled at every bar.

---

## 7. Order execution

### 7.1 Position sizing (`execution/sizing.py`)

The strategy only says "buy" or "sell", with a conviction between 0 and 1 (`self.order(side, conviction=...)`). Its **sizing rule** (`Sizing`) is built from its size parameters and frozen into the order. The router applies it at the signal's close, before the risk check:

| `size_mode` | Quantity |
|---|---|
| `fixed` (default) | `quantity` shares × conviction, at least 1 (10 by default, 5 for ARIMA) |
| `notional` | `size_value` dollars ÷ price |
| `equity_pct` | `size_value` % of the equity ÷ price |
| `risk` | the loss at the stop is capped at `size_value` % of the equity: dollar risk ÷ distance to the stop |

In `risk` mode, the distance to the stop is floored at `size_min_stop_pct` % of the price (0.1 %), so that a tiny stop cannot produce a huge size.

- Cascade's conviction is min(|predicted amplitude| / 5 %, 1). In `fixed` mode, its quantity is max(1, ⌊10 × conviction⌋).
- **Pair**: the group is sized as a whole, with **the same notional on each leg**.
  - In `fixed` mode, the first leg buys `quantity` shares, and the second leg the same dollar amount.
  - In `risk` mode, the dollar risk is divided by the combined stop.
  - If a leg rounds to 0 shares, the group is refused.
- A size that rounds to 0 refuses the entry. The `RiskManager` limits (§8) then apply, and may still reduce the order.
- An order with an explicit quantity (`self.order(side, 3)`) is sent as is.

### 7.2 Orders and lots

Every entry order carries an **exit plan** (`ExitPlan`, `execution/exits.py`): stop, target, maximum holding period. It opens a **lot** owned by its strategy, and the plan is **frozen** in it. The plan is saved with the lot and journaled with the signal, and it never changes, even after a restart. The `OrderRouter` runs the lot's life cycle:

```
 signal (close of t) + plan ──► sizing, risk check ──► MARKET order ──► fill (open of t+1)
                                                                           │
                       the lot's OCA pair is placed at the broker ◄────────┘
                       ├─ STOP  at the plan's level (e.g. fill × (1 − 2 %), fill − 1.5 ATR)
                       └─ LIMIT at the plan's level (e.g. fill × (1 + 5 %), the middle band)
                                    │
         ┌──────────────────────────┼──────────────────────────┬──────────────────────────┐
         ▼                          ▼                          ▼                          ▼
  the stop is hit           the target is hit          the plan's holding_bars:   the strategy asks
  (target cancelled)        (stop cancelled)           OCA pair cancelled,        to exit: same
                                                       exit at market
```

- **Plan levels**:
  - A level is a **% of the fill price**, a **distance** (for example k × ATR measured at the signal), or an **absolute price**.
  - The first two are anchored to the real fill price, which the strategy cannot know at the signal.
  - If the fill has already gone through an absolute level (a gap), the lot is closed at market at once, with reason `stop_gap` or `target_gap`.
- **Mandatory stop**: an entry **without a stop is refused** (`RiskLimits.require_stop`, on by default).
- **OCA** ("one cancels all"): when one of the two orders fills, the broker cancels the other. At a real broker these protective orders live at the broker, so **they stay active even if the application crashes**.
- **Prices**: protective prices start from the **real fill price** and are rounded to the tick (0.01).
- **Fills**:
  - they are attributed to lots by order id;
  - partial fills are handled: protection is placed only once the entry is complete, or done.
- **Several strategies on one symbol**: each can hold lots there. The broker position is their net sum, and closing a lot changes it by exactly that lot's quantity. Orders are not netted between strategies: each lot's exit stays exact, at the cost of a few extra fees.
- **Live race condition**: if the stop fills while a time exit is in flight, the lot is "over-closed". The excess becomes a correction lot, closed immediately.

### 7.3 Groups of lots (pairs)

Entries that share a `group_id` form a **group**: the two legs of a pair (`Strategy.group()`).

- **All or nothing**: if the risk check refuses or reduces one leg, none is sent, which avoids an unbalanced pair.
- **Common exit**:
  - the plan's stop and target apply to the **combined P&L** of the legs, as a fraction of their gross entry notional;
  - they are checked at each bar close, and all the legs exit together at market;
  - the time exit and a strategy-requested exit also close the whole group.
- **Leg stop**:
  - an order resting at a broker cannot watch a spread, so each leg also carries a wide stop (`leg_stop_pct`, 5 % for `pair`) that protects it if the application is down;
  - when it fills, the other legs are closed at once (`leg_stop`).
  - A pair needs **both stops**: the combined stop, which is the real stop of the pair, and the leg stop on top. Otherwise the entry is refused, and the `pair` parameter sheet rejects 0 for either one.
  - A warning is shown if the leg stop is tighter than the combined stop: it could fire on a common market move while the pair itself is flat.
- **Missing leg**: if the entry of one leg is rejected, or cancelled without a fill, the other legs are closed (`leg_missing`).

### 7.4 The simulated broker

The `SimulatedBroker` follows realistic, conservative rules:

- an order decided on the close of bar `t` is **never** filled at that close. A market order fills at the **open of `t+1`**, plus slippage against the order (1 bp by default);
- a **stop** triggers when the bar's high / low crosses it. It fills at the stop price, or at the open if the bar gapped through it, plus slippage;
- a **limit** fills at its price, or at a better open;
- protective orders placed at the open's fill are tested on the rest of that same bar;
- when a bar touches **both** the stop and the target, the **stop is taken** (the conservative choice);
- **fees** are 1 bp of the notional per side.

### 7.5 Portfolio

The `Portfolio` tracks the cash, the net position and the average price per symbol, and the fees paid. The **equity** is the cash plus the market value of the positions at the last close. It is recorded at every bar: this is the equity curve.

---

## 8. Risk management

The `RiskManager` checks **every entry order** before it is sent. Exits are never blocked: stops, targets, time exits, flattening. The stop, target and holding period of a position are not risk limits: they come from the strategy's exit plan (§7.2). The limits below protect the account.

| Limit (`RiskLimits`) | Effect |
|---|---|
| `require_stop` | an entry whose exit plan has no stop is refused (on by default) |
| `max_order_quantity` | the order's quantity is **reduced** to this cap |
| `max_position_quantity` | net position per symbol, pending entries included: the order is reduced, or refused when there is no room left |
| `max_daily_loss_pct` | **kill switch**: when the equity falls this percentage below its start-of-day value, trading stops, pending entries are cancelled and, by default (`flatten_on_halt`), every position is closed |

---

## 9. Measuring performance

### 9.1 Metrics (`compute_metrics`)

They are computed on the **equity curve**, not on a sum of per-trade percentages:

- `total_return_pct`: final equity against the initial capital;
- `sharpe_ratio`, `sortino_ratio`: on daily returns (New York sessions), annualized by √252;
- `max_drawdown_pct`: the largest peak-to-trough fall of the equity;
- per-trade statistics: `total_trades`, `win_rate`, `profit_factor`, `avg_trade_return_pct`, `total_pnl_net`, `total_fees`.

Positions still open at the end of a backtest are marked at the last close and counted as trades flagged `open`.

### 9.2 Walk-forward optimization

Each fold of the `WalkForwardOptimizer` follows this split:

```
[ train ][ embargo ][ validation ][ embargo ][ test ]
```

1. Each combination of the grid is trained on *train* and scored on *validation*.
2. The best combination is refitted on *train + validation*.
3. It is scored **once** on *test*, which was never used to choose it.

The embargo drops bars between the blocks, so that targets built on future prices cannot leak from one block into the next. The result gives the out-of-sample score per fold and the **stability of the chosen parameters**: a choice that changes at every fold reveals noise.

---

## 10. Sessions: journal, restart, replay

A live session is the same runner on a real-time `BarFeed`, with a `Journal` and a `StateStore`. These parts of the runtime are independent of any broker.

### 10.1 Session files

Each session has its own folder `sessions/<name>/`:

| File | Content |
|---|---|
| `session.json` | mode, symbols, capital, risk limits, simulator costs, full parameter values of each strategy |
| `strategies.pkl` | the strategies **right after training**: models and buffers |
| `warmup.parquet` | the training history |
| `journal.sqlite` | everything that happened (see below) |
| `state.json` | the runner state after the last bar |

### 10.2 The journal

The journal is a SQLite database in WAL mode. Rows are only appended, it survives crashes, and it can be read while the session runs. It holds:

- the `bars` table: every bar seen. A `catch_up` column flags the bars caught up after a restart;
- the `events` table, one row per event, with the bar, the wall-clock time and a JSON payload:
  - `features`: each strategy's feature row;
  - `decision`: its diagnostics, such as predictions;
  - `signal`: the order emitted, its sizing rule and exit plan, and the quantity the risk check let through;
  - `order`, `cancel`, `fill`, `status`: recorded at the broker boundary;
  - `trade`: closed trades;
  - `exit`: exits requested by a strategy;
  - `equity`: equity, cash, positions, kill-switch state, latency;
  - `session`: start, resume, stop.

### 10.3 Persisted state and restart

After **every bar**, the complete state is written atomically: to a temporary file, then renamed, so that a crash never leaves a half-written file. It covers:
- the portfolio;
- the open lots and their protective orders;
- the pending orders;
- the risk state (kill switch, start-of-day equity);
- the broker and the strategies.

A restart then works like this:

1. The **same models** are reloaded from `strategies.pkl`, without retraining.
2. The feature buffers are rebuilt from fresh history (`strategy.warm`).
3. The saved state is restored.
4. The broker recovers its working orders and the executions missed while down.
5. The missed bars are **caught up** (`runner.catch_up`).
   - The book advances: the simulator fills what was pending, holding periods progress and time exits are sent.
   - No new decision is taken on these past bars.

A test checks that **cutting a run in the middle of a trade and resuming it gives exactly the same trades and the same equity** as an uninterrupted run.

### 10.4 Replay: live against backtest

`quantlab replay --session <name>` reloads the strategies as they were before the first live bar, runs them on the journaled bars with the simulated broker, and compares both journals:

- **signal agreement**: for each (bar, strategy, symbol, side), is the signal present live, in the replay, or in both? Caught-up bars are reported apart, since the live run could not decide on them;
- **entry slippage**: live price against simulated price, in bp, positive when the live fill was worse;
- **feature differences** between live and replay. They should be zero: the same code ran on the same bars;
- P&L and trade count on both sides, p50 and p95 latency.

`bar_differences` also compares the live bars with history downloaded later, to spot data corrections made by the provider.

---

## 11. Known limitations

- Model cross-validation (`TimeSeriesSplit`) has no gap between training and validation, so labels at the boundary can overlap slightly. The walk-forward optimizer, with its embargo, does not have this flaw.
- Only stocks, and MKT, LMT and STP orders. The protective OCA pair is placed after the entry fills, not as a native bracket order.
- No netting of orders between strategies.
- Pair legs are dollar-neutral, not beta-neutral.
- Strategies are saved with `pickle`: only load sessions you produced yourself.
