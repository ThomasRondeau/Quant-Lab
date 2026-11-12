# How Quant Lab works

This document explains **how the system works**:
- the path of a market bar through the engine;
- how features stay causal;
- how models are trained without look-ahead;
- how orders are sized, simulated and protected;
- risk controls, metrics, and the session journal, restart and replay;
- what each page of the web interface shows.

For the code layout, the command-line reference and how to extend the platform, see [CODE_GUIDE.md](CODE_GUIDE.md).

---

## 1. Goal and guiding principle

Quant Lab is a research platform for intraday strategies built on technical indicators and machine learning. It is used to **backtest** them, **optimize** them with walk-forward validation, and run them **bar by bar exactly as they would run live**.

The guiding principle: **one code path for backtest and live**. Features, models, strategies, sizing, exits, risk and order routing are identical in every mode. Only two components change:

| Component | Backtest | Simulated live trading | Real live trading |
|---|---|---|---|
| Bar source (`BarFeed`) | `HistoricalBarFeed`, a replay of a DataFrame | `PacedBarFeed`, bars delivered in real time | a market-data feed (same protocol) |
| Execution (`Broker`) | `SimulatedBroker` | `SimulatedBroker` | a broker connector (same protocol) |

The first two modes ship with the platform. No market-data feed or broker connector is included yet: both are written against these two protocols ([CODE_GUIDE.md §5.5](CODE_GUIDE.md)).

Any backtest-only optimization, such as precomputing the features, must be **proven identical** to the live path by a parity test.

The system is driven from the command line (`python -m quantlab …`) or from the **web interface** (`python -m quantlab ui`, §11). Both call the same library functions. The interface opens on a data page, which analyzes a set of bars over a chosen period (price, returns, intraday profile, sessions, quality), and adds charts: equity curve, drawdown, per-trade P&L, scores per fold.

### 1.1 Hosted mode

The interface also runs as a public online demo. The environment variable `QUANTLAB_HOSTED` (`1`, `true`, `yes` or `on`; `ui/hosting.py`) turns the hosted mode on. The interface then:

- reads data from the bundled samples, from uploads (10 MB at most, `.streamlit/config.toml`) or from yfinance, and never from a path typed on the server;
- keeps the presets a visitor saves in that visitor's session, and writes nothing in the project folder. Uploaded files and live sessions go to the temporary folder of the server;
- caps one request (`Limits`). A request over a limit gets a message instead of running:

  | Limit | Value |
  |---|---|
  | Bars loaded | 12,000 |
  | Strategies in one backtest | 3 |
  | Test bars when ARIMA is selected | 200 |
  | Walk-forward runs (combinations × folds): `trend`, `mean_reversion`, `pair` | 60 |
  | Walk-forward runs: `ml_hybrid` | 24 |
  | Walk-forward runs: `cascade` | 4 |
  | Walk-forward runs: `arima` | none: it cannot be optimized online |

- runs simulated live sessions in a temporary folder: 3 at once across all visitors, 1 per visitor, 975 bars and 15 minutes at most. Ended sessions are deleted once they are an hour old, when a visitor starts a new one. A visitor is identified by a random id kept in the page URL, so a reload finds its sessions again;
- shows a note in the sidebar, with a link to the source code.

Two protections apply in every mode, hosted or local:

- at most 2 heavy computations (backtests, optimizations, session replays, and in hosted mode the training of a live session) run at once in the server process. The others wait up to 90 seconds, then get a "server busy" message;
- every backtest is cached by its full specification (64 entries), for all visitors, so the default runs are instant after the first one.

---

## 2. Overview

```
                 ┌───────────────────────────────────────────┐
  Historical     │ bundled samples · yfinance · parquet / csv│
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

Between steps 5 and 6, the runner passes to the router the close requests the strategy made while deciding (`close_positions()`, §6.2).

A **backtest** is this runner fed by a `HistoricalBarFeed` and wired to a `SimulatedBroker`. There is no separate backtesting engine.

---

## 3. Data

### 3.1 Sources

| Source | Resolution | Depth | Requirement |
|---|---|---|---|
| `sample` (default) | 2 min | 30 sessions (5,800 bars, about six weeks) for each of AAPL, MSFT, NVDA, SPY, QQQ | none: bundled in `quantlab/data/samples/`, offline |
| `yfinance` | 2 min | ~60 days | network (free, no API key) |
| `file` | any | any | parquet or csv |

The history window is a number of calendar days (`--days`, "Days of history"):

- **`yfinance`**: from that many days ago to today (`DataManager` follows the convention `(symbol, days_back, until)`: from `days_back` days ago to `until` days ago). Prices are not adjusted for splits or dividends.
- **`sample`**: the days count back from the end of the sample, so the default of 55 days returns it whole.
- **`file`**: the whole file is used, and the window is ignored. A file that already has the multi-symbol layout (§3.3) is read as is.

### 3.2 One format (`validate_ohlcv`)

Every source returns a normalized DataFrame:

- a **UTC** `DatetimeIndex` named `Datetime`, sorted and unique. A timestamp without a time zone is taken as UTC;
- lowercase `float` columns `open, high, low, close, volume` (any other column is kept);
- no row without a close.

When the timestamps are not the index, they are read from a `datetime`, `date` or `timestamp` column. A file without one of the five columns is refused.

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

**Consistency rules** link several parameters (`Strategy.check`). A violation is an error:

| Strategy | Rule |
|---|---|
| all | a `size_mode` other than `fixed` needs a `size_value` |
| `trend` | `sma_fast` < `sma_slow` |
| `ml_hybrid`, `cascade` | `macd_fast` < `macd_slow`; `training_margin` ≥ `horizon`; distinct `momentum_windows` |
| `ml_hybrid` | `rsi_buy` < `rsi_sell` |
| `cascade` | `volume_fast_window` < `volume_slow_window` |
| `pair` | `exit_zscore` < `zscore_threshold`; no ATR level; both a combined stop (`stop_loss_pct`) and a leg stop (`leg_stop_pct`) |

A **warning** flags a combination that is allowed but questionable. There are two:
- for `ml_hybrid` and `cascade`, a `holding_bars` different from the `horizon` of the models: positions are not held for the period the models predict;
- for `pair`, a leg stop no wider than the combined stop (§7.3).

**Presets** are versioned JSON files in `presets/`, named `<strategy>-<name>.json`: `{"strategy": "trend", "params": {...}}`. A preset only lists the values that differ from the defaults. One ships for each strategy:

| Preset | Changes |
|---|---|
| `trend-atr-exits` | stop at 1.5 ATR, target at 3 ATR, 20 bars of holding |
| `mean_reversion-middle-band` | target at the Bollinger middle band, RSI threshold 25, volume threshold 1.5 |
| `arima-slow` | order (1, 1, 1), window of 100 closes, minimum edge of 4 bp |
| `ml_hybrid-shallow` | 200 trees of depth 3, learning rate 0.03 |
| `cascade-horizon-10` | horizon and holding of 10 bars, training margin 40, minimum amplitude 0.1 % |
| `pair-wide` | entry at a z-score of 2.5, exit at 0.25, 30 bars of holding |

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
- fill `diagnostics` (for example its predictions). It is reset at every bar and journaled whenever the strategy filled it.

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

Every closed trade records why its lot was closed (the `reason` column of the trades table and of the CSV export):

| `reason` | The lot was closed by |
|---|---|
| `stop_loss`, `take_profit` | its protective stop or target; for a pair, the combined stop or target of the group |
| `time` | the end of its holding period |
| `stop_gap`, `target_gap` | a fill already beyond an absolute level |
| `signal` (default), or the name the strategy gives, such as `zscore` | a request of its strategy (`close_positions`) |
| `leg_stop` | a pair: the protective stop of a leg; the other legs carry the same reason |
| `leg_missing` | a pair: the entry of another leg was not filled |
| `kill switch` | the daily-loss limit (§8) |
| `correction` | the immediate close of an over-fill |
| `open` | nothing: the position was still open at the end of the backtest, and is marked at the last close |

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
- a **limit** fills at its price, or at a better open, without slippage;
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
| `max_daily_loss_pct` | **kill switch**: when the equity falls this fraction (0.01 = 1 %) below its start-of-day value, trading stops, pending entries are cancelled and, by default (`flatten_on_halt`), every position is closed |

Only `require_stop` is on by default. The three other limits are unset unless they are given: `--max-order`, `--max-position` and `--max-daily-loss` on the command line, or the "Execution and risk" panel of the interface. Both take the daily loss as a percentage.

The kill switch is checked at each bar close. Its day is the UTC date of the bar, which is the New York date during regular trading hours. Once triggered, it stays on for the rest of the run, the following days included: only `RiskManager.resume()` turns trading back on.

---

## 9. Measuring performance

### 9.1 Metrics (`compute_metrics`)

They are computed on the **equity curve**, not on a sum of per-trade percentages:

- `total_return_pct`: final equity against the initial capital;
- `sharpe_ratio`, `sortino_ratio`: on daily returns (New York sessions), annualized by √252;
- `max_drawdown_pct`: the largest peak-to-trough fall of the equity;
- per-trade statistics: `total_trades`, `win_rate`, `profit_factor`, `avg_trade_return_pct`, `total_pnl_net`, `total_fees`.

Details:
- a trade is one closed lot, so a pair counts two trades per round trip, one per leg;
- a trade's return is its net P&L over its entry notional, and a trade wins when its net P&L is positive;
- the profit factor is the sum of the gains over the sum of the losses, infinite when no trade loses;
- with fewer than two sessions in the test block, there is a single daily return: the Sharpe and Sortino ratios are then 0;
- the Sortino ratio divides by the downside deviation, computed over all the days. It is 0 when no day loses.

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

Settings:

| Setting | Default | Role |
|---|---|---|
| train, validation and test sizes | none: required | block sizes, in bars |
| `embargo` | 30 bars (command line and interface) | bars dropped between two blocks |
| `step` | the test size | shift between two folds; the default gives contiguous test blocks |
| `anchored` | off | the training block always starts at the first bar and grows, instead of sliding with a constant size |
| `objective` | `total_return_pct` | the metric of §9.1 to maximize |
| `min_trades` | 5 (command line and interface) | a combination with fewer trades on the scored block is discarded |

- A combination whose backtest fails is discarded too, and the error is logged. If every combination of a fold is discarded, a warning is logged and the first combination of the grid is used for that fold.
- The minimum number of trades also applies to the test block: a fold that trades less there has no test score, and is left out of the mean.
- The folds are cut as long as a whole fold fits in the data: train + validation + test + 2 × embargo bars for the first, one step more for each following one. The bundled sample of one symbol (5,800 bars) gives a single fold with blocks of 3,000 / 1,000 / 1,000 bars.
- Each backtest of the search uses the default capital and costs: $100,000, fees of 1 bp and slippage of 1 bp.

---

## 10. Live sessions: simulated trading, journal, restart, replay

A live session is the same runner on a real-time `BarFeed`, with a `Journal` and a `StateStore`. These parts of the runtime are independent of any broker.

### 10.0 Simulated live trading (`live/`)

`quantlab live` and the "Live trading" page (§11.4) run one strategy on a real-time stream without any market connection:

```
 bars (bundled sample, or any source) ──► split: history │ live bars
                                                  │            │ replay: the last N bars
                                                  │            │ synthetic: N new bars continuing the history
                                                  ▼            ▼
                              strategy.fit(history)    PacedBarFeed: one bar every 1 / speed seconds,
                              snapshot (session.json,  heartbeats in between, stop() from any thread,
                              strategies.pkl)          an optional wall-clock limit
                                                  │            │
                                                  ▼            ▼
                       StrategyRunner(SimulatedBroker, Journal, StateStore).run(feed)
```

- **Two markets**:
  - `replay` streams the last N bars of the data, and the strategy is trained on the bars before them;
  - `synthetic` trains the strategy on all the data, then streams N generated bars that continue it.

  N is 975 by default on the command line (five sessions of 2-minute bars) and 390 in the interface.
- **Pace** (`PacedBarFeed`): a number of bars per second, 2 by default. On the command line, 0 means as fast as possible. Between two bars, the feed sends a heartbeat at least every 0.25 second, so that a stop request is seen at once.
- **Data**: the bundled samples by default. The command line also takes yfinance (`--data yfinance`); the interface only uses the samples.
- **Synthetic market** (`synthetic_bars`):
  - log-normal returns with each symbol's realized volatility, correlated across symbols (0.8 by default, for pairs);
  - higher volatility near the open and the close;
  - volumes around the history's median;
  - timestamps in the New York regular sessions, weekdays only.

  It is reproducible with a seed. It is a plausible stream to trade on, not a forecast.
- **Background sessions** (`SimulatedSessions`):
  - the web interface reruns its script on every interaction, so sessions run in threads, in a registry that lives as long as the server process;
  - each session reports its status (loading data, training, running, finished, stopped, error) and its progress;
  - the page refreshes itself every 2 seconds while a session runs.
- **Live view** (`ui/live_view.py`): built from the session's files only, the journal and the state saved after every bar:
  - equity curve, realized and unrealized P&L;
  - open lots with their stop, target and holding time;
  - an activity feed of signals (with the size the risk checks let through), orders, fills and trades.
- **Replay**: a session ended this way replays with 100 % signal agreement and the same P&L. The same code runs on the same bars with the same simulated broker, which is the parity the platform guarantees.

### 10.1 Session files

Each session has its own folder `sessions/<name>/` (`--sessions-dir` to change the parent folder; `sessions/` is ignored by git). The name is `<strategy>_<symbols>_<date>_<time>` unless `--session` gives one. A name can be used once: starting a session in an existing folder is refused.

| File | Content |
|---|---|
| `session.json` | mode, market (replay or synthetic), symbols, pace, number of live bars, capital, risk limits, simulator costs, full parameter values of each strategy |
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
  - `session`: start and stop.

`read_journal(path)` returns both tables as DataFrames, and `.kind("fill")` expands the payloads of one kind into columns.

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

The restart is a library procedure, and no command runs it yet: `SessionFiles.load_strategies()`, `Strategy.warm(history)`, `StrategyRunner.set_state(StateStore.load())`, then `catch_up(bars)` and `run(feed)`. A simulated session that is stopped stays stopped; it can still be replayed.

### 10.4 Replay: live against backtest

`quantlab replay --session <name>` reloads the strategies as they were before the first live bar, runs them on the journaled bars with the simulated broker, and compares both journals:

- **signal agreement**: for each (bar, strategy, symbol, side), is the signal present live, in the replay, or in both? Caught-up bars are reported apart, since the live run could not decide on them;
- **entry slippage**: live price against simulated price, in bp, positive when the live fill was worse;
- **feature differences** between live and replay. They should be zero: the same code ran on the same bars;
- P&L and trade count on both sides, p50 and p95 latency.

`bar_differences` also compares the live bars with history downloaded later, to spot data corrections made by the provider.

---

## 11. The web interface

`python -m quantlab ui` opens the interface in the browser, on port 8501 (`--port` to change it). It has four pages. Each one calls the library functions that the command line calls, and shows times in New York time.

Three blocks are shared by several pages:

- **Data**: the symbols (comma-separated), the source (§3.1) and the days of history. The file source takes an upload, or a typed path when the interface runs locally.
- **Parameters**: the form of a strategy, generated from its parameter sheet (§6.1), with one tab per group. Above it, a preset can be loaded and the defaults restored. Under it, the values changed from the defaults, the errors and the warnings are listed, and the current values can be saved as a preset: a file in `presets/` locally, the visitor's session in hosted mode.
- **Execution and risk**: capital, fees, simulated slippage, and the three account limits of §8, unset by default.

### 11.1 Data (landing page)

The page analyzes a set of bars before it is traded. It loads the bars as soon as it opens, without a button: the bundled AAPL sample by default.

- **Symbols**: one to eight.
- **Period slider**: both ends move, by steps of one bar. The chart and every figure below follow the period.
- **Price chart**: the close, as a line. Nights and weekends take no room on the time axis.
  - With one symbol, the average volume per bar is drawn under the price.
  - With several, the closes are rebased to 100 at the start of the period, and a selector chooses the symbol analyzed below.
- **Tiles**: bars, sessions, bar interval, missing bars, return over the period, low and high, annualized volatility, average volume per session.
- **Tabs**:

  | Tab | Content |
  |---|---|
  | Returns | histogram of the bar returns, without the 0.5 % most extreme on each side; mean, standard deviation, worst, best, skewness, excess kurtosis, share of bars up; with several symbols, the correlation of their returns |
  | Intraday profile | average volume per bar and average absolute bar return, by time of day |
  | Sessions | one row per session (open, high, low, close, return, range, volume, bars), a chart of the session returns, a CSV download |
  | Quality | missing bars, incomplete sessions, largest gap inside a session, bars without volume, inconsistent bars, bars with a missing value |
  | Bars | the bars of the period, with a CSV download |

The figures come from `analysis/dataset.py`, with these definitions:

- a **session** is a New York date;
- the **bar interval** is the most frequent spacing between two bars;
- a **bar return** is close to close inside a session, so the overnight gap is left out;
- a **session return** is measured on the previous close, and on its own open for the first session; the **range** is the high over the low;
- the **volatility** is the standard deviation of the bar returns, annualized on the median number of bars per session and 252 sessions;
- a **missing bar** is a hole inside a session, between its first and its last bar, on the grid of the bar interval;
- an **incomplete session** holds fewer bars than the median session: a half day, or a session cut by the period;
- an **inconsistent bar** has a high under its open or its close, a low above them, or a low at or under zero.

### 11.2 Backtest

- **Inputs**:
  - one or several strategies; several form one portfolio on a single account;
  - the data block;
  - the test block: the last N calendar days (5 by default) or the last fraction of the bars. The bars before it train the strategies;
  - "Also each strategy alone", to compare the portfolio with its parts;
  - "Bar-by-bar live path", to compute the features as live does (§4.3);
  - the parameters of each strategy, and the execution and risk block.
- **Results**:
  - the training and test periods;
  - tiles: return, net P&L, trades, win rate, profit factor, Sharpe ratio, maximum drawdown;
  - the equity curve of each run, and the drawdown of the portfolio;
  - with several runs, a table of all the metrics of §9.1;
  - the P&L of each trade, the trades table with their exit reason (§7.2), and two CSV downloads: trades and equity curve.

### 11.3 Optimization

The page runs the walk-forward optimization of §9.2 on one strategy.

- **Inputs**:
  - the data block, and the parameters that stay fixed;
  - the grid: a minimum / maximum / step range for a number, a list of values for a boolean or a choice. Tuple parameters, such as the ARIMA order, cannot be put in the grid;
  - the sizes of the training, validation and test blocks (3,000 / 1,000 / 1,000 bars by default), the embargo, the step between folds, the objective, the minimum number of trades, and the anchored option.
- **Objectives**: `total_return_pct`, `sharpe_ratio`, `sortino_ratio`, `profit_factor`, `total_pnl_net`, `win_rate`.
- **Results**: the mean out-of-sample score, the test score of each fold, how many folds chose each combination, and the table of the folds with their chosen parameters and test metrics.

### 11.4 Live trading

The page runs the simulated sessions of §10.0.

- **Inputs**: one strategy, its symbols among the bundled samples (SPY and QQQ by default for the pair), its parameters, the market (replay or synthetic), the number of bars to stream (50 to 5,000, 390 by default: two sessions), the speed (1, 2, 5, 10 or 20 bars per second), the seed of the synthetic market, and the execution and risk block.
- **Sessions**: each one runs in the background and has its panel, with its status, its progress and a Stop button. The panel of the most recent session is open, and it refreshes every 2 seconds while a session runs.
- **An open panel shows**:
  - tiles: equity, realized and unrealized P&L, trades and winners, positions, last bar, median latency, kill switch;
  - the equity curve;
  - the open positions, with entry, last price, unrealized P&L, stop, target and bars held;
  - the last 40 events, newest first: signals, orders, fills, trades, exit requests, cancellations, statuses;
  - the P&L of each closed trade.
- **Replay**: once the session has ended, "Replay in the backtester" runs §10.4 and shows the signal agreement, the number of signals on each side, the entry slippage and the net P&L on each side.

Sessions started locally are saved in `sessions/`, so `quantlab replay --session <name>` reads them too. The list on the page lasts as long as the server process.

---

## 12. Known limitations

- Live trading is simulated only: no market-data feed and no broker connector ship with the platform, and no command resumes a session after a crash (§10.3).
- Model cross-validation (`TimeSeriesSplit`) has no gap between training and validation, so labels at the boundary can overlap slightly. The walk-forward optimizer, with its embargo, does not have this flaw.
- Only stocks, and MKT, LMT and STP orders. The protective OCA pair is placed after the entry fills, not as a native bracket order.
- No netting of orders between strategies.
- Pair legs are dollar-neutral, not beta-neutral.
- Strategies are saved with `pickle`: only load sessions you produced yourself.
