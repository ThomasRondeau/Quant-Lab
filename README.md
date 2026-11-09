# Quant Lab

A research platform for intraday trading strategies built on technical indicators and machine learning: features, models, strategies, a realistic bar-by-bar backtester, walk-forward optimization and a web interface.

The engine is **live-ready**. Backtest and live trading share one trading loop and the same features, strategies, sizing, exits and risk code. Only the bar source and the broker change, so a strategy moves from research to execution without a rewrite. Every shortcut taken for backtest speed is proven identical to the live path by tests.

## Highlights

- **One trading loop for backtest and live**: `StrategyRunner` runs a `BarFeed` against a `Broker`. A backtest is that runner fed historical bars with a `SimulatedBroker`.
- **Realistic execution**:
  - an order decided on the close of bar *t* fills at the open of *t+1*, with slippage and fees;
  - stops and targets trigger on high / low, with gaps;
  - when one bar touches both the stop and the target, the stop wins.
- **Causal features with declared warmups**: SMA, RSI, ATR, MACD, Bollinger Bands, ADX, session VWAP, z-scores, plus composite sets for the models. Tests check causality, warmup sufficiency, and **parity** between the batch path (3 to 9× faster) and the bar-by-bar live path.
- **Six strategies**:

  | Strategy | Logic |
  |---|---|
  | `trend` | moving-average crossover |
  | `mean_reversion` | filtered fade of moves outside the Bollinger Bands |
  | `arima` | statistical forecast |
  | `ml_hybrid` | technical trigger confirmed by XGBoost |
  | `cascade` | three models in a chain: opportunity, then direction, then amplitude |
  | `pair` | z-score of the price ratio of two assets |

- **Per-position exits**: each entry carries an exit plan, frozen into its lot:
  - a stop and a target, as a %, as ATR multiples or as an absolute price;
  - a holding period;
  - exits the strategy asks for itself.

  The stop is mandatory. A pair trades as one group: all or nothing, with a stop on the combined P&L and a protective stop on each leg.
- **Position sizing**: fixed quantity, notional, % of equity, or **fixed risk at the stop**, scaled by a conviction the strategy provides. The two legs of a pair always hold the same notional.
- **Risk controls**: maximum order and position sizes, and a daily-loss kill switch that flattens every position.
- **Every parameter is tunable and validated**: each strategy declares a typed parameter sheet (type, default, bounds, help). Up to 47 parameters per strategy: indicators, thresholds, model hyperparameters, target, sizing, exits. Rules check consistency, for example MACD fast < slow, or the training margin ≥ the prediction horizon. Presets are JSON files.
- **Walk-forward optimization**: train / validation / test folds with embargoes. The chosen parameters are scored once out of sample, and their stability across folds is reported.
- **Live-ready runtime**:
  - SQLite journal of every bar, feature row, prediction, order and fill;
  - atomic state snapshots after each bar;
  - restart that reproduces an uninterrupted run exactly;
  - a replay that compares a session with the backtester.
- **Web interface** (Streamlit):
  - forms generated from the parameter sheets;
  - equity, drawdown and per-trade P&L charts;
  - optimization ranges;
  - session charts with VWAP and support / resistance.
- **Engineering**: 170 tests (offline, no network), `mypy --strict`, ruff, pre-commit hooks, and a 100 ms per-bar latency budget enforced by a test.

## Quick start

Requires Python 3.10 or later.

```bash
git clone <repository-url> quant-lab && cd quant-lab
python -m pip install -e ".[dev]"
```

Run a backtest offline on the bundled sample (AAPL, 1-minute bars):

```bash
python -m quantlab backtest --strategy trend --strategy mean_reversion --symbol AAPL \
    --source file --file tests/fixtures/aapl_1m.parquet --test-fraction 0.3 --each
```

Use recent market data from yfinance (free, no API key; about 60 days of 2-minute bars):

```bash
python -m quantlab backtest --strategy cascade --symbol SPY --days 55 --test-days 5
python -m quantlab backtest --strategy pair --symbol AAPL --symbol MSFT --days 55 \
    --param size_mode=notional --param size_value=20000
```

Open the web interface:

```bash
python -m quantlab ui
```

Other commands:

```bash
python -m quantlab params --strategy cascade          # every parameter: group, default, bounds, help
python -m quantlab backtest --strategy trend --symbol AAPL --params-file presets/trend-atr-exits.json
python -m quantlab optimize --strategy trend --symbol AAPL --range sma_fast=5:20:5 --grid sma_slow=30,50,100 \
    --train-size 3000 --validation-size 1000 --test-size 1000 --embargo 30
python -m quantlab plot --symbol SPY --days 3           # one session: candles, VWAP, support / resistance
python -m quantlab --help
```

## How it works

```
 historical bars ──► strategy.fit(history): warm the feature buffer, train the models
                                  │
 ┌──────────── StrategyRunner: one loop, one thread, the same in backtest and live ──────────┐
 │  for each bar of the BarFeed:                                                              │
 │    1. broker.on_bar         the simulator fills pending orders (next open, stops, limits)  │
 │    2. portfolio.mark        mark positions at the close                                    │
 │    3. router.on_bar_close   time exits, combined stops of pairs, kill switch               │
 │    4. strategy.on_bar       features → decision → entry orders (side + conviction)         │
 │    5. router                size the entry, check the risk, send it, then place its        │
 │                             protective stop / target (one-cancels-other) once filled       │
 │    6. equity curve, latency, journal, state snapshot                                       │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
```

Details:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the system works (data, features, models, execution, risk, metrics, sessions);
- [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md): code layout, main interfaces, and how to add a feature, a strategy, a model or a broker;
- [docs/strategies/](docs/strategies/): one write-up per strategy.

## Results: a word of caution

The bundled fixtures cover about 2.5 days of test data. The tests use them to detect regressions, and they are far too short to measure performance:

| Strategy | Trades | Win rate | Return | Profit factor | Max drawdown |
|---|---|---|---|---|---|
| Pair trading (AAPL / MSFT, 2 min) | 56 | 42.9 % | +0.108 % | 1.64 | −0.08 % |
| Cascade | 25 | 52.0 % | +0.001 % | 1.22 | −0.005 % |
| ARIMA | 18 | 44.4 % | ≈ 0 % | 1.00 | −0.007 % |
| Mean reversion | 21 | 42.9 % | −0.056 % | 0.35 | −0.08 % |
| ML Hybrid | 31 | 38.7 % | −0.125 % | 0.26 | −0.16 % |
| Trend following | 72 | 19.4 % | −0.096 % | 0.36 | −0.10 % |

Fees are 1 bp per side and slippage 1 bp. No edge is claimed here. The point of the project is the research infrastructure: honest execution, no look-ahead, and walk-forward validation.

## Project layout

```
quantlab/
├── data/         yfinance and file loading, one OHLCV contract (UTC), multi-symbol layout
├── features/     causal features, dependency-resolving pipeline, composite sets for the models
├── models/       scikit-learn / XGBoost wrapper with time-ordered cross-validation, rolling ARIMA
├── strategies/   strategy base class, parameter sheets, the six strategies, registry
├── execution/    orders, sizing, exit plans, risk, order router (lots, groups), simulated broker
├── runtime/      bar buffer, feature feed, trading loop, journal, state, session files
├── backtest/     backtest entry point, equity-curve metrics, walk-forward optimizer
├── analysis/     session replay, session chart
├── ui/           Streamlit interface
└── cli.py        command line
presets/          parameter presets (JSON)
tests/            offline test suite and frozen data fixtures
```

## Development

```bash
python -m pytest                    # full suite (~3 min)
python -m pytest -m "not slow"      # without the end-to-end backtests
python -m ruff check . && python -m ruff format --check .
python -m mypy quantlab             # strict
python -m pre_commit install        # ruff + mypy before each commit
```

The characterization tests backtest every strategy on frozen fixtures, in both feature paths, and compare all metrics to `tests/fixtures/baseline_metrics.json`. A refactor must keep them green. Regenerate the baseline (`UPDATE_BASELINE=1 python -m pytest tests/test_characterization.py`) only when a change is meant to alter results, and review the diff.

## Roadmap

- Simulated live trading: run a strategy on a replayed or synthetic real-time feed, and watch its orders, fills and positions update in the interface.
- Longer research datasets and multi-month walk-forward studies.
- Broker connectors behind the existing `Broker` and `BarFeed` interfaces.
