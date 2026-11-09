# Pair trading and multi-symbol support

`quantlab/strategies/pair_trading.py` · `quantlab/data/pairs.py` · CLI name: `pair`

Pair trading plays the spread between two correlated assets, AAPL and MSFT by default. This document describes the strategy and how the platform handles several symbols.

## 1. Multi-symbol data

### Bar layout

A multi-symbol bar contains:

- for **each** symbol, its suffixed columns: `open_AAPL, high_AAPL, low_AAPL, close_AAPL, volume_AAPL, open_MSFT, …`;
- the plain `open … volume` columns, which repeat those of the **first** symbol, for code that knows a single asset only.

### Synchronization

Each symbol is loaded separately, then `combine_symbols` aligns them with an **inner join** on the timestamps. A minute missing for either symbol is dropped. A live feed produces the same layout.

### Execution price per symbol

The simulated broker and the portfolio valuation read the price of each order from its symbol's column: `bar_value(bar, "MSFT", "open")` reads `open_MSFT`. They fall back to the plain column for a single-symbol bar. An order on MSFT therefore fills at MSFT's price.

## 2. Features: the synthetic asset

The ratio is itself a feature, and the pipeline chains three composable features:

```
Ratio(close_AAPL, close_MSFT)            -> ratio_close_AAPL_close_MSFT
  └─ ZScore(ratio, 30)                   -> zscore_30_ratio_close_AAPL_close_MSFT
       └─ SMA(3, source=zscore)          -> sma_3_zscore_30_ratio_...   (3-bar smoothing)
```

- The pipeline resolves these dependencies, computes each stage once and sums the warmups: 0 + 29 + 2 bars.
- The z-score measures by how many standard deviations the ratio departs from its 30-bar moving average. The 3-bar smoothing filters isolated spikes.
- These features are causal and tested like all the others (causality, warmup, batch / bar-by-bar parity).
- The live buffer is **bounded** by the warmup plus a margin, so the cost per bar stays constant.

## 3. Decision rules

Let `z` be the smoothed z-score of the current bar:

| Condition | Reading | Orders |
|---|---|---|
| `z ≥ zscore_threshold` (2.2) | AAPL is expensive relative to MSFT | **sell** 10 AAPL and **buy** the same dollar amount of MSFT |
| `z ≤ −zscore_threshold` | AAPL is cheap relative to MSFT | **buy** 10 AAPL and **sell** the same dollar amount of MSFT |

**Same notional on both legs**: the router sizes the group as a whole.
- In `fixed` mode, the first leg takes `quantity` shares (10), and the second the same dollar amount, rounded to the share. The pair thus trades the price ratio, not the price of the more expensive stock.
- The other modes (`notional`, `equity_pct`, `risk`) split the group's notional equally.

**One group**: the two legs form a group of lots in the `OrderRouter` (`Strategy.group`).
- They pass the risk check **together or not at all**, so a pair is never unbalanced.
- They exit **together**:
  - when the z-score comes back within `±exit_zscore` (0.5; the strategy requests the exit);
  - after `holding_bars` (5);
  - or when the **combined P&L** reaches −`stop_loss_pct` (2 %) or +`take_profit_pct` (5 %) of the pair's gross notional, checked at each bar close.
- Each leg **also** carries a **wide protective stop** (`leg_stop_pct`, 5 %), which protects it if the application is down.
  - Both stops are mandatory. The combined stop is the real stop of the pair, because a leg stop cannot see the spread. For example, with A bought and B sold, if A drops 4.9 % and B rises 4.9 %, the pair loses 4.9 % without either leg stop firing.
  - When a leg stop fills, the other leg is closed at once.
- If a leg is not filled (rejected), the other leg is closed.

As long as the z-score stays beyond the threshold, a new pair is opened at every bar. The lots stack up, within the `RiskLimits` (`max_position_quantity`).

There is no ML model or target here: the rule is purely statistical. An ML version could predict the speed of convergence of the ratio, with a `ForwardTarget` computed on the ratio series.

## 4. Parameters and usage

| Parameter | Default | Example |
|---|---|---|
| `symbol_a`, `symbol_b` | AAPL, MSFT | the two `--symbol`, in order |
| `zscore_threshold` | 2.2 | `--param zscore_threshold=2.0` |
| `zscore_window` | 30 | `--param zscore_window=60` |
| `smoothing_window` | 3 | `--param smoothing_window=5` |
| `exit_zscore` | 0.5 | `--param exit_zscore=none` (no z-score exit) |
| `leg_stop_pct` | 5 (required, wider than the combined stop) | `--param leg_stop_pct=8` |
| `stop_loss_pct`, `take_profit_pct` | 2, 5 | in % of the pair's gross notional |
| `holding_bars` | 5 | `--param holding_bars=30` |
| `quantity`, `size_mode`, `size_value` | 10, fixed, none | `--param size_mode=notional --param size_value=20000` (gross notional of the pair) |

```bash
python -m quantlab backtest --strategy pair --symbol AAPL --symbol MSFT --days 55
python -m quantlab backtest --strategy pair --symbol AAPL --symbol MSFT --source file \
    --file tests/fixtures/aapl_msft_2m.parquet --test-fraction 0.3
```

## 5. Current results

On the regression fixture (`tests/fixtures/aapl_msft_2m.parquet`: 1,500 two-minute bars of training, 450 of test, $100,000 of capital):

| Trades (legs) | Win rate | Return | Profit factor | Max drawdown | Fees |
|---|---|---|---|---|---|
| 56 | 42.9 % | +0.108 % | 1.64 | −0.076 % | $36.82 |

The profit factor is above 1, but the sample (about 2 days) is too short to conclude anything.

## 6. Limitations and next steps

- **Dollar-neutral, not beta-neutral**: the legs hold the same notional, but not necessarily the same sensitivity to the market. A hedge ratio estimated by regression would address it.
- **Stability of the relationship**: test the pair for cointegration over the period (Engle-Granger) before trading it. A broken correlation turns the strategy into two directional bets.
- **Stacking**: limit the strategy to one open pair at a time, through `on_fill` and the strategy's state.
