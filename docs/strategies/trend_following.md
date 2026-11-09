# Trend following: moving-average crossover

`quantlab/strategies/trend_following.py` · CLI name: `trend`

A classic trend-following approach: take a position in the direction of the crossover of two simple moving averages.

## 1. Indicators

| Feature | Role |
|---|---|
| `SMA(sma_fast)`, `sma_5` by default | fast average, reacts to recent moves |
| `SMA(sma_slow)`, `sma_20` by default | slow average, the short-term underlying trend |

Both averages are computed at each bar on the live buffer, or precomputed in a backtest, with the same values. The declared warmup is `sma_slow − 1` bars, and the strategy decides nothing before that history is available.

## 2. Decision rules

The strategy compares the last two feature rows (`self.feed.latest(2)`):

- **buy** (10 shares by default) when the fast SMA crosses **above** the slow SMA (it was below or equal on the previous bar);
- **sell** when it crosses **below**.

The strategy only emits entries. As for every strategy, the router handles the exit:

- at market after `holding_bars` = 5 bars;
- or earlier, by the −2 % stop loss or the +5 % take profit placed as soon as the entry fills.

See [ARCHITECTURE.md §7](../ARCHITECTURE.md).

## 3. Parameters

| Parameter | Default | Example |
|---|---|---|
| `sma_fast` | 5 | `--param sma_fast=10` |
| `sma_slow` | 20 | `--param sma_slow=50` |
| sizing (`size_mode`, `quantity`, `size_value`) | fixed, 10, none | `--param size_mode=risk --param size_value=0.5` |

Every parameter (indicators, sizing, exits): `quantlab params --strategy trend`, or the form in the interface.

```bash
python -m quantlab backtest --strategy trend --symbol AAPL --days 55 --test-days 5
python -m quantlab optimize --strategy trend --symbol AAPL --range sma_fast=5:20:5 --grid sma_slow=30,50,100 \
    --train-size 3000 --validation-size 1000 --test-size 1000 --embargo 30
```

## 4. Current results

On the regression fixture (`tests/fixtures/aapl_1m.parquet`: 4,000 one-minute bars of training, 1,000 of test, $100,000 of capital):

| Trades | Win rate | Return | Profit factor | Max drawdown | Fees |
|---|---|---|---|---|---|
| 72 | 19.4 % | −0.096 % | 0.36 | −0.10 % | $39.86 |

This fixture detects regressions; it does not measure performance, since it covers only about 2.5 days.

## 5. Analysis and next steps

- On 1- to 2-minute bars, a crossover of short averages mostly reacts to noise. It gives many false signals, and the fees (1 bp per side) and slippage (1 bp) weigh heavily.
- An ADX feature (`ADX(14)`) is available to keep only strong trends.
- Ideas to try:
  - longer horizons (15-minute or 1-hour bars, adjusting `holding_bars`);
  - a trend-regime filter;
  - an exit on the opposite crossover (`close_positions`) rather than after a fixed number of bars.

  Evaluate each variant with `optimize` (walk-forward) rather than on a single split.
