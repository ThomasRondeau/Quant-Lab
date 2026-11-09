# Mean reversion: filtered reversal

`quantlab/strategies/mean_reversion.py` · CLI name: `mean_reversion`

The strategy fades temporary price excesses outside the Bollinger Bands. It enters only when several signs of an exhausted move line up.

## 1. Indicators

| Feature | Role |
|---|---|
| `BollingerBands(20, 2)` (`bb_upper_20_2`, `bb_lower_20_2`) | statistical envelope of the "normal" move |
| `ATR(14)` | volatility: measures how far the price leaves the bands |
| `RSI(14)` | overbought / oversold state |
| `VolumeRatio(20)` | current volume divided by its 20-bar average |

## 2. Decision rules

The price must **clearly break** the band, not just touch it:

- **buy** (10 shares by default) when all three conditions hold:
  - close < lower band − 0.2 × ATR;
  - RSI < `rsi_threshold` (30);
  - volume ratio > 1.2;
- **sell** when all three conditions hold:
  - close > upper band + 0.2 × ATR;
  - RSI > 100 − `rsi_threshold` (70);
  - volume ratio > 1.2.

The volume spike hints at a buying or selling climax.

By default the router closes the position: after 5 bars (`holding_bars`), or earlier at the −2 % stop loss or the +5 % take profit (`stop_loss_pct`, `take_profit_pct`). With `target_middle_band=true`, the target of each position is instead the Bollinger middle band at the signal: the mean the price is expected to revert to.

## 3. Parameters

| Parameter | Default | Example |
|---|---|---|
| `rsi_threshold` | 30 | `--param rsi_threshold=25` |
| `target_middle_band` | false | `--param target_middle_band=true` |
| `volume_threshold`, `band_atr_multiple` | 1.2, 0.2 | `--param volume_threshold=1.5` |
| `bb_window`, `bb_std`, `rsi_window`, `volume_window`, `band_atr_window` | 20, 2, 14, 20, 14 | `--param bb_window=30` |
| sizing (`size_mode`, `quantity`, `size_value`) | fixed, 10, none | `--param size_mode=risk --param size_value=0.5` |
| exits (`stop_loss_pct`, `take_profit_pct`, `stop_atr`, `holding_bars`…) | 2, 5, none, 5 | `--param stop_atr=1.5` |

Every parameter: `quantlab params --strategy mean_reversion`, or the form in the interface. The preset `presets/mean_reversion-middle-band.json` targets the middle band with stricter filters.

## 4. Current results

On the regression fixture (AAPL, 1-minute bars, 1,000 test bars, $100,000 of capital):

| Trades | Win rate | Return | Profit factor | Max drawdown | Fees |
|---|---|---|---|---|---|
| 21 | 42.9 % | −0.056 % | 0.35 | −0.08 % | $11.64 |

Execution happens at the next bar's open, with slippage: a close is only known once its bar is over. On a reversal strategy, this delay costs part of the rebound.

## 5. Analysis and next steps

- The strategy trades rarely (21 trades in about 2.5 days), so it is sensitive to the sample. It needs a walk-forward validation over several months of data.
- Ideas to try:
  - an exit at the middle band (`target_middle_band`);
  - limit orders at entry, to avoid paying the slippage;
  - optimizing the ATR multiple and the volume threshold.
- The **ML Hybrid** strategy ([ml_hybrid.md](ml_hybrid.md)) reuses this trigger and filters it with a direction model.
