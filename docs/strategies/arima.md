# ARIMA: very short-term statistical forecast

`quantlab/strategies/arima.py` (strategy) · `quantlab/models/arima.py` (`ArimaForecaster`) · CLI name: `arima`

The strategy exploits the short-term autocorrelation of prices with a classic time-series model, refitted at every bar.

## 1. Model

- **ARIMA(p, d, q)**, (2, 1, 0) by default, through `statsmodels`.
- **Rolling forecast**: at each bar, the model is refitted on the last `window` closes (50 by default) and forecasts the change of the next close.
- The raw bars come from `self.feed.bars(window)`. The strategy therefore declares `min_history = window`, so that the live buffer keeps enough history.
- If the fit fails to converge, the forecast is 0: no trade, and the engine keeps running.

## 2. Decision rule

The strategy trades only when the forecast move **covers the costs**:

- if \|forecast change\| / price < `min_edge_bps` (2 bp by default, the round-trip fees of 2 × 1 bp), it does nothing;
- otherwise it buys on an upward forecast and sells on a downward one. The size is 5 shares by default (`quantity`), or another sizing mode.

The forecast is journaled at every bar (`diagnostics = {"forecast_change": …}`). The router handles the exit: after 5 bars, or at the −2 % stop loss or the +5 % take profit.

## 3. Parameters

| Parameter | Default | Example |
|---|---|---|
| `order` | (2, 1, 0) | `--param order=1,1,1` |
| `min_edge_bps` | 2.0 | `--param min_edge_bps=4` |
| `window` | 50 | `--param window=100` |
| sizing (`size_mode`, `quantity`, `size_value`) | fixed, 5, none | `--param size_mode=risk --param size_value=0.5` |
| exits (`stop_loss_pct`, `take_profit_pct`, `stop_atr`, `holding_bars`…) | 2, 5, none, 5 | `--param holding_bars=3` |

The strategy has 13 parameters: `quantlab params --strategy arima` lists them, and the interface shows them as a form. The preset `presets/arima-slow.json` uses the order (1, 1, 1), a window of 100 closes and a minimum edge of 4 bp.

The `order` is a tuple, so it cannot be searched by `optimize` or by the optimization page; `min_edge_bps` and `window` can.

## 4. Current results

On the regression fixture (`tests/fixtures/aapl_1m.parquet`: 1-minute AAPL bars, 4,000 training bars, $100,000 of capital). Only 200 test bars are used, about three and a half hours, because refitting at every bar is slow. They fall within one session, so the Sharpe ratio of this run is 0:

| Trades | Win rate | Return | Profit factor | Max drawdown | Fees |
|---|---|---|---|---|---|
| 18 | 44.4 % | ≈ 0 % | 1.00 | −0.007 % | $4.97 |

The strategy just breaks even once the fees (1 bp per side) and the simulated slippage (1 bp) are paid.

## 5. Computational cost

Refitting an ARIMA at every bar is expensive. It is fine for 1-minute bars live, but it makes backtests long. For that reason the online demo tests ARIMA on 200 bars at most, and does not optimize it. For large-scale research, consider:
- a Kalman filter;
- less frequent refits, every N bars, checking the effect on the signals.

## 6. Next steps

- Require an edge that also covers the slippage, for example `min_edge_bps=4`.
- Choose the (p, d, q) order and the window by walk-forward rather than fixing them.
- Combine the forecast with a regime filter: short-term autocorrelation changes with volatility and the time of day.
