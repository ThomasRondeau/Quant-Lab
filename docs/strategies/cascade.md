# Cascade: three models in a chain

`quantlab/strategies/cascade.py` · CLI name: `cascade`

The most elaborate strategy. Three specialized models answer three questions in turn, and a trade is taken only when the three answers agree.

## 1. The three models

All three models look at the same horizon of **5 bars**.

| Step | Question | Model | Target (`ForwardTarget`) | Score |
|---|---|---|---|---|
| 1. Opportunity | will a significant move happen? | `RandomForestClassifier`: 100 trees, depth 5, ≥ 15 samples per leaf, balanced classes | `move`: \|5-bar return\| > 0.05 % | balanced accuracy |
| 2. Direction | up or down? | `XGBClassifier`: 100 trees, depth 4, learning rate 0.05, subsample 0.8 | `direction` | accuracy |
| 3. Amplitude | by how much, in %? | `RandomForestRegressor`: 100 trees, depth 5, ≥ 10 samples per leaf | `return_pct` (signed) | R² |

The shallow trees and the minimum samples per leaf are regularization choices. Early versions overfitted: a score of 1.0 in training, about 0.5 in validation.

## 2. Features

Each model has its own composite feature set, built on **shared** indicators computed once: ATR(14), SMA(5), SMA(20), Bollinger(20, 2), RSI(14), MACD(12, 26, 9) and the volume ratio (14).

- **`opportunity_*`** (10 columns):
  - band width;
  - volatility regime (ATR over its 20-bar mean) and the change of the ATR;
  - volume ratio and volume trend;
  - momentum alignment (MACD, RSI, fast SMA);
  - distances to the bands and to the SMA;
  - price acceleration.
- **`direction_*`** (10 columns): RSI, MACD and distances normalized by the ATR, SMA slope, position within the bands, signed volume, momentum over 5, 10 and 20 bars.
- **`amplitude_*`** (7 columns): relative volatility, distance to the SMA, band width, RSI distance to 50, \|MACD histogram\|, \|5-bar momentum\|, volume ratio.

Most of these columns are unitless (relative to the price or to the ATR), so they compare from one period to the next. Every window is a parameter, and the column names follow the windows.

## 3. Decision rule

1. The opportunity model must predict 1; otherwise, nothing.
2. Read the direction (up or down) and the predicted amplitude, in %.
3. Require \|amplitude\| ≥ `min_amplitude_threshold`, 0.05 % by default, in the same unit as the targets.
4. The sign of the amplitude must agree with the direction.
5. When everything agrees, the strategy buys (up) or sells (down).
   - The conviction is min(|amplitude| / `conviction_cap_pct`, 1), with a cap of 5 % by default.
   - The router turns it into a size. In `fixed` mode, that is `max(1, int(conviction × quantity))` shares, with `quantity` = 10 by default.

The three predictions are journaled at every bar (`opportunity`, `direction_up`, `amplitude_pct`). The router handles the exit: after 5 bars, or at the −2 % stop loss or the +5 % take profit.

## 4. Training and validation

- **Training**: on the history, excluding the last 30 bars (`training_margin`, for a horizon of 5), to rule out any look-ahead.
- **Overfitting check**: `strategy.evaluate()` reports, for each model, the in-sample score and the time-ordered cross-validated score (`cv_folds`).
- **Latency**: this is the heaviest strategy (features, then three predictions). Single-bar predictions run on one thread. The latency test requires less than 100 ms per bar at the median, and it runs around 50 ms.

## 5. Parameters

Cascade exposes 47 parameters.

| Parameter | Default | Example |
|---|---|---|
| `min_amplitude_threshold` | 0.05 (%) | `--param min_amplitude_threshold=0.1` |
| `conviction_cap_pct` | 5 (%) | `--param conviction_cap_pct=2` |
| hyperparameters (`amplitude_*`, `direction_*`, `opportunity_*`), `random_state` | see §1 | `--param direction_max_depth=3` |
| target (`horizon`, `opportunity_threshold_pct`, `training_margin`, `cv_folds`) | 5, 0.05, 30, 5 | `--param horizon=10 --param training_margin=40` |
| indicator and composite-feature windows | see §2 | `--param momentum_windows=3,7,30` |
| sizing (`size_mode`, `quantity`, `size_value`) | fixed, 10, none | `--param size_mode=risk --param size_value=0.5` |

Every parameter: `quantlab params --strategy cascade`, or the form in the interface.

Changing `horizon` without `holding_bars` raises a warning: positions would not be held for the period the models predict. The preset `presets/cascade-horizon-10.json` changes both.

## 6. Current results

On the regression fixture (AAPL, 1-minute bars, 4,000 training bars and 1,000 test bars, $100,000 of capital):

| Trades | Win rate | Return | Profit factor | Max drawdown | Fees |
|---|---|---|---|---|---|
| 25 | 52.0 % | +0.0014 % | 1.22 | −0.005 % | $1.39 |

It is the only single-symbol strategy with a profit factor above 1 on this fixture. But the sample (25 trades in about 2.5 days) is far too small to conclude anything.

## 7. Analysis and next steps

- **Size**: in `fixed` mode, the predicted amplitude must reach at least 1 % to go above 1 share. Intraday, the strategy therefore almost always trades **1 share**. A `risk` or `equity_pct` sizing mode gives meaningful positions.
- **Opportunity threshold**: 0.05 % is very low for 1-minute bars. It does not cover the round-trip costs (about 4 bp with fees and slippage). Raise it, or express it as a multiple of the costs.
- **Next**: validate by walk-forward over several months, then measure the gap between live and backtest with `quantlab replay`.
