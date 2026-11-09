# ML Hybrid: technical signal confirmed by a model

`quantlab/strategies/ml_hybrid.py` · CLI name: `ml_hybrid`

The strategy combines a mean-reversion trigger with an XGBoost model that predicts the price direction. It trades only when both agree.

## 1. Technical trigger

The trigger is close to the one of [mean reversion](mean_reversion.md), a little wider:

- **buy candidate** when all three conditions hold: close ≤ lower Bollinger band (20, 2), RSI(14) < 35, volume ratio (20) > 1.1;
- **sell candidate** when all three conditions hold: close ≥ upper band, RSI > 65, volume ratio > 1.1.

## 2. Model confirmation

- **Model**: an `XGBClassifier` wrapped in a `SklearnModel`:
  - 100 trees, `max_depth` = 4, `learning_rate` = 0.05;
  - `subsample` and `colsample_bytree` at 0.8;
  - a fixed seed.
- **Target**: `ForwardTarget(5, "direction")`, 1 when the close 5 bars ahead is higher.
- **Inputs**: the `DirectionFeatures` set, 10 `direction_*` columns normalized by the ATR so that they compare over time:
  - RSI;
  - MACD histogram / ATR;
  - distance to the 20-bar SMA / ATR, and the 3-bar slope of that SMA / ATR;
  - position within the bands;
  - signed volume;
  - ATR / price;
  - momentum over 5, 10 and 20 bars.
- **Rule**:
  - a buy candidate is executed (10 shares by default) only when the model predicts a rise;
  - a sell candidate only when it predicts a fall;
  - otherwise the signal is dropped.
- The prediction is journaled at every bar (`diagnostics = {"direction_up": …}`), so it can be analyzed after a session.

## 3. Training without look-ahead

- `strategy.fit(history)` fills the feature buffer, then trains the model on the whole training history.
- The last 30 bars are excluded from the training set (`training_margin` = 30, for a horizon of 5). No label uses prices after the end of the history.
- Labels still unknown at the end of the history are NaN and dropped.
- Exponential averages (MACD) are computed with a full warmup, and the live buffer starts with no gap after training.

## 4. Parameters

| Parameter | Default | Example |
|---|---|---|
| `max_depth` | 4 | `--param max_depth=3` |
| `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `random_state` | 100, 0.05, 0.8, 0.8, 42 | `--param learning_rate=0.03` |
| `rsi_buy`, `rsi_sell`, `volume_threshold` | 35, 65, 1.1 | `--param rsi_buy=30 --param rsi_sell=70` |
| `horizon`, `training_margin` | 5, 30 | `--param horizon=10 --param training_margin=40` |
| indicator and composite-feature windows | RSI 14, Bollinger 20 / 2, volume 20, MACD 12/26/9… | `--param rsi_window=21` |
| sizing (`size_mode`, `quantity`, `size_value`) | fixed, 10, none | `--param size_mode=risk --param size_value=0.5` |

Every parameter: `quantlab params --strategy ml_hybrid`, or the form in the interface.

## 5. Current results

On the regression fixture (AAPL, 1-minute bars, 4,000 training bars and 1,000 test bars, $100,000 of capital):

| Trades | Win rate | Return | Profit factor | Max drawdown | Fees |
|---|---|---|---|---|---|
| 31 | 38.7 % | −0.125 % | 0.26 | −0.16 % | $17.18 |

## 6. Analysis and next steps

- The model is trained on **all** bars, but it is only consulted on the rare bars where the technical trigger fires. Training it on reversal situations only, or weighting them, would match its use better.
- Compare the in-sample score with the time-ordered cross-validated score (`model.cross_validate`) to detect overfitting, then validate by walk-forward over several months.
- `max_depth` lends itself to `optimize`: `--range max_depth=2:6:1`.
